"""
Build the ASSAM pilot TRAINING MATRIX (features only -- this script NEVER trains a
model). It is the Assam analogue of the feature-building portion of
scripts/train_real_models.py (Sikkim), with exactly ONE deliberate substitution:
the degenerate elevation land_cover_class proxy is replaced by the REAL, categorical
ESA WorldCover land cover produced by scripts/prepare_assam_landcover.py.

FEATURE SCHEMA (11 model features, identical layout to the Sikkim trainer):
    5 terrain (float32) : elevation, slope, aspect, roughness, tpi
        -> sampled from the on-disk Assam rasters (assam_pilot_dem.tif +
           assam_pilot_{slope,aspect,roughness,tpi}.tif) with the SAME
           scripts/train_real_models.sample_rasters_at_points helper (nearest).
    1 land cover (int32, CATEGORICAL) : land_cover_class
        -> app.services.worldcover.assign_assam_land_cover_from_raster (grouped,
           nominal, NEVER ordinal). Declared categorical via
           worldcover.landcover_categorical_feature() so a future trainer passes it
           to LightGBM's categorical_feature=. This REPLACES assign_land_cover_proxy.
    5 rainfall (float32) : rain_1d, rain_3d, rain_7d, antecedent_rain_14d,
                           rain_intensity_max_3d
        -> the SAME scripts/train_real_models.fetch_historical_rainfall_series
           (Open-Meteo ERA5 archive, daily precipitation_sum, strictly T-14..T-1).
           This is the identical rainfall source Sikkim TRAINING already uses;
           NASA IMERG (app.services.weather_ingestion) is the SERVING/real-time path
           and is NOT a training feature -- it is left completely untouched.
           Requests use LOW concurrency + exponential backoff on HTTP 429 ("Too many
           concurrent requests") via fetch_rainfall_with_backoff below; that only
           spaces out identical retries and NEVER alters the fetched values, the
           antecedent dates, or the source, and never substitutes/fills a value.

SAMPLES: all real GLC positives from data/models/assam_events.json (target=1) plus
spatially-buffered negatives from app.models.ml_pipeline.generate_spatial_negative_samples
(target=0, 3:1, >= 0.05 deg from any positive, seed 42), drawn from exactly the
canonical Assam AOI rectangle (== the DEM/land-cover extent). No sample is dropped by
this builder: every positive and every negative is kept.

DATA-INTEGRITY CONTRACT (nothing is fabricated or filled):
    * Terrain nodata (-9999) and any non-finite terrain sample -> set to NaN and
      counted as MISSING. Never zero/mean/interpolated.
    * Land cover nodata / outside-coverage / unknown code -> UNAVAILABLE_SENTINEL
      (-1) and counted as MISSING. Never assigned a real class.
    * Rainfall: on ANY failure to obtain a complete real antecedent series for ANY
      sample, RainfallUnavailableError aborts the whole build and NO matrix artifact
      is written (mirrors the Sikkim trainer). Missing rainfall never becomes 0 mm.
    A row with any missing feature is retained here (so the matrix is a faithful,
    auditable record) but the written schema sidecar reports exactly how many such
    rows exist; a downstream trainer must DROP them, never fill them.

HOST-ONLY: needs rasterio (to read the rasters) and outbound network (Open-Meteo).
Neither exists in the offline sandbox, so this does not run there; it still imports
and py_compiles offline because every heavy/host-only dependency is imported lazily
inside main(). It does NOT train, does NOT commit, and touches neither the Sikkim
pipeline, IMERG, nor the frontend.

Usage (from backend/ on the host):
    python scripts/build_assam_training_matrix.py
    python scripts/build_assam_training_matrix.py --max-workers 8
"""
import argparse
import datetime as _dt
import json
import os
import random
import sys
import time

# Only pure-stdlib config/service helpers are imported at module load so this file
# compiles/imports offline. rasterio, requests (network), sklearn (ml_pipeline) and
# the Sikkim trainer's rainfall/terrain helpers are imported lazily inside main().
_BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
for _p in (_BACKEND_DIR, _SCRIPTS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from app.core.config_states import (  # noqa: E402  (path set up above)
    aoi_bounds_tuple,
    assert_pilot_aoi_consistency,
    bbox_contains,
    get_pilot_aoi_bounds,
)
from app.services import worldcover as wc  # noqa: E402

STATE_NAME = "Assam"
PILOT_AREA = "Guwahati-Kamrup + western Karbi Anglong"

EVENTS_FILENAME = "assam_events.json"
DEM_FILENAME = "assam_pilot_dem.tif"
LANDCOVER_FILENAME = "assam_pilot_landcover.tif"
TERRAIN_DERIVATIVE_FILENAMES = {
    "slope": "assam_pilot_slope.tif",
    "aspect": "assam_pilot_aspect.tif",
    "roughness": "assam_pilot_roughness.tif",
    "tpi": "assam_pilot_tpi.tif",
}
MATRIX_FILENAME = "assam_pilot_training_matrix.parquet"
SCHEMA_FILENAME = "assam_pilot_training_matrix_schema.json"

# Feature layout -- identical order/semantics to the Sikkim trainer's
# static_features + rainfall block, so an Assam trainer can reuse the same code.
TERRAIN_FEATURES = ["elevation", "slope", "aspect", "roughness", "tpi"]
RAINFALL_FEATURES = ["rain_1d", "rain_3d", "rain_7d", "antecedent_rain_14d", "rain_intensity_max_3d"]
STATIC_FEATURES = TERRAIN_FEATURES + [wc.LANDCOVER_FEATURE_NAME]        # 6 static
ALL_FEATURES = STATIC_FEATURES + RAINFALL_FEATURES                     # 11 total

# Terrain nodata sentinel written by the terrain-derivative rasters (GDAL_NODATA).
# The DEM itself carries no nodata; both are still guarded for non-finite values.
TERRAIN_NODATA = -9999.0

# 3:1 spatially-buffered negatives, >= 0.05 deg (~5 km) from any positive, seed 42.
# Restated only so the report is self-describing; the actual values come from
# app.models.ml_pipeline.generate_spatial_negative_samples.
NEGATIVE_COUNT_RATIO = 3
NEGATIVE_BUFFER_DEG = 0.05

# --------------------------------------------------------------------------- #
# Polite Open-Meteo fetching (concurrency + HTTP-429 backoff).
#
# The reused scripts/train_real_models.fetch_historical_rainfall_series makes one
# archive-api.open-meteo.com request per sample. Fetching all 236 samples with high
# parallelism makes Open-Meteo answer HTTP 429 "Too many concurrent requests". Two
# measures fix that WITHOUT changing the request, the computed values, the antecedent
# dates, or the source:
#   1. a deliberately LOW default worker count (below), and
#   2. an exponential-backoff-with-jitter retry that fires ONLY on HTTP 429.
# Everything else -- other HTTP errors, network errors, a genuinely incomplete real
# series -- is re-raised immediately (never retried, never filled), so the build
# still aborts fast and no matrix is written on real data loss. A 429 that persists
# past the retry budget is also re-raised (never substituted with a fabricated value).
RAINFALL_DEFAULT_MAX_WORKERS = 2      # was 8; low enough to avoid concurrent-429s
RAINFALL_MAX_RETRIES = 5              # retries AFTER the first 429, per sample
RAINFALL_BACKOFF_BASE_S = 2.0         # 2s, 4s, 8s, 16s, 32s (full-jitter upper bounds)
RAINFALL_BACKOFF_CAP_S = 60.0         # never wait more than a minute between tries
# fetch_historical_rainfall_series stamps the HTTP status into its error message
# ("... request returned HTTP 429 ..."), so a 429 is detectable here WITHOUT touching
# that (Sikkim) function.
_HTTP_429_MARKER = "HTTP 429"


def _is_http_429(err):
    """True iff this rainfall error was caused by an Open-Meteo HTTP 429 response.
    Detected from the status the reused fetcher stamps into its message; only these
    (concurrency/rate-limit) failures are worth retrying."""
    return _HTTP_429_MARKER in str(err)


def fetch_rainfall_with_backoff(fetch_fn, lat, lon, event_date, retryable_error_type,
                                max_retries=RAINFALL_MAX_RETRIES,
                                base_delay=RAINFALL_BACKOFF_BASE_S,
                                cap_delay=RAINFALL_BACKOFF_CAP_S,
                                sleep=None, jitter_uniform=None):
    """
    Call the REUSED Open-Meteo fetcher and return its result COMPLETELY UNCHANGED.
    The only added behaviour: on an HTTP 429 ("Too many concurrent requests"), wait
    with full-jitter exponential backoff and retry the identical request. This never
    alters the fetched rainfall values, the antecedent window/dates, or the source,
    and never substitutes or fills a value.

    * Non-429 failures (other HTTP codes, network/timeout, incomplete real series)
      are re-raised immediately -- the build aborts fast, exactly as before.
    * A 429 that persists past `max_retries` is also re-raised (never filled).

    `sleep` and `jitter_uniform` are injectable purely so the retry logic can be
    unit-tested offline without real waiting; production uses time.sleep /
    random.uniform.
    """
    _sleep = sleep if sleep is not None else time.sleep
    _uniform = jitter_uniform if jitter_uniform is not None else random.uniform
    attempt = 0
    while True:
        try:
            return fetch_fn(lat, lon, event_date)
        except retryable_error_type as err:
            if not _is_http_429(err) or attempt >= max_retries:
                raise
            delay = min(cap_delay, base_delay * (2 ** attempt))
            _sleep(_uniform(0.0, delay))
            attempt += 1


def load_positive_events(models_dir):
    """
    Load the real Assam GLC positives from the committed events snapshot into a
    DataFrame with latitude/longitude/event_date (datetime) and target=1. All event
    metadata columns are preserved for provenance; they are NOT model features.
    """
    import pandas as pd

    path = os.path.join(models_dir, EVENTS_FILENAME)
    if not os.path.exists(path):
        raise SystemExit("Assam events snapshot not found at %s." % path)
    with open(path, "r", encoding="utf-8") as fh:
        doc = json.load(fh)
    events = doc.get("events", [])
    if not events:
        raise SystemExit("Assam events snapshot at %s contains zero events." % path)
    pos_df = pd.DataFrame(events)
    pos_df["latitude"] = pos_df["latitude"].astype(float)
    pos_df["longitude"] = pos_df["longitude"].astype(float)
    pos_df["event_date"] = pd.to_datetime(pos_df["event_date"], errors="coerce")
    n_bad_date = int(pos_df["event_date"].isna().sum())
    if n_bad_date:
        # A positive with an unparseable date has no valid antecedent window; that is
        # a data problem to surface, not to silently drop or paper over.
        raise SystemExit(
            "%d Assam positive event(s) have an unparseable event_date; refusing to "
            "build a matrix with an undefined antecedent rainfall window." % n_bad_date
        )
    pos_df["target"] = 1
    return pos_df


def terrain_raster_map(raw_dir, proc_dir):
    """Map feature name -> raster path for the 5 terrain features; assert each exists."""
    raster_map = {"elevation": os.path.join(raw_dir, DEM_FILENAME)}
    for feat, fname in TERRAIN_DERIVATIVE_FILENAMES.items():
        raster_map[feat] = os.path.join(proc_dir, fname)
    missing = [name for name, p in raster_map.items() if not os.path.exists(p)]
    if missing:
        raise SystemExit(
            "Missing terrain raster(s) for %s: %s. Run scripts/prepare_assam_terrain.py first."
            % (", ".join(missing), [raster_map[m] for m in missing])
        )
    return raster_map


def mask_terrain_missing(full_df):
    """
    Convert terrain nodata (-9999) and non-finite terrain values to NaN so they are
    honestly MISSING rather than a fake numeric feature, and return per-feature
    missing counts. Never fills.
    """
    import numpy as np

    missing_counts = {}
    for feat in TERRAIN_FEATURES:
        col = full_df[feat].astype(np.float32)
        bad = (~np.isfinite(col)) | (np.isclose(col, TERRAIN_NODATA))
        col = col.mask(bad)  # set bad entries to NaN
        full_df[feat] = col.astype(np.float32)
        missing_counts[feat] = int(bad.sum())
    return missing_counts


def _feature_missing_report(full_df, landcover_unavailable):
    """Per-feature missing counts + rows-with-any-missing-feature, for the sidecar."""
    import numpy as np

    report = {}
    any_missing = np.zeros(len(full_df), dtype=bool)
    for feat in ALL_FEATURES:
        if feat == wc.LANDCOVER_FEATURE_NAME:
            bad = (full_df[feat].values == wc.UNAVAILABLE_SENTINEL)
        else:
            bad = ~np.isfinite(full_df[feat].astype("float64").values)
        report[feat] = int(bad.sum())
        any_missing |= bad
    report["_rows_with_any_missing_feature"] = int(any_missing.sum())
    report["_landcover_unavailable_rows"] = int(landcover_unavailable)
    return report


def write_schema_json(out_path, document):
    """Write the schema/missing sidecar as CRLF/2-space JSON, matching data/models/."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="\r\n") as fh:
        json.dump(document, fh, indent=2, ensure_ascii=False)
        fh.write("\n")


def main():
    parser = argparse.ArgumentParser(
        description="Build the Assam pilot training feature matrix (no model training)."
    )
    parser.add_argument(
        "--max-workers", type=int, default=RAINFALL_DEFAULT_MAX_WORKERS,
        help="Parallel workers for the Open-Meteo antecedent-rainfall fetch (default "
             "%d, deliberately low: Open-Meteo returns HTTP 429 'Too many concurrent "
             "requests' under high parallelism, which is then retried with backoff). "
             "Set to 1 for fully serial fetching." % RAINFALL_DEFAULT_MAX_WORKERS,
    )
    args = parser.parse_args()

    # Heavy / host-only dependencies imported here so the module still imports offline.
    import numpy as np
    import pandas as pd
    from concurrent.futures import ThreadPoolExecutor, as_completed

    from app.models.ml_pipeline import generate_spatial_negative_samples
    # Reuse the Sikkim trainer's terrain sampler + Open-Meteo rainfall fetch verbatim
    # (single definition site); importing the module runs no pipeline (guarded by
    # __main__) and defines no Assam-specific behaviour of its own.
    import train_real_models as trm

    raw_dir = os.path.join(_BACKEND_DIR, "data", "raw")
    proc_dir = os.path.join(_BACKEND_DIR, "data", "processed")
    models_dir = os.path.join(_BACKEND_DIR, "data", "models")
    os.makedirs(proc_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    # 0. Canonical AOI (fail fast if the pilot AOI is not inside Assam's admin bbox).
    consistency = assert_pilot_aoi_consistency(STATE_NAME)
    aoi = get_pilot_aoi_bounds(STATE_NAME)
    landcover_path = os.path.join(raw_dir, LANDCOVER_FILENAME)
    if not os.path.exists(landcover_path):
        raise SystemExit(
            "WorldCover raster not found at %s. Run scripts/prepare_assam_landcover.py first."
            % landcover_path
        )
    raster_map = terrain_raster_map(raw_dir, proc_dir)

    print("=" * 72)
    print(" ASSAM PILOT TRAINING MATRIX (features only; NO training)")
    print(" state=%s  pilot_area=%s" % (STATE_NAME, PILOT_AREA))
    print(" canonical AOI      : %s" % consistency["pilot_aoi"])
    print(" pilot within state : %s" % consistency["pilot_within_state"])
    print("=" * 72)

    # 1. Real positives + spatially-buffered negatives (kept in full).
    pos_df = load_positive_events(models_dir)
    n_pos = len(pos_df)
    neg_df = generate_spatial_negative_samples(
        pos_df, dict(aoi), count_ratio=NEGATIVE_COUNT_RATIO, buffer_deg=NEGATIVE_BUFFER_DEG
    )
    n_neg = len(neg_df)
    full_df = pd.concat([pos_df, neg_df], ignore_index=True)
    print("\n[1] samples: %d positives + %d negatives = %d rows" % (n_pos, n_neg, len(full_df)))

    # 2. Terrain (reused sampler) -> nodata/non-finite to NaN (never filled).
    full_df = trm.sample_rasters_at_points(full_df, raster_map)
    terrain_missing = mask_terrain_missing(full_df)
    print("[2] terrain sampled from 5 rasters; missing (nodata/non-finite): %s" % terrain_missing)

    # 3. Land cover -> REAL WorldCover, grouped, categorical (replaces elevation proxy).
    full_df, lc_unavailable_mask = wc.assign_assam_land_cover_from_raster(full_df, landcover_path)
    n_lc_unavailable = int(lc_unavailable_mask.sum())
    print("[3] land_cover_class from WorldCover (categorical, grouped); UNAVAILABLE rows: %d"
          % n_lc_unavailable)

    # 4. Rainfall -> Open-Meteo ERA5 antecedent series (reused). Abort, never fill.
    #    Kept polite: low concurrency (args.max_workers) + exponential backoff that
    #    retries ONLY on HTTP 429 ("Too many concurrent requests"). The fetched
    #    values, antecedent dates, and source are never altered by the retry.
    print("[4] fetching Open-Meteo ERA5 antecedent rainfall for %d samples "
          "(T-%d..T-1) with max_workers=%d + HTTP-429 backoff..."
          % (len(full_df), trm.ANTECEDENT_WINDOW_DAYS, args.max_workers))

    def fetch_row_rain(idx_row):
        idx, row = idx_row
        rf = fetch_rainfall_with_backoff(
            trm.fetch_historical_rainfall_series,
            row["latitude"], row["longitude"], row["event_date"],
            trm.RainfallUnavailableError,
        )
        return idx, rf

    rain_results = [None] * len(full_df)
    try:
        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            futures = [executor.submit(fetch_row_rain, (idx, row)) for idx, row in full_df.iterrows()]
            for future in as_completed(futures):
                idx, rf = future.result()
                rain_results[idx] = rf
    except trm.RainfallUnavailableError as e:
        # Mirror the Sikkim trainer: do NOT fabricate rainfall and do NOT persist a
        # matrix built on incomplete rainfall. No artifact is written.
        raise trm.RainfallUnavailableError(
            "ABORTING: real Open-Meteo ERA5 antecedent rainfall is unavailable for one "
            "or more Assam samples, so a scientifically valid training matrix cannot be "
            "built. No zero/synthetic rainfall is substituted and no matrix artifact is "
            "written. First failure: %s" % e
        ) from e

    rain_df = pd.DataFrame(rain_results)
    for col in RAINFALL_FEATURES:
        full_df[col] = rain_df[col].astype(np.float32)

    # 5. dtypes: terrain + rainfall float32; land_cover_class int32 (categorical); target int.
    for c in TERRAIN_FEATURES + RAINFALL_FEATURES:
        full_df[c] = full_df[c].astype(np.float32)
    full_df[wc.LANDCOVER_FEATURE_NAME] = full_df[wc.LANDCOVER_FEATURE_NAME].astype(np.int32)
    full_df["target"] = full_df["target"].astype(np.int64)

    # 6. Persist the matrix (all columns; metadata kept for provenance) + schema sidecar.
    matrix_path = os.path.join(proc_dir, MATRIX_FILENAME)
    full_df.to_parquet(matrix_path)

    missing_report = _feature_missing_report(full_df, n_lc_unavailable)
    dtypes = {feat: str(full_df[feat].dtype) for feat in ALL_FEATURES}
    generated_at = _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    document = {
        "schema_version": "1.0.0",
        "state": STATE_NAME,
        "pilot_area": PILOT_AREA,
        "aoi": aoi,
        "matrix": {
            "path": "backend/data/processed/%s" % MATRIX_FILENAME,
            "n_rows": int(len(full_df)),
            "n_positives": int(n_pos),
            "n_negatives": int(n_neg),
            "n_model_features": len(ALL_FEATURES),
            "n_columns_total": int(len(full_df.columns)),
        },
        "feature_schema": {
            "terrain_features": TERRAIN_FEATURES,
            "land_cover_feature": wc.LANDCOVER_FEATURE_NAME,
            "rainfall_features": RAINFALL_FEATURES,
            "ordered_model_features": ALL_FEATURES,
            "dtypes": dtypes,
            "categorical_features": wc.landcover_categorical_feature(),
            "target_column": "target",
        },
        "missing_values": missing_report,
        "sources": {
            "terrain": {
                "dem": "backend/data/raw/%s" % DEM_FILENAME,
                "derivatives": {k: "backend/data/processed/%s" % v
                                for k, v in TERRAIN_DERIVATIVE_FILENAMES.items()},
                "sampler": "scripts/train_real_models.sample_rasters_at_points (nearest)",
                "nodata_sentinel": TERRAIN_NODATA,
            },
            "land_cover": {
                "product": "ESA WorldCover %s %d (grouped, NOMINAL, categorical)"
                            % (wc.WORLDCOVER_VERSION, wc.WORLDCOVER_YEAR),
                "raster": "backend/data/raw/%s" % LANDCOVER_FILENAME,
                "assigner": "app.services.worldcover.assign_assam_land_cover_from_raster",
                "unavailable_sentinel": wc.UNAVAILABLE_SENTINEL,
                "group_labels": {str(k): v for k, v in wc.ASSAM_LANDCOVER_GROUP_LABELS.items()},
            },
            "rainfall": {
                "product": "Open-Meteo ERA5 archive (daily precipitation_sum)",
                "endpoint": "https://archive-api.open-meteo.com/v1/archive",
                "antecedent_window_days": trm.ANTECEDENT_WINDOW_DAYS,
                "window": "strictly T-%d .. T-1 (no future leakage)" % trm.ANTECEDENT_WINDOW_DAYS,
                "fetcher": "scripts/train_real_models.fetch_historical_rainfall_series",
                "on_failure": "RainfallUnavailableError aborts the build; never zero/synthetic",
                "imerg_note": "NASA IMERG (app.services.weather_ingestion) is the serving path only; NOT a training feature; untouched",
            },
        },
        "negative_sampling": {
            "method": "app.models.ml_pipeline.generate_spatial_negative_samples",
            "count_ratio": NEGATIVE_COUNT_RATIO,
            "buffer_deg": NEGATIVE_BUFFER_DEG,
            "seed": 42,
            "domain": "canonical Assam AOI rectangle (== DEM/land-cover extent)",
        },
        "integrity": {
            "fabrication": "none: terrain nodata/non-finite -> NaN; land cover nodata -> sentinel; rainfall failure aborts",
            "rows_retained": "all positives + all negatives kept; downstream trainer must DROP missing-feature rows, never fill",
            "land_cover_is_categorical": wc.LANDCOVER_IS_CATEGORICAL,
        },
        "aoi_consistency": consistency,
        "generated_at": generated_at,
    }
    schema_path = os.path.join(models_dir, SCHEMA_FILENAME)
    write_schema_json(schema_path, document)

    # 7. Report.
    print("\n--- FEATURE SCHEMA (11 model features) ---")
    for feat in ALL_FEATURES:
        kind = "categorical" if feat == wc.LANDCOVER_FEATURE_NAME else "numeric"
        print("    %-22s %-8s %s" % (feat, dtypes[feat], kind))
    print("    target column          : target (int64)")
    print("    categorical_feature    : %s" % wc.landcover_categorical_feature())
    print("\n--- MISSING VALUES (per model feature) ---")
    for feat in ALL_FEATURES:
        print("    %-22s %d" % (feat, missing_report[feat]))
    print("    rows with ANY missing feature : %d" % missing_report["_rows_with_any_missing_feature"])
    print("\n--- RAINFALL AVAILABILITY ---")
    print("    source: Open-Meteo ERA5 archive (daily precipitation_sum), T-%d..T-1"
          % trm.ANTECEDENT_WINDOW_DAYS)
    print("    status: COMPLETE real antecedent series retrieved for all %d samples "
          "(any failure would have aborted before this point)." % len(full_df))
    print("\n--- FINAL MATRIX SIZE ---")
    print("    rows           : %d (%d positives + %d negatives)" % (len(full_df), n_pos, n_neg))
    print("    model features : %d (%d terrain + 1 land cover + %d rainfall)"
          % (len(ALL_FEATURES), len(TERRAIN_FEATURES), len(RAINFALL_FEATURES)))
    print("    total columns  : %d (incl. target + event metadata)" % len(full_df.columns))
    print("\nMatrix : %s" % matrix_path)
    print("Schema : %s" % schema_path)
    print("Done. (No model was trained.)")


if __name__ == "__main__":
    main()
