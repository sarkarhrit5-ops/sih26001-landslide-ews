"""
MEGHALAYA GRID PREDICTION SERVICE

Runs the persisted rainfall-coupled Meghalaya LightGBM over a coarse grid tiling the
canonical Meghalaya pilot AOI (East Khasi + Jaintia Hills belt) and returns, per grid
cell, the model's raw susceptibility probability and the system warning class. It is
the Meghalaya analogue of app.services.assam_prediction and app.services.
arunachal_prediction and, exactly like those modules, deliberately REUSES
app.services.sikkim_prediction's state-agnostic building blocks (grid construction,
the real IMERG antecedent-rainfall provider, model-evidence resolution/validation,
rainfall feature derivation, warning-band mapping, report assembly and the UNAVAILABLE
marker) so the four pilots cannot drift apart. Sikkim, Assam, Arunachal Pradesh, IMERG
(weather_ingestion) and the trained models are NOT modified by this module.

TWO -- AND ONLY TWO -- THINGS DIFFER FROM THE SIKKIM PATH (identical to Assam / Arunachal):

  1. LAND COVER IS REAL, NOT AN ELEVATION PROXY. Sikkim derives land_cover_class
     from an elevation binning (risk_inputs.land_cover_class_from_elevation).
     Meghalaya instead samples REAL ESA WorldCover v200 (2021) at each cell via
     app.services.worldcover.sample_worldcover_at_points over the Meghalaya land-cover
     raster, grouped into the nominal classes 1..6. A cell whose WorldCover sample
     is nodata / outside coverage (sentinel -1) is returned UNAVAILABLE and dropped
     -- never back-filled with a class -- mirroring the training pipeline, which
     drops such rows rather than imputing them.

  2. LAND COVER IS SCORED AS A CATEGORICAL. The persisted Meghalaya model was fit
     with land_cover_class as a pandas categorical over the FIXED WorldCover
     vocabulary list(worldcover.ASSAM_LANDCOVER_GROUP_CODES) == (1,2,3,4,5,6) and
     categorical_feature=['land_cover_class']. Inference therefore applies the SAME
     categorical view (NOT the int32 cast the Sikkim path uses); terrain and
     rainfall stay float32. This treatment matches the Assam / Arunachal pilots exactly.

Everything else -- the 11-feature order, the coarse-grid controls, the antecedent
T-1..T-14 IMERG rainfall applied AOI-uniformly, the decision threshold, and the
"answer only from real inputs or refuse" data-integrity contract -- is shared,
verbatim, with sikkim_prediction.

DATA-INTEGRITY CONTRACT (identical in spirit to routes.py / risk_inputs.py /
sikkim_prediction.py / assam_prediction.py / arunachal_prediction.py): this module
answers only from real, resolved inputs or it refuses. It never fabricates a
coordinate, terrain value, land-cover class, rainfall value or prediction, never
back-fills a nodata cell with a placeholder, and never hard-codes a probability. A
cell whose terrain OR land cover is missing / nodata / out-of-coverage is returned
with status UNAVAILABLE and NO probability; if the model artifacts or the real
rainfall cannot be obtained, the whole request is refused (PredictionUnavailable ->
HTTP 503).

Heavy, host-only dependencies (rasterio for the terrain + WorldCover rasters, joblib
via model_artifacts, pandas, and lightgbm via the model object +
ml_pipeline.calculate_warning_level) are imported LAZILY, so this module imports
cleanly in the offline test sandbox. The full pipeline is exercised offline by
injecting fake `model_evidence`, `terrain_sampler`, `land_cover_resolver` and
`rainfall_provider` collaborators; the real end-to-end run (load .pkl, read the
Meghalaya rasters, fetch real IMERG, predict_proba) is host-only.
"""

import os

from app.services import risk_inputs, worldcover as wc
from app.services import sikkim_prediction as sp


# --- Shared, state-agnostic contract re-exported from sikkim_prediction --------
# Re-using these (rather than re-declaring them) guarantees the Meghalaya path stays
# in lock-step with the Sikkim path for everything that is genuinely common.
PredictionUnavailable = sp.PredictionUnavailable
MODEL_FEATURE_ORDER = sp.MODEL_FEATURE_ORDER          # the 11 features, exact order
TERRAIN_FEATURES = sp.TERRAIN_FEATURES                # elevation, slope, aspect, roughness, tpi
LAND_COVER_FEATURE = sp.LAND_COVER_FEATURE            # "land_cover_class"
RAINFALL_FEATURES = sp.RAINFALL_FEATURES              # rain_1d .. rain_intensity_max_3d
RAINFALL_WINDOW_DAYS = sp.RAINFALL_WINDOW_DAYS
DEFAULT_STEP_DEG = sp.DEFAULT_STEP_DEG
MIN_STEP_DEG = sp.MIN_STEP_DEG
MAX_STEP_DEG = sp.MAX_STEP_DEG
MAX_CELLS = sp.MAX_CELLS
DECISION_THRESHOLD = sp.DECISION_THRESHOLD
POSITIVE_CLASS_LABEL = sp.POSITIVE_CLASS_LABEL

# --- Meghalaya-specific identity + on-disk artifacts ---------------------------
STATE_NAME = "Meghalaya"
PILOT_AREA = "East Khasi + Jaintia Hills belt"

# Meghalaya raster filenames -- byte-for-byte the ones
# scripts/build_meghalaya_training_matrix.py consumed to build the matrix the model
# was fit on (DEM + land cover under data/raw, terrain derivatives under
# data/processed).
MEGHALAYA_DEM_FILENAME = "meghalaya_pilot_dem.tif"
MEGHALAYA_LANDCOVER_FILENAME = "meghalaya_pilot_landcover.tif"
MEGHALAYA_TERRAIN_DERIVATIVE_FILENAMES = {
    "slope": "meghalaya_pilot_slope.tif",
    "aspect": "meghalaya_pilot_aspect.tif",
    "roughness": "meghalaya_pilot_roughness.tif",
    "tpi": "meghalaya_pilot_tpi.tif",
}

# The fixed categorical level set the persisted model was fit on. Building the
# inference-time categorical over THIS vocabulary (not over whatever codes happen to
# appear in one request) is what makes the category->code mapping identical to
# training, so the model reads the same integer category codes it learned on. This is
# a SHARED, state-agnostic WorldCover attribute (historically named for Assam); it is
# used verbatim, never renamed.
LAND_COVER_CATEGORIES = list(wc.ASSAM_LANDCOVER_GROUP_CODES)  # (1,2,3,4,5,6)


# ---------------------------------------------------------------------------
# Meghalaya raster path helpers (mirror risk_inputs.terrain_raster_paths, Meghalaya names)
# ---------------------------------------------------------------------------
def meghalaya_terrain_raster_paths(data_dir=None):
    """
    Absolute paths of the five real Meghalaya terrain rasters, keyed by feature name.

    Uses the SAME data root as the rest of the backend (risk_inputs.default_data_dir
    -> backend/data): the DEM at data/raw/meghalaya_pilot_dem.tif and the derivative
    rasters at data/processed/meghalaya_pilot_<name>.tif.
    """
    root = os.path.abspath(data_dir or risk_inputs.default_data_dir())
    paths = {"elevation": os.path.join(root, "raw", MEGHALAYA_DEM_FILENAME)}
    for name, fname in MEGHALAYA_TERRAIN_DERIVATIVE_FILENAMES.items():
        paths[name] = os.path.join(root, "processed", fname)
    return paths


def meghalaya_landcover_raster_path(data_dir=None):
    """Absolute path of the real Meghalaya ESA WorldCover raster (data/raw)."""
    root = os.path.abspath(data_dir or risk_inputs.default_data_dir())
    return os.path.join(root, "raw", MEGHALAYA_LANDCOVER_FILENAME)


def missing_meghalaya_terrain_rasters(data_dir=None):
    """[(feature_name, path)] for every Meghalaya terrain raster that is absent or empty."""
    return [
        (name, path)
        for name, path in sorted(meghalaya_terrain_raster_paths(data_dir).items())
        if not (os.path.exists(path) and os.path.getsize(path) > 0)
    ]


# ---------------------------------------------------------------------------
# Terrain (real Meghalaya rasters; batched to open each raster once)
# ---------------------------------------------------------------------------
def _meghalaya_terrain_sampler(centers, data_dir=None, rasterio_module=None):
    """
    Batched real-terrain sampler over the Meghalaya rasters. Same rules as
    sikkim_prediction._default_terrain_sampler (AOI/raster coverage, NODATA_SENTINEL,
    finite check), only the raster paths differ. Returns a list parallel to
    `centers`, each entry {"values": {feat: float} | None, "problems": [str]}.

    Raises PredictionUnavailable if any Meghalaya terrain raster is missing/empty or
    rasterio is unavailable -- a SYSTEMIC condition. A single nodata / out-of-coverage
    cell is reported per-cell (values None) without a placeholder, never as a systemic
    refusal.
    """
    missing = missing_meghalaya_terrain_rasters(data_dir)
    if missing:
        raise PredictionUnavailable(
            "Missing or empty Meghalaya terrain raster(s): "
            + ", ".join("%s" % name for name, _ in missing),
            details={"missing_rasters": {name: path for name, path in missing}},
        )
    if rasterio_module is None:
        try:
            import rasterio as rasterio_module  # lazy, host-only
        except ImportError as exc:
            raise PredictionUnavailable(
                "rasterio is not installed, so the real Meghalaya terrain rasters "
                "cannot be read (%s)." % exc
            )

    paths = meghalaya_terrain_raster_paths(data_dir)
    n = len(centers)
    values = [dict() for _ in range(n)]
    problems = [list() for _ in range(n)]

    for feat in TERRAIN_FEATURES:
        path = paths[feat]
        with rasterio_module.open(path) as src:
            b = src.bounds
            nodata = src.nodata
            coords = [(float(lon), float(lat)) for (lat, lon) in centers]
            samples = [row[0] for row in src.sample(coords)]
        for idx, (lat, lon) in enumerate(centers):
            if not (b.left <= float(lon) <= b.right and b.bottom <= float(lat) <= b.top):
                problems[idx].append(
                    "'%s' raster does not cover (lat=%s, lon=%s)." % (feat, lat, lon)
                )
                continue
            number = sp._finite(samples[idx])
            if number is None:
                problems[idx].append(
                    "'%s' sampled a non-finite value at (lat=%s, lon=%s)." % (feat, lat, lon)
                )
                continue
            if number == risk_inputs.NODATA_SENTINEL or (
                nodata is not None and sp._finite(nodata) == number
            ):
                problems[idx].append(
                    "'%s' is nodata at (lat=%s, lon=%s) (value %s)." % (feat, lat, lon, number)
                )
                continue
            values[idx][feat] = number

    result = []
    for idx in range(n):
        if problems[idx] or len(values[idx]) != len(TERRAIN_FEATURES):
            result.append({
                "values": None,
                "problems": problems[idx] or ["incomplete terrain sample at this cell"],
            })
        else:
            result.append({"values": values[idx], "problems": []})
    return result


# ---------------------------------------------------------------------------
# Land cover (REAL ESA WorldCover; nodata cells refused, never filled)
# ---------------------------------------------------------------------------
def _meghalaya_land_cover(centers, data_dir=None, reader=None):
    """
    Real WorldCover land-cover group per cell, sampled from the Meghalaya WorldCover
    raster via worldcover.sample_worldcover_at_points (NEAREST pixel). Returns a list
    parallel to `centers`, each entry {"value": int(group 1..6) | None, "problems":
    [str]}.

    A cell whose WorldCover sample is UNAVAILABLE (nodata / outside coverage / unknown
    code, sentinel -1) comes back value=None with a reason -- the caller marks it
    UNAVAILABLE and drops it. This module NEVER substitutes a land-cover class, exactly
    as the Meghalaya training pipeline drops such rows instead of imputing them.

    Raises PredictionUnavailable only for the SYSTEMIC condition of a missing/empty
    raster (mirroring the terrain sampler); `reader` is injectable for offline tests.
    """
    path = meghalaya_landcover_raster_path(data_dir)
    if not (os.path.exists(path) and os.path.getsize(path) > 0):
        raise PredictionUnavailable(
            "Missing or empty Meghalaya WorldCover land-cover raster: %s" % path,
            details={"missing_rasters": {"land_cover": path}},
        )
    lats = [float(lat) for (lat, lon) in centers]
    lons = [float(lon) for (lat, lon) in centers]
    try:
        groups, unavailable_mask = wc.sample_worldcover_at_points(
            path, lats, lons, reader=reader
        )
    except PredictionUnavailable:
        raise
    except Exception as exc:  # noqa: BLE001 - surfaced honestly, never swallowed
        raise PredictionUnavailable(
            "Real ESA WorldCover land cover could not be read (%s: %s)."
            % (type(exc).__name__, exc)
        )

    result = []
    for grp, bad in zip(list(groups), list(unavailable_mask)):
        if bool(bad) or int(grp) == wc.UNAVAILABLE_SENTINEL:
            result.append({
                "value": None,
                "problems": [
                    "WorldCover land cover is nodata / outside coverage at this cell "
                    "(sentinel %d); refusing to substitute a class." % wc.UNAVAILABLE_SENTINEL
                ],
            })
        else:
            result.append({"value": int(grp), "problems": []})
    return result


# ---------------------------------------------------------------------------
# Scoring (CATEGORICAL land cover, matching how the model was fit)
# ---------------------------------------------------------------------------
def _score_cells_categorical(model, feature_rows):
    """
    feature_rows: list of dicts, each holding all 11 MODEL_FEATURE_ORDER keys.
    Returns positive-class probabilities aligned to feature_rows.

    Unlike the Sikkim path (which casts land_cover_class to int32), this builds
    land_cover_class as a pandas categorical over the FIXED WorldCover vocabulary
    LAND_COVER_CATEGORIES (1..6) with terrain/rainfall as float32 -- the exact view
    the persisted Meghalaya model was fit on (categorical_feature=['land_cover_class']).
    """
    import pandas as pd  # available offline; lazy to match the module's ethos

    frame = pd.DataFrame(feature_rows, columns=list(MODEL_FEATURE_ORDER))
    for col in MODEL_FEATURE_ORDER:
        if col == LAND_COVER_FEATURE:
            frame[col] = pd.Categorical(
                frame[col].astype("int64"),
                categories=list(LAND_COVER_CATEGORIES),
            )
        else:
            frame[col] = frame[col].astype("float32")

    proba = model.predict_proba(frame)
    classes = list(getattr(model, "classes_", [0, 1]))
    if POSITIVE_CLASS_LABEL in classes:
        pos_idx = classes.index(POSITIVE_CLASS_LABEL)
    else:
        pos_idx = len(classes) - 1
    return [float(row[pos_idx]) for row in proba]


# ---------------------------------------------------------------------------
# Disclosures
# ---------------------------------------------------------------------------
def _meghalaya_disclosures():
    return [
        "Output is the model's RAW positive-class probability (susceptibility), "
        "NOT the Option-C fused final_risk_score served by /risk/current; here "
        "rainfall is a model feature, not a separate trigger multiplier.",
        "TRAIN/SERVE RAINFALL-SOURCE SHIFT: the Meghalaya model was trained on "
        "Open-Meteo ERA5 reanalysis rainfall (provenance records IMERG as NOT_USED at "
        "training); this endpoint serves NASA GPM IMERG. The sources differ, which may "
        "affect probability calibration.",
        "Rainfall is a single AOI-mean IMERG series applied UNIFORMLY to every grid "
        "cell (IMERG's ~0.1 deg grid is coarser than the prediction grid), so per-cell "
        "variation is driven by terrain and land cover, not by spatial rainfall "
        "differences.",
        "land_cover_class is REAL observed land cover (NOT an elevation proxy): ESA "
        "WorldCover %s (%d), sampled per cell and grouped into the nominal classes "
        "1=forest, 2=shrub/grass/herbaceous, 3=cropland, 4=built-up, 5=bare/sparse/snow, "
        "6=water/wetland. It is fed to the LightGBM primary model as a CATEGORICAL "
        "feature (categorical_feature=['land_cover_class']), exactly as at training. "
        "This categorical, real land cover (matching the Assam / Arunachal pilots' "
        "treatment exactly) is the ONLY methodological difference from the Sikkim pilot, "
        "which uses an ordered elevation proxy left numeric."
        % (wc.WORLDCOVER_VERSION, wc.WORLDCOVER_YEAR),
        "risk_class maps the probability onto the system warning bands via "
        "ml_pipeline.calculate_warning_level (LOW <0.40, MEDIUM 0.40-0.65, HIGH "
        "0.65-0.85, EXTREME >=0.85); exceeds_decision_threshold uses the validated "
        "operating threshold %.2f." % DECISION_THRESHOLD,
        "Cells whose terrain OR land cover is missing, nodata or outside coverage are "
        "returned with status UNAVAILABLE and no probability -- never a placeholder.",
        "This is a COARSE grid over the pilot AOI, not the native 30 m DEM resolution; "
        "each cell reports the model output sampled at its center.",
    ]


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def predict_meghalaya_grid(target_date, step_deg=DEFAULT_STEP_DEG, run_type="Early",
                           state_name=STATE_NAME, data_dir=None, artifact_dir=None,
                           model_evidence=None, terrain_sampler=None,
                           land_cover_resolver=None, rainfall_provider=None):
    """
    Full read-only prediction over the coarse Meghalaya pilot-AOI grid. See the module
    docstring for the data-integrity contract and the two Meghalaya-specific
    differences (real WorldCover land cover + categorical scoring). The
    `model_evidence`, `terrain_sampler`, `land_cover_resolver` and `rainfall_provider`
    collaborators are injectable so the assembly and the no-fabrication invariants are
    testable offline with fakes; the defaults run the real host-only pipeline.

    Returns a JSON-safe dict. Raises PredictionUnavailable (-> HTTP 503) when the
    model or real rainfall cannot be obtained, or ValueError (-> HTTP 400) for a bad
    grid step.
    """
    target_date = sp._as_datetime(target_date)

    # 1. Model (real Meghalaya artifacts, correct 11-feature order) -- refuse if unusable.
    evidence = sp._resolve_model_evidence(model_evidence, state_name, artifact_dir)
    model = evidence["model"]

    # 2. Grid over the canonical Meghalaya AOI.
    bounds, grid_meta, cells = sp.build_grid(step_deg=step_deg, state_name=state_name)
    centers = [(c["latitude"], c["longitude"]) for c in cells]

    # 3. Real antecedent rainfall (AOI-uniform, real IMERG) -- required for every cell.
    provider = rainfall_provider or sp._default_rainfall_provider
    try:
        rainfall = provider(bounds, target_date, run_type)
    except PredictionUnavailable:
        raise
    except Exception as exc:  # noqa: BLE001 - surfaced honestly as 503, never swallowed
        raise PredictionUnavailable(
            "Real IMERG antecedent rainfall could not be obtained (%s: %s)."
            % (type(exc).__name__, exc)
        )
    if not isinstance(rainfall, dict) or "features" not in rainfall:
        raise PredictionUnavailable("Rainfall provider returned no 'features'.")
    rain_features = rainfall["features"]
    sp._validate_rainfall_features(rain_features)

    # 4. Terrain per cell (real Meghalaya rasters).
    sampler = terrain_sampler or _meghalaya_terrain_sampler
    samples = sampler(centers, data_dir)
    if len(samples) != len(cells):
        raise PredictionUnavailable(
            "terrain sampler returned %d rows for %d cells." % (len(samples), len(cells))
        )

    # 5. Real land cover per cell (ESA WorldCover; NOT an elevation proxy).
    resolver = land_cover_resolver or _meghalaya_land_cover
    land_cover = resolver(centers, data_dir)
    if len(land_cover) != len(cells):
        raise PredictionUnavailable(
            "land-cover resolver returned %d rows for %d cells."
            % (len(land_cover), len(cells))
        )

    # 6. Assemble usable feature rows; mark unusable cells UNAVAILABLE (no fill). A
    #    cell needs BOTH complete terrain AND a real land-cover class to be scorable.
    scorable_idx = []
    feature_rows = []
    for idx, (cell, sample, lc) in enumerate(zip(cells, samples, land_cover)):
        terrain_values = sample.get("values")
        if not terrain_values:
            sp._mark_unavailable(
                cell, sample.get("problems") or ["terrain unavailable at this cell"]
            )
            continue
        lc_value = lc.get("value")
        if lc_value is None:
            sp._mark_unavailable(
                cell, lc.get("problems") or ["land cover unavailable at this cell"]
            )
            continue
        row = {feat: float(terrain_values[feat]) for feat in TERRAIN_FEATURES}
        row[LAND_COVER_FEATURE] = int(lc_value)
        for feat in RAINFALL_FEATURES:
            row[feat] = float(rain_features[feat])
        cell["_features"] = row
        scorable_idx.append(idx)
        feature_rows.append(row)

    # 7. Score the usable cells with the CATEGORICAL land-cover view the model was fit on.
    probabilities = _score_cells_categorical(model, feature_rows) if feature_rows else []

    risk_counts = {"LOW": 0, "MEDIUM": 0, "HIGH": 0, "EXTREME": 0}
    probs_only = []
    for pos, idx in enumerate(scorable_idx):
        cell = cells[idx]
        prob = float(probabilities[pos])
        level = sp._warning_level(prob)
        cell["status"] = "OK"
        cell["susceptibility_probability"] = round(prob, 6)
        cell["risk_class"] = level
        cell["exceeds_decision_threshold"] = bool(prob >= DECISION_THRESHOLD)
        cell["features"] = sp._round_features(cell.pop("_features"))
        cell["reasons"] = []
        risk_counts[level] = risk_counts.get(level, 0) + 1
        probs_only.append(prob)

    scored = len(probs_only)
    summary = {
        "cells_total": len(cells),
        "cells_scored": scored,
        "cells_unavailable": len(cells) - scored,
        "risk_class_counts": risk_counts,
        "cells_exceeding_threshold": sum(1 for p in probs_only if p >= DECISION_THRESHOLD),
        "max_probability": round(max(probs_only), 6) if probs_only else None,
        "mean_probability": round(sum(probs_only) / scored, 6) if scored else None,
    }

    return {
        "state": state_name,
        "pilot_area": PILOT_AREA,
        "generated_from": (
            "persisted Meghalaya LightGBM (static_plus_rainfall, 11 features; "
            "land cover scored as a categorical) + real ESA WorldCover land cover + "
            "real IMERG antecedent rainfall"
        ),
        "target_date": target_date.strftime("%Y-%m-%d"),
        "aoi": bounds,
        "grid": grid_meta,
        "decision_threshold": DECISION_THRESHOLD,
        "model": sp._model_report(evidence),
        "rainfall": sp._rainfall_report(rainfall),
        "summary": summary,
        "disclosures": _meghalaya_disclosures(),
        "cells": cells,
    }
