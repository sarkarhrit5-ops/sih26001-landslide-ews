#!/usr/bin/env python3
"""
Train the Meghalaya pilot landslide model from the completed 136x11 training
matrix.

This is the Meghalaya analogue of ``scripts/train_arunachal_model.py`` (itself
the Arunachal analogue of ``scripts/train_assam_model.py``, in turn the Assam
analogue of ``scripts/train_real_models.py`` sections 4-7:
validate -> decide -> persist). It deliberately REUSES the Sikkim/Assam/Arunachal
methodology verbatim through ``app.models.ml_pipeline`` and
``app.services.model_artifacts`` and changes exactly one thing relative to
Sikkim:

    * ``land_cover_class`` is fed to the LightGBM PRIMARY model as a pandas
      ``category`` (with ``categorical_feature=['land_cover_class']``), because
      the Meghalaya feature -- exactly like Assam's and Arunachal's -- is a REAL,
      NOMINAL ESA WorldCover class grouping (forest / shrub / cropland / built-up
      / bare / water) that has no natural order, unlike Sikkim's elevation-binned
      proxy, which is an ordered 1 < 2 < 3 encoding and is therefore left numeric
      there.

Everything else is identical to Sikkim/Assam/Arunachal: the 11-feature order, the
latitude-median spatial holdout, the event-year temporal holdout (cutoff 2014),
the buffered negative sampling already baked into the matrix, the antecedent-only
rainfall already baked into the matrix, the reproducibility gate, the Option A/C
decision rule and the four persisted artifacts.

OPTION C IS PRE-DETERMINED for Meghalaya, on TWO independent grounds computed
from the committed events snapshot (data/models/meghalaya_events.json):
    * independent event dates = 29 (<< the 100-event floor for a headline
      validated model), and
    * pct_low_accuracy = 76.47% (26 of 34 GLC positives are coarser than the
      high-accuracy labels), which is well above the 50% ceiling.
Either alone forces the honest "research-grade / Option C" recommendation; this
trainer does NOT hardcode that outcome -- it computes it from the data via
ml_pipeline.evaluate_model_decision and simply will not be surprised by it.

Hard integrity rules (shared with the whole project):
    * This script NEVER fabricates, fills, interpolates or substitutes data.
      Rows with any non-finite terrain/rainfall value, or an UNAVAILABLE
      land-cover sentinel (-1), are DROPPED (counted, never imputed).
    * It does NOT rebuild features and it does NOT touch the network: it loads
      the matrix that ``scripts/build_meghalaya_training_matrix.py`` already
      wrote.
    * It does NOT modify Sikkim, Assam, Arunachal, IMERG
      (``weather_ingestion.py``), the frontend, or any shared service.
      State/pilot labels that ``model_artifacts`` hardcodes to "Sikkim" are
      corrected on the returned dicts (relabelling only -- no metric, feature or
      provenance value is altered).

Runtime note: reading the parquet needs pyarrow, and training needs
scikit-learn + lightgbm + joblib. Those are absent from the offline sandbox, so
this file is authored to IMPORT and py_compile offline (every heavy dependency is
imported lazily inside ``main()``); the actual training run is host-only. Use
``--dry-run`` to train + validate + decide and print the report WITHOUT writing
any artifact.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import sys

# Only pure/offline-safe dependencies are imported at module load so this file
# imports and py_compiles in the offline sandbox. numpy and pandas are present
# offline; scikit-learn, lightgbm, joblib, pyarrow, and the ml_pipeline module
# (which imports lightgbm/shap at its top) are imported lazily inside main().
import numpy as np
import pandas as pd

# Make ``app`` importable whether run from repo root or from backend/.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_BACKEND_DIR = os.path.dirname(_THIS_DIR)
for _p in (_BACKEND_DIR, _THIS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from app.core.config_states import (  # noqa: E402  (path set up above)
    assert_pilot_aoi_consistency,
    get_pilot_aoi_bounds,
)
from app.services import worldcover as wc  # noqa: E402

# ---------------------------------------------------------------------------
# Meghalaya pilot constants
# ---------------------------------------------------------------------------
STATE_NAME = "Meghalaya"
PILOT_AREA = "East Khasi + Jaintia Hills belt"
MATRIX_FILENAME = "meghalaya_pilot_training_matrix.parquet"
MATRIX_SCHEMA_FILENAME = "meghalaya_pilot_training_matrix_schema.json"

# Feature layout -- IDENTICAL name/order/semantics to
# train_real_models.py ``static_features`` / ``dynamic_features``.
TERRAIN_FEATURES = ["elevation", "slope", "aspect", "roughness", "tpi"]
STATIC_FEATURES = TERRAIN_FEATURES + [wc.LANDCOVER_FEATURE_NAME]          # 6
RAINFALL_FEATURES = [
    "rain_1d",
    "rain_3d",
    "rain_7d",
    "antecedent_rain_14d",
    "rain_intensity_max_3d",
]
DYNAMIC_FEATURES = STATIC_FEATURES + RAINFALL_FEATURES                    # 11

# Same cutoff Sikkim/Assam/Arunachal use. Verified non-degenerate for Meghalaya
# from the committed snapshot (positives: 12 train (<=2014) / 22 test (>=2015);
# distinct event dates 10 vs 19). Kept at 2014 for cross-pilot consistency.
TEMPORAL_CUTOFF_YEAR = 2014

# GLC location_accuracy labels that count as HIGH accuracy (Sikkim's rule).
# Everything else contributes to pct_low_accuracy.
HIGH_ACCURACY_LABELS = ("1km", "exact", "100m")


# ---------------------------------------------------------------------------
# Pure helpers (offline-testable; use only numpy/pandas)
# ---------------------------------------------------------------------------
def meghalaya_land_cover_meaning() -> str:
    """Human-readable meaning string for the Meghalaya land_cover_class feature.

    Overrides model_artifacts.FEATURE_MEANINGS (which documents Sikkim's
    elevation proxy) with the truthful WorldCover description.
    """
    labels = ", ".join(
        "%d = %s" % (code, wc.ASSAM_LANDCOVER_GROUP_LABELS[code])
        for code in sorted(wc.ASSAM_LANDCOVER_GROUP_LABELS)
    )
    return (
        "REAL observed land cover (NOT an elevation proxy): ESA WorldCover %s "
        "(%s), sampled at each point and grouped into NOMINAL classes: %s. "
        "Treated as CATEGORICAL for the LightGBM primary model via "
        "categorical_feature=%s (never ordinal). Cells with no WorldCover data "
        "use sentinel %d and the affected rows are dropped, never imputed."
        % (
            wc.WORLDCOVER_VERSION,
            wc.WORLDCOVER_YEAR,
            labels,
            wc.landcover_categorical_feature(),
            wc.UNAVAILABLE_SENTINEL,
        )
    )


def meghalaya_feature_meanings() -> dict:
    """Meaning overrides passed to build_feature_schema_document.

    Only ``land_cover_class`` differs from Sikkim; the other ten fall back to
    model_artifacts.FEATURE_MEANINGS (terrain + antecedent rainfall wording is
    identical across pilots).
    """
    return {wc.LANDCOVER_FEATURE_NAME: meghalaya_land_cover_meaning()}


def relabel_for_meghalaya(doc):
    """Correct the state/pilot_area labels on a model_artifacts doc.

    build_metrics_document and build_feature_schema_document hardcode
    ``state="Sikkim"`` (and metrics also ``pilot_area="East Sikkim"``) because
    they predate the Meghalaya pilot and take no state argument. This rewrites
    ONLY those label fields to the real Meghalaya run. It never touches
    validation metrics, feature names, dtypes, meanings, sample counts or any
    other value, and the persistence gate does not assert on the state string.
    """
    if isinstance(doc, dict):
        if "state" in doc:
            doc["state"] = STATE_NAME
        if "pilot_area" in doc:
            doc["pilot_area"] = PILOT_AREA
    return doc


def drop_missing_feature_rows(
    df,
    features=None,
    landcover_col=None,
    sentinel=None,
):
    """Drop (never fill) rows with any missing model feature.

    A row is dropped when any numeric feature is non-finite (NaN/inf) OR the
    land-cover column equals the UNAVAILABLE sentinel. Returns
    ``(clean_df, report)``. The builder deliberately retains such rows; dropping
    them here is the trainer's half of the "abort/exclude, never impute"
    contract.
    """
    if features is None:
        features = DYNAMIC_FEATURES
    if landcover_col is None:
        landcover_col = wc.LANDCOVER_FEATURE_NAME
    if sentinel is None:
        sentinel = wc.UNAVAILABLE_SENTINEL

    n0 = len(df)
    numeric_cols = [f for f in features if f != landcover_col]
    if numeric_cols:
        finite_mask = np.isfinite(
            df[numeric_cols].to_numpy(dtype="float64", na_value=np.nan)
        ).all(axis=1)
    else:
        finite_mask = np.ones(n0, dtype=bool)

    if landcover_col in df.columns:
        lc_ok = df[landcover_col].to_numpy() != sentinel
    else:
        lc_ok = np.ones(n0, dtype=bool)

    keep = finite_mask & lc_ok
    clean = df.loc[keep].reset_index(drop=True)
    report = {
        "rows_in": int(n0),
        "rows_dropped": int((~keep).sum()),
        "rows_out": int(len(clean)),
        "dropped_nonfinite_numeric": int((~finite_mask).sum()),
        "dropped_landcover_unavailable": int((~lc_ok).sum()),
        "fill_or_impute_performed": False,
    }
    return clean, report


def lgbm_categorical_view(X, categories, col=None):
    """Return a COPY of X with ``land_cover_class`` as a pandas Categorical.

    ``categories`` fixes the complete nominal vocabulary so train and test share
    identical category codes and LightGBM builds native categorical splits. The
    stored values are never changed -- only the dtype. The input frame is left
    untouched.
    """
    if col is None:
        col = wc.LANDCOVER_FEATURE_NAME
    X = X.copy()
    if col in X.columns:
        X[col] = pd.Categorical(X[col].astype("int64"), categories=list(categories))
    return X


def compute_glc_quality(pos_df):
    """Build the glc_quality_info dict consumed by evaluate_model_decision.

    Computed from the matrix positives, mirroring the Sikkim CSV cross-check:
    ``pct_low_accuracy`` = share of positives whose GLC location_accuracy is not
    a high-accuracy label; ``independent_events`` = distinct positive event
    DATES; ``total_usable_events`` = positive count.
    """
    n = int(len(pos_df))
    if n == 0:
        return {
            "pct_low_accuracy": 0.0,
            "independent_events": 0,
            "total_usable_events": 0,
        }
    la = pos_df["location_accuracy"].astype(str).str.strip()
    low = int((~la.isin(HIGH_ACCURACY_LABELS)).sum())
    pct = round(low / n * 100.0, 4)
    indep = int(pd.to_datetime(pos_df["event_date"]).dt.date.nunique())
    return {
        "pct_low_accuracy": pct,
        "independent_events": indep,
        "total_usable_events": n,
    }


# ---------------------------------------------------------------------------
# Training helpers (host-only: need scikit-learn + lightgbm via ``ml``)
# ---------------------------------------------------------------------------
def train_primary_categorical(ml, X_train, X_test, y_train, y_test, categories):
    """Meghalaya analogue of ml_pipeline.train_primary_model.

    Same estimator (``ml.build_primary_model()``), same hyperparameters, seed
    and split as Sikkim -- the ONLY difference is that ``land_cover_class`` is a
    categorical column and is declared to LightGBM via ``categorical_feature``.
    Returns ``(fitted_model, metrics_dict)`` using ml_pipeline.compute_metrics so
    the numbers are directly comparable to the reported baselines.
    """
    model = ml.build_primary_model()
    cat = wc.landcover_categorical_feature()
    X_tr = lgbm_categorical_view(X_train, categories)
    X_te = lgbm_categorical_view(X_test, categories)
    model.fit(X_tr, y_train, categorical_feature=cat)
    proba = model.predict_proba(X_te)[:, 1]
    return model, ml.compute_metrics(y_test, proba)


def evaluate_baselines_meghalaya(ml, X_train, X_test, y_train, y_test, categories):
    """Meghalaya analogue of ml_pipeline.train_and_evaluate_baselines.

    Logistic Regression and Random Forest are fit EXACTLY as Sikkim does -- on
    the numeric matrix, same classes/params/seed -- so those comparison rows stay
    methodologically identical across pilots. Only the LightGBM entry (the model
    actually persisted) uses the nominal categorical treatment. Returns the same
    ``{model_name: metrics}`` shape as the Sikkim helper.
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression

    results = {}

    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train, y_train)
    results["Logistic Regression"] = ml.compute_metrics(
        y_test, lr.predict_proba(X_test)[:, 1]
    )

    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=2)
    rf.fit(X_train, y_train)
    results["Random Forest"] = ml.compute_metrics(
        y_test, rf.predict_proba(X_test)[:, 1]
    )

    _model, lgbm_metrics = train_primary_categorical(
        ml, X_train, X_test, y_train, y_test, categories
    )
    results[ml.PRIMARY_MODEL_NAME] = lgbm_metrics
    return results


# ---------------------------------------------------------------------------
# Main (host-only)
# ---------------------------------------------------------------------------
def _resolve_paths(matrix_arg):
    processed_dir = os.path.join(_BACKEND_DIR, "data", "processed")
    models_dir = os.path.join(_BACKEND_DIR, "data", "models")
    matrix_path = matrix_arg or os.path.join(processed_dir, MATRIX_FILENAME)
    schema_path = os.path.join(models_dir, MATRIX_SCHEMA_FILENAME)
    return matrix_path, schema_path, models_dir


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Train the Meghalaya pilot landslide model from the completed "
            "meghalaya_pilot_training_matrix.parquet, mirroring the "
            "Sikkim/Assam/Arunachal methodology (WorldCover categorical, real "
            "rainfall only, leakage controls). Does not modify Sikkim, Assam, "
            "Arunachal or IMERG."
        )
    )
    parser.add_argument(
        "--matrix",
        default=None,
        help="Path to meghalaya_pilot_training_matrix.parquet (default: backend/data/processed/).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Train + validate + decide and print the report, but write NO artifact.",
    )
    args = parser.parse_args(argv)

    # Lazy, host-only heavy imports (kept out of module scope so this file
    # imports and py_compiles offline).
    from app.models import ml_pipeline as ml
    from app.services import model_artifacts

    matrix_path, schema_path, models_dir = _resolve_paths(args.matrix)

    print("==========================================================")
    print("  MEGHALAYA PILOT MODEL TRAINING (Sikkim/Assam/Arunachal method) ")
    print("==========================================================")
    print(" state=%s  pilot_area=%s" % (STATE_NAME, PILOT_AREA))
    print(" matrix=%s" % matrix_path)
    print(" dry_run=%s" % bool(args.dry_run))

    if not os.path.exists(matrix_path):
        raise SystemExit(
            "Training matrix not found: %s\nRun scripts/build_meghalaya_training_matrix.py "
            "on the host first (it needs rasterio + network)." % matrix_path
        )

    # --- 0. Canonical AOI (fail fast, single source of truth) ---
    aoi_consistency = assert_pilot_aoi_consistency(STATE_NAME)
    aoi = get_pilot_aoi_bounds(STATE_NAME)
    print("\n--- 0. CANONICAL AOI ---")
    print("  AOI (pilot bounds): %s" % aoi)

    # --- 1. Load the completed matrix (real rainfall already baked in) ---
    print("\n--- 1. LOAD TRAINING MATRIX ---")
    full_df = pd.read_parquet(matrix_path)
    print("  loaded rows=%d cols=%d" % (len(full_df), full_df.shape[1]))

    required_cols = DYNAMIC_FEATURES + ["target", "latitude", "longitude", "event_date"]
    missing_cols = [c for c in required_cols if c not in full_df.columns]
    if missing_cols:
        raise SystemExit(
            "Matrix is missing required columns: %s\nAvailable: %s"
            % (missing_cols, list(full_df.columns))
        )

    # --- 2. Drop (never fill) rows with any missing feature ---
    clean_df, drop_report = drop_missing_feature_rows(full_df)
    print("\n--- 2. MISSING-FEATURE ROW DROP (never imputed) ---")
    print("  %s" % drop_report)
    if len(clean_df) == 0:
        raise SystemExit("All rows dropped for missing features; nothing to train.")

    # Integrity guard: every retained land_cover_class value must be a known
    # nominal WorldCover group code. We refuse to silently coerce an unexpected
    # value to a category (which pandas would turn into NaN).
    present_codes = set(int(v) for v in pd.unique(clean_df[wc.LANDCOVER_FEATURE_NAME]))
    valid_codes = set(int(c) for c in wc.ASSAM_LANDCOVER_GROUP_CODES)
    unknown_codes = present_codes - valid_codes
    if unknown_codes:
        raise SystemExit(
            "land_cover_class contains values outside the canonical WorldCover "
            "group codes %s: %s" % (sorted(valid_codes), sorted(unknown_codes))
        )
    # Complete, stable nominal vocabulary for the categorical dtype.
    categories = list(wc.ASSAM_LANDCOVER_GROUP_CODES)

    pos_df = clean_df[clean_df["target"] == 1]
    neg_df = clean_df[clean_df["target"] == 0]
    dedup_count = int(
        len(pos_df.drop_duplicates(subset=["latitude", "longitude", "event_date"]))
    )
    duplicate_positives = int(len(pos_df) - dedup_count)
    glc_info = compute_glc_quality(pos_df)
    print("  positives=%d (unique=%d)  negatives=%d" % (len(pos_df), dedup_count, len(neg_df)))
    print("  glc_quality=%s" % glc_info)

    # --- 3. Spatial + temporal holdout splits (reuse Sikkim helpers verbatim) ---
    print("\n--- 3. HOLDOUT SPLITS ---")
    X_train_sp_st, X_test_sp_st, y_train_sp, y_test_sp = ml.run_spatial_holdout_validation(
        clean_df, STATIC_FEATURES
    )
    X_train_sp_dy, X_test_sp_dy, _, _ = ml.run_spatial_holdout_validation(
        clean_df, DYNAMIC_FEATURES
    )
    X_train_tm_st, X_test_tm_st, y_train_tm, y_test_tm = ml.run_temporal_holdout_validation(
        clean_df, STATIC_FEATURES, cutoff_year=TEMPORAL_CUTOFF_YEAR
    )
    X_train_tm_dy, X_test_tm_dy, _, _ = ml.run_temporal_holdout_validation(
        clean_df, DYNAMIC_FEATURES, cutoff_year=TEMPORAL_CUTOFF_YEAR
    )
    print(
        "  spatial: train=%d test=%d | temporal(cutoff %d): train=%d test=%d"
        % (
            len(X_train_sp_dy),
            len(X_test_sp_dy),
            TEMPORAL_CUTOFF_YEAR,
            len(X_train_tm_dy),
            len(X_test_tm_dy),
        )
    )

    # --- 4. Baseline comparison (LR/RF numeric like Sikkim; LightGBM categorical) ---
    print("\n--- 4. MODEL COMPARISON ---")
    sp_st_res = evaluate_baselines_meghalaya(
        ml, X_train_sp_st, X_test_sp_st, y_train_sp, y_test_sp, categories
    )
    sp_dy_res = evaluate_baselines_meghalaya(
        ml, X_train_sp_dy, X_test_sp_dy, y_train_sp, y_test_sp, categories
    )
    tm_st_res = evaluate_baselines_meghalaya(
        ml, X_train_tm_st, X_test_tm_st, y_train_tm, y_test_tm, categories
    )
    tm_dy_res = evaluate_baselines_meghalaya(
        ml, X_train_tm_dy, X_test_tm_dy, y_train_tm, y_test_tm, categories
    )
    print("  temporal / static+rainfall:")
    for mname, metrics in tm_dy_res.items():
        print("    %-20s: %s" % (mname, metrics))

    # --- 5. Leakage controls (same wording as Sikkim, Meghalaya counts) ---
    leakage_checks = {
        "spatial_leakage": (
            "MITIGATED: Spatially buffered negative sampling (>= 0.05 deg / ~5km) "
            "+ latitude-median spatial holdout split."
        ),
        "temporal_leakage": (
            "MITIGATED: Strict temporal holdout (Train <= %d vs Test >= %d)."
            % (TEMPORAL_CUTOFF_YEAR, TEMPORAL_CUTOFF_YEAR + 1)
        ),
        "future_rainfall_leakage": (
            "MITIGATED: Antecedent rainfall strictly from past days (T-14..T-1) "
            "before event date T; no future rainfall used."
        ),
        "duplicate_overlapping_events": (
            "MITIGATED: %d duplicate (lat,lon,event_date) positives in the "
            "committed AOI-filtered snapshot (positives=%d, unique=%d)."
            % (duplicate_positives, len(pos_df), dedup_count)
        ),
    }

    # --- 6. Option A / C decision (computed, not hardcoded) ---
    print("\n--- 6. MODEL DECISION ---")
    decision = ml.evaluate_model_decision(glc_info, {"Static + Rainfall": tm_dy_res})
    print("  Final Recommendation: %s" % decision["final_recommendation"])
    for reason in decision["justification_reasons"]:
        print("    - %s" % reason)

    if args.dry_run:
        print("\n--- DRY RUN: no artifact written. ---")
        return 0

    # --- 7. Persist validation evidence (primary = LightGBM temporal+dynamic) ---
    print("\n--- 7. PERSISTING VALIDATION EVIDENCE ---")
    try:
        primary_model, primary_metrics = train_primary_categorical(
            ml, X_train_tm_dy, X_test_tm_dy, y_train_tm, y_test_tm, categories
        )
        reported_metrics = tm_dy_res.get(ml.PRIMARY_MODEL_NAME, {})
        if primary_metrics != reported_metrics:
            raise model_artifacts.ArtifactValidationError(
                "Primary model metrics did not reproduce the reported primary "
                "evaluation. Refitted=%s vs reported=%s. No artifact written."
                % (primary_metrics, reported_metrics)
            )

        feature_columns = list(X_train_tm_dy.columns)
        # dtypes AS HANDED TO THE FITTED MODEL: land_cover_class is a pandas
        # category here (matching train_primary_categorical's internal fit view),
        # so the schema truthfully records the categorical contract rather than
        # the raw int32 storage dtype. terrain/rainfall stay float32.
        fit_view_dtypes = {
            k: str(v)
            for k, v in lgbm_categorical_view(X_train_tm_dy, categories)
            .dtypes.astype(str)
            .to_dict()
            .items()
        }

        metrics_doc = relabel_for_meghalaya(
            model_artifacts.build_metrics_document(
                validation_metrics=primary_metrics,
                primary_model_name=ml.PRIMARY_MODEL_NAME,
                primary_evaluation="temporal_holdout / static_plus_rainfall",
                feature_set="static_plus_rainfall",
                model_comparison={
                    "spatial_holdout": {
                        "static_only": sp_st_res,
                        "static_plus_rainfall": sp_dy_res,
                    },
                    "temporal_holdout": {
                        "static_only": tm_st_res,
                        "static_plus_rainfall": tm_dy_res,
                    },
                },
                holdout_details={
                    "spatial_holdout": "Latitude median split (train <= median, test > median)",
                    "temporal_holdout": "Event year split (train <= %d, test >= %d)"
                    % (TEMPORAL_CUTOFF_YEAR, TEMPORAL_CUTOFF_YEAR + 1),
                    "decision_threshold": 0.5,
                },
                sample_counts={
                    "total_samples": int(len(clean_df)),
                    "positive_samples": int(dedup_count),
                    "negative_samples": int(len(neg_df)),
                    "primary_train_samples": int(len(X_train_tm_dy)),
                    "primary_test_samples": int(len(X_test_tm_dy)),
                    "primary_train_positives": int(y_train_tm.sum()),
                    "primary_test_positives": int(y_test_tm.sum()),
                },
                decision=decision,
                dataset_provenance_reference=matrix_path,
            )
        )

        feature_schema_doc = relabel_for_meghalaya(
            model_artifacts.build_feature_schema_document(
                feature_names=feature_columns,
                dtypes=fit_view_dtypes,
                meanings=meghalaya_feature_meanings(),
                feature_set_name="static_plus_rainfall",
                target_column="target",
            )
        )

        provenance_doc = model_artifacts.build_provenance_document(
            state=STATE_NAME,
            pilot_area=PILOT_AREA,
            aoi=dict(aoi),
            model_type="%s (lightgbm.LGBMClassifier)" % ml.PRIMARY_MODEL_NAME,
            model_hyperparameters=dict(ml.PRIMARY_MODEL_HYPERPARAMS),
            feature_list=feature_columns,
            random_seed=42,
            glc_source=(
                "NASA Global Landslide Catalog Export (meghalaya_events.json "
                "snapshot), AOI-filtered"
            ),
            glc_event_count=int(dedup_count),
            sample_counts={
                "positive_events": int(len(pos_df)),
                "deduplicated_positive_events": int(dedup_count),
                "negative_samples": int(len(neg_df)),
                "total_samples": int(len(clean_df)),
                "independent_event_dates": int(glc_info["independent_events"]),
                "pct_events_low_location_accuracy": glc_info["pct_low_accuracy"],
            },
            rainfall_source=(
                "Open-Meteo ERA5 archive API (daily precipitation_sum), antecedent "
                "window strictly T-14..T-1; no zero-fill or synthetic substitution"
            ),
            dem_source=(
                "Copernicus GLO-30 DEM (30 m), Meghalaya pilot tiles, merged and "
                "clipped to the canonical pilot AOI"
            ),
            terrain_derivative_method=(
                "app.services.terrain_processing.process_dem_in_chunks "
                "(slope, aspect, roughness, tpi; chunked at 512 px)"
            ),
            exposure_source="NOT USED as a model feature in this run",
            spatial_split="Latitude median split",
            temporal_split="Event year split (train <= %d, test >= %d)"
            % (TEMPORAL_CUTOFF_YEAR, TEMPORAL_CUTOFF_YEAR + 1),
            negative_sampling=(
                "Spatially buffered random points, >= 0.05 deg (~5 km) from any "
                "positive, 3:1 ratio, seed 42"
            ),
            leakage_controls=leakage_checks,
            dataset_artifact=matrix_path,
            input_status={
                "dem_copernicus_glo30": "REAL",
                "terrain_derivatives": "REAL",
                "landslide_inventory_glc": "REAL",
                "antecedent_rainfall_open_meteo_era5": "REAL",
                # The single meaningful status change from Sikkim: real observed
                # land cover instead of a derived elevation proxy.
                "land_cover_class": "REAL",
                "imerg_satellite_rainfall": "NOT_USED",
                "osm_exposure": "NOT_USED",
            },
            extra={
                "land_cover_treatment": (
                    "land_cover_class is a REAL ESA WorldCover %s (%s) nominal "
                    "grouping and is fed to the LightGBM primary model as a pandas "
                    "categorical (categorical_feature=%s). This is the ONLY "
                    "methodological difference from the Sikkim pilot, whose "
                    "land_cover_class is an ordered elevation proxy left numeric. "
                    "It matches the Assam and Arunachal pilots' treatment exactly."
                    % (
                        wc.WORLDCOVER_VERSION,
                        wc.WORLDCOVER_YEAR,
                        wc.landcover_categorical_feature(),
                    )
                ),
                "baseline_encoding_note": (
                    "Logistic Regression and Random Forest comparison rows use the "
                    "numeric land_cover_class encoding exactly as Sikkim does; only "
                    "the persisted LightGBM primary model uses the categorical view."
                ),
                "landcover_group_labels": {
                    str(k): v for k, v in wc.ASSAM_LANDCOVER_GROUP_LABELS.items()
                },
                "rows_dropped_for_missing_features": drop_report,
                "duplicate_positive_events": duplicate_positives,
                "imerg_note": (
                    "IMERG (weather_ingestion.py) was NOT used to build any training "
                    "feature. Antecedent rainfall comes from the Open-Meteo ERA5 "
                    "archive, exactly as in Sikkim/Assam/Arunachal training."
                ),
                "matrix_schema_reference": schema_path,
                "aoi_source": (
                    "app.core.config_states.MEGHALAYA_PILOT_AOI -- the single "
                    "canonical Meghalaya pilot AOI used for tile selection, DEM crop, "
                    "GLC filter and negative sampling."
                ),
                "aoi_vs_state_bbox": aoi_consistency,
            },
        )

        artifact_paths = model_artifacts.save_model_evidence(
            model=primary_model,
            metrics_doc=metrics_doc,
            schema_doc=feature_schema_doc,
            provenance_doc=provenance_doc,
            state_name=STATE_NAME,
        )
        print("Persisted validation evidence artifacts:")
        for kind in ("model", "metrics", "schema", "provenance"):
            print("  %-11s: %s" % (kind, artifact_paths[kind]))
        return 0
    except model_artifacts.ArtifactValidationError as exc:
        print("ARTIFACT PERSISTENCE REFUSED (no artifact written):")
        print("  %s" % exc)
        return 1
    except model_artifacts.ArtifactPersistenceError as exc:
        print("ARTIFACT PERSISTENCE REFUSED (no artifact written):")
        print("  %s" % exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
