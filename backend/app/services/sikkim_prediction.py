"""
SIKKIM GRID PREDICTION SERVICE

Runs the persisted rainfall-coupled Sikkim LightGBM over a coarse grid tiling the
canonical East Sikkim pilot AOI and returns, per grid cell, the model's raw
susceptibility probability and the system warning class.

WHY THIS IS A SEPARATE PATH (and does not reuse app.services.risk_inputs).
`risk_inputs.resolve_model_input` DELIBERATELY refuses rainfall-coupled models:
its Option-C design treats rainfall as a separate trigger multiplier applied on
top of a static-only susceptibility, so rainfall is never double-counted. The
persisted Sikkim artifact, however, IS the 11-feature `static_plus_rainfall`
model. Running THAT model with real antecedent rainfall is a distinct, valid
operation, so it lives here on its own path and never touches Option-C fusion.
The number returned here is the model's RAW positive-class probability, NOT the
Option-C fused final_risk_score served by /risk/current.

DATA-INTEGRITY CONTRACT (identical in spirit to routes.py / risk_inputs.py):
this module answers only from real, resolved inputs or it refuses. It never
fabricates a coordinate, a terrain value, a rainfall value or a prediction, never
back-fills a nodata cell with a placeholder, and never hard-codes a probability.
A cell whose terrain is missing/nodata/out-of-coverage is returned with status
UNAVAILABLE and NO probability; if the model artifacts or the real rainfall
cannot be obtained, the whole request is refused (PredictionUnavailable -> 503).

Heavy, host-only dependencies (rasterio, joblib via model_artifacts, pandas, and
lightgbm via the model object + ml_pipeline.calculate_warning_level) are imported
LAZILY so this module imports cleanly in the offline test sandbox. The full
pipeline is exercised offline by injecting fake `model_evidence`,
`terrain_sampler` and `rainfall_provider` collaborators; the real end-to-end run
(load .pkl, read the DEM rasters, fetch real IMERG, predict_proba) is host-only.
"""

import math
from datetime import date as _date_cls, datetime, timedelta

from app.core.config_states import get_pilot_aoi_bounds
from app.services import risk_inputs


# The 11 features in the EXACT order the persisted model was fit on
# (data/models/sikkim_feature_schema.json). Inference must supply this order.
MODEL_FEATURE_ORDER = (
    "elevation",
    "slope",
    "aspect",
    "roughness",
    "tpi",
    "land_cover_class",
    "rain_1d",
    "rain_3d",
    "rain_7d",
    "antecedent_rain_14d",
    "rain_intensity_max_3d",
)
# Single-sourced from risk_inputs so the two paths cannot drift apart.
TERRAIN_FEATURES = risk_inputs.TERRAIN_FEATURE_NAMES        # elevation, slope, aspect, roughness, tpi
LAND_COVER_FEATURE = "land_cover_class"
RAINFALL_FEATURES = risk_inputs.RAINFALL_FEATURE_NAMES      # rain_1d, rain_3d, rain_7d, antecedent_rain_14d, rain_intensity_max_3d

# float32 terrain + rainfall, int32 land_cover_class (per the schema dtype block).
_INT_FEATURES = (LAND_COVER_FEATURE,)

# Antecedent rainfall window: schema semantics are T-1..T-14 (no event-day leakage).
RAINFALL_WINDOW_DAYS = 14

# Coarse grid controls. The full 30 m DEM grid (~3960 x 3240 ~ 12.8M pixels) is
# infeasible to score per request; a coarse grid over the AOI is required.
DEFAULT_STEP_DEG = 0.05
MIN_STEP_DEG = 0.02
MAX_STEP_DEG = 0.25
MAX_CELLS = 1500

# The validated operating threshold recorded in the metrics artifact.
DECISION_THRESHOLD = 0.5
POSITIVE_CLASS_LABEL = 1

DATA_UNAVAILABLE = risk_inputs.DATA_UNAVAILABLE
DEFAULT_STATE_NAME = risk_inputs.DEFAULT_STATE_NAME


class PredictionUnavailable(Exception):
    """
    Raised when the request cannot be answered from real inputs (missing/invalid
    model artifacts, a feature-order mismatch, or real rainfall that could not be
    obtained). Carries a machine-readable `reason` and optional `details`; the API
    layer renders it as HTTP 503 DATA_UNAVAILABLE. It never carries a fabricated
    prediction.
    """

    def __init__(self, reason, details=None):
        super().__init__(reason)
        self.reason = reason
        self.details = details or {}


def _finite(value):
    """float(value) if finite, else None (mirrors risk_inputs._finite_number)."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _as_datetime(value):
    """Coerce a datetime / date / 'YYYY-MM-DD' string into a datetime."""
    if isinstance(value, datetime):
        return value
    if isinstance(value, _date_cls):
        return datetime(value.year, value.month, value.day)
    if isinstance(value, str):
        return datetime.strptime(value, "%Y-%m-%d")
    raise ValueError(
        "target_date must be a datetime, date, or 'YYYY-MM-DD' string, got %r" % (value,)
    )


# ---------------------------------------------------------------------------
# Grid
# ---------------------------------------------------------------------------
def build_grid(step_deg=DEFAULT_STEP_DEG, state_name=DEFAULT_STATE_NAME):
    """
    Tiles the canonical pilot AOI into (near-)square cells of side ~`step_deg`.

    Returns (bounds, grid_meta, cells). Each cell carries its integer row/col, a
    center lat/lon placed STRICTLY inside the AOI (so point_within_pilot_aoi
    holds) and its bbox footprint. Raises ValueError for an out-of-range step or a
    cell count over MAX_CELLS (the API maps ValueError to HTTP 400).
    """
    step = float(step_deg)
    if not (MIN_STEP_DEG <= step <= MAX_STEP_DEG):
        raise ValueError(
            "step_deg=%s is out of range [%s, %s]." % (step, MIN_STEP_DEG, MAX_STEP_DEG)
        )
    bounds = get_pilot_aoi_bounds(state_name)
    span_lat = bounds["max_lat"] - bounds["min_lat"]
    span_lon = bounds["max_lon"] - bounds["min_lon"]
    n_lat = max(1, int(round(span_lat / step)))
    n_lon = max(1, int(round(span_lon / step)))
    if n_lat * n_lon > MAX_CELLS:
        raise ValueError(
            "grid %dx%d = %d cells exceeds MAX_CELLS=%d; use a coarser step."
            % (n_lat, n_lon, n_lat * n_lon, MAX_CELLS)
        )
    cell_h = span_lat / n_lat
    cell_w = span_lon / n_lon
    cells = []
    for i in range(n_lat):
        lo_lat = bounds["min_lat"] + i * cell_h
        c_lat = lo_lat + 0.5 * cell_h
        for j in range(n_lon):
            lo_lon = bounds["min_lon"] + j * cell_w
            c_lon = lo_lon + 0.5 * cell_w
            cells.append({
                "cell_id": "r%02dc%02d" % (i, j),
                "row": i,
                "col": j,
                "latitude": round(c_lat, 6),
                "longitude": round(c_lon, 6),
                "bbox": {
                    "min_lat": round(lo_lat, 6),
                    "max_lat": round(lo_lat + cell_h, 6),
                    "min_lon": round(lo_lon, 6),
                    "max_lon": round(lo_lon + cell_w, 6),
                },
            })
    grid_meta = {
        "step_deg": step,
        "n_lat": n_lat,
        "n_lon": n_lon,
        "cell_count": len(cells),
        "cell_height_deg": round(cell_h, 6),
        "cell_width_deg": round(cell_w, 6),
    }
    return bounds, grid_meta, cells


# ---------------------------------------------------------------------------
# Terrain (real rasters; batched to open each raster once)
# ---------------------------------------------------------------------------
def _default_terrain_sampler(centers, data_dir=None, rasterio_module=None):
    """
    Batched real-terrain sampler: opens each of the five terrain rasters ONCE and
    samples every grid-cell center, replicating resolve_terrain's exact rules
    (AOI/raster coverage, NODATA_SENTINEL, finite check). Returns a list parallel
    to `centers`, each entry {"values": {feat: float} | None, "problems": [str]}.

    Raises PredictionUnavailable if any terrain raster is missing/empty or rasterio
    is unavailable -- a SYSTEMIC condition, mirroring resolve_terrain refusing for
    a missing raster. A single nodata / out-of-coverage cell is reported per-cell
    (values None) without a placeholder, never as a systemic refusal.
    """
    missing = risk_inputs.missing_terrain_rasters(data_dir)
    if missing:
        raise PredictionUnavailable(
            "Missing or empty terrain raster(s): "
            + ", ".join("%s" % name for name, _ in missing),
            details={"missing_rasters": {name: path for name, path in missing}},
        )
    if rasterio_module is None:
        try:
            import rasterio as rasterio_module  # lazy, host-only
        except ImportError as exc:
            raise PredictionUnavailable(
                "rasterio is not installed, so the real terrain rasters cannot be "
                "read (%s)." % exc
            )

    paths = risk_inputs.terrain_raster_paths(data_dir)
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
            number = _finite(samples[idx])
            if number is None:
                problems[idx].append(
                    "'%s' sampled a non-finite value at (lat=%s, lon=%s)." % (feat, lat, lon)
                )
                continue
            if number == risk_inputs.NODATA_SENTINEL or (
                nodata is not None and _finite(nodata) == number
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
# Rainfall (real IMERG antecedent series, AOI-uniform)
# ---------------------------------------------------------------------------
def _derive_rainfall_features(daily):
    """
    daily[k] = precip mm for day T-(k+1); returns the five schema features.

        rain_1d               = day T-1
        rain_3d               = sum(T-1..T-3)
        rain_7d               = sum(T-1..T-7)
        antecedent_rain_14d   = sum(T-1..T-14)
        rain_intensity_max_3d = max(T-1..T-3)
    """
    if len(daily) < RAINFALL_WINDOW_DAYS:
        raise ValueError(
            "need %d antecedent days, got %d" % (RAINFALL_WINDOW_DAYS, len(daily))
        )
    vals = [float(v) for v in daily]
    return {
        "rain_1d": vals[0],
        "rain_3d": float(sum(vals[0:3])),
        "rain_7d": float(sum(vals[0:7])),
        "antecedent_rain_14d": float(sum(vals[0:14])),
        "rain_intensity_max_3d": float(max(vals[0:3])),
    }


def _default_rainfall_provider(bounds, target_date, run_type="Early", session=None):
    """
    Derives the model's five antecedent-rainfall features from REAL IMERG daily
    means over the pilot AOI.

    If the requested target date is too recent for the selected IMERG run,
    search backward for the latest available observation. No synthetic values,
    zero-filling, or duplicated rainfall days are allowed.
    """
    from app.services import weather_ingestion

    sess = session if session is not None else weather_ingestion.get_earthdata_session()

    # Find the latest available IMERG observation before the target date.
    latest_date = None
    max_probe_days = 30

    for offset in range(1, max_probe_days + 1):
        candidate = target_date - timedelta(days=offset)
        try:
            weather_ingestion._fetch_imerg_day(
                sess, candidate, bounds, run_type
            )
            latest_date = candidate
            break
        except Exception as exc:
            # Only continue searching for unavailable granules.
            if "404" not in str(exc):
                raise

    if latest_date is None:
        raise PredictionUnavailable(
            "No available IMERG %s observation found within %d days before %s."
            % (run_type, max_probe_days, target_date.date())
        )

    # Fetch a consecutive 14-day antecedent window ending at the
    # latest available observation. Never duplicate a day.
    daily = []
    for k in range(RAINFALL_WINDOW_DAYS):
        day = latest_date - timedelta(days=k)
        daily.append(
            float(
                weather_ingestion._fetch_imerg_day(
                    sess, day, bounds, run_type
                )
            )
        )

    return {
        "source": "IMERG_%s" % run_type,
        "run_type": run_type,
        "aoi_uniform": True,
        "window_days": RAINFALL_WINDOW_DAYS,
        "requested_date": target_date.strftime("%Y-%m-%d"),
        "rainfall_observation_date": latest_date.strftime("%Y-%m-%d"),
        "daily_series_mm": [round(v, 4) for v in daily],
        "features": _derive_rainfall_features(daily),
    }


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
def _resolve_model_evidence(model_evidence, state_name, artifact_dir):
    """Load (or accept an injected) evidence bundle and validate it can be used."""
    if model_evidence is None:
        from app.services import model_artifacts  # lazy: joblib on host

        model_evidence = model_artifacts.load_model_evidence(
            state_name=state_name, base_dir=artifact_dir,
            load_model=True, require_full_metrics=False,
        )
    status = model_evidence.get("status")
    if status != "VALID" or model_evidence.get("model") is None:
        raise PredictionUnavailable(
            "The persisted %s model artifacts are not usable (status=%s)."
            % (state_name, status),
            details={"status": status, "problems": model_evidence.get("problems", [])},
        )
    schema = model_evidence.get("feature_schema") or {}
    names = tuple(schema.get("feature_names") or ())
    if names != MODEL_FEATURE_ORDER:
        raise PredictionUnavailable(
            "Persisted feature schema does not match the expected 11-feature "
            "order; refusing to guess the column order.",
            details={"expected": list(MODEL_FEATURE_ORDER), "actual": list(names)},
        )
    return model_evidence


def _score_cells(model, feature_rows):
    """
    feature_rows: list of dicts, each holding all 11 MODEL_FEATURE_ORDER keys.
    Returns positive-class probabilities aligned to feature_rows, computed from a
    typed pandas DataFrame built in the exact model column order.
    """
    import pandas as pd  # available offline; lazy to match the module's ethos

    frame = pd.DataFrame(feature_rows, columns=list(MODEL_FEATURE_ORDER))
    for col in MODEL_FEATURE_ORDER:
        frame[col] = frame[col].astype("int32" if col in _INT_FEATURES else "float32")

    proba = model.predict_proba(frame)
    classes = list(getattr(model, "classes_", [0, 1]))
    if POSITIVE_CLASS_LABEL in classes:
        pos_idx = classes.index(POSITIVE_CLASS_LABEL)
    else:
        pos_idx = len(classes) - 1
    return [float(row[pos_idx]) for row in proba]


def _warning_level(probability):
    from app.models.ml_pipeline import calculate_warning_level  # lazy: lightgbm/shap on host

    return calculate_warning_level(probability)


# ---------------------------------------------------------------------------
# Report assembly
# ---------------------------------------------------------------------------
def _validate_rainfall_features(features):
    for feat in RAINFALL_FEATURES:
        if feat not in features or _finite(features[feat]) is None:
            raise PredictionUnavailable(
                "Rainfall feature '%s' is missing or non-finite; refusing to "
                "substitute a value." % feat,
                details={"rainfall_features": {k: features.get(k) for k in RAINFALL_FEATURES}},
            )


def _round_features(row):
    out = {}
    for feat in MODEL_FEATURE_ORDER:
        value = row[feat]
        out[feat] = int(value) if feat in _INT_FEATURES else round(float(value), 4)
    return out


def _model_report(evidence):
    metrics = evidence.get("metrics") or {}
    vmetrics = metrics.get("validation_metrics") if isinstance(metrics, dict) else None
    report = {
        "feature_order": list(MODEL_FEATURE_ORDER),
        "n_features": len(MODEL_FEATURE_ORDER),
        "decision_threshold": DECISION_THRESHOLD,
        "artifact_status": evidence.get("status"),
    }
    # Pass the real validation metrics through unchanged; never synthesise keys.
    if isinstance(vmetrics, dict):
        report["validation_metrics"] = vmetrics
    return report


def _rainfall_report(rainfall):
    return {
        "source": rainfall.get("source"),
        "run_type": rainfall.get("run_type"),
        "aoi_uniform": rainfall.get("aoi_uniform", True),
        "window_days": rainfall.get("window_days", RAINFALL_WINDOW_DAYS),
        "daily_series_mm": rainfall.get("daily_series_mm"),
        "features": {k: round(float(rainfall["features"][k]), 4) for k in RAINFALL_FEATURES},
        "note": (
            "Antecedent-only (T-1..T-14, event day excluded). One AOI-mean IMERG "
            "series applied uniformly to all cells."
        ),
    }


def _disclosures():
    return [
        "Output is the model's RAW positive-class probability (susceptibility), "
        "NOT the Option-C fused final_risk_score served by /risk/current; here "
        "rainfall is a model feature, not a separate trigger multiplier.",
        "TRAIN/SERVE RAINFALL-SOURCE SHIFT: the model was trained on Open-Meteo "
        "ERA5 reanalysis rainfall (provenance records IMERG as NOT_USED at "
        "training); this endpoint serves NASA GPM IMERG. The sources differ, which "
        "may affect probability calibration.",
        "Rainfall is a single AOI-mean IMERG series applied UNIFORMLY to every "
        "grid cell (IMERG's ~0.1 deg grid is coarser than the prediction grid), so "
        "per-cell variation is driven by terrain, not by spatial rainfall "
        "differences.",
        "land_cover_class is the documented ELEVATION-BINNED PROXY (not an observed "
        "land-cover product): <3000 m -> 1, [3000,4200) m -> 2, >=4200 m -> 3.",
        "risk_class maps the probability onto the system warning bands via "
        "ml_pipeline.calculate_warning_level (LOW <0.40, MEDIUM 0.40-0.65, HIGH "
        "0.65-0.85, EXTREME >=0.85); exceeds_decision_threshold uses the validated "
        "operating threshold %.2f." % DECISION_THRESHOLD,
        "Cells whose terrain is missing, nodata or outside raster coverage are "
        "returned with status UNAVAILABLE and no probability -- never a placeholder.",
        "This is a COARSE grid over the pilot AOI, not the native 30 m DEM "
        "resolution; each cell reports the model output sampled at its center.",
    ]


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def predict_sikkim_grid(target_date, step_deg=DEFAULT_STEP_DEG, run_type="Early",
                        state_name=DEFAULT_STATE_NAME, data_dir=None, artifact_dir=None,
                        model_evidence=None, terrain_sampler=None, rainfall_provider=None):
    """
    Full read-only prediction over the coarse pilot-AOI grid. See the module
    docstring for the data-integrity contract. The `model_evidence`,
    `terrain_sampler` and `rainfall_provider` collaborators are injectable so the
    assembly and the no-fabrication invariants are testable offline with fakes; the
    defaults run the real host-only pipeline.

    Returns a JSON-safe dict. Raises PredictionUnavailable (-> HTTP 503) when the
    model or real rainfall cannot be obtained, or ValueError (-> HTTP 400) for a
    bad grid step.
    """
    target_date = _as_datetime(target_date)

    # 1. Model (real artifacts, correct feature order) -- refuse if unusable.
    evidence = _resolve_model_evidence(model_evidence, state_name, artifact_dir)
    model = evidence["model"]

    # 2. Grid.
    bounds, grid_meta, cells = build_grid(step_deg=step_deg, state_name=state_name)
    centers = [(c["latitude"], c["longitude"]) for c in cells]

    # 3. Real antecedent rainfall (AOI-uniform) -- required for every cell.
    provider = rainfall_provider or _default_rainfall_provider
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
    _validate_rainfall_features(rain_features)

    # 4. Terrain per cell.
    sampler = terrain_sampler or _default_terrain_sampler
    samples = sampler(centers, data_dir)
    if len(samples) != len(cells):
        raise PredictionUnavailable(
            "terrain sampler returned %d rows for %d cells." % (len(samples), len(cells))
        )

    # 5. Assemble usable feature rows; mark unusable cells UNAVAILABLE (no fill).
    scorable_idx = []
    feature_rows = []
    for idx, (cell, sample) in enumerate(zip(cells, samples)):
        terrain_values = sample.get("values")
        if not terrain_values:
            _mark_unavailable(cell, sample.get("problems") or ["terrain unavailable at this cell"])
            continue
        try:
            land_cover = risk_inputs.land_cover_class_from_elevation(terrain_values["elevation"])
        except (ValueError, KeyError) as exc:
            _mark_unavailable(cell, [str(exc)])
            continue
        row = {feat: float(terrain_values[feat]) for feat in TERRAIN_FEATURES}
        row[LAND_COVER_FEATURE] = int(land_cover)
        for feat in RAINFALL_FEATURES:
            row[feat] = float(rain_features[feat])
        cell["_features"] = row
        scorable_idx.append(idx)
        feature_rows.append(row)

    # 6. Score the usable cells.
    probabilities = _score_cells(model, feature_rows) if feature_rows else []

    risk_counts = {"LOW": 0, "MEDIUM": 0, "HIGH": 0, "EXTREME": 0}
    probs_only = []
    for pos, idx in enumerate(scorable_idx):
        cell = cells[idx]
        prob = float(probabilities[pos])
        level = _warning_level(prob)
        cell["status"] = "OK"
        cell["susceptibility_probability"] = round(prob, 6)
        cell["risk_class"] = level
        cell["exceeds_decision_threshold"] = bool(prob >= DECISION_THRESHOLD)
        cell["features"] = _round_features(cell.pop("_features"))
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
        "pilot_area": "East Sikkim",
        "generated_from": (
            "persisted LightGBM (static_plus_rainfall, 11 features) + real IMERG "
            "antecedent rainfall"
        ),
        "target_date": target_date.strftime("%Y-%m-%d"),
        "aoi": bounds,
        "grid": grid_meta,
        "decision_threshold": DECISION_THRESHOLD,
        "model": _model_report(evidence),
        "rainfall": _rainfall_report(rainfall),
        "summary": summary,
        "disclosures": _disclosures(),
        "cells": cells,
    }


def _mark_unavailable(cell, reasons):
    cell["status"] = "UNAVAILABLE"
    cell["susceptibility_probability"] = None
    cell["risk_class"] = None
    cell["exceeds_decision_threshold"] = None
    cell["reasons"] = list(reasons)
    cell.pop("_features", None)
