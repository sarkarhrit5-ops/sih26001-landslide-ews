"""
REAL-INPUT RESOLUTION FOR THE LIVE RISK SERVING PATH
====================================================

Before this module existed, `/risk/current` and `/risk/forecast` built their
answer out of four invented numbers -- susceptibility 0.65, current rainfall
55.0 mm, slope 35.0 deg, exposure 0.5 -- and then declared
`has_real_dem=True, has_real_rainfall=True`, so the API reported HIGH confidence
for a result that contained no measurement whatsoever.

This module resolves each of those inputs either from a real artifact / real
service, or reports it UNAVAILABLE together with the specific reason. It never:

  * substitutes a default, a mean, a zero or a "typical" value for a missing
    measurement,
  * silently swallows a failure from a data source,
  * uses the quarantined documentary reference metrics, or any hardcoded score,
  * extrapolates the pilot model outside the AOI it was actually fitted on.

The status vocabulary is the same one the persisted-artifact contract uses
(`model_artifacts.INPUT_STATUS_VALUES`): REAL, UNAVAILABLE, NOT_USED,
DERIVED_PROXY. `test_risk_inputs.py` pins the two definitions together.

Import cost is deliberately tiny (stdlib + the pure-python AOI config). rasterio,
pandas, requests, xarray, joblib and the estimator libraries are imported lazily
inside the resolver that needs them, so this module -- and its tests -- stay
importable in an environment where those packages are absent. That is also why a
missing package is reported as an ordinary UNAVAILABLE reason rather than an
import-time crash.
"""

import math
import os
from datetime import datetime, timedelta, timezone

from app.core.config_states import get_pilot_aoi_bounds

# ---------------------------------------------------------------------------
# Status vocabulary
# ---------------------------------------------------------------------------
STATUS_REAL = "REAL"
STATUS_UNAVAILABLE = "UNAVAILABLE"
STATUS_NOT_USED = "NOT_USED"
STATUS_DERIVED_PROXY = "DERIVED_PROXY"

# Must stay equal to model_artifacts.INPUT_STATUS_VALUES (asserted by a test).
INPUT_STATUS_VALUES = (
    STATUS_REAL, STATUS_UNAVAILABLE, STATUS_NOT_USED, STATUS_DERIVED_PROXY,
)

# A resolved input may be fed to the risk model only with one of these statuses.
# DERIVED_PROXY is accepted because the persisted model was itself *trained* on
# the documented elevation-binned land-cover proxy: refusing it at inference
# while using it at training would be the inconsistency, not the honesty.
USABLE_STATUSES = (STATUS_REAL, STATUS_DERIVED_PROXY)

# Response-level token the API uses when a required measurement is absent.
DATA_UNAVAILABLE = "DATA_UNAVAILABLE"

DEFAULT_STATE_NAME = "Sikkim"

# ---------------------------------------------------------------------------
# Artifact layout -- mirrors what scripts/train_real_models.py actually writes
# ---------------------------------------------------------------------------
DEM_FILENAME = "east_sikkim_dem.tif"
TERRAIN_DERIVATIVE_PREFIX = "real_"
TERRAIN_DERIVATIVE_NAMES = ("slope", "aspect", "roughness", "tpi")
TERRAIN_FEATURE_NAMES = ("elevation",) + TERRAIN_DERIVATIVE_NAMES

# terrain_processing.process_dem_in_chunks writes this fill value.
NODATA_SENTINEL = -9999.0

# Feature-name groups. The rainfall group is the reason susceptibility cannot be
# taken from a rainfall-coupled model (see resolve_susceptibility).
RAINFALL_FEATURE_NAMES = (
    "rain_1d", "rain_3d", "rain_7d", "antecedent_rain_14d",
    "rain_intensity_max_3d",
)
PROXY_FEATURE_NAMES = ("land_cover_class",)
SUPPORTED_SERVING_FEATURES = TERRAIN_FEATURE_NAMES + PROXY_FEATURE_NAMES

# ---------------------------------------------------------------------------
# Land-cover proxy -- SINGLE definition site
# ---------------------------------------------------------------------------
# scripts/train_real_models.assign_land_cover_proxy reads these same constants,
# so the class boundaries used at inference cannot drift from the ones used at
# training (a silent drift would mean serving a feature the model never saw).
LAND_COVER_ELEVATION_BREAKS_M = (3000.0, 4200.0)
LAND_COVER_PROXY_CLASSES = (1, 2, 3)
LAND_COVER_PROXY_LABELS = {
    1: "tree cover / dense forest (<3000 m)",
    2: "shrubland / alpine scrub (3000-4200 m)",
    3: "bare rock / sparse vegetation / snow (>=4200 m)",
}

# ---------------------------------------------------------------------------
# Service parameters
# ---------------------------------------------------------------------------
# IMERG is a 0.1-degree product; +/-0.05 deg selects the cell containing the
# point instead of averaging rainfall over the whole AOI.
POINT_BBOX_HALF_WIDTH_DEG = 0.05
# GPM_3IMERGDE daily files exist only for completed UTC days.
IMERG_LATENCY_DAYS = 1
DEFAULT_IMERG_RUN_TYPE = "Early"
DEFAULT_FORECAST_HORIZON_HOURS = 72
CURRENT_RAINFALL_WINDOW_HOURS = 24.0

EXPOSURE_UNAVAILABLE_REASON = (
    "No validated mapping from OSM asset counts to a 0-1 exposure score exists "
    "anywhere in this repository, so any exposure_score the API emitted would be "
    "an invented number. exposure_score is also not an input to final_risk_score "
    "(final_risk_score = min(1, susceptibility * trigger_multiplier)), so it is "
    "reported as UNAVAILABLE rather than blocking the whole response."
)

RISK_MODE_CURRENT = "current"
RISK_MODE_FORECAST = "forecast"
RISK_MODES = (RISK_MODE_CURRENT, RISK_MODE_FORECAST)

INPUT_SUSCEPTIBILITY = "susceptibility"
INPUT_MODEL_INPUT = "model_input"
INPUT_SLOPE = "slope_deg"
INPUT_CURRENT_RAINFALL = "current_rainfall_mm"
INPUT_FORECAST_RAINFALL = "forecast_rainfall_mm"
INPUT_EXPOSURE = "exposure_score"

REQUIRED_INPUTS_BY_MODE = {
    RISK_MODE_CURRENT: (INPUT_SUSCEPTIBILITY, INPUT_SLOPE, INPUT_CURRENT_RAINFALL),
    RISK_MODE_FORECAST: (
        INPUT_SUSCEPTIBILITY, INPUT_SLOPE, INPUT_CURRENT_RAINFALL,
        INPUT_FORECAST_RAINFALL,
    ),
}
# Resolved but never blocking: not consumed by the risk arithmetic.
NON_BLOCKING_INPUTS = (INPUT_EXPOSURE,)


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def default_data_dir():
    """backend/data -- the directory the training pipeline reads and writes."""
    return os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "data")
    )


def terrain_raster_paths(data_dir=None):
    """
    Absolute paths of the five real terrain rasters, keyed by feature name.

    These are byte-for-byte the paths scripts/train_real_models.py consumes:
    the DEM at data/raw/east_sikkim_dem.tif and the Horn-derivative rasters at
    data/processed/real_<name>.tif.
    """
    root = os.path.abspath(data_dir or default_data_dir())
    paths = {"elevation": os.path.join(root, "raw", DEM_FILENAME)}
    for name in TERRAIN_DERIVATIVE_NAMES:
        paths[name] = os.path.join(
            root, "processed", "%s%s.tif" % (TERRAIN_DERIVATIVE_PREFIX, name)
        )
    return paths


def missing_terrain_rasters(data_dir=None):
    """[(feature_name, path)] for every terrain raster that is absent or empty."""
    return [
        (name, path)
        for name, path in sorted(terrain_raster_paths(data_dir).items())
        if not (os.path.exists(path) and os.path.getsize(path) > 0)
    ]


def _display_path(path):
    """
    Repo-relative form of a path, for text that is returned to API clients.
    Absolute server paths belong in logs and in the `details` block, not in a
    public error message.
    """
    backend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    try:
        relative = os.path.relpath(os.path.abspath(path), backend_dir)
    except ValueError:  # pragma: no cover - different drive on Windows
        return os.path.basename(path)
    if relative.startswith(os.pardir):
        return os.path.basename(path)
    return "backend/" + relative.replace(os.sep, "/")


def input_record(name, status, value=None, source=None, reasons=None, details=None):
    """Normalised per-input record. `value` is None unless the input is usable."""
    if status not in INPUT_STATUS_VALUES:
        raise ValueError(
            "Unknown input status %r; allowed: %s" % (status, list(INPUT_STATUS_VALUES))
        )
    if status not in USABLE_STATUSES:
        value = None
    record = {
        "name": name,
        "status": status,
        "value": value,
        "source": source,
        "reasons": [str(reason) for reason in (reasons or [])],
    }
    if details is not None:
        record["details"] = details
    return record


def _unavailable(name, reasons, details=None):
    return input_record(name, STATUS_UNAVAILABLE, reasons=reasons, details=details)


def _finite_number(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def point_within_pilot_aoi(lat, lon, state_name=DEFAULT_STATE_NAME):
    """True if (lat, lon) is inside the canonical pilot AOI rectangle."""
    bounds = get_pilot_aoi_bounds(state_name)
    return (
        bounds["min_lat"] <= float(lat) <= bounds["max_lat"]
        and bounds["min_lon"] <= float(lon) <= bounds["max_lon"]
    )


def point_bounding_box(lat, lon, half_width_deg=POINT_BBOX_HALF_WIDTH_DEG):
    """
    A single-grid-cell bbox around a point, in the {min_lat,...} shape the
    rainfall subsetting helpers expect. Used so point rainfall is not silently
    replaced by an AOI-wide average.
    """
    half = float(half_width_deg)
    return {
        "min_lat": max(-90.0, float(lat) - half),
        "max_lat": min(90.0, float(lat) + half),
        "min_lon": max(-180.0, float(lon) - half),
        "max_lon": min(180.0, float(lon) + half),
    }


def land_cover_class_from_elevation(elevation_m):
    """
    The documented elevation-binned land-cover proxy, as a scalar.

    Matches scripts/train_real_models.assign_land_cover_proxy exactly:
    <3000 m -> 1, [3000, 4200) m -> 2, >=4200 m -> 3.

    Difference from the training-time version, on purpose: that one is vectorised
    with `np.select(..., default=1)`, so a nodata elevation silently becomes
    class 1. Here a non-finite elevation raises instead, because at serving time
    an unusable elevation must surface as UNAVAILABLE, not as "tree cover".
    """
    elevation = _finite_number(elevation_m)
    if elevation is None:
        raise ValueError(
            "Cannot derive land_cover_class from a non-numeric or non-finite "
            "elevation (%r)." % (elevation_m,)
        )
    lower, upper = LAND_COVER_ELEVATION_BREAKS_M
    if elevation < lower:
        return LAND_COVER_PROXY_CLASSES[0]
    if elevation < upper:
        return LAND_COVER_PROXY_CLASSES[1]
    return LAND_COVER_PROXY_CLASSES[2]


# ---------------------------------------------------------------------------
# Terrain
# ---------------------------------------------------------------------------
def resolve_terrain(lat, lon, data_dir=None, state_name=DEFAULT_STATE_NAME):
    """
    Samples the five real terrain rasters at a point.

    REAL only if every raster exists, covers the point, and returns a finite
    non-nodata value. A nodata or out-of-coverage sample is reported, never
    replaced.
    """
    name = "terrain"
    if not point_within_pilot_aoi(lat, lon, state_name):
        bounds = get_pilot_aoi_bounds(state_name)
        return _unavailable(name, [
            "Point (lat=%s, lon=%s) lies outside the canonical pilot AOI "
            "(min_lat=%s, max_lat=%s, min_lon=%s, max_lon=%s). The DEM mosaic and "
            "the pilot model cover only that rectangle, so there is no terrain "
            "measurement here and extrapolating the model would be fabrication."
            % (lat, lon, bounds["min_lat"], bounds["max_lat"],
               bounds["min_lon"], bounds["max_lon"])
        ], details={"pilot_aoi": bounds})

    paths = terrain_raster_paths(data_dir)
    missing = missing_terrain_rasters(data_dir)
    if missing:
        return _unavailable(name, [
            "Missing or empty terrain raster '%s' (expected at %s)."
            % (key, _display_path(path))
            for key, path in missing
        ], details={"expected_paths": paths})

    try:
        import rasterio
    except ImportError as exc:
        return _unavailable(name, [
            "rasterio is not installed, so the real terrain rasters cannot be "
            "read (%s)." % exc
        ], details={"expected_paths": paths})

    values = {}
    problems = []
    for key in TERRAIN_FEATURE_NAMES:
        path = paths[key]
        try:
            with rasterio.open(path) as src:
                bounds = src.bounds
                if not (bounds.left <= float(lon) <= bounds.right
                        and bounds.bottom <= float(lat) <= bounds.top):
                    problems.append(
                        "'%s' raster does not cover (lat=%s, lon=%s); its extent is "
                        "lon %s..%s, lat %s..%s."
                        % (key, lat, lon, bounds.left, bounds.right,
                           bounds.bottom, bounds.top)
                    )
                    continue
                sample = next(src.sample([(float(lon), float(lat))]))[0]
                nodata = src.nodata
        except Exception as exc:
            problems.append(
                "'%s' raster could not be sampled at (lat=%s, lon=%s): %s: %s"
                % (key, lat, lon, type(exc).__name__, exc)
            )
            continue

        number = _finite_number(sample)
        if number is None:
            problems.append(
                "'%s' sampled a non-finite value at (lat=%s, lon=%s); refusing to "
                "substitute a placeholder." % (key, lat, lon)
            )
            continue
        if number == NODATA_SENTINEL or (
            nodata is not None and _finite_number(nodata) == number
        ):
            problems.append(
                "'%s' is nodata at (lat=%s, lon=%s) (value %s); the point is inside "
                "the raster extent but has no measurement." % (key, lat, lon, number)
            )
            continue
        values[key] = number

    if problems:
        return _unavailable(name, problems, details={"expected_paths": paths})

    return input_record(
        name,
        STATUS_REAL,
        value=values,
        source="Real terrain rasters derived from the Copernicus GLO-30 DEM",
        details={"paths": {key: paths[key] for key in TERRAIN_FEATURE_NAMES}},
    )


def resolve_slope(lat, lon, data_dir=None, state_name=DEFAULT_STATE_NAME,
                  terrain=None):
    """Slope in degrees at a point, taken from the real slope raster."""
    terrain = terrain if terrain is not None else resolve_terrain(
        lat, lon, data_dir=data_dir, state_name=state_name
    )
    if terrain["status"] not in USABLE_STATUSES:
        return _unavailable(
            INPUT_SLOPE,
            ["Real slope raster unavailable."] + terrain["reasons"],
        )
    slope = _finite_number(terrain["value"].get("slope"))
    if slope is None:
        return _unavailable(
            INPUT_SLOPE, ["Terrain sample contained no usable 'slope' value."]
        )
    return input_record(
        INPUT_SLOPE, STATUS_REAL, value=slope, source=terrain["source"]
    )


# ---------------------------------------------------------------------------
# Susceptibility (persisted model only)
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Persisted-model inputs
# ---------------------------------------------------------------------------
def resolve_model_input(lat, lon, data_dir=None, artifact_dir=None,
                        state_name=DEFAULT_STATE_NAME, terrain=None):
    """
    Assembles everything needed to run the persisted model at a point: the
    deserialised estimator and a one-row feature frame in the persisted feature
    order -- or UNAVAILABLE with the reason.

    On success `value` holds LIVE OBJECTS (an estimator and a DataFrame), so it
    must never be serialised into an API response; the JSON-safe facts are in
    `details`.

    Four refusals are deliberate:

    1. No VALID artifact bundle in backend/data/models -> UNAVAILABLE. The gate in
       model_artifacts.load_model_evidence has no fallback, and neither does this.
    2. A model whose feature schema contains rainfall features -> UNAVAILABLE.
       dynamic_risk_module implements Option C: susceptibility must be
       rainfall-INDEPENDENT because rainfall is applied separately as a trigger
       multiplier. Feeding a rainfall-coupled model's output in as
       susceptibility_score would count rainfall twice and inflate the score. If a
       static-only model is persisted later, this path unblocks itself.
    3. A feature the serving path cannot source from a real artifact ->
       UNAVAILABLE, naming the feature.
    4. Terrain that cannot be sampled at the point -> UNAVAILABLE, never a
       substituted value.
    """
    name = INPUT_MODEL_INPUT
    try:
        from app.services import model_artifacts
    except ImportError as exc:  # pragma: no cover - app-internal module
        return _unavailable(name, ["model_artifacts is unavailable (%s)." % exc])

    evidence = model_artifacts.load_model_evidence(
        state_name=state_name, base_dir=artifact_dir, load_model=True
    )
    if evidence["status"] != model_artifacts.ARTIFACT_STATUS_VALID:
        return _unavailable(name, [
            "No usable persisted model: artifact status is %s." % evidence["status"]
        ] + list(evidence["problems"]), details={
            "artifact_status": evidence["status"],
            "artifact_paths": evidence["paths"],
            "missing_artifacts": evidence["missing"],
        })

    model = evidence["model"]
    if model is None or not hasattr(model, "predict_proba"):
        return _unavailable(name, [
            "The persisted model exposes no predict_proba(), so a calibrated "
            "susceptibility probability cannot be produced from it."
        ])

    schema = evidence["feature_schema"] or {}
    feature_names = schema.get("feature_names")
    if not isinstance(feature_names, list) or not feature_names:
        return _unavailable(name, [
            "The persisted feature schema declares no feature_names, so the "
            "inference feature order is unknown."
        ])

    coupled = [f for f in feature_names if f in RAINFALL_FEATURE_NAMES]
    if coupled:
        return _unavailable(name, [
            "The persisted model is rainfall-coupled (its features include %s). "
            "dynamic_risk_module applies rainfall separately as a trigger "
            "multiplier, so using this model's output as susceptibility_score "
            "would double-count rainfall. A rainfall-independent (static-only) "
            "model must be persisted for this endpoint."
            % ", ".join(sorted(coupled))
        ], details={"feature_names": list(feature_names),
                    "rainfall_features": sorted(coupled)})

    unsupported = [f for f in feature_names if f not in SUPPORTED_SERVING_FEATURES]
    if unsupported:
        return _unavailable(name, [
            "The persisted model needs feature(s) the serving path cannot source "
            "from a real artifact: %s. Supported: %s."
            % (", ".join(sorted(unsupported)), ", ".join(SUPPORTED_SERVING_FEATURES))
        ], details={"feature_names": list(feature_names)})

    terrain = terrain if terrain is not None else resolve_terrain(
        lat, lon, data_dir=data_dir, state_name=state_name
    )
    if terrain["status"] not in USABLE_STATUSES:
        return _unavailable(
            name,
            ["Model inputs unavailable: the real terrain features could not be "
             "sampled at this point."] + terrain["reasons"],
            details=terrain.get("details"),
        )

    row = {}
    proxy_used = []
    for feature in feature_names:
        if feature in terrain["value"]:
            row[feature] = terrain["value"][feature]
        elif feature in PROXY_FEATURE_NAMES:
            try:
                row[feature] = land_cover_class_from_elevation(
                    terrain["value"]["elevation"]
                )
            except (KeyError, ValueError) as exc:
                return _unavailable(name, [
                    "Derived-proxy feature '%s' could not be computed: %s"
                    % (feature, exc)
                ])
            proxy_used.append(feature)
        else:  # pragma: no cover - guarded by the `unsupported` check above
            return _unavailable(name, [
                "Feature '%s' has no real source at serving time." % feature
            ])

    try:
        import pandas as pd
    except ImportError as exc:
        return _unavailable(name, [
            "pandas is not installed, so the model input frame cannot be built "
            "with the persisted feature names (%s)." % exc
        ])

    try:
        frame = pd.DataFrame([[row[f] for f in feature_names]], columns=feature_names)
    except Exception as exc:
        return _unavailable(name, [
            "The model input frame could not be built: %s: %s"
            % (type(exc).__name__, exc)
        ])

    metrics = evidence["metrics"] or {}
    details = {
        "feature_names": list(feature_names),
        "feature_values": row,
        "derived_proxy_features": proxy_used,
        "metrics_source": metrics.get("metrics_source"),
        "gate_compatible": evidence["gate_compatible"],
    }
    return input_record(
        name,
        STATUS_DERIVED_PROXY if proxy_used else STATUS_REAL,
        value={
            "model": model,
            "frame": frame,
            "feature_names": list(feature_names),
            "feature_values": row,
        },
        source="Persisted %s model at %s" % (
            schema.get("feature_set_name") or "pilot",
            _display_path(evidence["paths"]["model"]),
        ),
        details=details,
    )


def resolve_susceptibility(lat, lon, data_dir=None, artifact_dir=None,
                           state_name=DEFAULT_STATE_NAME, terrain=None,
                           model_input=None):
    """
    Rainfall-independent susceptibility from the persisted, evidence-gated model
    -- or UNAVAILABLE. Every refusal is inherited from resolve_model_input; the
    only addition here is that an out-of-range probability is reported rather than
    clamped into plausibility.
    """
    name = INPUT_SUSCEPTIBILITY
    prepared = model_input if model_input is not None else resolve_model_input(
        lat, lon, data_dir=data_dir, artifact_dir=artifact_dir,
        state_name=state_name, terrain=terrain,
    )
    if prepared["status"] not in USABLE_STATUSES:
        return _unavailable(
            name, prepared["reasons"], details=prepared.get("details")
        )

    bundle = prepared["value"]
    try:
        proba = bundle["model"].predict_proba(bundle["frame"])
        score = float(proba[0][1])
    except Exception as exc:
        return _unavailable(name, [
            "The persisted model failed to score this point: %s: %s"
            % (type(exc).__name__, exc)
        ])

    checked = _finite_number(score)
    if checked is None or not (0.0 <= checked <= 1.0):
        return _unavailable(name, [
            "The persisted model returned %r, which is not a probability in "
            "[0, 1]; refusing to clamp an implausible value into range." % (score,)
        ])

    return input_record(
        name, prepared["status"], value=checked, source=prepared["source"],
        details=prepared.get("details"),
    )


# ---------------------------------------------------------------------------
# Rainfall
# ---------------------------------------------------------------------------
def resolve_current_rainfall(lat, lon, target_date=None,
                             run_type=DEFAULT_IMERG_RUN_TYPE,
                             half_width_deg=POINT_BBOX_HALF_WIDTH_DEG):
    """
    Observed 24 h rainfall at a point from NASA GPM IMERG.

    Requires Earthdata credentials and network access; when either is absent the
    input is UNAVAILABLE. The previous serving path used the literal 55.0 mm here,
    which is above the pilot's 24 h critical accumulation and therefore
    manufactured a trigger exceedance on every request.
    """
    name = INPUT_CURRENT_RAINFALL
    try:
        from app.services.weather_ingestion import fetch_imerg_precipitation
    except ImportError as exc:
        return _unavailable(name, [
            "The IMERG client could not be imported (%s); its dependencies "
            "(requests, xarray, h5netcdf) are not installed." % exc
        ])

    if target_date is None:
        target_date = datetime.now(timezone.utc) - timedelta(days=IMERG_LATENCY_DAYS)
    bounds = point_bounding_box(lat, lon, half_width_deg)

    try:
        result = fetch_imerg_precipitation(
            bounds, target_date, run_type=run_type, windows=[1]
        )
    except Exception as exc:
        return _unavailable(name, [
            "IMERG rainfall could not be retrieved for %s: %s: %s"
            % (target_date.strftime("%Y-%m-%d"), type(exc).__name__, exc)
        ], details={"bounds": bounds})

    accumulation = _finite_number(
        (result or {}).get("accumulations", {}).get("accumulation_1d_mm")
    )
    if accumulation is None or accumulation < 0.0:
        return _unavailable(name, [
            "IMERG returned no usable 1-day accumulation for %s (payload: %r)."
            % (target_date.strftime("%Y-%m-%d"), result)
        ], details={"bounds": bounds})

    return input_record(
        name, STATUS_REAL, value=accumulation,
        source=(result or {}).get("source") or ("IMERG_%s" % run_type),
        details={
            "bounds": bounds,
            "target_date": (result or {}).get("target_date"),
            "window_hours": CURRENT_RAINFALL_WINDOW_HOURS,
        },
    )


def resolve_forecast_rainfall(lat, lon, hours=DEFAULT_FORECAST_HORIZON_HOURS):
    """
    Forecast rainfall accumulation at a point from the Open-Meteo API.

    The previous serving path wrapped this call in `except Exception:
    forecast_rain = 0.0`, so an unreachable forecast service was reported as a
    confident "no rain expected". Failures are now surfaced.
    """
    name = INPUT_FORECAST_RAINFALL
    try:
        from app.services.weather_ingestion import fetch_open_meteo_forecast
    except ImportError as exc:
        return _unavailable(name, [
            "The Open-Meteo client could not be imported (%s)." % exc
        ])

    try:
        accumulation = fetch_open_meteo_forecast(lat, lon, hours)
    except Exception as exc:
        return _unavailable(name, [
            "Open-Meteo forecast rainfall could not be retrieved: %s: %s"
            % (type(exc).__name__, exc)
        ])

    number = _finite_number(accumulation)
    if number is None or number < 0.0:
        return _unavailable(name, [
            "Open-Meteo returned %r, which is not a usable rainfall accumulation."
            % (accumulation,)
        ])

    return input_record(
        name, STATUS_REAL, value=number, source="Open-Meteo forecast API",
        details={"horizon_hours": int(hours)},
    )


def resolve_exposure(lat, lon):
    """
    Always UNAVAILABLE, and that is the honest answer -- see
    EXPOSURE_UNAVAILABLE_REASON. Kept as a resolver so the day a validated
    normalisation exists, only this function changes.
    """
    return _unavailable(INPUT_EXPOSURE, [EXPOSURE_UNAVAILABLE_REASON])


# ---------------------------------------------------------------------------
# Aggregation for the API layer
# ---------------------------------------------------------------------------
def resolve_risk_inputs(lat, lon, mode=RISK_MODE_CURRENT, data_dir=None,
                        artifact_dir=None, state_name=DEFAULT_STATE_NAME,
                        forecast_hours=DEFAULT_FORECAST_HORIZON_HOURS):
    """
    Resolves every input the risk fusion needs at a point.

    Returns {"location", "mode", "inputs", "blocking_inputs", "blocking_reasons",
    "usable", "has_real_dem", "has_real_rainfall"}. `usable` is True only when
    every required input for the mode resolved to a usable status; the API must
    refuse to answer otherwise.
    """
    if mode not in RISK_MODES:
        raise ValueError("Unknown risk mode %r; allowed: %s" % (mode, list(RISK_MODES)))

    terrain = resolve_terrain(lat, lon, data_dir=data_dir, state_name=state_name)
    # Resolved once and shared: loading the persisted model twice per request
    # would double the I/O for no benefit.
    model_input = resolve_model_input(
        lat, lon, data_dir=data_dir, artifact_dir=artifact_dir,
        state_name=state_name, terrain=terrain,
    )
    inputs = {
        INPUT_SUSCEPTIBILITY: resolve_susceptibility(
            lat, lon, model_input=model_input,
        ),
        INPUT_SLOPE: resolve_slope(
            lat, lon, data_dir=data_dir, state_name=state_name, terrain=terrain,
        ),
        INPUT_CURRENT_RAINFALL: resolve_current_rainfall(lat, lon),
        INPUT_EXPOSURE: resolve_exposure(lat, lon),
    }
    if mode == RISK_MODE_FORECAST:
        inputs[INPUT_FORECAST_RAINFALL] = resolve_forecast_rainfall(
            lat, lon, hours=forecast_hours
        )
    else:
        inputs[INPUT_FORECAST_RAINFALL] = input_record(
            INPUT_FORECAST_RAINFALL, STATUS_NOT_USED,
            reasons=["/risk/current evaluates the observed 24 h trigger only; no "
                     "forecast is fetched, and none is assumed."],
        )

    required = REQUIRED_INPUTS_BY_MODE[mode]
    blocking = [
        key for key in required
        if inputs[key]["status"] not in USABLE_STATUSES
    ]
    blocking_reasons = []
    for key in blocking:
        for reason in inputs[key]["reasons"]:
            entry = "%s: %s" % (key, reason)
            if entry not in blocking_reasons:
                blocking_reasons.append(entry)

    return {
        "location": [float(lat), float(lon)],
        "mode": mode,
        "inputs": inputs,
        "required_inputs": list(required),
        "non_blocking_inputs": list(NON_BLOCKING_INPUTS),
        "blocking_inputs": blocking,
        "blocking_reasons": blocking_reasons,
        "usable": not blocking,
        # Truthful flags for dynamic_risk_module's confidence calculation.
        "has_real_dem": terrain["status"] in USABLE_STATUSES,
        "has_real_rainfall": (
            inputs[INPUT_CURRENT_RAINFALL]["status"] in USABLE_STATUSES
        ),
    }


def input_status_summary(resolution):
    """{input_name: status} -- the compact form the API returns to clients."""
    return {
        key: record["status"]
        for key, record in sorted(resolution["inputs"].items())
    }


def build_unavailable_detail(resolution, message=None):
    """
    The structured body the API returns when a required measurement is absent.
    Contains no risk numbers at all: a partially-fabricated risk score is exactly
    what this phase removed.
    """
    return {
        "status": DATA_UNAVAILABLE,
        "message": message or (
            "Risk cannot be computed for this location because required real "
            "inputs are unavailable. No substituted or default values were used."
        ),
        "location": resolution["location"],
        "mode": resolution["mode"],
        "required_inputs": input_status_summary(resolution),
        "blocking_inputs": resolution["blocking_inputs"],
        "blocking_reasons": resolution["blocking_reasons"],
    }
