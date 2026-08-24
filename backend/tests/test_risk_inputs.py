"""
Focused tests for app.services.risk_inputs -- the real-input resolver that
replaced the hardcoded serving inputs.

DEPENDENCY BUDGET: stdlib + numpy/pandas only. No DEM, no rainfall, no network,
no LightGBM/shap/sklearn, no rasterio, no geopandas. Network-backed resolvers are
exercised by injecting a fake weather_ingestion module into sys.modules, which is
also why risk_inputs imports it lazily.

The point of these tests is not that the resolvers succeed -- offline they mostly
cannot -- but that when they cannot, they say so instead of inventing a number.
"""

import io
import json
import os
import pickle
import sys
import tokenize
import types

import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.core.config_states import get_pilot_aoi_bounds
from app.services import model_artifacts
from app.services import risk_inputs as ri

BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# A point well inside the canonical pilot AOI, and one well outside it.
INSIDE_LAT, INSIDE_LON = 27.3314, 88.6138
OUTSIDE_LAT, OUTSIDE_LON = 28.15, 88.95

STATIC_FEATURES = ["elevation", "slope", "aspect", "roughness", "tpi",
                   "land_cover_class"]
RAINFALL_COUPLED_FEATURES = STATIC_FEATURES + ["rain_1d", "rain_3d", "rain_7d"]

# Fixture metrics are deliberately NOT the documentary pilot figures
# (PR-AUC 0.7762 / ROC-AUC 0.9190) so a test can never be mistaken for evidence.
FIXTURE_METRICS = {"PR-AUC": 0.5, "ROC-AUC": 0.6}


# ---------------------------------------------------------------------------
# Test doubles (module level so pickle can resolve them)
# ---------------------------------------------------------------------------
class ConstantProbaModel:
    """
    Minimal estimator stand-in: no sklearn/LightGBM needed.

    `expected_columns` makes column ORDER testable across pickling -- the check has
    to live inside the estimator, because the object the resolver scores is a
    deserialized copy, not the instance the test holds.
    """

    def __init__(self, positive_proba=0.7, expected_columns=None):
        self.positive_proba = positive_proba
        self.expected_columns = expected_columns

    def predict_proba(self, frame):
        columns = list(getattr(frame, "columns", []))
        if self.expected_columns is not None and columns != list(self.expected_columns):
            raise ValueError(
                "frame columns %r are not the persisted feature order %r"
                % (columns, list(self.expected_columns))
            )
        return [[1.0 - self.positive_proba, self.positive_proba]]


class OutOfRangeProbaModel:
    def predict_proba(self, frame):
        return [[0.0, 1.5]]


class ExplodingModel:
    def predict_proba(self, frame):
        raise ValueError("feature mismatch in test double")


class NoProbaModel:
    def predict(self, frame):
        return [1]


def _write_artifacts(artifact_dir, model, feature_names):
    """
    Writes a gate-VALID artifact bundle into a temporary directory using the
    project's own document builders, so the fixture cannot drift from the
    contract it is testing.
    """
    os.makedirs(artifact_dir, exist_ok=True)
    paths = model_artifacts.canonical_artifact_paths(base_dir=artifact_dir)
    with open(paths["model"], "wb") as handle:
        pickle.dump(model, handle, protocol=4)
    metrics = model_artifacts.build_metrics_document(
        validation_metrics=FIXTURE_METRICS,
        primary_model_name="TestDouble",
        primary_evaluation="unit-test fixture",
    )
    schema = model_artifacts.build_feature_schema_document(
        feature_names=feature_names,
        dtypes={name: "float64" for name in feature_names},
        feature_set_name="unit-test fixture",
    )
    for kind, doc in (("metrics", metrics), ("schema", schema)):
        with open(paths[kind], "w", encoding="utf-8") as handle:
            json.dump(doc, handle)
    return paths


def _real_terrain_double():
    """
    An explicitly labelled test double standing in for a real terrain sample.
    It exists only in memory, is never written to disk, and is never used by
    production code paths -- it lets the model-scoring wiring be tested without a
    DEM or rasterio.
    """
    return ri.input_record(
        "terrain", ri.STATUS_REAL,
        value={"elevation": 3500.0, "slope": 41.0, "aspect": 120.0,
               "roughness": 12.0, "tpi": 3.0},
        source="unit-test double (not real data)",
    )


def _fake_weather_module(monkeypatch, **attributes):
    module = types.ModuleType("app.services.weather_ingestion")
    for name, value in attributes.items():
        setattr(module, name, value)
    monkeypatch.setitem(sys.modules, "app.services.weather_ingestion", module)
    return module


def _code_without_strings_or_comments(path):
    """
    Source text with every comment and string literal removed, so a source scan
    cannot be fooled -- and cannot raise false alarms against documentation that
    quotes the very literals it warns about.
    """
    with open(path, "rb") as handle:
        tokens = tokenize.tokenize(handle.readline)
        return " ".join(
            token.string for token in tokens
            if token.type not in (tokenize.COMMENT, tokenize.STRING)
        )


# ---------------------------------------------------------------------------
# Contract: status vocabulary and paths
# ---------------------------------------------------------------------------
def test_status_vocabulary_matches_artifact_contract():
    assert set(ri.INPUT_STATUS_VALUES) == set(model_artifacts.INPUT_STATUS_VALUES)


def test_only_real_and_proxy_statuses_are_usable():
    assert set(ri.USABLE_STATUSES) == {ri.STATUS_REAL, ri.STATUS_DERIVED_PROXY}
    assert ri.STATUS_UNAVAILABLE not in ri.USABLE_STATUSES
    assert ri.STATUS_NOT_USED not in ri.USABLE_STATUSES


def test_input_record_drops_values_for_unusable_statuses():
    record = ri.input_record("x", ri.STATUS_UNAVAILABLE, value=0.5)
    assert record["value"] is None, "an unusable input must not carry a value"


def test_input_record_rejects_unknown_status():
    with pytest.raises(ValueError):
        ri.input_record("x", "PROBABLY_FINE", value=1.0)


def test_terrain_paths_match_the_training_pipeline_layout():
    paths = ri.terrain_raster_paths()
    assert sorted(paths) == sorted(ri.TERRAIN_FEATURE_NAMES)
    assert paths["elevation"].endswith(os.path.join("data", "raw",
                                                    "east_sikkim_dem.tif"))
    for name in ri.TERRAIN_DERIVATIVE_NAMES:
        assert paths[name].endswith(
            os.path.join("data", "processed", "real_%s.tif" % name)
        )


@pytest.mark.parametrize("marker", ["east_sikkim_dem.tif", "real_"])
def test_training_script_and_resolver_agree_on_artifact_names(marker):
    """The serving resolver must look where the training script actually writes."""
    with open(os.path.join(BACKEND_DIR, "scripts", "train_real_models.py"),
              "r", encoding="utf-8", errors="replace") as handle:
        source = handle.read()
    assert marker in source


# ---------------------------------------------------------------------------
# Terrain
# ---------------------------------------------------------------------------
def test_terrain_unavailable_when_rasters_are_absent(tmp_path):
    record = ri.resolve_terrain(INSIDE_LAT, INSIDE_LON, data_dir=str(tmp_path))
    assert record["status"] == ri.STATUS_UNAVAILABLE
    assert record["value"] is None
    joined = " ".join(record["reasons"])
    for name in ri.TERRAIN_FEATURE_NAMES:
        assert name in joined, "the missing raster must be named: %s" % name


def test_terrain_reasons_do_not_leak_absolute_server_paths(tmp_path):
    record = ri.resolve_terrain(INSIDE_LAT, INSIDE_LON, data_dir=str(tmp_path))
    for reason in record["reasons"]:
        assert BACKEND_DIR not in reason


def test_terrain_refuses_points_outside_the_canonical_aoi(tmp_path):
    record = ri.resolve_terrain(OUTSIDE_LAT, OUTSIDE_LON, data_dir=str(tmp_path))
    assert record["status"] == ri.STATUS_UNAVAILABLE
    assert "outside the canonical pilot AOI" in " ".join(record["reasons"])
    assert record["details"]["pilot_aoi"] == get_pilot_aoi_bounds("Sikkim")


def test_aoi_containment_uses_the_canonical_bounds():
    bounds = get_pilot_aoi_bounds("Sikkim")
    assert ri.point_within_pilot_aoi(bounds["min_lat"], bounds["min_lon"])
    assert ri.point_within_pilot_aoi(bounds["max_lat"], bounds["max_lon"])
    assert not ri.point_within_pilot_aoi(bounds["max_lat"] + 0.01,
                                         bounds["max_lon"])
    assert not ri.point_within_pilot_aoi(bounds["min_lat"],
                                         bounds["max_lon"] + 0.01)


def test_slope_propagates_terrain_unavailability(tmp_path):
    record = ri.resolve_slope(INSIDE_LAT, INSIDE_LON, data_dir=str(tmp_path))
    assert record["status"] == ri.STATUS_UNAVAILABLE
    assert record["value"] is None


def test_slope_is_taken_from_the_real_raster_sample():
    record = ri.resolve_slope(INSIDE_LAT, INSIDE_LON,
                              terrain=_real_terrain_double())
    assert record["status"] == ri.STATUS_REAL
    assert record["value"] == 41.0


# ---------------------------------------------------------------------------
# Land-cover proxy
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("elevation,expected", [
    (-100.0, 1), (0.0, 1), (2999.99, 1),
    (3000.0, 2), (4199.99, 2),
    (4200.0, 3), (8848.0, 3),
])
def test_land_cover_proxy_bins(elevation, expected):
    assert ri.land_cover_class_from_elevation(elevation) == expected


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), None, "high"])
def test_land_cover_proxy_refuses_unusable_elevation(bad):
    with pytest.raises(ValueError):
        ri.land_cover_class_from_elevation(bad)


def test_land_cover_break_constants_match_the_documented_meaning():
    meaning = model_artifacts.FEATURE_MEANINGS["land_cover_class"]
    for break_value in ri.LAND_COVER_ELEVATION_BREAKS_M:
        assert str(int(break_value)) in meaning


# ---------------------------------------------------------------------------
# Persisted-model inputs and susceptibility
# ---------------------------------------------------------------------------
def test_model_input_unavailable_without_persisted_artifacts(tmp_path):
    record = ri.resolve_model_input(
        INSIDE_LAT, INSIDE_LON, artifact_dir=str(tmp_path / "models")
    )
    assert record["status"] == ri.STATUS_UNAVAILABLE
    assert record["value"] is None
    assert model_artifacts.ARTIFACT_STATUS_MISSING in " ".join(record["reasons"])


def test_model_input_refuses_a_rainfall_coupled_model(tmp_path):
    """
    Option C applies rainfall separately as a trigger multiplier, so a
    rainfall-coupled model cannot supply susceptibility without double-counting.
    """
    _write_artifacts(str(tmp_path), ConstantProbaModel(), RAINFALL_COUPLED_FEATURES)
    record = ri.resolve_model_input(
        INSIDE_LAT, INSIDE_LON, artifact_dir=str(tmp_path)
    )
    assert record["status"] == ri.STATUS_UNAVAILABLE
    reasons = " ".join(record["reasons"])
    assert "double-count" in reasons
    assert "rain_1d" in reasons
    assert record["details"]["rainfall_features"] == ["rain_1d", "rain_3d", "rain_7d"]


def test_model_input_refuses_features_it_cannot_source(tmp_path):
    _write_artifacts(str(tmp_path), ConstantProbaModel(),
                     ["elevation", "distance_to_fault_km"])
    record = ri.resolve_model_input(
        INSIDE_LAT, INSIDE_LON, artifact_dir=str(tmp_path)
    )
    assert record["status"] == ri.STATUS_UNAVAILABLE
    assert "distance_to_fault_km" in " ".join(record["reasons"])


def test_model_input_refuses_a_model_without_predict_proba(tmp_path):
    _write_artifacts(str(tmp_path), NoProbaModel(), STATIC_FEATURES)
    record = ri.resolve_model_input(
        INSIDE_LAT, INSIDE_LON, artifact_dir=str(tmp_path)
    )
    assert record["status"] == ri.STATUS_UNAVAILABLE
    assert "predict_proba" in " ".join(record["reasons"])


def test_model_input_reports_missing_terrain_rather_than_substituting(tmp_path):
    _write_artifacts(str(tmp_path / "models"), ConstantProbaModel(), STATIC_FEATURES)
    record = ri.resolve_model_input(
        INSIDE_LAT, INSIDE_LON,
        data_dir=str(tmp_path / "data"), artifact_dir=str(tmp_path / "models"),
    )
    assert record["status"] == ri.STATUS_UNAVAILABLE
    assert "terrain" in " ".join(record["reasons"]).lower()


def test_susceptibility_uses_the_persisted_feature_order(tmp_path):
    # The estimator itself rejects any other column order, so a mis-ordered frame
    # would surface here as UNAVAILABLE rather than as a plausible score.
    model = ConstantProbaModel(positive_proba=0.7, expected_columns=STATIC_FEATURES)
    _write_artifacts(str(tmp_path), model, STATIC_FEATURES)
    record = ri.resolve_susceptibility(
        INSIDE_LAT, INSIDE_LON, artifact_dir=str(tmp_path),
        terrain=_real_terrain_double(),
    )
    assert record["status"] == ri.STATUS_DERIVED_PROXY, (
        "land_cover_class is a documented proxy, so the result must not claim REAL"
    )
    assert record["value"] == 0.7
    assert record["details"]["feature_names"] == STATIC_FEATURES
    assert record["details"]["derived_proxy_features"] == ["land_cover_class"]
    # elevation 3500 m falls in the middle proxy bin.
    assert record["details"]["feature_values"]["land_cover_class"] == 2


def test_susceptibility_is_real_when_no_proxy_feature_is_needed(tmp_path):
    _write_artifacts(str(tmp_path), ConstantProbaModel(0.25),
                     ["elevation", "slope", "roughness"])
    record = ri.resolve_susceptibility(
        INSIDE_LAT, INSIDE_LON, artifact_dir=str(tmp_path),
        terrain=_real_terrain_double(),
    )
    assert record["status"] == ri.STATUS_REAL
    assert record["value"] == 0.25


def test_susceptibility_refuses_an_out_of_range_probability(tmp_path):
    _write_artifacts(str(tmp_path), OutOfRangeProbaModel(), STATIC_FEATURES)
    record = ri.resolve_susceptibility(
        INSIDE_LAT, INSIDE_LON, artifact_dir=str(tmp_path),
        terrain=_real_terrain_double(),
    )
    assert record["status"] == ri.STATUS_UNAVAILABLE
    assert record["value"] is None
    assert "1.5" in " ".join(record["reasons"])


def test_susceptibility_reports_a_scoring_failure(tmp_path):
    _write_artifacts(str(tmp_path), ExplodingModel(), STATIC_FEATURES)
    record = ri.resolve_susceptibility(
        INSIDE_LAT, INSIDE_LON, artifact_dir=str(tmp_path),
        terrain=_real_terrain_double(),
    )
    assert record["status"] == ri.STATUS_UNAVAILABLE
    assert "feature mismatch in test double" in " ".join(record["reasons"])


# ---------------------------------------------------------------------------
# Rainfall
# ---------------------------------------------------------------------------
def test_current_rainfall_reports_earthdata_failure_without_zero_filling(monkeypatch):
    def boom(bounds, date, run_type="Early", windows=None):
        raise PermissionError("EARTHDATA AUTHENTICATION REJECTED (HTTP 401)")

    _fake_weather_module(monkeypatch, fetch_imerg_precipitation=boom)
    record = ri.resolve_current_rainfall(INSIDE_LAT, INSIDE_LON)
    assert record["status"] == ri.STATUS_UNAVAILABLE
    assert record["value"] is None, "a failed rainfall fetch must not become 0 mm"
    assert "EARTHDATA AUTHENTICATION REJECTED" in " ".join(record["reasons"])


def test_current_rainfall_returns_the_real_accumulation(monkeypatch):
    captured = {}

    def fetch(bounds, date, run_type="Early", windows=None):
        captured["bounds"] = bounds
        captured["windows"] = windows
        return {"source": "IMERG_Early", "target_date": "2026-08-23",
                "accumulations": {"accumulation_1d_mm": 12.5}}

    _fake_weather_module(monkeypatch, fetch_imerg_precipitation=fetch)
    record = ri.resolve_current_rainfall(INSIDE_LAT, INSIDE_LON)
    assert record["status"] == ri.STATUS_REAL
    assert record["value"] == 12.5
    assert record["source"] == "IMERG_Early"
    assert captured["windows"] == [1]
    # A point-sized window, not the whole AOI averaged together.
    box = captured["bounds"]
    assert abs((box["max_lat"] - box["min_lat"]) - 2 * ri.POINT_BBOX_HALF_WIDTH_DEG) < 1e-9
    assert abs((box["max_lon"] - box["min_lon"]) - 2 * ri.POINT_BBOX_HALF_WIDTH_DEG) < 1e-9


@pytest.mark.parametrize("payload", [
    {},
    {"accumulations": {}},
    {"accumulations": {"accumulation_1d_mm": None}},
    {"accumulations": {"accumulation_1d_mm": -3.0}},
    {"accumulations": {"accumulation_1d_mm": "wet"}},
])
def test_current_rainfall_rejects_unusable_payloads(monkeypatch, payload):
    _fake_weather_module(
        monkeypatch,
        fetch_imerg_precipitation=lambda *a, **k: payload,
    )
    record = ri.resolve_current_rainfall(INSIDE_LAT, INSIDE_LON)
    assert record["status"] == ri.STATUS_UNAVAILABLE
    assert record["value"] is None


def test_forecast_rainfall_failure_is_not_reported_as_no_rain(monkeypatch):
    def boom(lat, lon, hours):
        raise RuntimeError("forecast service unreachable")

    _fake_weather_module(monkeypatch, fetch_open_meteo_forecast=boom)
    record = ri.resolve_forecast_rainfall(INSIDE_LAT, INSIDE_LON)
    assert record["status"] == ri.STATUS_UNAVAILABLE
    assert record["value"] is None, (
        "the removed behaviour was `except Exception: forecast_rain = 0.0`"
    )
    assert "forecast service unreachable" in " ".join(record["reasons"])


def test_forecast_rainfall_returns_the_real_value(monkeypatch):
    _fake_weather_module(
        monkeypatch,
        fetch_open_meteo_forecast=lambda lat, lon, hours: 88.0,
    )
    record = ri.resolve_forecast_rainfall(INSIDE_LAT, INSIDE_LON, hours=72)
    assert record["status"] == ri.STATUS_REAL
    assert record["value"] == 88.0
    assert record["details"]["horizon_hours"] == 72


# ---------------------------------------------------------------------------
# Exposure
# ---------------------------------------------------------------------------
def test_exposure_is_unavailable_and_never_a_default_of_half():
    record = ri.resolve_exposure(INSIDE_LAT, INSIDE_LON)
    assert record["status"] == ri.STATUS_UNAVAILABLE
    assert record["value"] is None
    assert "0-1 exposure score" in " ".join(record["reasons"])


def test_exposure_never_blocks_the_response():
    assert ri.INPUT_EXPOSURE in ri.NON_BLOCKING_INPUTS
    for mode in ri.RISK_MODES:
        assert ri.INPUT_EXPOSURE not in ri.REQUIRED_INPUTS_BY_MODE[mode]


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------
def test_resolve_risk_inputs_rejects_an_unknown_mode():
    with pytest.raises(ValueError):
        ri.resolve_risk_inputs(INSIDE_LAT, INSIDE_LON, mode="vibes")


def test_resolve_risk_inputs_is_unusable_and_honest_when_nothing_is_present(tmp_path):
    resolution = ri.resolve_risk_inputs(
        INSIDE_LAT, INSIDE_LON, mode=ri.RISK_MODE_CURRENT,
        data_dir=str(tmp_path / "data"), artifact_dir=str(tmp_path / "models"),
    )
    assert resolution["usable"] is False
    assert resolution["has_real_dem"] is False
    assert resolution["has_real_rainfall"] is False
    assert set(resolution["blocking_inputs"]) == set(
        ri.REQUIRED_INPUTS_BY_MODE[ri.RISK_MODE_CURRENT]
    )
    assert resolution["blocking_reasons"], "a refusal must state its reasons"
    assert ri.INPUT_EXPOSURE not in resolution["blocking_inputs"]


def test_current_mode_marks_the_forecast_not_used_rather_than_zero(tmp_path):
    resolution = ri.resolve_risk_inputs(
        INSIDE_LAT, INSIDE_LON, mode=ri.RISK_MODE_CURRENT,
        data_dir=str(tmp_path / "data"), artifact_dir=str(tmp_path / "models"),
    )
    forecast = resolution["inputs"][ri.INPUT_FORECAST_RAINFALL]
    assert forecast["status"] == ri.STATUS_NOT_USED
    assert forecast["value"] is None


def test_forecast_mode_requires_a_real_forecast(tmp_path):
    resolution = ri.resolve_risk_inputs(
        INSIDE_LAT, INSIDE_LON, mode=ri.RISK_MODE_FORECAST,
        data_dir=str(tmp_path / "data"), artifact_dir=str(tmp_path / "models"),
    )
    assert ri.INPUT_FORECAST_RAINFALL in resolution["blocking_inputs"]


def test_unavailable_detail_is_json_safe_and_carries_no_risk_numbers(tmp_path):
    resolution = ri.resolve_risk_inputs(
        INSIDE_LAT, INSIDE_LON, mode=ri.RISK_MODE_FORECAST,
        data_dir=str(tmp_path / "data"), artifact_dir=str(tmp_path / "models"),
    )
    detail = ri.build_unavailable_detail(resolution)
    encoded = json.dumps(detail)
    assert detail["status"] == ri.DATA_UNAVAILABLE
    assert detail["location"] == [INSIDE_LAT, INSIDE_LON]
    for forbidden in ("final_risk_score", "warning_level", "susceptibility_score",
                      "confidence"):
        assert forbidden not in encoded, (
            "a DATA_UNAVAILABLE body must not contain a risk verdict"
        )


# ---------------------------------------------------------------------------
# Source scans: the removed literals must not creep back
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("literal", ["0.65", "55.0", "has_real_dem", "has_real_rainfall"])
def test_routes_no_longer_hardcode_risk_inputs(literal):
    """
    The serving path must not contain the invented susceptibility/rainfall values,
    and must not assert the provenance flags as constants. Comments and strings are
    stripped first, so the module docstring that documents the removal is not a
    false positive.
    """
    code = _code_without_strings_or_comments(
        os.path.join(BACKEND_DIR, "app", "api", "routes.py")
    )
    if literal in ("has_real_dem", "has_real_rainfall"):
        assert "%s = True" % literal not in code
        assert "%s=True" % literal not in code.replace(" ", "")
    else:
        assert literal not in code


@pytest.mark.parametrize("importance", ["0.42", "0.28", "0.18"])
def test_ml_pipeline_no_longer_hardcodes_feature_importances(importance):
    code = _code_without_strings_or_comments(
        os.path.join(BACKEND_DIR, "app", "models", "ml_pipeline.py")
    )
    assert importance not in code


def test_dynamic_risk_module_defaults_are_fail_closed():
    """
    Checked by source inspection so this test needs neither LightGBM nor shap:
    ml_pipeline cannot be imported in a minimal environment, but its signature can
    still be read.
    """
    import ast

    path = os.path.join(BACKEND_DIR, "app", "models", "ml_pipeline.py")
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        tree = ast.parse(handle.read())
    functions = {
        node.name: node for node in tree.body
        if isinstance(node, ast.FunctionDef)
    }
    assert "dynamic_risk_module" in functions
    args = functions["dynamic_risk_module"].args
    defaults = dict(zip(
        [a.arg for a in args.args][-len(args.defaults):],
        [ast.literal_eval(d) for d in args.defaults],
    ))
    assert defaults["exposure_score"] is None, "0.5 was a fabricated exposure"
    assert defaults["has_real_dem"] is False
    assert defaults["has_real_rainfall"] is False
