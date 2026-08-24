"""
Focused offline tests for app.services.sikkim_prediction -- the read-only grid
prediction service that runs the persisted 11-feature Sikkim LightGBM with real
IMERG antecedent rainfall.

DEPENDENCY BUDGET: stdlib + numpy/pandas only. No DEM/rasterio, no network/IMERG,
no real LightGBM. The model, terrain sampler and rainfall provider are injected as
fakes, so these tests exercise the ASSEMBLY and the NO-FABRICATION invariants --
not a real prediction (which is host-only). The only heavy module touched is
app.models.ml_pipeline (for the real calculate_warning_level banding); its
top-level lightgbm/shap/sklearn imports are stubbed here IF absent, exactly as the
offline shim harness does, so the REAL banding function is what runs.

The point, as with test_risk_inputs, is that when an input is unavailable the
service SAYS SO (status UNAVAILABLE / HTTP-503-mapped refusal) instead of
inventing a probability, a coordinate, or a rainfall value.
"""

import math
import os
import sys
import types

import numpy as np
import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# --- Make the REAL app.models.ml_pipeline importable offline (for the real
#     calculate_warning_level). Stub its heavy top-level deps ONLY if absent. ---
try:  # pragma: no cover - host has the real libs
    import lightgbm  # noqa: F401
    _HEAVY_PRESENT = True
except Exception:  # pragma: no cover - offline sandbox
    _HEAVY_PRESENT = False

if not _HEAVY_PRESENT:
    class _PermissiveStub(types.ModuleType):
        def __getattr__(self, item):
            return type(str(item), (object,), {})

    for _name in (
        "lightgbm", "shap", "sklearn", "sklearn.linear_model",
        "sklearn.ensemble", "sklearn.metrics", "sklearn.model_selection",
    ):
        sys.modules.setdefault(_name, _PermissiveStub(_name))

from app.services import risk_inputs
from app.services import sikkim_prediction as sp


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------
class _FakeModel:
    """
    predict_proba maps each row's 'slope' feature to a probability (slope/100),
    so the caller can drive specific warning bands, and records every DataFrame it
    is handed so the test can assert the exact 11-column typed contract.
    """

    classes_ = [0, 1]

    def __init__(self):
        self.frames = []

    def predict_proba(self, frame):
        self.frames.append(frame.copy())
        probs = []
        for slope in frame["slope"].tolist():
            p = max(0.0, min(0.999, float(slope) / 100.0))
            probs.append([1.0 - p, p])
        return np.array(probs, dtype=float)


def _valid_evidence(model):
    return {
        "status": "VALID",
        "model": model,
        "feature_schema": {"feature_names": list(sp.MODEL_FEATURE_ORDER)},
        "metrics": {"validation_metrics": {"pr_auc": 0.6926, "roc_auc": 0.8914}},
        "problems": [],
    }


# slope -> band, and an elevation that also exercises all three land-cover classes.
_PATTERN = [
    (20.0, 2500.0),   # p=0.20 LOW,     class 1 (<3000)
    (50.0, 3500.0),   # p=0.50 MEDIUM,  class 2 ([3000,4200))
    (70.0, 5000.0),   # p=0.70 HIGH,    class 3 (>=4200)
    (90.0, 3000.0),   # p=0.90 EXTREME, class 2 (boundary -> 2)
    (None, None),     # nodata cell -> UNAVAILABLE
]


def _pattern_terrain_sampler(centers, data_dir=None):
    out = []
    for i, _center in enumerate(centers):
        slope, elevation = _PATTERN[i % len(_PATTERN)]
        if slope is None:
            out.append({"values": None, "problems": ["synthetic nodata cell"]})
            continue
        out.append({
            "values": {
                "elevation": elevation, "slope": slope, "aspect": 123.0,
                "roughness": 4.5, "tpi": -1.2,
            },
            "problems": [],
        })
    return out


def _all_nodata_sampler(centers, data_dir=None):
    return [{"values": None, "problems": ["nodata"]} for _ in centers]


def _fake_rainfall_provider(bounds, target_date, run_type="Early"):
    daily = [float(v) for v in range(1, sp.RAINFALL_WINDOW_DAYS + 1)]  # 1..14 mm
    return {
        "source": "IMERG_%s" % run_type,
        "run_type": run_type,
        "aoi_uniform": True,
        "window_days": sp.RAINFALL_WINDOW_DAYS,
        "daily_series_mm": [round(v, 4) for v in daily],
        "features": sp._derive_rainfall_features(daily),
    }


def _run(step=0.25, date="2025-09-19", sampler=_pattern_terrain_sampler, model=None):
    model = model if model is not None else _FakeModel()
    result = sp.predict_sikkim_grid(
        date,
        step_deg=step,
        model_evidence=_valid_evidence(model),
        terrain_sampler=sampler,
        rainfall_provider=_fake_rainfall_provider,
    )
    return result, model


# ---------------------------------------------------------------------------
# Grid geometry
# ---------------------------------------------------------------------------
def test_grid_centers_are_strictly_inside_the_pilot_aoi():
    _bounds, meta, cells = sp.build_grid(step_deg=sp.DEFAULT_STEP_DEG)
    assert meta["cell_count"] == len(cells) == meta["n_lat"] * meta["n_lon"]
    assert cells, "grid must not be empty"
    for cell in cells:
        assert risk_inputs.point_within_pilot_aoi(cell["latitude"], cell["longitude"]), cell
        bbox = cell["bbox"]
        assert bbox["min_lat"] < bbox["max_lat"] and bbox["min_lon"] < bbox["max_lon"]


def test_grid_rejects_out_of_range_or_too_fine_step():
    with pytest.raises(ValueError):
        sp.build_grid(step_deg=0.001)           # below MIN_STEP_DEG
    with pytest.raises(ValueError):
        sp.build_grid(step_deg=1.0)             # above MAX_STEP_DEG
    # A step inside the [MIN,MAX] range but implying > MAX_CELLS must also refuse.
    with pytest.raises(ValueError):
        sp.build_grid(step_deg=0.02)


def test_grid_cell_count_never_exceeds_cap():
    _bounds, meta, _cells = sp.build_grid(step_deg=0.03)
    assert meta["cell_count"] <= sp.MAX_CELLS


# ---------------------------------------------------------------------------
# Rainfall feature derivation (schema semantics: T-1..T-14)
# ---------------------------------------------------------------------------
def test_rainfall_features_match_schema_definitions():
    daily = [float(v) for v in range(1, 15)]  # day T-1=1, T-2=2, ... T-14=14
    feats = sp._derive_rainfall_features(daily)
    assert feats["rain_1d"] == 1.0
    assert feats["rain_3d"] == 1 + 2 + 3
    assert feats["rain_7d"] == sum(range(1, 8))
    assert feats["antecedent_rain_14d"] == sum(range(1, 15))
    assert feats["rain_intensity_max_3d"] == 3.0
    assert set(feats) == set(sp.RAINFALL_FEATURES)


def test_rainfall_derivation_requires_full_window():
    with pytest.raises(ValueError):
        sp._derive_rainfall_features([1.0, 2.0, 3.0])  # < 14 days


# ---------------------------------------------------------------------------
# Full assembly with injected fakes
# ---------------------------------------------------------------------------
def test_full_prediction_shape_bands_and_typed_feature_contract():
    result, model = _run()

    assert result["state"] == "Sikkim"
    assert result["target_date"] == "2025-09-19"
    assert result["decision_threshold"] == sp.DECISION_THRESHOLD
    assert result["model"]["feature_order"] == list(sp.MODEL_FEATURE_ORDER)
    assert result["model"]["n_features"] == 11
    # Real validation metrics passed through unchanged (never synthesised).
    assert result["model"]["validation_metrics"]["pr_auc"] == 0.6926

    cells = result["cells"]
    summary = result["summary"]
    assert summary["cells_total"] == len(cells)
    assert summary["cells_scored"] + summary["cells_unavailable"] == summary["cells_total"]
    assert sum(summary["risk_class_counts"].values()) == summary["cells_scored"]

    # The typed 11-column contract handed to the model.
    frame = model.frames[0]
    assert list(frame.columns) == list(sp.MODEL_FEATURE_ORDER)
    assert str(frame["land_cover_class"].dtype) == "int32"
    for col in ("elevation", "slope", "rain_7d", "antecedent_rain_14d"):
        assert str(frame[col].dtype) == "float32"

    # Band + threshold + land-cover-proxy correctness on the scored cells.
    by_slope = {}
    for cell in cells:
        if cell["status"] != "OK":
            continue
        slope = cell["features"]["slope"]
        by_slope[slope] = cell
        # land_cover echoed as an int and equal to the proxy of the elevation used.
        assert isinstance(cell["features"]["land_cover_class"], int)
        expected_lc = risk_inputs.land_cover_class_from_elevation(cell["features"]["elevation"])
        assert cell["features"]["land_cover_class"] == expected_lc
        # rainfall features are the AOI-uniform derived values, identical per cell.
        assert cell["features"]["rain_1d"] == 1.0
        assert cell["features"]["antecedent_rain_14d"] == float(sum(range(1, 15)))

    assert by_slope[20.0]["risk_class"] == "LOW"
    assert by_slope[20.0]["exceeds_decision_threshold"] is False
    assert by_slope[50.0]["risk_class"] == "MEDIUM"
    assert by_slope[50.0]["exceeds_decision_threshold"] is True
    assert by_slope[70.0]["risk_class"] == "HIGH"
    assert by_slope[90.0]["risk_class"] == "EXTREME"
    assert abs(by_slope[70.0]["susceptibility_probability"] - 0.70) < 1e-6

    # Rainfall echoed in the response matches the derivation.
    assert result["rainfall"]["source"] == "IMERG_Early"
    assert result["rainfall"]["aoi_uniform"] is True
    assert result["rainfall"]["features"]["rain_3d"] == 6.0

    # Honesty disclosures are present and cover the key caveats.
    text = " ".join(result["disclosures"]).lower()
    assert len(result["disclosures"]) >= 6
    assert "era5" in text and "imerg" in text          # train/serve shift
    assert "proxy" in text                             # land-cover proxy
    assert "raw" in text                               # raw probability, not Option-C


def test_unavailable_cells_carry_no_probability_and_a_reason():
    result, _model = _run()
    unavailable = [c for c in result["cells"] if c["status"] == "UNAVAILABLE"]
    assert unavailable, "the pattern sampler must produce some nodata cells"
    for cell in unavailable:
        assert cell["susceptibility_probability"] is None
        assert cell["risk_class"] is None
        assert cell["exceeds_decision_threshold"] is None
        assert cell["reasons"]                         # a stated reason, never silent
        assert "features" not in cell                  # no fabricated feature vector


def test_all_nodata_grid_scores_nothing_and_never_calls_the_model():
    result, model = _run(sampler=_all_nodata_sampler)
    assert result["summary"]["cells_scored"] == 0
    assert result["summary"]["cells_unavailable"] == result["summary"]["cells_total"]
    assert result["summary"]["mean_probability"] is None
    assert result["summary"]["max_probability"] is None
    assert model.frames == []                          # predict_proba never invoked


# ---------------------------------------------------------------------------
# Refusals (mapped to HTTP 503 by the route)
# ---------------------------------------------------------------------------
def test_refuses_when_model_artifacts_are_not_valid():
    bad = {"status": "INVALID", "model": None, "feature_schema": None, "problems": ["x"]}
    with pytest.raises(sp.PredictionUnavailable):
        sp.predict_sikkim_grid(
            "2025-09-19", step_deg=0.25,
            model_evidence=bad,
            terrain_sampler=_pattern_terrain_sampler,
            rainfall_provider=_fake_rainfall_provider,
        )


def test_refuses_on_feature_order_mismatch():
    model = _FakeModel()
    evidence = _valid_evidence(model)
    evidence["feature_schema"]["feature_names"] = list(sp.MODEL_FEATURE_ORDER)[::-1]
    with pytest.raises(sp.PredictionUnavailable) as exc:
        sp.predict_sikkim_grid(
            "2025-09-19", step_deg=0.25,
            model_evidence=evidence,
            terrain_sampler=_pattern_terrain_sampler,
            rainfall_provider=_fake_rainfall_provider,
        )
    assert exc.value.details["expected"] == list(sp.MODEL_FEATURE_ORDER)


def test_refuses_when_real_rainfall_cannot_be_obtained():
    def _boom(bounds, target_date, run_type="Early"):
        raise RuntimeError("EARTHDATA IMERG FETCH FAILED")

    with pytest.raises(sp.PredictionUnavailable) as exc:
        sp.predict_sikkim_grid(
            "2025-09-19", step_deg=0.25,
            model_evidence=_valid_evidence(_FakeModel()),
            terrain_sampler=_pattern_terrain_sampler,
            rainfall_provider=_boom,
        )
    assert "rainfall" in exc.value.reason.lower()


def test_refuses_on_non_finite_rainfall_feature():
    def _nan_rain(bounds, target_date, run_type="Early"):
        feats = sp._derive_rainfall_features([float(v) for v in range(1, 15)])
        feats["rain_7d"] = float("nan")               # a poisoned feature
        return {"source": "IMERG_Early", "run_type": "Early", "aoi_uniform": True,
                "window_days": 14, "daily_series_mm": [], "features": feats}

    with pytest.raises(sp.PredictionUnavailable):
        sp.predict_sikkim_grid(
            "2025-09-19", step_deg=0.25,
            model_evidence=_valid_evidence(_FakeModel()),
            terrain_sampler=_pattern_terrain_sampler,
            rainfall_provider=_nan_rain,
        )


def test_bad_target_date_string_raises_valueerror():
    with pytest.raises(ValueError):
        sp.predict_sikkim_grid(
            "not-a-date", step_deg=0.25,
            model_evidence=_valid_evidence(_FakeModel()),
            terrain_sampler=_pattern_terrain_sampler,
            rainfall_provider=_fake_rainfall_provider,
        )
