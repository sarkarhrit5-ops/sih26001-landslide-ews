"""
Focused offline tests for app.services.meghalaya_prediction -- the read-only grid
prediction service that runs the persisted 11-feature MEGHALAYA LightGBM with real
IMERG antecedent rainfall.

This mirrors test_arunachal_prediction.py / test_assam_prediction.py (same dependency
budget: stdlib + numpy/pandas only; model, terrain sampler, land-cover resolver and
rainfall provider are injected as fakes, so these tests exercise the ASSEMBLY and the
NO-FABRICATION invariants -- not a real prediction, which is host-only). The only heavy
module touched is app.models.ml_pipeline (for the real calculate_warning_level
banding); its top-level lightgbm/shap/sklearn imports are stubbed here IF absent,
exactly as the offline shim harness does, so the REAL banding function is what runs.

It additionally pins the TWO -- and only two -- ways the Meghalaya path differs from
Sikkim (identical to the Assam / Arunachal pilots):

  1. land_cover_class is scored as a pandas CATEGORICAL over the FIXED WorldCover
     vocabulary (1..6), NOT the int32 cast the Sikkim path uses.
  2. land_cover_class is REAL ESA WorldCover, NOT an elevation proxy -- and a cell
     whose WorldCover sample is UNAVAILABLE is dropped, never back-filled.

As with the other pilots, the point is that when an input is unavailable the service
SAYS SO (status UNAVAILABLE / HTTP-503-mapped refusal) instead of inventing a
probability, a coordinate, a land-cover class, or a rainfall value.
"""

import os
import sys
import types

import numpy as np
import pandas as pd
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
from app.services import worldcover as wc
from app.services import meghalaya_prediction as mp
from app.services import sikkim_prediction as sp


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------
class _FakeModel:
    """
    predict_proba maps each row's 'slope' feature to a probability (slope/100),
    so the caller can drive specific warning bands, and records every DataFrame it
    is handed so the test can assert the exact 11-column CATEGORICAL contract.
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
        "feature_schema": {"feature_names": list(mp.MODEL_FEATURE_ORDER)},
        "metrics": {"validation_metrics": {"pr_auc": 0.41, "roc_auc": 0.66}},
        "problems": [],
    }


# (slope -> band) and (real WorldCover group) per cell. Every terrain elevation is a
# FIXED 1000.0 m -- which the Sikkim elevation proxy would map to class 1 for every
# cell -- so any scored land_cover_class other than 1 proves the class came from the
# injected real land-cover resolver, NOT from an elevation proxy.
#   slope None  -> terrain nodata            -> UNAVAILABLE (land cover irrelevant)
#   group None  -> WorldCover UNAVAILABLE    -> UNAVAILABLE (never back-filled)
_PATTERN = [
    (20.0, 1),      # p=0.20 LOW,     forest
    (50.0, 4),      # p=0.50 MEDIUM,  built-up
    (70.0, 3),      # p=0.70 HIGH,    cropland
    (90.0, 6),      # p=0.90 EXTREME, water / wetland
    (None, 1),      # terrain nodata -> UNAVAILABLE
    (30.0, None),   # land cover UNAVAILABLE -> UNAVAILABLE (never filled)
]

_FIXED_ELEVATION_M = 1000.0  # < 3000 -> Sikkim proxy would call this class 1 everywhere


def _pattern_terrain_sampler(centers, data_dir=None):
    out = []
    for i, _center in enumerate(centers):
        slope, _group = _PATTERN[i % len(_PATTERN)]
        if slope is None:
            out.append({"values": None, "problems": ["synthetic terrain nodata cell"]})
            continue
        out.append({
            "values": {
                "elevation": _FIXED_ELEVATION_M, "slope": slope, "aspect": 123.0,
                "roughness": 4.5, "tpi": -1.2,
            },
            "problems": [],
        })
    return out


def _pattern_land_cover(centers, data_dir=None):
    out = []
    for i, _center in enumerate(centers):
        _slope, group = _PATTERN[i % len(_PATTERN)]
        if group is None:
            out.append({"value": None, "problems": ["synthetic worldcover nodata cell"]})
            continue
        out.append({"value": int(group), "problems": []})
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


def _run(step=0.25, date="2025-09-19", terrain_sampler=_pattern_terrain_sampler,
         land_cover_resolver=_pattern_land_cover, model=None):
    model = model if model is not None else _FakeModel()
    result = mp.predict_meghalaya_grid(
        date,
        step_deg=step,
        model_evidence=_valid_evidence(model),
        terrain_sampler=terrain_sampler,
        land_cover_resolver=land_cover_resolver,
        rainfall_provider=_fake_rainfall_provider,
    )
    return result, model


# ---------------------------------------------------------------------------
# Grid geometry (over the canonical MEGHALAYA AOI, not Sikkim's)
# ---------------------------------------------------------------------------
def test_grid_centers_are_strictly_inside_the_meghalaya_pilot_aoi():
    _bounds, meta, cells = sp.build_grid(step_deg=0.25, state_name=mp.STATE_NAME)
    assert meta["cell_count"] == len(cells) == meta["n_lat"] * meta["n_lon"]
    assert cells, "grid must not be empty"
    assert meta["cell_count"] <= sp.MAX_CELLS
    for cell in cells:
        assert risk_inputs.point_within_pilot_aoi(
            cell["latitude"], cell["longitude"], state_name=mp.STATE_NAME
        ), cell
        bbox = cell["bbox"]
        assert bbox["min_lat"] < bbox["max_lat"] and bbox["min_lon"] < bbox["max_lon"]
    # This AOI really is the Meghalaya one, disjoint from Sikkim's DEM extent.
    for cell in cells:
        assert not risk_inputs.point_within_pilot_aoi(
            cell["latitude"], cell["longitude"], state_name="Sikkim"
        )


# ---------------------------------------------------------------------------
# Identity + model contract
# ---------------------------------------------------------------------------
def test_state_pilot_area_and_model_contract():
    result, _model = _run()
    assert result["state"] == "Meghalaya"
    assert result["pilot_area"] == mp.PILOT_AREA == "East Khasi + Jaintia Hills belt"
    assert result["target_date"] == "2025-09-19"
    assert result["decision_threshold"] == mp.DECISION_THRESHOLD
    assert result["model"]["feature_order"] == list(mp.MODEL_FEATURE_ORDER)
    assert result["model"]["n_features"] == 11
    # Real validation metrics passed through unchanged (never synthesised).
    assert result["model"]["validation_metrics"]["pr_auc"] == 0.41
    # The 11-feature order is genuinely the shared one.
    assert list(mp.MODEL_FEATURE_ORDER) == list(sp.MODEL_FEATURE_ORDER)


# ---------------------------------------------------------------------------
# THE Meghalaya difference #1: land cover scored as a CATEGORICAL over 1..6
# ---------------------------------------------------------------------------
def test_land_cover_scored_as_categorical_over_fixed_worldcover_vocabulary():
    result, model = _run()

    frame = model.frames[0]
    assert list(frame.columns) == list(mp.MODEL_FEATURE_ORDER)

    # land_cover_class is a pandas categorical (NOT int32 as in the Sikkim path)...
    dtype = frame["land_cover_class"].dtype
    assert isinstance(dtype, pd.CategoricalDtype)
    # ...over the FIXED WorldCover level set 1..6, regardless of which groups actually
    # appear in this request -- this is what makes the category->code mapping identical
    # to how the persisted model was fit.
    assert list(dtype.categories) == [1, 2, 3, 4, 5, 6]
    assert list(dtype.categories) == list(wc.ASSAM_LANDCOVER_GROUP_CODES)

    # Terrain + rainfall stay float32.
    for col in ("elevation", "slope", "aspect", "roughness", "tpi",
                "rain_1d", "rain_7d", "antecedent_rain_14d", "rain_intensity_max_3d"):
        assert str(frame[col].dtype) == "float32", col

    _ = result  # assembled without error


# ---------------------------------------------------------------------------
# THE Meghalaya difference #2: land cover is REAL WorldCover, not an elevation proxy
# ---------------------------------------------------------------------------
def test_bands_threshold_and_real_land_cover_not_elevation_proxy():
    result, _model = _run()
    cells = result["cells"]
    summary = result["summary"]

    assert summary["cells_total"] == len(cells)
    assert summary["cells_scored"] + summary["cells_unavailable"] == summary["cells_total"]
    assert sum(summary["risk_class_counts"].values()) == summary["cells_scored"]

    by_slope = {}
    for cell in cells:
        if cell["status"] != "OK":
            continue
        slope = cell["features"]["slope"]
        by_slope[slope] = cell
        # land cover echoed as an int...
        assert isinstance(cell["features"]["land_cover_class"], int)
        # ...and it is a valid WorldCover group, never the sentinel.
        assert cell["features"]["land_cover_class"] in wc.ASSAM_LANDCOVER_GROUP_CODES
        # rainfall features are the AOI-uniform derived values, identical per cell.
        assert cell["features"]["rain_1d"] == 1.0
        assert cell["features"]["antecedent_rain_14d"] == float(sum(range(1, 15)))

    # Every scored cell has elevation 1000 m -> the elevation proxy would say class 1
    # for ALL of them. The real land cover is 4/3/6 for slopes 50/70/90, PROVING the
    # class comes from the injected WorldCover resolver, not from elevation.
    assert by_slope[50.0]["features"]["land_cover_class"] == 4
    assert by_slope[70.0]["features"]["land_cover_class"] == 3
    assert by_slope[90.0]["features"]["land_cover_class"] == 6
    proxy_would_say = risk_inputs.land_cover_class_from_elevation(_FIXED_ELEVATION_M)
    assert by_slope[50.0]["features"]["land_cover_class"] != proxy_would_say

    # Warning bands + decision threshold on the scored cells.
    assert by_slope[20.0]["risk_class"] == "LOW"
    assert by_slope[20.0]["exceeds_decision_threshold"] is False
    assert by_slope[50.0]["risk_class"] == "MEDIUM"
    assert by_slope[70.0]["risk_class"] == "HIGH"
    assert by_slope[90.0]["risk_class"] == "EXTREME"
    assert by_slope[90.0]["exceeds_decision_threshold"] is True
    assert abs(by_slope[70.0]["susceptibility_probability"] - 0.70) < 1e-6

    # Rainfall echoed in the response matches the derivation.
    assert result["rainfall"]["source"] == "IMERG_Early"
    assert result["rainfall"]["aoi_uniform"] is True
    assert result["rainfall"]["features"]["rain_3d"] == 6.0


def test_unavailable_terrain_and_unavailable_land_cover_both_drop_never_fill():
    result, _model = _run()
    unavailable = [c for c in result["cells"] if c["status"] == "UNAVAILABLE"]
    assert unavailable, "the pattern must produce nodata cells"
    for cell in unavailable:
        assert cell["susceptibility_probability"] is None
        assert cell["risk_class"] is None
        assert cell["exceeds_decision_threshold"] is None
        assert cell["reasons"]                         # a stated reason, never silent
        assert "features" not in cell                  # no fabricated feature vector

    # At least one cell is UNAVAILABLE specifically because its REAL land cover was
    # unavailable -- proving land cover is dropped, never back-filled with a class.
    reasons_text = " ".join(
        r for c in unavailable for r in c["reasons"]
    ).lower()
    assert "worldcover" in reasons_text or "land cover" in reasons_text


def test_all_nodata_grid_scores_nothing_and_never_calls_the_model():
    result, model = _run(terrain_sampler=_all_nodata_sampler)
    assert result["summary"]["cells_scored"] == 0
    assert result["summary"]["cells_unavailable"] == result["summary"]["cells_total"]
    assert result["summary"]["mean_probability"] is None
    assert result["summary"]["max_probability"] is None
    assert model.frames == []                          # predict_proba never invoked


# ---------------------------------------------------------------------------
# _score_cells_categorical unit-level: fixed vocabulary + positive-class pick
# ---------------------------------------------------------------------------
class _RecordingModel:
    """Records the frame and returns a fixed per-class probability by CLASS LABEL."""

    def __init__(self, classes):
        self.classes_ = list(classes)
        self.frames = []

    def predict_proba(self, frame):
        self.frames.append(frame.copy())
        n = len(frame)
        cols = [[0.8 if cls == 1 else 0.2] * n for cls in self.classes_]
        return np.array(cols, dtype=float).T


def _feature_row(slope, land_cover_group):
    row = {"elevation": 1000.0, "slope": slope, "aspect": 100.0,
           "roughness": 2.0, "tpi": 0.5, "land_cover_class": land_cover_group}
    for feat in sp.RAINFALL_FEATURES:
        row[feat] = 3.0
    return row


def test_score_cells_categorical_builds_fixed_vocabulary_and_picks_positive_class():
    rows = [_feature_row(40.0, 4), _feature_row(80.0, 6)]

    # classes_ in natural order [0, 1] -> positive class is column 1.
    m1 = _RecordingModel([0, 1])
    out1 = mp._score_cells_categorical(m1, rows)
    assert out1 == [0.8, 0.8]

    frame = m1.frames[0]
    assert isinstance(frame["land_cover_class"].dtype, pd.CategoricalDtype)
    assert list(frame["land_cover_class"].dtype.categories) == [1, 2, 3, 4, 5, 6]
    # underlying real values preserved through the categorical view
    assert frame["land_cover_class"].astype("int64").tolist() == [4, 6]
    assert str(frame["slope"].dtype) == "float32"

    # classes_ REVERSED [1, 0] -> positive class must still be found by LABEL (col 0).
    m2 = _RecordingModel([1, 0])
    out2 = mp._score_cells_categorical(m2, rows)
    assert out2 == [0.8, 0.8]


# ---------------------------------------------------------------------------
# _meghalaya_land_cover: real groups mapped, nodata refused (never filled)
# ---------------------------------------------------------------------------
def test_meghalaya_land_cover_maps_real_groups_and_refuses_nodata(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    (raw_dir / mp.MEGHALAYA_LANDCOVER_FILENAME).write_bytes(b"not-a-real-tif-but-nonzero")

    # reader(raster_path, coords) -> raw WorldCover codes, one per (lon, lat) coord.
    def _reader(raster_path, coords):
        return [10, 0, 50][:len(coords)]   # Tree cover, nodata, Built-up

    centers = [(25.5, 91.5), (25.6, 91.6), (25.7, 91.7)]
    res = mp._meghalaya_land_cover(centers, data_dir=str(tmp_path), reader=_reader)

    assert res[0]["value"] == 1 and res[0]["problems"] == []      # 10 -> forest
    assert res[1]["value"] is None and res[1]["problems"]         # 0  -> UNAVAILABLE
    assert res[2]["value"] == 4 and res[2]["problems"] == []      # 50 -> built-up


def test_meghalaya_land_cover_refuses_when_raster_missing(tmp_path):
    # No land-cover raster on disk -> a SYSTEMIC refusal (mapped to HTTP 503), not a
    # per-cell None.
    with pytest.raises(mp.PredictionUnavailable):
        mp._meghalaya_land_cover([(25.5, 91.5)], data_dir=str(tmp_path))


# ---------------------------------------------------------------------------
# Path helpers point at the MEGHALAYA rasters (DEM+landcover in raw/, derivatives in processed/)
# ---------------------------------------------------------------------------
def test_path_helpers_point_at_meghalaya_rasters():
    paths = mp.meghalaya_terrain_raster_paths(data_dir="/tmp/somewhere")
    assert set(paths) == {"elevation", "slope", "aspect", "roughness", "tpi"}
    assert paths["elevation"].endswith(os.path.join("raw", "meghalaya_pilot_dem.tif"))
    for name in ("slope", "aspect", "roughness", "tpi"):
        assert paths[name].endswith(
            os.path.join("processed", "meghalaya_pilot_%s.tif" % name)
        )
    lc = mp.meghalaya_landcover_raster_path(data_dir="/tmp/somewhere")
    assert lc.endswith(os.path.join("raw", "meghalaya_pilot_landcover.tif"))


# ---------------------------------------------------------------------------
# Honesty disclosures
# ---------------------------------------------------------------------------
def test_disclosures_describe_real_worldcover_categorical_land_cover():
    result, _model = _run()
    disclosures = result["disclosures"]
    assert len(disclosures) >= 6
    text = " ".join(disclosures).lower()
    assert "worldcover" in text                        # real product named
    assert "categorical" in text                       # scored categorically
    assert "not an elevation proxy" in text            # explicitly NOT the Sikkim proxy
    assert "era5" in text and "imerg" in text          # train/serve rainfall shift
    assert "raw" in text                               # raw probability, not Option-C


# ---------------------------------------------------------------------------
# Refusals (mapped to HTTP 503 / 400 by the route) -- the Meghalaya path enforces them too
# ---------------------------------------------------------------------------
def test_refuses_when_model_artifacts_are_not_valid():
    bad = {"status": "INVALID", "model": None, "feature_schema": None, "problems": ["x"]}
    with pytest.raises(mp.PredictionUnavailable):
        mp.predict_meghalaya_grid(
            "2025-09-19", step_deg=0.25,
            model_evidence=bad,
            terrain_sampler=_pattern_terrain_sampler,
            land_cover_resolver=_pattern_land_cover,
            rainfall_provider=_fake_rainfall_provider,
        )


def test_refuses_on_feature_order_mismatch():
    model = _FakeModel()
    evidence = _valid_evidence(model)
    evidence["feature_schema"]["feature_names"] = list(mp.MODEL_FEATURE_ORDER)[::-1]
    with pytest.raises(mp.PredictionUnavailable) as exc:
        mp.predict_meghalaya_grid(
            "2025-09-19", step_deg=0.25,
            model_evidence=evidence,
            terrain_sampler=_pattern_terrain_sampler,
            land_cover_resolver=_pattern_land_cover,
            rainfall_provider=_fake_rainfall_provider,
        )
    assert exc.value.details["expected"] == list(mp.MODEL_FEATURE_ORDER)


def test_refuses_when_real_rainfall_cannot_be_obtained():
    def _boom(bounds, target_date, run_type="Early"):
        raise RuntimeError("EARTHDATA IMERG FETCH FAILED")

    with pytest.raises(mp.PredictionUnavailable) as exc:
        mp.predict_meghalaya_grid(
            "2025-09-19", step_deg=0.25,
            model_evidence=_valid_evidence(_FakeModel()),
            terrain_sampler=_pattern_terrain_sampler,
            land_cover_resolver=_pattern_land_cover,
            rainfall_provider=_boom,
        )
    assert "rainfall" in exc.value.reason.lower()


def test_refuses_on_non_finite_rainfall_feature():
    def _nan_rain(bounds, target_date, run_type="Early"):
        feats = sp._derive_rainfall_features([float(v) for v in range(1, 15)])
        feats["rain_7d"] = float("nan")               # a poisoned feature
        return {"source": "IMERG_Early", "run_type": "Early", "aoi_uniform": True,
                "window_days": 14, "daily_series_mm": [], "features": feats}

    with pytest.raises(mp.PredictionUnavailable):
        mp.predict_meghalaya_grid(
            "2025-09-19", step_deg=0.25,
            model_evidence=_valid_evidence(_FakeModel()),
            terrain_sampler=_pattern_terrain_sampler,
            land_cover_resolver=_pattern_land_cover,
            rainfall_provider=_nan_rain,
        )


def test_bad_target_date_string_raises_valueerror():
    with pytest.raises(ValueError):
        mp.predict_meghalaya_grid(
            "not-a-date", step_deg=0.25,
            model_evidence=_valid_evidence(_FakeModel()),
            terrain_sampler=_pattern_terrain_sampler,
            land_cover_resolver=_pattern_land_cover,
            rainfall_provider=_fake_rainfall_provider,
        )
