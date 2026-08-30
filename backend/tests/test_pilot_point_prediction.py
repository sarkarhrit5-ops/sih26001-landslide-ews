"""
Offline tests for app.services.pilot_point_prediction -- the read-only POINT
prediction service that /risk/current uses for the four canonical pilot AOIs.

Dependency budget is the same as test_sikkim_prediction / test_assam_prediction:
stdlib + numpy/pandas only. The model, terrain sampler, land-cover resolver and
rainfall provider are injected as fakes, so these tests exercise the ASSEMBLY and
the NO-FABRICATION invariants -- not a real prediction, which is host-only.

What is pinned here:

  1. State resolution is EXPLICIT. One containing AOI -> that state; two (the
     Assam/Meghalaya and Assam/Arunachal overlap bands) -> PilotStateAmbiguous
     unless `state` is named; none -> PointOutsidePilotAoi. There is no silent
     Sikkim default anywhere.
  2. The rainfall-coupled model probability is NEVER emitted as
     susceptibility_score or final_risk_score, and the Option-C trigger multiplier
     is never applied to it.
  3. REAL vs FALLBACK rainfall provenance is passed through from the producer and
     labelled, never rewritten by this layer.
  4. Sikkim scores land_cover_class as an int32 from the documented elevation
     proxy; the three WorldCover pilots score it as a pandas Categorical over
     1..6 -- exactly as their grid paths do.
  5. Unavailable terrain, nodata land cover, an unusable model, an unobtainable
     rainfall window or an out-of-range model output all REFUSE
     (PredictionUnavailable -> HTTP 503) instead of inventing a number.
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

from app.core.config_states import PILOT_AOIS, get_pilot_aoi_bounds
from app.services import risk_inputs
from app.services import worldcover as wc
from app.services import arunachal_prediction as arp
from app.services import assam_prediction as ap
from app.services import meghalaya_prediction as mp
from app.services import pilot_point_prediction as ppp
from app.services import sikkim_prediction as sp


# ---------------------------------------------------------------------------
# Representative points. The first four are the verifier's own points; each is
# inside exactly ONE canonical pilot AOI (asserted below, so this table cannot
# silently rot if an AOI is ever re-cut).
# ---------------------------------------------------------------------------
POINTS = {
    "Sikkim": (27.33, 88.62),
    "Assam": (26.14, 91.77),
    "Arunachal Pradesh": (27.10, 93.60),
    "Meghalaya": (25.57, 91.88),
}

# Inside BOTH Assam and Meghalaya (lat 25.6-25.99 x lon 91.3-92.8).
OVERLAP_ASSAM_MEGHALAYA = (25.80, 92.00)
# Inside BOTH Assam and Arunachal Pradesh (lat 26.5-26.6 x lon 92.0-93.7).
OVERLAP_ASSAM_ARUNACHAL = (26.55, 93.00)
# Inside no pilot AOI, but still inside routes.validate_coordinates' envelope.
OUTSIDE_ALL = (22.00, 82.00)

WORLDCOVER_STATES = ("Assam", "Arunachal Pradesh", "Meghalaya")
ALL_STATES = tuple(sorted(POINTS))

_ELEVATION_M = 1000.0          # -> elevation proxy class 1 (<3000 m) for Sikkim
_SLOPE_DEG = 42.0              # -> _FakeModel probability 0.42
_WORLDCOVER_GROUP = 4


class _FakeModel:
    """
    Records the exact DataFrame it was handed (so the dtype contract is testable)
    and returns slope/100 as the positive-class probability.
    """

    classes_ = [0, 1]

    def __init__(self):
        self.frames = []

    def predict_proba(self, frame):
        self.frames.append(frame.copy())
        slope = np.asarray(frame["slope"], dtype="float64") / 100.0
        return np.column_stack([1.0 - slope, slope])


def _valid_evidence(model):
    return {
        "status": "VALID",
        "model": model,
        "feature_schema": {"feature_names": list(ppp.MODEL_FEATURE_ORDER)},
        "metrics": {"validation_metrics": {"pr_auc": 0.2977, "roc_auc": 0.7113}},
        "problems": [],
    }


def _point_terrain_sampler(centers, data_dir=None):
    return [
        {
            "values": {
                "elevation": _ELEVATION_M, "slope": _SLOPE_DEG, "aspect": 123.0,
                "roughness": 4.5, "tpi": -1.2,
            },
            "problems": [],
        }
        for _center in centers
    ]


def _nodata_terrain_sampler(centers, data_dir=None):
    return [{"values": None, "problems": ["synthetic terrain nodata cell"]}
            for _center in centers]


def _point_land_cover(centers, data_dir=None):
    return [{"value": _WORLDCOVER_GROUP, "problems": []} for _center in centers]


def _nodata_land_cover(centers, data_dir=None):
    return [{"value": None, "problems": ["synthetic worldcover nodata cell"]}
            for _center in centers]


def _daily_series():
    return [float(v) for v in range(1, ppp.RAINFALL_WINDOW_DAYS + 1)]  # 1..14 mm


def _real_rainfall_provider(bounds, target_date, run_type="Early"):
    """IMERG, data_quality_status=REAL -- what the service prefers."""
    daily = _daily_series()
    return {
        "source": "NASA GPM IMERG %s (V07)" % run_type,
        "run_type": run_type,
        "aoi_uniform": True,
        "window_days": ppp.RAINFALL_WINDOW_DAYS,
        "daily_series_mm": [round(v, 4) for v in daily],
        "features": sp._derive_rainfall_features(daily),
        "source_kind": "IMERG",
        "is_fallback": False,
        "data_quality_status": "REAL",
        "units": "mm",
        "requested_date": "2025-09-19",
        "rainfall_observation_date": "2025-09-18",
        "fetched_at_utc": "2025-09-19T00:10:00Z",
        "freshness": {"cache_hit": False, "age_seconds": 0},
        "coverage": {"aoi_bounds": dict(bounds), "cells": 1},
        "caveats": [],
    }


def _fallback_rainfall_provider(bounds, target_date, run_type="Early"):
    """Open-Meteo ERA5 archive, explicitly labelled FALLBACK."""
    daily = _daily_series()
    return {
        "source": "Open-Meteo ERA5 archive",
        "run_type": run_type,
        "aoi_uniform": True,
        "window_days": ppp.RAINFALL_WINDOW_DAYS,
        "daily_series_mm": [round(v, 4) for v in daily],
        "features": sp._derive_rainfall_features(daily),
        "source_kind": "FALLBACK",
        "is_fallback": True,
        "data_quality_status": "FALLBACK",
        "units": "mm",
        "requested_date": "2025-09-19",
        "rainfall_observation_date": "2025-09-18",
        "fetched_at_utc": "2025-09-19T00:10:00Z",
        "freshness": {"cache_hit": True, "age_seconds": 120},
        "coverage": {"aoi_bounds": dict(bounds), "cells": 1},
        "caveats": ["IMERG unavailable; ERA5 reanalysis substituted and labelled."],
    }


def _run(state, lat=None, lon=None, requested_state=None, date="2025-09-19",
         terrain_sampler=_point_terrain_sampler, land_cover_resolver="default",
         rainfall_provider=_real_rainfall_provider, model=None):
    """
    Run the point service for `state` with injected fakes. land_cover_resolver
    defaults to the state's own wiring (None for Sikkim -> the elevation proxy;
    the WorldCover fake for the other three).
    """
    if lat is None or lon is None:
        lat, lon = POINTS[state]
    if land_cover_resolver == "default":
        land_cover_resolver = None if state == "Sikkim" else _point_land_cover
    model = model if model is not None else _FakeModel()
    result = ppp.predict_pilot_point(
        lat, lon, date,
        state=requested_state,
        model_evidence=_valid_evidence(model),
        terrain_sampler=terrain_sampler,
        land_cover_resolver=land_cover_resolver,
        rainfall_provider=rainfall_provider,
    )
    return result, model


# ---------------------------------------------------------------------------
# The representative points really are where these tests claim they are
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("state", ALL_STATES)
def test_representative_point_is_inside_exactly_its_own_pilot_aoi(state):
    lat, lon = POINTS[state]
    assert ppp.pilot_states_containing(lat, lon) == [state]
    bounds = get_pilot_aoi_bounds(state)
    assert bounds["min_lat"] <= lat <= bounds["max_lat"]
    assert bounds["min_lon"] <= lon <= bounds["max_lon"]


def test_overlap_points_really_are_in_two_pilot_aois():
    assert ppp.pilot_states_containing(*OVERLAP_ASSAM_MEGHALAYA) == ["Assam", "Meghalaya"]
    assert ppp.pilot_states_containing(*OVERLAP_ASSAM_ARUNACHAL) == [
        "Arunachal Pradesh", "Assam"
    ]
    assert ppp.pilot_states_containing(*OUTSIDE_ALL) == []


def test_pilot_specs_cover_exactly_the_canonical_pilot_states():
    assert set(ppp.PILOT_SPECS) == set(PILOT_AOIS) == set(POINTS)


# ---------------------------------------------------------------------------
# State resolution: explicit, never defaulted
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("state", ALL_STATES)
def test_unqualified_point_resolves_to_its_single_containing_state(state):
    lat, lon = POINTS[state]
    assert ppp.resolve_pilot_state(lat, lon) == state


@pytest.mark.parametrize("point,expected", [
    (OVERLAP_ASSAM_MEGHALAYA, ["Assam", "Meghalaya"]),
    (OVERLAP_ASSAM_ARUNACHAL, ["Arunachal Pradesh", "Assam"]),
])
def test_overlapping_aoi_without_state_is_ambiguous_and_assumes_nothing(point, expected):
    with pytest.raises(ppp.PilotStateAmbiguous) as excinfo:
        ppp.resolve_pilot_state(point[0], point[1])
    err = excinfo.value
    assert err.details["pilot_states_containing_point"] == expected
    for name in expected:
        assert name in err.reason
    # The ambiguity error must not pick a winner, and must not mention a default.
    assert "No state was assumed." in err.reason


@pytest.mark.parametrize("requested,expected", [
    ("assam", "Assam"),
    ("Assam", "Assam"),
    ("  MEGHALAYA ", "Meghalaya"),
    ("meghalaya", "Meghalaya"),
])
def test_overlap_is_resolved_by_the_explicit_state_parameter(requested, expected):
    lat, lon = OVERLAP_ASSAM_MEGHALAYA
    assert ppp.resolve_pilot_state(lat, lon, state=requested) == expected


@pytest.mark.parametrize("requested,expected", [
    ("arunachal", "Arunachal Pradesh"),
    ("arunachal pradesh", "Arunachal Pradesh"),
    ("arunachal_pradesh", "Arunachal Pradesh"),
    ("Arunachal-Pradesh", "Arunachal Pradesh"),
])
def test_arunachal_spellings_all_resolve(requested, expected):
    lat, lon = POINTS["Arunachal Pradesh"]
    assert ppp.resolve_pilot_state(lat, lon, state=requested) == expected


def test_point_outside_every_pilot_aoi_raises_the_fallthrough_error():
    with pytest.raises(ppp.PointOutsidePilotAoi) as excinfo:
        ppp.resolve_pilot_state(*OUTSIDE_ALL)
    assert set(excinfo.value.details["pilot_aois"]) == set(PILOT_AOIS)


def test_unknown_state_is_invalid_not_silently_ignored():
    lat, lon = POINTS["Sikkim"]
    with pytest.raises(ppp.PilotStateInvalid) as excinfo:
        ppp.resolve_pilot_state(lat, lon, state="Nagaland")
    assert excinfo.value.details["pilot_states"] == sorted(PILOT_AOIS)


def test_named_state_whose_aoi_excludes_the_point_is_invalid():
    lat, lon = POINTS["Sikkim"]
    with pytest.raises(ppp.PilotStateInvalid) as excinfo:
        ppp.resolve_pilot_state(lat, lon, state="assam")
    err = excinfo.value
    assert err.details["requested_state"] == "Assam"
    assert err.details["pilot_states_containing_point"] == ["Sikkim"]
    # A wrong state must NOT be silently rewritten to the containing one.
    assert "outside the canonical Assam pilot AOI" in err.reason


def test_blank_state_parameter_is_treated_as_absent():
    lat, lon = POINTS["Meghalaya"]
    assert ppp.resolve_pilot_state(lat, lon, state="   ") == "Meghalaya"


# ---------------------------------------------------------------------------
# The happy path, for all four pilots
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("state", ALL_STATES)
def test_point_prediction_shape_and_labels(state):
    result, model = _run(state)
    lat, lon = POINTS[state]

    assert result["state"] == state
    assert result["pilot_area"] == ppp.PILOT_SPECS[state]["pilot_area"]
    assert result["method"] == ppp.METHOD == "pilot_rainfall_coupled_model_point"
    assert result["point"] == {"latitude": lat, "longitude": lon}
    assert result["target_date"] == "2025-09-19"
    assert result["aoi"] == get_pilot_aoi_bounds(state)
    assert result["decision_threshold"] == ppp.DECISION_THRESHOLD
    assert result["state_resolution"] == {
        "resolved_state": state,
        "requested_state": None,
        "pilot_states_containing_point": [state],
    }

    hazard = result["hazard"]
    assert hazard["status"] == "OK"
    # slope 42 -> 0.42 from _FakeModel: the model really consumed the sampled slope.
    assert hazard["rainfall_conditioned_probability"] == pytest.approx(0.42, abs=1e-9)
    assert hazard["exceeds_decision_threshold"] is False
    assert hazard["is_option_c_fused_risk"] is False
    assert hazard["is_rainfall_independent_susceptibility"] is False
    assert isinstance(hazard["risk_class"], str) and hazard["risk_class"]
    assert model.frames, "the persisted model must actually be called"


@pytest.mark.parametrize("state", ALL_STATES)
def test_all_eleven_features_are_real_and_none_are_defaulted(state):
    result, _model = _run(state)
    features = result["hazard"]["features"]
    assert list(features) == list(ppp.MODEL_FEATURE_ORDER)
    assert features["elevation"] == pytest.approx(_ELEVATION_M)
    assert features["slope"] == pytest.approx(_SLOPE_DEG)
    expected_rain = sp._derive_rainfall_features(_daily_series())
    for feat, value in expected_rain.items():
        assert features[feat] == pytest.approx(round(value, 4))


@pytest.mark.parametrize("state", ALL_STATES)
def test_probability_moves_with_rainfall_driven_model_input(state):
    """
    The row handed to the model carries the live rainfall series, so a different
    series reaches the model. (_FakeModel keys off slope, so this asserts the
    PLUMBING -- the recorded frame -- not a fabricated sensitivity.)
    """
    _low, low_model = _run(state)
    low_frame = low_model.frames[0]
    assert float(low_frame["rain_1d"].iloc[0]) == pytest.approx(1.0)
    assert float(low_frame["antecedent_rain_14d"].iloc[0]) == pytest.approx(sum(_daily_series()))


# ---------------------------------------------------------------------------
# Option-C fusion is reported as NOT applied, and never faked
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("state", ALL_STATES)
def test_option_c_fusion_is_explicitly_unavailable_and_not_applied(state):
    result, _model = _run(state)
    fusion = result["option_c_fusion"]
    assert fusion["available"] is False
    assert fusion["applied"] is False
    assert fusion["susceptibility_score"] is None
    assert fusion["trigger_multiplier"] is None
    assert fusion["final_risk_score"] is None
    assert "double-count rainfall" in fusion["reason"]
    for feat in ppp.RAINFALL_FEATURES:
        assert feat in fusion["reason"]


@pytest.mark.parametrize("state", ALL_STATES)
def test_the_model_probability_is_never_emitted_as_susceptibility_or_fused_risk(state):
    result, _model = _run(state)
    probability = result["hazard"]["rainfall_conditioned_probability"]

    def _walk(node, path=""):
        if isinstance(node, dict):
            for key, value in node.items():
                where = "%s.%s" % (path, key)
                if key in ("susceptibility_score", "final_risk_score",
                           "trigger_multiplier"):
                    assert value is None, "%s must stay None, got %r" % (where, value)
                    continue
                assert value != probability or key in (
                    "rainfall_conditioned_probability",
                ), "%s leaks the coupled probability" % where
                _walk(value, where)
        elif isinstance(node, list):
            for i, value in enumerate(node):
                _walk(value, "%s[%d]" % (path, i))

    _walk(result)


@pytest.mark.parametrize("state", ALL_STATES)
def test_disclosures_name_the_method_and_the_fusion_refusal(state):
    result, _model = _run(state)
    disclosures = result["disclosures"]
    blob = " ".join(disclosures)
    assert ppp.METHOD in blob
    assert "NOT a rainfall-independent" in blob
    assert ppp.OPTION_C_UNAVAILABLE_REASON in disclosures
    # The state's own grid disclosures are carried through verbatim, not edited.
    for line in ppp.PILOT_SPECS[state]["disclosures"]():
        assert line in disclosures


# ---------------------------------------------------------------------------
# Rainfall provenance: REAL vs FALLBACK, passed through, never rewritten
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("state", ALL_STATES)
def test_real_imerg_rainfall_is_reported_as_real(state):
    result, _model = _run(state, rainfall_provider=_real_rainfall_provider)
    rainfall = result["rainfall"]
    assert rainfall["data_quality_status"] == "REAL"
    assert rainfall["source_kind"] == "IMERG"
    assert rainfall["is_fallback"] is False
    assert result["resolved_inputs"]["rainfall"] == {
        "status": "REAL",
        "source": rainfall["source"],
        "is_fallback": False,
    }
    assert "FALLBACK" not in (rainfall.get("note") or "")


@pytest.mark.parametrize("state", ALL_STATES)
def test_fallback_rainfall_is_labelled_fallback_and_never_called_official(state):
    result, _model = _run(state, rainfall_provider=_fallback_rainfall_provider)
    rainfall = result["rainfall"]
    assert rainfall["data_quality_status"] == "FALLBACK"
    assert rainfall["source_kind"] == "FALLBACK"
    assert rainfall["is_fallback"] is True
    assert "Open-Meteo ERA5" in rainfall["source"]
    assert result["resolved_inputs"]["rainfall"]["status"] == "FALLBACK"
    assert result["resolved_inputs"]["rainfall"]["is_fallback"] is True
    note = rainfall["note"]
    assert "FALLBACK" in note and "NOT an official live IMERG observation" in note


@pytest.mark.parametrize("state", ALL_STATES)
def test_fallback_changes_only_provenance_not_the_resolved_inputs_of_other_kinds(state):
    real, _m1 = _run(state, rainfall_provider=_real_rainfall_provider)
    fallback, _m2 = _run(state, rainfall_provider=_fallback_rainfall_provider)
    for key in ("model", "terrain", "land_cover"):
        assert real["resolved_inputs"][key] == fallback["resolved_inputs"][key]
    assert real["hazard"]["features"] == fallback["hazard"]["features"]
    assert real["resolved_inputs"]["rainfall"] != fallback["resolved_inputs"]["rainfall"]


@pytest.mark.parametrize("state", ALL_STATES)
def test_rainfall_uses_the_canonical_aoi_bounds_so_it_shares_the_grid_cache_entry(state):
    seen = {}

    def _recording_provider(bounds, target_date, run_type="Early"):
        seen["bounds"] = bounds
        seen["run_type"] = run_type
        return _real_rainfall_provider(bounds, target_date, run_type)

    result, _model = _run(state, rainfall_provider=_recording_provider)
    assert seen["bounds"] == get_pilot_aoi_bounds(state)
    assert seen["run_type"] == "Early"
    assert result["aoi"] == seen["bounds"]


# ---------------------------------------------------------------------------
# The land-cover / dtype contract, per state family
# ---------------------------------------------------------------------------
def test_sikkim_uses_the_documented_elevation_proxy_scored_as_int32():
    result, model = _run("Sikkim")
    frame = model.frames[0]
    assert str(frame[sp.LAND_COVER_FEATURE].dtype) == "int32"
    expected = int(risk_inputs.land_cover_class_from_elevation(_ELEVATION_M))
    assert result["hazard"]["features"][sp.LAND_COVER_FEATURE] == expected
    assert result["resolved_inputs"]["land_cover"]["status"] == risk_inputs.STATUS_DERIVED_PROXY
    assert "elevation-binned proxy" in result["resolved_inputs"]["land_cover"]["source"]


@pytest.mark.parametrize("state", WORLDCOVER_STATES)
def test_worldcover_pilots_score_land_cover_as_a_fixed_categorical(state):
    result, model = _run(state)
    frame = model.frames[0]
    column = frame[sp.LAND_COVER_FEATURE]
    assert isinstance(column.dtype, pd.CategoricalDtype)
    assert list(column.cat.categories) == list(wc.ASSAM_LANDCOVER_GROUP_CODES)
    assert int(column.iloc[0]) == _WORLDCOVER_GROUP
    assert result["hazard"]["features"][sp.LAND_COVER_FEATURE] == _WORLDCOVER_GROUP
    assert result["resolved_inputs"]["land_cover"]["status"] == risk_inputs.STATUS_REAL
    assert "WorldCover" in result["resolved_inputs"]["land_cover"]["source"]


@pytest.mark.parametrize("state", WORLDCOVER_STATES)
def test_worldcover_land_cover_is_not_the_elevation_proxy(state):
    """
    The fake sampler reports 1000 m, whose proxy class is 1; the fake resolver
    reports group 4. Seeing 4 proves the real resolver was consulted.
    """
    result, _model = _run(state)
    assert int(risk_inputs.land_cover_class_from_elevation(_ELEVATION_M)) == 1
    assert result["hazard"]["features"][sp.LAND_COVER_FEATURE] == 4


@pytest.mark.parametrize("state", ALL_STATES)
def test_the_model_frame_is_exactly_one_row_in_the_persisted_feature_order(state):
    _result, model = _run(state)
    assert len(model.frames) == 1
    frame = model.frames[0]
    assert list(frame.columns) == list(ppp.MODEL_FEATURE_ORDER)
    assert len(frame) == 1, "a point request must score ONE row, not a grid"


# ---------------------------------------------------------------------------
# Refusals: unavailable inputs surface as UNAVAILABLE, never as a number
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("state", ALL_STATES)
def test_terrain_nodata_at_the_point_refuses(state):
    with pytest.raises(ppp.PredictionUnavailable) as excinfo:
        _run(state, terrain_sampler=_nodata_terrain_sampler)
    err = excinfo.value
    assert "Real terrain is unavailable" in err.reason
    assert "synthetic terrain nodata cell" in err.details["problems"]


@pytest.mark.parametrize("state", WORLDCOVER_STATES)
def test_land_cover_nodata_at_the_point_refuses_instead_of_back_filling(state):
    with pytest.raises(ppp.PredictionUnavailable) as excinfo:
        _run(state, land_cover_resolver=_nodata_land_cover)
    err = excinfo.value
    assert "Land cover is unavailable" in err.reason
    assert "no\nclass was substituted" in err.reason.replace(" ", "\n") or \
        "class was substituted" in err.reason
    assert "synthetic worldcover nodata cell" in err.details["problems"]


def test_sikkim_refuses_when_the_elevation_cannot_be_binned():
    def _nan_elevation_sampler(centers, data_dir=None):
        rows = _point_terrain_sampler(centers, data_dir)
        rows[0]["values"]["elevation"] = float("nan")
        return rows

    with pytest.raises(ppp.PredictionUnavailable) as excinfo:
        _run("Sikkim", terrain_sampler=_nan_elevation_sampler)
    assert "Land cover is unavailable" in excinfo.value.reason


@pytest.mark.parametrize("state", ALL_STATES)
def test_unusable_model_artifact_refuses(state):
    evidence = {"status": "UNAVAILABLE", "model": None,
                "feature_schema": None, "metrics": None,
                "problems": ["synthetic missing artifact"]}
    lat, lon = POINTS[state]
    with pytest.raises(ppp.PredictionUnavailable):
        ppp.predict_pilot_point(
            lat, lon, "2025-09-19",
            model_evidence=evidence,
            terrain_sampler=_point_terrain_sampler,
            land_cover_resolver=None if state == "Sikkim" else _point_land_cover,
            rainfall_provider=_real_rainfall_provider,
        )


@pytest.mark.parametrize("state", ALL_STATES)
def test_rainfall_provider_failure_refuses_and_never_imputes_zero(state):
    def _boom(bounds, target_date, run_type="Early"):
        raise RuntimeError("synthetic IMERG and fallback both unreachable")

    with pytest.raises(ppp.PredictionUnavailable) as excinfo:
        _run(state, rainfall_provider=_boom)
    reason = excinfo.value.reason
    assert "Real antecedent rainfall could not be obtained" in reason
    assert "synthetic IMERG and fallback both unreachable" in reason


@pytest.mark.parametrize("state", ALL_STATES)
def test_non_finite_rainfall_feature_refuses(state):
    def _nan_rain(bounds, target_date, run_type="Early"):
        payload = _real_rainfall_provider(bounds, target_date, run_type)
        payload["features"]["rain_3d"] = float("nan")
        return payload

    with pytest.raises(ppp.PredictionUnavailable) as excinfo:
        _run(state, rainfall_provider=_nan_rain)
    assert "rain_3d" in excinfo.value.reason


@pytest.mark.parametrize("state", ALL_STATES)
def test_rainfall_payload_without_features_refuses(state):
    def _no_features(bounds, target_date, run_type="Early"):
        return {"source": "synthetic", "window_days": ppp.RAINFALL_WINDOW_DAYS}

    with pytest.raises(ppp.PredictionUnavailable) as excinfo:
        _run(state, rainfall_provider=_no_features)
    assert "features" in excinfo.value.reason


@pytest.mark.parametrize("state", ALL_STATES)
def test_out_of_range_model_output_is_refused_not_clamped(state):
    class _RogueModel(_FakeModel):
        def predict_proba(self, frame):
            self.frames.append(frame.copy())
            return np.array([[0.0, 1.7]])

    with pytest.raises(ppp.PredictionUnavailable) as excinfo:
        _run(state, model=_RogueModel())
    assert "refusing to clamp" in excinfo.value.reason
    assert excinfo.value.details["raw_output"] == pytest.approx(1.7)


@pytest.mark.parametrize("state", ALL_STATES)
def test_a_sampler_that_returns_the_wrong_row_count_refuses(state):
    def _two_rows(centers, data_dir=None):
        return _point_terrain_sampler(centers, data_dir) * 2

    with pytest.raises(ppp.PredictionUnavailable) as excinfo:
        _run(state, terrain_sampler=_two_rows)
    assert "for 1 point" in excinfo.value.reason


def test_predict_pilot_point_propagates_state_errors_without_predicting():
    lat, lon = OVERLAP_ASSAM_MEGHALAYA
    with pytest.raises(ppp.PilotStateAmbiguous):
        ppp.predict_pilot_point(
            lat, lon, "2025-09-19",
            model_evidence=_valid_evidence(_FakeModel()),
            terrain_sampler=_point_terrain_sampler,
            land_cover_resolver=_point_land_cover,
            rainfall_provider=_real_rainfall_provider,
        )


def test_state_resolution_is_echoed_when_the_caller_named_a_state():
    lat, lon = OVERLAP_ASSAM_MEGHALAYA
    result, _model = _run("Assam", lat=lat, lon=lon, requested_state="assam")
    assert result["state"] == "Assam"
    assert result["state_resolution"] == {
        "resolved_state": "Assam",
        "requested_state": "assam",
        "pilot_states_containing_point": ["Assam", "Meghalaya"],
    }


# ---------------------------------------------------------------------------
# Non-interference: the point path reuses the grid path's collaborators and
# changes nothing about them.
# ---------------------------------------------------------------------------
def test_point_service_reuses_the_existing_grid_collaborators_by_identity():
    assert ppp.PredictionUnavailable is sp.PredictionUnavailable
    assert ppp.MODEL_FEATURE_ORDER is sp.MODEL_FEATURE_ORDER
    assert ppp.RAINFALL_WINDOW_DAYS == sp.RAINFALL_WINDOW_DAYS
    assert ppp.DECISION_THRESHOLD == sp.DECISION_THRESHOLD
    assert ppp.PILOT_SPECS["Sikkim"]["disclosures"] is sp._disclosures
    assert ppp.PILOT_SPECS["Assam"]["disclosures"] is ap._assam_disclosures
    assert ppp.PILOT_SPECS["Arunachal Pradesh"]["disclosures"] is arp._arunachal_disclosures
    assert ppp.PILOT_SPECS["Meghalaya"]["disclosures"] is mp._meghalaya_disclosures


def test_the_four_grid_entry_points_are_untouched_and_still_callable():
    for module, name in (
        (sp, "predict_sikkim_grid"), (ap, "predict_assam_grid"),
        (arp, "predict_arunachal_grid"), (mp, "predict_meghalaya_grid"),
    ):
        assert callable(getattr(module, name))


def test_the_grid_path_still_produces_its_own_unchanged_response():
    """
    The point path must not have altered the grid path: run the real Assam grid
    with the same fakes and check its own contract (cells + grid keys, and NO
    point-only keys) still holds.
    """
    model = _FakeModel()
    grid = ap.predict_assam_grid(
        "2025-09-19",
        step_deg=0.25,
        model_evidence=_valid_evidence(model),
        terrain_sampler=_point_terrain_sampler,
        land_cover_resolver=_point_land_cover,
        rainfall_provider=_real_rainfall_provider,
    )
    assert grid["state"] == "Assam"
    assert grid["cells"], "grid must still return cells"
    assert "grid" in grid
    for point_only in ("hazard", "option_c_fusion", "state_resolution", "point",
                       "method"):
        assert point_only not in grid
