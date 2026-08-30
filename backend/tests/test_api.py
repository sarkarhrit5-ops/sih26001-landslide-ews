import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from fastapi import HTTPException
from fastapi.testclient import TestClient
from app.main import app
from app.services import risk_inputs

client = TestClient(app)

# The risk endpoints are now data-dependent: they answer only when every required
# input resolves to a real measurement, and otherwise refuse with HTTP 503. Both
# outcomes are correct, so these helpers assert the contract rather than a fixed
# status code -- which also means the suite does not require a DEM, credentials or
# a persisted model to pass.
DATA_UNAVAILABLE_STATUS_CODE = 503
USABLE_STATUSES = set(risk_inputs.USABLE_STATUSES)


def _assert_honest_refusal(response):
    assert response.status_code == DATA_UNAVAILABLE_STATUS_CODE
    detail = response.json()["detail"]
    assert detail["status"] == risk_inputs.DATA_UNAVAILABLE
    assert detail["blocking_inputs"], "a refusal must name the missing input(s)"
    assert detail["blocking_reasons"], "a refusal must explain itself"
    # A refusal must not smuggle a risk verdict out anyway.
    assert "final_risk_score" not in response.text
    assert "warning_level" not in response.text


def _assert_inputs_were_real(data, mode):
    resolved = data["resolved_inputs"]
    for name in risk_inputs.REQUIRED_INPUTS_BY_MODE[mode]:
        assert resolved[name]["status"] in USABLE_STATUSES, (
            "a 200 response may only be built from real inputs"
        )


def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}

def test_current_risk_valid():
    # OUTSIDE every canonical pilot AOI, so this exercises the (unchanged) Option-C
    # fusion path. A point inside a pilot AOI now takes the pilot-point path, whose
    # contract is asserted separately below.
    response = client.get("/api/v1/risk/current?lat=22.0&lon=82.0")
    if response.status_code != 200:
        _assert_honest_refusal(response)
        return
    data = response.json()
    assert "location" in data
    assert "risk" in data
    risk = data["risk"]
    assert "susceptibility_score" in risk
    assert "current_trigger_score" in risk
    assert "forecast_trigger_score" in risk
    assert "exposure_score" in risk
    assert "final_risk_score" in risk
    assert "warning_level" in risk
    assert "confidence" in risk
    _assert_inputs_were_real(data, risk_inputs.RISK_MODE_CURRENT)
    # /risk/current asserts nothing about the future.
    assert risk["forecast_evaluated"] is False
    assert risk["forecast_trigger_score"] is None

def test_current_risk_invalid_coordinates():
    # Test latitude out of bounds (e.g. lat=10.0 outside 20-35 range)
    response = client.get("/api/v1/risk/current?lat=10.0&lon=88.61")
    assert response.status_code == 400
    assert "Invalid coordinates" in response.json()["detail"]

def test_forecast_risk_valid():
    response = client.get("/api/v1/risk/forecast?lat=27.33&lon=88.61")
    if response.status_code != 200:
        _assert_honest_refusal(response)
        return
    data = response.json()
    assert "location" in data
    assert "forecast_accumulation_mm" in data
    assert "risk_forecast" in data
    risk = data["risk_forecast"]
    assert "final_risk_score" in risk
    assert "warning_level" in risk
    _assert_inputs_were_real(data, risk_inputs.RISK_MODE_FORECAST)
    # An unreachable forecast service is a refusal, never a 0 mm forecast.
    assert data["forecast_accumulation_mm"] is not None

def test_cell_explain():
    response = client.get("/api/v1/cell/cell_123/explain")
    assert response.status_code == 200
    data = response.json()
    assert data["cell_id"] == "cell_123"
    assert "explanation" in data

def test_cell_explain_without_coordinates_is_unavailable_not_invented():
    from app.models.ml_pipeline import EXPLANATION_STATUS_UNAVAILABLE

    response = client.get("/api/v1/cell/cell_123/explain")
    assert response.status_code == 200
    data = response.json()
    # There is no cell registry, so a bare cell_id cannot be explained. It used to
    # return invented importances (slope 0.42 / rain_3d 0.28 / roughness 0.18).
    assert data["explanation"]["status"] == EXPLANATION_STATUS_UNAVAILABLE
    assert data["explanation"]["top_features"] == []
    assert data["explanation"]["reasons"]
    assert "unresolved_cell" in data
    for invented in ("0.42", "0.28", "0.18"):
        assert invented not in response.text

def test_cell_explain_with_coordinates_reports_missing_model():
    from app.models.ml_pipeline import (
        EXPLANATION_STATUS_REAL,
        EXPLANATION_STATUS_UNAVAILABLE,
    )

    response = client.get("/api/v1/cell/cell_123/explain?lat=27.33&lon=88.61")
    assert response.status_code == 200
    explanation = response.json()["explanation"]
    assert explanation["status"] in (
        EXPLANATION_STATUS_REAL, EXPLANATION_STATUS_UNAVAILABLE
    )
    if explanation["status"] == EXPLANATION_STATUS_UNAVAILABLE:
        assert explanation["reasons"]
        assert explanation["top_features"] == []

def test_cell_explain_rejects_invalid_coordinates():
    response = client.get("/api/v1/cell/cell_123/explain?lat=10.0&lon=88.61")
    assert response.status_code == 400

def test_exposure_alerts():
    response = client.get("/api/v1/exposure/alerts")
    assert response.status_code == 200
    data = response.json()
    assert "exposed_assets" in data
    assert len(data["exposed_assets"]) > 0
    # The payload is a hand-written fixture, not a real OSM/Overpass query, and
    # must say so explicitly so it cannot be mistaken for measured exposure data.
    assert data["is_mock"] is True
    assert data["data_source"] == "MOCK_FIXTURE"
    assert "not a real OSM" in data["provenance"]

def test_validation_status_endpoint():
    response = client.get("/api/v1/validation/status")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) == 8
    state_names = [item["state"] for item in data]
    assert "Sikkim" in state_names
    assert "Arunachal Pradesh" in state_names
    assert "Assam" in state_names
    assert "Manipur" in state_names
    assert "Meghalaya" in state_names
    assert "Mizoram" in state_names
    assert "Nagaland" in state_names
    assert "Tripura" in state_names


# ---------------------------------------------------------------------------
# Rainfall provenance exposure
#
# The route layer neither acquires rainfall nor computes risk; it reshapes the
# provenance that rainfall_service already stamped onto every read into one
# consistently-named block. These tests therefore drive the route functions
# directly with canned producer payloads: no network, no Earthdata credentials,
# no DEM and no persisted model are required, and nothing here can be mistaken
# for evidence that a real fetch succeeded.
# ---------------------------------------------------------------------------
import pytest  # noqa: E402 - after the sys.path fixup

from app.api import routes  # noqa: E402
from app.services import (  # noqa: E402
    arunachal_prediction,
    assam_prediction,
    meghalaya_prediction,
    sikkim_prediction,
)

PILOT_ENDPOINTS = [
    ("sikkim", sikkim_prediction, "predict_sikkim_grid", routes.predict_sikkim_grid),
    ("assam", assam_prediction, "predict_assam_grid", routes.predict_assam_grid),
    ("arunachal", arunachal_prediction, "predict_arunachal_grid",
     routes.predict_arunachal_grid),
    ("meghalaya", meghalaya_prediction, "predict_meghalaya_grid",
     routes.predict_meghalaya_grid),
]

_REAL_RAINFALL_REPORT = {
    "source": "IMERG_Early",
    "source_kind": "IMERG",
    "is_fallback": False,
    "data_quality_status": "REAL",
    "requested_date": "2025-09-19",
    "rainfall_observation_date": "2025-09-18",
    "fetched_at_utc": "2025-09-19T06:00:00Z",
    "freshness": {"cache_hit": False, "age_seconds": 0.0, "ttl_seconds": 1800.0},
    "units": "mm",
    "run_type": "Early",
    "window_days": 14,
    "daily_series_mm": [1.0] * 14,
    "features": {"rain_1d": 1.0},
    "note": "Antecedent-only (T-1..T-14, event day excluded).",
}

_FALLBACK_RAINFALL_REPORT = dict(
    _REAL_RAINFALL_REPORT,
    source="Open-Meteo ERA5 archive (FALLBACK)",
    source_kind="OPEN_METEO_FALLBACK",
    is_fallback=True,
    data_quality_status="FALLBACK",
    caveats=["Reanalysis, not a live satellite observation."],
)


def _prediction_payload(rainfall):
    """A minimal stand-in for a pilot prediction response."""
    return {
        "state": "Sikkim",
        "target_date": "2025-09-19",
        "grid": {"cells": 4},
        "model": {"artifact": "fixture"},
        "rainfall": dict(rainfall),
        "summary": {"scored_cells": 4},
        "disclosures": ["fixture"],
        "cells": [],
    }


def _canned(module, attr, monkeypatch, payload):
    monkeypatch.setattr(module, attr, lambda *a, **k: payload)


@pytest.mark.parametrize("name,module,attr,route", PILOT_ENDPOINTS)
def test_pilot_endpoints_expose_real_rainfall_provenance(name, module, attr, route,
                                                         monkeypatch):
    _canned(module, attr, monkeypatch, _prediction_payload(_REAL_RAINFALL_REPORT))
    body = route()
    block = body["rainfall_provenance"]
    assert block["data_quality_status"] == "REAL"
    assert block["source_kind"] == "IMERG"
    assert block["is_fallback"] is False
    assert "fallback_warning" not in block
    for field in routes.RAINFALL_PROVENANCE_FIELDS:
        assert field in block, "%s must expose %s" % (name, field)
    assert block["units"] == "mm"
    assert block["requested_date"] == "2025-09-19"
    assert block["rainfall_observation_date"] == "2025-09-18"
    assert block["fetched_at_utc"] == "2025-09-19T06:00:00Z"
    assert block["freshness"]["ttl_seconds"] == 1800.0


@pytest.mark.parametrize("name,module,attr,route", PILOT_ENDPOINTS)
def test_pilot_endpoints_mark_fallback_unmistakably(name, module, attr, route,
                                                    monkeypatch):
    _canned(module, attr, monkeypatch, _prediction_payload(_FALLBACK_RAINFALL_REPORT))
    body = route()
    block = body["rainfall_provenance"]
    assert block["data_quality_status"] == "FALLBACK"
    assert block["is_fallback"] is True
    assert block["source_kind"] == "OPEN_METEO_FALLBACK"
    assert "FALLBACK" in block["fallback_warning"]
    assert "Open-Meteo ERA5" in block["fallback_warning"]
    assert "Open-Meteo ERA5" in block["source"]
    assert block["caveats"] == ["Reanalysis, not a live satellite observation."]


@pytest.mark.parametrize("name,module,attr,route", PILOT_ENDPOINTS)
def test_pilot_endpoints_preserve_every_existing_field(name, module, attr, route,
                                                       monkeypatch):
    payload = _prediction_payload(_REAL_RAINFALL_REPORT)
    expected = {k: v for k, v in payload.items()}
    _canned(module, attr, monkeypatch, payload)
    body = route()
    for key, value in expected.items():
        assert body[key] == value, "%s dropped or altered %r" % (name, key)
    # Purely additive.
    assert set(body) - set(expected) == {"rainfall_provenance"}
    # The producer's own richer report is untouched.
    assert body["rainfall"]["note"] == _REAL_RAINFALL_REPORT["note"]
    assert body["rainfall"]["daily_series_mm"] == _REAL_RAINFALL_REPORT["daily_series_mm"]


@pytest.mark.parametrize("name,module,attr,route", PILOT_ENDPOINTS)
def test_pilot_endpoints_still_refuse_unavailable_rainfall(name, module, attr, route,
                                                           monkeypatch):
    def boom(*a, **k):
        raise module.PredictionUnavailable(
            "real antecedent rainfall could not be obtained",
            details={"source": "IMERG"},
        )

    monkeypatch.setattr(module, attr, boom)
    with pytest.raises(HTTPException) as excinfo:
        route()
    assert excinfo.value.status_code == DATA_UNAVAILABLE_STATUS_CODE
    assert excinfo.value.detail["status"] == "DATA_UNAVAILABLE"
    assert "rainfall" in excinfo.value.detail["reason"]
    # A refusal carries no probability.
    assert "susceptibility_probability" not in str(excinfo.value.detail)


@pytest.mark.parametrize("name,module,attr,route", PILOT_ENDPOINTS)
def test_pilot_endpoints_tolerate_a_producer_without_provenance(name, module, attr,
                                                                route, monkeypatch):
    """A legacy/injected producer that predates provenance must not get a fake block."""
    payload = _prediction_payload({"source": None, "run_type": "Early"})
    payload["rainfall"] = {"run_type": "Early", "window_days": 14}
    _canned(module, attr, monkeypatch, payload)
    body = route()
    assert "rainfall_provenance" not in body


# --- /risk/current and /risk/forecast -------------------------------------
# The Option-C fusion path of /risk/current is now reached only for points OUTSIDE
# every canonical pilot AOI (inside one, the pilot-point path answers instead), so
# these Option-C tests use such a point. /risk/forecast is unchanged and ignores
# the distinction.
_OPTION_C_POINT = (22.0, 82.0)


def _resolution(rainfall_record, mode):
    """A resolution shaped like risk_inputs.resolve_risk_inputs' return value."""
    inputs = {
        risk_inputs.INPUT_SUSCEPTIBILITY: risk_inputs.input_record(
            risk_inputs.INPUT_SUSCEPTIBILITY, risk_inputs.STATUS_REAL, value=0.4,
            source="fixture model",
        ),
        risk_inputs.INPUT_SLOPE: risk_inputs.input_record(
            risk_inputs.INPUT_SLOPE, risk_inputs.STATUS_REAL, value=22.0,
            source="fixture DEM",
        ),
        risk_inputs.INPUT_EXPOSURE: risk_inputs.input_record(
            risk_inputs.INPUT_EXPOSURE, risk_inputs.STATUS_REAL, value=0.3,
            source="fixture OSM",
        ),
        risk_inputs.INPUT_CURRENT_RAINFALL: rainfall_record,
        risk_inputs.INPUT_FORECAST_RAINFALL: risk_inputs.input_record(
            risk_inputs.INPUT_FORECAST_RAINFALL, risk_inputs.STATUS_REAL, value=8.0,
            source="fixture forecast",
        ),
    }
    blocking = [
        key for key in risk_inputs.REQUIRED_INPUTS_BY_MODE[mode]
        if inputs[key]["status"] not in risk_inputs.USABLE_STATUSES
    ]
    return {
        "location": list(_OPTION_C_POINT),
        "mode": mode,
        "inputs": inputs,
        "required_inputs": list(risk_inputs.REQUIRED_INPUTS_BY_MODE[mode]),
        "non_blocking_inputs": list(risk_inputs.NON_BLOCKING_INPUTS),
        "blocking_inputs": blocking,
        "blocking_reasons": [
            "%s: %s" % (k, r) for k in blocking for r in inputs[k]["reasons"]
        ],
        "usable": not blocking,
        "has_real_dem": True,
        "has_real_rainfall": (
            inputs[risk_inputs.INPUT_CURRENT_RAINFALL]["status"]
            == risk_inputs.STATUS_REAL
        ),
    }


def _rainfall_record(status, value, details, source, reasons=None):
    record = risk_inputs.input_record(
        risk_inputs.INPUT_CURRENT_RAINFALL, status, value=value, source=source,
        reasons=reasons, details=details,
    )
    return record


_REAL_DETAILS = {
    "window_hours": 24,
    "data_quality_status": "REAL",
    "source_kind": "IMERG",
    "is_fallback": False,
    "units": "mm",
    "fetched_at_utc": "2025-09-19T06:00:00Z",
    "freshness": {"cache_hit": True, "age_seconds": 12.0, "ttl_seconds": 1800.0},
    "requested_date": "2025-09-18",
    "target_date": "2025-09-18",
}

_FALLBACK_DETAILS = dict(
    _REAL_DETAILS,
    data_quality_status="FALLBACK",
    source_kind="OPEN_METEO_FALLBACK",
    is_fallback=True,
    caveats=["Reanalysis, not a live satellite observation."],
)


def _patch_resolution(monkeypatch, resolution):
    monkeypatch.setattr(
        routes.risk_inputs, "resolve_risk_inputs",
        lambda lat, lon, mode=risk_inputs.RISK_MODE_CURRENT, **k: resolution,
    )


@pytest.mark.parametrize("route,mode", [
    (routes.get_current_risk, risk_inputs.RISK_MODE_CURRENT),
    (routes.get_forecast_risk, risk_inputs.RISK_MODE_FORECAST),
])
def test_risk_routes_mark_imerg_rainfall_as_real(route, mode, monkeypatch):
    record = _rainfall_record(
        risk_inputs.STATUS_REAL, 12.5, _REAL_DETAILS, "IMERG_Early",
    )
    _patch_resolution(monkeypatch, _resolution(record, mode))
    body = route(*_OPTION_C_POINT)
    rainfall = body["rainfall"]
    assert rainfall["accumulation_mm"] == 12.5
    assert rainfall["window_hours"] == 24
    assert rainfall["status"] == risk_inputs.STATUS_REAL
    assert rainfall["data_quality_status"] == "REAL"
    assert rainfall["source_kind"] == "IMERG"
    assert rainfall["source"] == "IMERG_Early"
    assert rainfall["is_fallback"] is False
    assert rainfall["has_real_rainfall"] is True
    assert rainfall["units"] == "mm"
    assert rainfall["fetched_at_utc"] == "2025-09-19T06:00:00Z"
    assert rainfall["freshness"]["cache_hit"] is True
    assert rainfall["rainfall_observation_date"] == "2025-09-18"
    assert "fallback_warning" not in rainfall
    # Existing fields are untouched.
    assert body["location"] == list(_OPTION_C_POINT)
    assert "resolved_inputs" in body


@pytest.mark.parametrize("route,mode", [
    (routes.get_current_risk, risk_inputs.RISK_MODE_CURRENT),
    (routes.get_forecast_risk, risk_inputs.RISK_MODE_FORECAST),
])
def test_risk_routes_mark_open_meteo_rainfall_as_fallback(route, mode, monkeypatch):
    record = _rainfall_record(
        risk_inputs.STATUS_DERIVED_PROXY, 9.5, _FALLBACK_DETAILS,
        "Open-Meteo ERA5 archive (FALLBACK)",
        reasons=["FALLBACK SOURCE: NASA GPM IMERG was unavailable."],
    )
    _patch_resolution(monkeypatch, _resolution(record, mode))
    body = route(*_OPTION_C_POINT)
    rainfall = body["rainfall"]
    assert rainfall["accumulation_mm"] == 9.5
    assert rainfall["status"] == risk_inputs.STATUS_DERIVED_PROXY
    assert rainfall["data_quality_status"] == "FALLBACK"
    assert rainfall["is_fallback"] is True
    assert "Open-Meteo ERA5" in rainfall["source"]
    assert "FALLBACK" in rainfall["fallback_warning"]
    assert "Open-Meteo ERA5" in rainfall["fallback_warning"]
    # A fallback is usable, so the endpoint still answers ...
    risk = body.get("risk") or body["risk_forecast"]
    assert risk["final_risk_score"] is not None
    # ... but it must not be presented as a live observation, and it degrades
    # confidence rather than inflating it.
    assert rainfall["has_real_rainfall"] is False
    assert risk["confidence"] == "MEDIUM"
    assert risk["inputs_are_real"]["rainfall"] is False
    assert rainfall["reasons"], "a fallback must explain itself"


@pytest.mark.parametrize("route,mode", [
    (routes.get_current_risk, risk_inputs.RISK_MODE_CURRENT),
    (routes.get_forecast_risk, risk_inputs.RISK_MODE_FORECAST),
])
def test_risk_routes_still_refuse_when_rainfall_is_unavailable(route, mode, monkeypatch):
    record = _rainfall_record(
        risk_inputs.STATUS_UNAVAILABLE, None, {"window_hours": 24}, None,
        reasons=["both NASA IMERG and the Open-Meteo fallback failed"],
    )
    _patch_resolution(monkeypatch, _resolution(record, mode))
    with pytest.raises(HTTPException) as excinfo:
        route(*_OPTION_C_POINT)
    assert excinfo.value.status_code == DATA_UNAVAILABLE_STATUS_CODE
    detail = excinfo.value.detail
    assert detail["status"] == risk_inputs.DATA_UNAVAILABLE
    assert risk_inputs.INPUT_CURRENT_RAINFALL in detail["blocking_inputs"]
    assert detail["blocking_reasons"]
    # A refusal still carries no risk numbers and no rainfall value.
    assert "final_risk_score" not in str(detail)
    assert "accumulation_mm" not in str(detail)


def test_risk_routes_hold_no_rainfall_constant_of_their_own():
    """
    The rainfall the route reports is whatever risk_inputs resolved -- there is no
    serving-side constant left. Two different resolved values must produce two
    different reported accumulations, and an absent value must stay absent.
    """
    for resolved in (0.0, 3.25, 118.75):
        record = _rainfall_record(
            risk_inputs.STATUS_REAL, resolved, _REAL_DETAILS, "IMERG_Early",
        )
        block = routes._current_rainfall_block(
            _resolution(record, risk_inputs.RISK_MODE_CURRENT)
        )
        assert block["accumulation_mm"] == resolved

    missing = _rainfall_record(
        risk_inputs.STATUS_UNAVAILABLE, None, {"window_hours": 24}, None,
        reasons=["IMERG and the Open-Meteo fallback both failed"],
    )
    block = routes._current_rainfall_block(
        _resolution(missing, risk_inputs.RISK_MODE_CURRENT)
    )
    assert block["accumulation_mm"] is None
    assert block["status"] == risk_inputs.STATUS_UNAVAILABLE
    assert block["has_real_rainfall"] is False


def test_rainfall_provenance_helper_never_invents_a_block():
    assert routes._rainfall_provenance(None) is None
    assert routes._rainfall_provenance("not-a-dict") is None
    assert routes._rainfall_provenance({}) is None
    assert routes._rainfall_provenance({"run_type": "Early"}) is None
    block = routes._rainfall_provenance({"units": "mm"})
    assert block["units"] == "mm"
    assert block["source"] is None, "a missing field is null, never guessed"
    assert block["is_fallback"] is False


# ---------------------------------------------------------------------------
# /risk/current PILOT POINT path
#
# Inside a canonical pilot AOI the endpoint answers from
# app.services.pilot_point_prediction instead of Option-C fusion. These tests
# drive the route FUNCTION directly with a canned point payload (the real service
# is unit-tested offline in test_pilot_point_prediction.py), so they need no DEM,
# model, credentials or network. What is pinned here is the ROUTING and the error
# mapping -- not the prediction.
# ---------------------------------------------------------------------------
from app.services import pilot_point_prediction as ppp  # noqa: E402

PILOT_POINTS = {
    "Sikkim": (27.33, 88.62),
    "Assam": (26.14, 91.77),
    "Arunachal Pradesh": (27.10, 93.60),
    "Meghalaya": (25.57, 91.88),
}
# Inside BOTH the Assam and Meghalaya canonical AOIs.
OVERLAP_POINT = (25.80, 92.00)


def _point_payload(state, rainfall):
    """A minimal stand-in for a pilot_point_prediction response."""
    return {
        "state": state,
        "method": ppp.METHOD,
        "point": {"latitude": PILOT_POINTS[state][0],
                  "longitude": PILOT_POINTS[state][1]},
        "target_date": "2025-09-19",
        "model": {"artifact_status": "VALID"},
        "rainfall": dict(rainfall),
        "hazard": {
            "status": "OK",
            "rainfall_conditioned_probability": 0.42,
            "risk_class": "MODERATE",
            "is_option_c_fused_risk": False,
            "is_rainfall_independent_susceptibility": False,
        },
        "option_c_fusion": {
            "available": False, "applied": False,
            "reason": ppp.OPTION_C_UNAVAILABLE_REASON,
            "susceptibility_score": None, "trigger_multiplier": None,
            "final_risk_score": None,
        },
        "disclosures": ["fixture"],
    }


def _canned_point(monkeypatch, payload, calls=None):
    def _fake(lat, lon, target_date, state=None, **kwargs):
        if calls is not None:
            calls.append({"lat": lat, "lon": lon, "state": state})
        return payload
    monkeypatch.setattr(routes.pilot_point_prediction, "predict_pilot_point", _fake)


def _no_fusion(monkeypatch):
    """Option-C fusion must not run on the pilot path; make it fail loudly if it does."""
    def _boom(*a, **k):
        raise AssertionError("dynamic_risk_module must NOT be called for a pilot point")
    monkeypatch.setattr(routes, "dynamic_risk_module", _boom)


@pytest.mark.parametrize("state", sorted(PILOT_POINTS))
def test_risk_current_uses_the_pilot_point_path_inside_each_pilot_aoi(state, monkeypatch):
    calls = []
    _canned_point(monkeypatch, _point_payload(state, _REAL_RAINFALL_REPORT), calls)
    _no_fusion(monkeypatch)
    lat, lon = PILOT_POINTS[state]
    body = routes.get_current_risk(lat, lon)
    assert calls == [{"lat": lat, "lon": lon, "state": state}], (
        "the state must be resolved explicitly and passed through"
    )
    assert body["state"] == state
    assert body["method"] == ppp.METHOD
    assert body["hazard"]["rainfall_conditioned_probability"] == 0.42
    # The coupled probability is never dressed up as fused risk or susceptibility.
    assert body["option_c_fusion"]["applied"] is False
    assert body["option_c_fusion"]["susceptibility_score"] is None
    assert body["option_c_fusion"]["final_risk_score"] is None
    assert "risk" not in body, "the pilot path must not emit an Option-C risk block"
    # Additive provenance, same shape as the four /predict/<state>/grid endpoints.
    provenance = body["rainfall_provenance"]
    assert provenance["data_quality_status"] == "REAL"
    assert provenance["is_fallback"] is False


@pytest.mark.parametrize("state", sorted(PILOT_POINTS))
def test_risk_current_pilot_path_labels_a_fallback_unmistakably(state, monkeypatch):
    _canned_point(monkeypatch, _point_payload(state, _FALLBACK_RAINFALL_REPORT))
    _no_fusion(monkeypatch)
    body = routes.get_current_risk(*PILOT_POINTS[state])
    provenance = body["rainfall_provenance"]
    assert provenance["is_fallback"] is True
    assert provenance["data_quality_status"] == "FALLBACK"
    assert "FALLBACK" in provenance["fallback_warning"]
    assert "Open-Meteo ERA5" in provenance["fallback_warning"]


def test_risk_current_requires_state_where_two_pilot_aois_overlap(monkeypatch):
    _no_fusion(monkeypatch)
    with pytest.raises(HTTPException) as excinfo:
        routes.get_current_risk(*OVERLAP_POINT)
    assert excinfo.value.status_code == 400
    detail = excinfo.value.detail
    assert detail["status"] == "AMBIGUOUS_PILOT_STATE"
    assert detail["details"]["pilot_states_containing_point"] == ["Assam", "Meghalaya"]
    # An ambiguity is refused, never resolved to an assumed state or a number.
    assert "final_risk_score" not in str(detail)
    assert "rainfall_conditioned_probability" not in str(detail)


@pytest.mark.parametrize("requested,expected", [
    ("assam", "Assam"),
    ("Meghalaya", "Meghalaya"),
])
def test_risk_current_resolves_the_overlap_with_an_explicit_state(requested, expected,
                                                                 monkeypatch):
    calls = []
    _canned_point(monkeypatch, _point_payload(expected, _REAL_RAINFALL_REPORT), calls)
    _no_fusion(monkeypatch)
    body = routes.get_current_risk(OVERLAP_POINT[0], OVERLAP_POINT[1], state=requested)
    assert calls[0]["state"] == expected
    assert body["state"] == expected


@pytest.mark.parametrize("bad_state", ["Nagaland", "Bihar", "not-a-state"])
def test_risk_current_rejects_a_non_pilot_state(bad_state, monkeypatch):
    _no_fusion(monkeypatch)
    with pytest.raises(HTTPException) as excinfo:
        routes.get_current_risk(PILOT_POINTS["Sikkim"][0], PILOT_POINTS["Sikkim"][1],
                                state=bad_state)
    assert excinfo.value.status_code == 400
    assert excinfo.value.detail["status"] == "INVALID_PILOT_STATE"


def test_risk_current_rejects_a_state_whose_aoi_excludes_the_point(monkeypatch):
    _no_fusion(monkeypatch)
    with pytest.raises(HTTPException) as excinfo:
        routes.get_current_risk(PILOT_POINTS["Sikkim"][0], PILOT_POINTS["Sikkim"][1],
                                state="assam")
    assert excinfo.value.status_code == 400
    detail = excinfo.value.detail
    assert detail["status"] == "INVALID_PILOT_STATE"
    assert detail["details"]["requested_state"] == "Assam"


def test_risk_current_maps_a_pilot_refusal_to_503_without_numbers(monkeypatch):
    def _refuse(lat, lon, target_date, state=None, **kwargs):
        raise ppp.PredictionUnavailable(
            "Real terrain is unavailable at this point; no placeholder was substituted.",
            details={"problems": ["nodata"]},
        )
    monkeypatch.setattr(routes.pilot_point_prediction, "predict_pilot_point", _refuse)
    _no_fusion(monkeypatch)
    with pytest.raises(HTTPException) as excinfo:
        routes.get_current_risk(*PILOT_POINTS["Meghalaya"])
    assert excinfo.value.status_code == DATA_UNAVAILABLE_STATUS_CODE
    detail = excinfo.value.detail
    assert detail["status"] == risk_inputs.DATA_UNAVAILABLE
    assert detail["details"]["problems"] == ["nodata"]
    assert "rainfall_conditioned_probability" not in str(detail)
    assert "final_risk_score" not in str(detail)


def test_risk_current_outside_every_pilot_aoi_still_takes_the_option_c_path(monkeypatch):
    """The pre-existing behaviour for non-pilot points is unchanged."""
    def _must_not_run(*a, **k):
        raise AssertionError("the pilot path must not run outside a pilot AOI")
    monkeypatch.setattr(routes.pilot_point_prediction, "predict_pilot_point",
                        _must_not_run)
    record = _rainfall_record(
        risk_inputs.STATUS_REAL, 12.5, _REAL_DETAILS, "IMERG_Early",
    )
    _patch_resolution(monkeypatch, _resolution(record, risk_inputs.RISK_MODE_CURRENT))
    body = routes.get_current_risk(*_OPTION_C_POINT)
    assert "risk" in body and "hazard" not in body
    assert body["risk"]["final_risk_score"] is not None
    assert body["location"] == list(_OPTION_C_POINT)


def test_risk_current_invalid_coordinates_are_rejected_before_state_resolution(monkeypatch):
    def _must_not_run(*a, **k):
        raise AssertionError("coordinates must be validated first")
    monkeypatch.setattr(routes.pilot_point_prediction, "resolve_pilot_state",
                        _must_not_run)
    with pytest.raises(HTTPException) as excinfo:
        routes.get_current_risk(10.0, 88.61)
    assert excinfo.value.status_code == 400


def test_risk_forecast_is_unchanged_and_never_consults_the_pilot_path(monkeypatch):
    def _must_not_run(*a, **k):
        raise AssertionError("/risk/forecast must not use the pilot point path")
    monkeypatch.setattr(routes.pilot_point_prediction, "predict_pilot_point",
                        _must_not_run)
    monkeypatch.setattr(routes.pilot_point_prediction, "resolve_pilot_state",
                        _must_not_run)
    record = _rainfall_record(
        risk_inputs.STATUS_REAL, 12.5, _REAL_DETAILS, "IMERG_Early",
    )
    _patch_resolution(monkeypatch, _resolution(record, risk_inputs.RISK_MODE_FORECAST))
    # A point INSIDE a pilot AOI, to prove the forecast route ignores the split.
    body = routes.get_forecast_risk(*PILOT_POINTS["Sikkim"])
    assert "risk_forecast" in body
    assert body["risk_forecast"]["final_risk_score"] is not None
    assert "hazard" not in body


def test_get_current_risk_accepts_an_optional_state_parameter():
    import inspect

    signature = inspect.signature(routes.get_current_risk)
    assert list(signature.parameters) == ["lat", "lon", "state"]
    assert signature.parameters["state"].default is None
    # /risk/forecast keeps its original signature.
    assert list(inspect.signature(routes.get_forecast_risk).parameters) == ["lat", "lon"]

