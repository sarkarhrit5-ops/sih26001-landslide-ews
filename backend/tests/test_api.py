import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

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
    response = client.get("/api/v1/risk/current?lat=27.33&lon=88.61")
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

