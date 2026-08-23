import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}

def test_current_risk_valid():
    response = client.get("/api/v1/risk/current?lat=27.33&lon=88.61")
    assert response.status_code == 200
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

def test_current_risk_invalid_coordinates():
    # Test latitude out of bounds (e.g. lat=10.0 outside 20-35 range)
    response = client.get("/api/v1/risk/current?lat=10.0&lon=88.61")
    assert response.status_code == 400
    assert "Invalid coordinates" in response.json()["detail"]

def test_forecast_risk_valid():
    response = client.get("/api/v1/risk/forecast?lat=27.33&lon=88.61")
    assert response.status_code == 200
    data = response.json()
    assert "location" in data
    assert "forecast_accumulation_mm" in data
    assert "risk_forecast" in data
    risk = data["risk_forecast"]
    assert "final_risk_score" in risk
    assert "warning_level" in risk

def test_cell_explain():
    response = client.get("/api/v1/cell/cell_123/explain")
    assert response.status_code == 200
    data = response.json()
    assert data["cell_id"] == "cell_123"
    assert "explanation" in data

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

