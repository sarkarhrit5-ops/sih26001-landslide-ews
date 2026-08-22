from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}

def test_current_risk():
    response = client.get("/api/v1/risk/current?lat=27.33&lon=88.61")
    assert response.status_code == 200
    data = response.json()
    assert "location" in data
    assert "risk" in data
    assert data["risk"]["current_hazard"] > 0

def test_exposure_alerts():
    response = client.get("/api/v1/exposure/alerts")
    assert response.status_code == 200
    data = response.json()
    assert "exposed_assets" in data
    assert len(data["exposed_assets"]) > 0
