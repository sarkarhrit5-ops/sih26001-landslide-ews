from fastapi import APIRouter
from app.models.ml_pipeline import dynamic_risk_module, explain_risk
from app.services.exposure import mock_get_osm_assets
from app.services.weather_ingestion import fetch_open_meteo_forecast

router = APIRouter()

@router.get("/risk/current")
def get_current_risk(lat: float, lon: float):
    # Mock susceptibility and current rain
    base_susceptibility = 0.65
    current_rain = 55.0
    slope = 35.0
    
    risk = dynamic_risk_module(base_susceptibility, current_rain, 0.0, slope)
    return {"location": [lat, lon], "risk": risk}

@router.get("/risk/forecast")
def get_forecast_risk(lat: float, lon: float):
    base_susceptibility = 0.65
    current_rain = 55.0
    slope = 35.0
    
    forecast_rain = fetch_open_meteo_forecast(lat, lon, 24)
    risk = dynamic_risk_module(base_susceptibility, current_rain, forecast_rain, slope)
    return {"location": [lat, lon], "risk_forecast": risk}

@router.get("/cell/{cell_id}/explain")
def explain_cell_risk(cell_id: str):
    # Mock explanation
    explanation = explain_risk(None, None)
    return {"cell_id": cell_id, "explanation": explanation}
    
@router.get("/exposure/alerts")
def get_exposure_alerts():
    assets = mock_get_osm_assets()
    # Convert geometry to WKT for JSON serialization
    assets["geometry"] = assets["geometry"].apply(lambda geom: geom.wkt)
    alert_list = assets.to_dict(orient="records")
    return {"exposed_assets": alert_list}
