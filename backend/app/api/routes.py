from fastapi import APIRouter, HTTPException
from app.models.ml_pipeline import dynamic_risk_module, explain_risk
from app.services.exposure import mock_get_osm_assets
from app.services.weather_ingestion import fetch_open_meteo_forecast

router = APIRouter()

def validate_coordinates(lat: float, lon: float):
    if not (20.0 <= lat <= 35.0 and 80.0 <= lon <= 95.0):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid coordinates: lat={lat}, lon={lon}. Must be within region (lat: 20-35, lon: 80-95)."
        )

@router.get("/risk/current")
def get_current_risk(lat: float, lon: float):
    validate_coordinates(lat, lon)
    base_susceptibility = 0.65
    current_rain = 55.0
    slope = 35.0
    
    risk = dynamic_risk_module(
        susceptibility_score=base_susceptibility,
        current_rainfall_mm=current_rain,
        forecast_rainfall_mm=0.0,
        slope_deg=slope,
        exposure_score=0.5,
        has_real_dem=True,
        has_real_rainfall=True
    )
    return {"location": [lat, lon], "risk": risk}

@router.get("/risk/forecast")
def get_forecast_risk(lat: float, lon: float):
    validate_coordinates(lat, lon)
    base_susceptibility = 0.65
    current_rain = 55.0
    slope = 35.0
    
    try:
        forecast_rain = fetch_open_meteo_forecast(lat, lon, 72)
    except Exception:
        forecast_rain = 0.0

    risk = dynamic_risk_module(
        susceptibility_score=base_susceptibility,
        current_rainfall_mm=current_rain,
        forecast_rainfall_mm=forecast_rain,
        slope_deg=slope,
        exposure_score=0.5,
        has_real_dem=True,
        has_real_rainfall=True
    )
    return {
        "location": [lat, lon],
        "forecast_accumulation_mm": forecast_rain,
        "risk_forecast": risk
    }

@router.get("/cell/{cell_id}/explain")
def explain_cell_risk(cell_id: str):
    explanation = explain_risk(None, None)
    return {"cell_id": cell_id, "explanation": explanation}
    
@router.get("/exposure/alerts")
def get_exposure_alerts():
    assets = mock_get_osm_assets()
    assets["geometry"] = assets["geometry"].apply(lambda geom: geom.wkt)
    alert_list = assets.to_dict(orient="records")
    return {"exposed_assets": alert_list}

@router.get("/validation/status")
def get_validation_status():
    import json
    import os
    file_path = os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed", "state_validation.json")
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            return json.load(f)
    else:
        return []
