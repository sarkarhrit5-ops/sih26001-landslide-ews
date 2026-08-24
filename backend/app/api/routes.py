"""
API ROUTES

DATA-INTEGRITY CONTRACT FOR THIS LAYER: an endpoint either answers from real,
resolved inputs, or it refuses and says which input was missing and why. It never
fills a gap with a default, a plausible constant or a zero.

Concretely, `/risk/current` and `/risk/forecast` used to build their answer from
susceptibility 0.65, current rainfall 55.0 mm, slope 35.0 deg and exposure 0.5 --
none measured -- while asserting `has_real_dem=True, has_real_rainfall=True`, so
every response claimed HIGH confidence. They now resolve each input through
app.services.risk_inputs and return HTTP 503 with a structured
DATA_UNAVAILABLE body when any required measurement is absent.
"""

from typing import Optional

from fastapi import APIRouter, HTTPException
from app.models.ml_pipeline import dynamic_risk_module, explain_risk
from app.services import risk_inputs
from app.services.exposure import mock_get_osm_assets

router = APIRouter()

# HTTP 503: the request is well-formed and the endpoint exists, but the upstream
# real-data inputs it depends on are not available right now.
DATA_UNAVAILABLE_STATUS_CODE = 503


def validate_coordinates(lat: float, lon: float):
    if not (20.0 <= lat <= 35.0 and 80.0 <= lon <= 95.0):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid coordinates: lat={lat}, lon={lon}. Must be within region (lat: 20-35, lon: 80-95)."
        )


def _resolved_input_report(resolution):
    """JSON-safe echo of what each input resolved to, for successful responses."""
    report = {}
    for key, record in resolution["inputs"].items():
        entry = {"status": record["status"], "source": record["source"]}
        if record["reasons"]:
            entry["reasons"] = record["reasons"]
        report[key] = entry
    return report


def _refuse(resolution, message):
    """Raises the structured 503. Carries no risk numbers -- there are none."""
    raise HTTPException(
        status_code=DATA_UNAVAILABLE_STATUS_CODE,
        detail=risk_inputs.build_unavailable_detail(resolution, message=message),
    )


@router.get("/risk/current")
def get_current_risk(lat: float, lon: float):
    validate_coordinates(lat, lon)
    resolution = risk_inputs.resolve_risk_inputs(
        lat, lon, mode=risk_inputs.RISK_MODE_CURRENT
    )
    if not resolution["usable"]:
        _refuse(
            resolution,
            "Current risk cannot be computed for this location: one or more "
            "required real inputs are unavailable. No substituted, default or "
            "placeholder values were used.",
        )

    inputs = resolution["inputs"]
    risk = dynamic_risk_module(
        susceptibility_score=inputs[risk_inputs.INPUT_SUSCEPTIBILITY]["value"],
        current_rainfall_mm=inputs[risk_inputs.INPUT_CURRENT_RAINFALL]["value"],
        # None (not 0.0): this endpoint evaluates the observed trigger only, so no
        # forecast is asserted either way.
        forecast_rainfall_mm=None,
        slope_deg=inputs[risk_inputs.INPUT_SLOPE]["value"],
        exposure_score=inputs[risk_inputs.INPUT_EXPOSURE]["value"],
        has_real_dem=resolution["has_real_dem"],
        has_real_rainfall=resolution["has_real_rainfall"],
    )
    return {
        "location": [lat, lon],
        "risk": risk,
        "resolved_inputs": _resolved_input_report(resolution),
    }


@router.get("/risk/forecast")
def get_forecast_risk(lat: float, lon: float):
    validate_coordinates(lat, lon)
    resolution = risk_inputs.resolve_risk_inputs(
        lat, lon, mode=risk_inputs.RISK_MODE_FORECAST
    )
    if not resolution["usable"]:
        # The old implementation swallowed a forecast failure into
        # `forecast_rain = 0.0`, reporting "no rain expected" when the forecast
        # service was simply unreachable. That is now a refusal.
        _refuse(
            resolution,
            "Forecast risk cannot be computed for this location: one or more "
            "required real inputs are unavailable. In particular, an unreachable "
            "forecast service is no longer reported as 0 mm of rainfall.",
        )

    inputs = resolution["inputs"]
    forecast_rain = inputs[risk_inputs.INPUT_FORECAST_RAINFALL]["value"]
    risk = dynamic_risk_module(
        susceptibility_score=inputs[risk_inputs.INPUT_SUSCEPTIBILITY]["value"],
        current_rainfall_mm=inputs[risk_inputs.INPUT_CURRENT_RAINFALL]["value"],
        forecast_rainfall_mm=forecast_rain,
        slope_deg=inputs[risk_inputs.INPUT_SLOPE]["value"],
        exposure_score=inputs[risk_inputs.INPUT_EXPOSURE]["value"],
        has_real_dem=resolution["has_real_dem"],
        has_real_rainfall=resolution["has_real_rainfall"],
    )
    return {
        "location": [lat, lon],
        "forecast_accumulation_mm": forecast_rain,
        "risk_forecast": risk,
        "resolved_inputs": _resolved_input_report(resolution),
    }


@router.get("/cell/{cell_id}/explain")
def explain_cell_risk(cell_id: str, lat: Optional[float] = None, lon: Optional[float] = None):
    """
    Real SHAP attribution for a cell, or an explicit UNAVAILABLE explanation.

    This used to call `explain_risk(None, None)`, which took the hardcoded-
    importances branch on every single request: the "explanation" the UI showed
    (slope 0.42 / rain_3d 0.28 / roughness 0.18) was invented. That branch no
    longer exists.

    `lat`/`lon` are optional and additive: the backend holds no cell registry, so
    a bare `cell_id` cannot be resolved to coordinates and therefore cannot be
    explained. Supplying coordinates lets the endpoint build the real feature
    vector and run SHAP over the persisted model.

    Returns HTTP 200 with `explanation.status == "UNAVAILABLE"` rather than an
    error status, because the response shape stays valid and the caller is told
    precisely what is missing.
    """
    if lat is None or lon is None:
        return {
            "cell_id": cell_id,
            "explanation": explain_risk(None, None),
            "unresolved_cell": (
                "No cell registry exists in the backend, so cell_id '%s' cannot be "
                "mapped to coordinates. Pass ?lat=&lon= to request a real "
                "explanation for a point." % cell_id
            ),
        }

    validate_coordinates(lat, lon)
    prepared = risk_inputs.resolve_model_input(lat, lon)
    if prepared["status"] not in risk_inputs.USABLE_STATUSES:
        from app.models.ml_pipeline import explanation_unavailable
        return {
            "cell_id": cell_id,
            "location": [lat, lon],
            "explanation": explanation_unavailable(prepared["reasons"]),
        }

    bundle = prepared["value"]
    return {
        "cell_id": cell_id,
        "location": [lat, lon],
        "explanation": explain_risk(
            bundle["model"], bundle["frame"], feature_names=bundle["feature_names"]
        ),
    }


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
    from app.services.state_validation import reconcile_validation_report
    file_path = os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed", "state_validation.json")
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            records = json.load(f)
        # Do not blindly trust stale validation claims persisted in the JSON: a
        # record may say VALIDATED_PILOT even though the required persisted
        # model/metrics evidence is absent. Reconcile against on-disk evidence so
        # the API cannot present an unbacked claim as current truth. The file
        # itself is left unchanged (historical evidence is preserved).
        return reconcile_validation_report(records)
    else:
        return []
