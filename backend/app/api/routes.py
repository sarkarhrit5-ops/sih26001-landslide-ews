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

`/risk/current` additionally has a PILOT POINT path (app.services.
pilot_point_prediction): inside one of the four canonical pilot AOIs it reports the
persisted pilot model's RAINFALL-CONDITIONED hazard probability, explicitly labelled,
with Option-C fusion reported as NOT applied and the gate's reason -- because those
models are rainfall-coupled and feeding their output into dynamic_risk_module as
susceptibility_score would double-count rainfall. `/risk/forecast` is unchanged.
"""

from datetime import datetime
from typing import Optional

from fastapi import APIRouter, HTTPException
from app.models.ml_pipeline import dynamic_risk_module, explain_risk
from app.services import (
    arunachal_prediction,
    assam_prediction,
    live_rainfall,
    meghalaya_prediction,
    pilot_events,
    pilot_map_view,
    pilot_point_prediction,
    risk_inputs,
    sikkim_prediction,
)
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


# ---------------------------------------------------------------------------
# Rainfall provenance (additive)
#
# app.services.rainfall_service already stamps every rainfall read with its
# source, quality status and freshness; risk_inputs carries that through in the
# input record's `details`, and the four pilot prediction services carry it in
# their `rainfall` block. This layer only RESHAPES what is already there into one
# consistently-named block so a client does not have to know which producer it is
# talking to. It computes nothing, defaults nothing, and never invents a value:
# a field the producer did not supply comes back as None.
# ---------------------------------------------------------------------------
RAINFALL_PROVENANCE_FIELDS = (
    "source",
    "source_kind",
    "is_fallback",
    "data_quality_status",
    "requested_date",
    "rainfall_observation_date",
    "fetched_at_utc",
    "freshness",
    "units",
)

# Emitted verbatim whenever is_fallback is true, so a fallback is unmistakable in
# the response body itself and not only inferable from a status enum.
FALLBACK_WARNING = (
    "FALLBACK RAINFALL: NASA GPM IMERG was unavailable, so this rainfall comes "
    "from the Open-Meteo ERA5 archive (reanalysis, not a live satellite "
    "observation). It is labelled data_quality_status=FALLBACK and must not be "
    "presented as official live rainfall."
)


def _rainfall_provenance(rainfall):
    """
    Normalise a producer's rainfall metadata into RAINFALL_PROVENANCE_FIELDS.

    `rainfall` may be the prediction services' `rainfall` report or a
    risk_inputs input record's `details`; the only shape difference is that the
    latter names the observation day `target_date`. Returns None when the producer
    supplied no rainfall metadata at all, rather than an all-None block that could
    be mistaken for a real read.
    """
    if not isinstance(rainfall, dict):
        return None
    block = {}
    for field in RAINFALL_PROVENANCE_FIELDS:
        block[field] = rainfall.get(field)
    if block["rainfall_observation_date"] is None:
        block["rainfall_observation_date"] = rainfall.get("target_date")
    if not any(v is not None for v in block.values()):
        return None
    block["is_fallback"] = bool(block["is_fallback"])
    if block["is_fallback"]:
        block["fallback_warning"] = FALLBACK_WARNING
    caveats = rainfall.get("caveats")
    if caveats:
        block["caveats"] = list(caveats)
    return block


def _current_rainfall_block(resolution):
    """
    Additive `rainfall` block for the risk endpoints: the observed accumulation
    that WAS used, with its status and provenance. The value is whatever
    risk_inputs resolved -- this layer holds no rainfall constant of its own.
    """
    record = resolution["inputs"].get(risk_inputs.INPUT_CURRENT_RAINFALL) or {}
    block = {
        "accumulation_mm": record.get("value"),
        "window_hours": (record.get("details") or {}).get("window_hours"),
        "status": record.get("status"),
        "has_real_rainfall": resolution["has_real_rainfall"],
    }
    provenance = _rainfall_provenance(record.get("details"))
    if provenance is not None:
        block.update(provenance)
    # The producer keeps the human-readable source label on the record itself, not
    # in `details`, so it is filled in from there rather than left null.
    if block.get("source") is None:
        block["source"] = record.get("source")
    if record.get("reasons"):
        block["reasons"] = list(record["reasons"])
    return block


def _with_rainfall_provenance(payload):
    """
    Attach the normalised block to a pilot prediction response under one
    consistent key, so all four /predict/<state>/grid endpoints expose rainfall
    provenance identically. Every pre-existing field, including the producer's own
    richer `rainfall` report, is left exactly as the service returned it.
    """
    if not isinstance(payload, dict):
        return payload
    provenance = _rainfall_provenance(payload.get("rainfall"))
    if provenance is not None:
        payload["rainfall_provenance"] = provenance
    return payload


@router.get("/risk/current")
def get_current_risk(lat: float, lon: float, state: Optional[str] = None):
    """
    Current risk at a point.

    TWO PATHS, chosen by whether the point is inside a canonical pilot AOI, and
    never silently interchanged:

      * INSIDE a pilot AOI -> app.services.pilot_point_prediction runs that state's
        persisted 11-feature model at the point with the live rainfall_service
        series, and the response reports the model's RAINFALL-CONDITIONED hazard
        probability under `hazard`, with `option_c_fusion.applied = False` and the
        gate's reason. The Option-C trigger multiplier is NOT applied to it and it
        is NEVER presented as susceptibility_score or final_risk_score, because the
        persisted pilot models are rainfall-coupled and doing so would double-count
        rainfall.
      * OUTSIDE every pilot AOI -> the pre-existing Option-C path via
        app.services.risk_inputs, unchanged, which refuses with HTTP 503
        DATA_UNAVAILABLE when a required real input is missing.

    Query params:
      * state -- optional pilot state ('Sikkim', 'Assam', 'Arunachal Pradesh',
        'Meghalaya'). Required where two pilot AOIs overlap (the Assam/Meghalaya and
        Assam/Arunachal bands): those requests are refused with HTTP 400 and the
        candidate list rather than resolved to an assumed state.

    HTTP 400 for an unknown/contradictory `state` or an ambiguous point; HTTP 503
    DATA_UNAVAILABLE when the real inputs cannot be obtained.
    """
    validate_coordinates(lat, lon)

    # 1. Establish the pilot state EXPLICITLY. There is no Sikkim default: a point
    #    in no pilot AOI falls through to the pre-existing Option-C path below.
    pilot_state = None
    try:
        pilot_state = pilot_point_prediction.resolve_pilot_state(lat, lon, state=state)
    except pilot_point_prediction.PointOutsidePilotAoi:
        # No pilot raster set or model covers this point; fall through to the
        # pre-existing Option-C path (whose behaviour is unchanged).
        pilot_state = None
    except pilot_point_prediction.PilotStateAmbiguous as exc:
        raise HTTPException(
            status_code=400,
            detail={
                "status": "AMBIGUOUS_PILOT_STATE",
                "reason": exc.reason,
                "details": exc.details,
            },
        )
    except pilot_point_prediction.PilotStateInvalid as exc:
        raise HTTPException(
            status_code=400,
            detail={
                "status": "INVALID_PILOT_STATE",
                "reason": exc.reason,
                "details": exc.details,
            },
        )

    # 2. Pilot point path: the real per-state model + live rainfall, no fusion.
    if pilot_state is not None:
        try:
            return _with_rainfall_provenance(
                pilot_point_prediction.predict_pilot_point(
                    lat, lon, datetime.utcnow(), state=pilot_state,
                )
            )
        except pilot_point_prediction.PredictionUnavailable as exc:
            raise HTTPException(
                status_code=DATA_UNAVAILABLE_STATUS_CODE,
                detail={
                    "status": "DATA_UNAVAILABLE",
                    "reason": exc.reason,
                    "details": exc.details,
                },
            )

    # 3. Pre-existing Option-C path, byte-for-byte unchanged.
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
        # Additive: which rainfall product answered, how fresh it is, and whether it
        # was the labelled Open-Meteo ERA5 fallback rather than live IMERG.
        "rainfall": _current_rainfall_block(resolution),
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
        # Additive, and about the OBSERVED trigger only: the forecast accumulation
        # keeps its own existing top-level field above.
        "rainfall": _current_rainfall_block(resolution),
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
    # NOTE: mock_get_osm_assets() returns a small hand-written fixture, NOT a real
    # OSM/Overpass query. The response is explicitly marked as such so a consumer
    # cannot mistake these placeholder assets for measured exposure data. The
    # provenance fields are additive and do not change the `exposed_assets` shape.
    assets = mock_get_osm_assets()
    assets["geometry"] = assets["geometry"].apply(lambda geom: geom.wkt)
    alert_list = assets.to_dict(orient="records")
    return {
        "exposed_assets": alert_list,
        "data_source": "MOCK_FIXTURE",
        "is_mock": True,
        "provenance": (
            "SYNTHETIC PLACEHOLDER exposure fixture (app.services.exposure."
            "mock_get_osm_assets); not a real OSM/Overpass query. Do not treat "
            "as measured exposure."
        ),
    }


@router.get("/validation/status")
def get_validation_status():
    import json
    import os
    from app.services.state_validation import (
        reconcile_validation_report,
        refresh_assam_data_status,
        refresh_arunachal_data_status,
        refresh_meghalaya_data_status,
    )
    file_path = os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed", "state_validation.json")
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            records = json.load(f)
        # Do not blindly trust stale validation claims persisted in the JSON: a
        # record may say VALIDATED_PILOT even though the required persisted
        # model/metrics evidence is absent. Reconcile against on-disk evidence so
        # the API cannot present an unbacked claim as current truth. The file
        # itself is left unchanged (historical evidence is preserved).
        #
        # Then refresh the ASSAM record UP against the real Assam artifacts that
        # now exist on disk (the assam_pilot_* terrain rasters, assam_osm.geojson,
        # and the persisted assam_model.pkl + metrics + schema evidence). This
        # file was written by an early NER sweep before any of those existed, and
        # neither reconcile_validation_report (downgrade-only) nor
        # determine_overall_status (model-evidence gate is pilot-only; Assam is
        # is_pilot=False) will lift the stale "Missing" / "Not Trained" values.
        # Assam-scoped and read-only: every other record is returned unchanged and
        # the on-disk file is never rewritten. Fabricates nothing.
        #
        # The Arunachal Pradesh record has the identical stale-JSON problem and its
        # real artifacts (arunachal_pilot_* rasters, arunachal_pradesh_osm.geojson,
        # and the persisted arunachal_pradesh_model.pkl + metrics + schema evidence)
        # now exist too, so refresh it the same way. The Meghalaya record (the 4th
        # pilot) is refreshed identically against its own real meghalaya_pilot_*
        # rasters, meghalaya_osm.geojson and persisted meghalaya_model.pkl + metrics +
        # schema. Each refresh touches ONLY its own state's record; composing them
        # leaves every other record unchanged.
        return refresh_meghalaya_data_status(
            refresh_arunachal_data_status(
                refresh_assam_data_status(reconcile_validation_report(records))
            )
        )
    else:
        return []


# ---------------------------------------------------------------------------
# Read-only Sikkim pilot evidence endpoints (ADDITIVE).
#
# These serve ONLY real, already-persisted artifacts and the real AOI-filtered
# landslide inventory. They do not compute, retrain, download, or fabricate
# anything. They expose what the reproduction run already wrote to disk:
#   * the persisted model-evidence bundle under backend/data/models/
#     (sikkim_metrics.json / sikkim_feature_schema.json / sikkim_provenance.json),
#     read through app.services.model_artifacts.verify_artifact_set(), which
#     returns an honest MISSING / INVALID / VALID verdict and never substitutes
#     hardcoded numbers; and
#   * the NASA Global Landslide Catalog positives inside the canonical pilot AOI,
#     resolved by app.services.pilot_events (snapshot-first, raw-CSV fallback)
#     using the EXACT filter used by scripts/train_real_models.py (bbox from
#     config_states.get_pilot_aoi_bounds -> drop rows with no event_date ->
#     de-duplicate on (latitude, longitude, event_date)). This yields the same
#     82 events recorded in sikkim_provenance.json (glc_event_count = 82).
# ---------------------------------------------------------------------------

# Fields copied verbatim from the persisted evidence bundle. Absolute filesystem
# paths recorded inside the artifacts (dataset_artifact /
# dataset_provenance_reference) are deliberately NOT exposed by the API.
_EVIDENCE_METRICS_FIELDS = (
    "validation_metrics", "metrics_source", "status", "primary_model",
    "primary_evaluation", "feature_set", "holdout_details", "sample_counts",
    "model_comparison", "model_decision", "generated_at",
)
_EVIDENCE_SCHEMA_FIELDS = (
    "feature_set_name", "feature_names", "feature_order", "n_features",
    "dtype", "meaning", "target_column",
)
_EVIDENCE_PROVENANCE_FIELDS = (
    "aoi", "glc_source", "glc_event_count", "sample_counts", "rainfall_source",
    "dem_source", "terrain_derivative_method", "exposure_source", "model_type",
    "model_hyperparameters", "model_serialization", "feature_list",
    "spatial_split", "temporal_split", "negative_sampling", "leakage_controls",
    "random_seed", "code_version", "software_versions", "input_status",
    "generation_timestamp", "additional_context",
)


def _pick(doc, fields):
    """Whitelisted, JSON-safe projection of a persisted artifact document."""
    if not isinstance(doc, dict):
        return None
    return {key: doc[key] for key in fields if key in doc}


@router.get("/validation/sikkim/evidence")
def get_sikkim_evidence():
    """
    Serve the persisted Sikkim model-evidence bundle, read-only.

    Mirrors the honesty of /cell/{id}/explain: returns HTTP 200 with an explicit
    `status` ("VALID" / "MISSING" / "INVALID") rather than an error status, so the
    response shape stays valid and the caller is told exactly what is on disk. All
    numbers are read from the artifacts written by the reproduction run; this
    endpoint never computes, defaults, or back-fills a value.
    """
    from app.services import model_artifacts
    verdict = model_artifacts.verify_artifact_set(
        state_name="Sikkim", require_full_metrics=False
    )
    return {
        "state": "Sikkim",
        "pilot_area": "East Sikkim",
        "status": verdict["status"],
        "gate_compatible": verdict["gate_compatible"],
        "problems": verdict["problems"],
        "metrics": _pick(verdict.get("metrics"), _EVIDENCE_METRICS_FIELDS),
        "feature_schema": _pick(verdict.get("feature_schema"), _EVIDENCE_SCHEMA_FIELDS),
        "provenance": _pick(verdict.get("provenance"), _EVIDENCE_PROVENANCE_FIELDS),
    }


@router.get("/validation/assam/evidence")
def get_assam_evidence():
    """
    Serve the persisted Assam model-evidence bundle, read-only.

    Identical contract to /validation/sikkim/evidence: returns HTTP 200 with an
    explicit `status` ("VALID" / "MISSING" / "INVALID") and reads every number from
    the artifacts written by the Assam training run (assam_metrics.json /
    assam_feature_schema.json / assam_provenance.json, resolved by
    model_artifacts.verify_artifact_set(state_name="Assam")). It never computes,
    defaults, or back-fills a value. The ONLY Assam-specific facts here are the
    state label and pilot area; the field projections are shared with Sikkim.

    The Assam pilot's single methodological difference from Sikkim -- land_cover_class
    is REAL ESA WorldCover treated as a categorical feature, not an elevation proxy
    -- is carried through faithfully in the returned feature_schema/provenance.
    """
    from app.services import model_artifacts
    verdict = model_artifacts.verify_artifact_set(
        state_name="Assam", require_full_metrics=False
    )
    return {
        "state": "Assam",
        "pilot_area": "Guwahati-Kamrup + western Karbi Anglong",
        "status": verdict["status"],
        "gate_compatible": verdict["gate_compatible"],
        "problems": verdict["problems"],
        "metrics": _pick(verdict.get("metrics"), _EVIDENCE_METRICS_FIELDS),
        "feature_schema": _pick(verdict.get("feature_schema"), _EVIDENCE_SCHEMA_FIELDS),
        "provenance": _pick(verdict.get("provenance"), _EVIDENCE_PROVENANCE_FIELDS),
    }


@router.get("/validation/arunachal/evidence")
def get_arunachal_evidence():
    """
    Serve the persisted Arunachal Pradesh model-evidence bundle, read-only.

    Identical contract to /validation/sikkim/evidence and /validation/assam/evidence:
    returns HTTP 200 with an explicit `status` ("VALID" / "MISSING" / "INVALID") and
    reads every number from the artifacts written by the Arunachal training run
    (arunachal_pradesh_metrics.json / arunachal_pradesh_feature_schema.json /
    arunachal_pradesh_provenance.json, resolved by
    model_artifacts.verify_artifact_set(state_name="Arunachal Pradesh")). It never
    computes, defaults, or back-fills a value. The ONLY Arunachal-specific facts here
    are the state label and pilot area; the field projections are shared with Sikkim.

    Like the Assam pilot, Arunachal's single methodological difference from Sikkim --
    land_cover_class is REAL ESA WorldCover treated as a categorical feature, not an
    elevation proxy -- is carried through faithfully in the returned
    feature_schema/provenance.
    """
    from app.services import model_artifacts
    verdict = model_artifacts.verify_artifact_set(
        state_name="Arunachal Pradesh", require_full_metrics=False
    )
    return {
        "state": "Arunachal Pradesh",
        "pilot_area": "central Subansiri-Siang belt",
        "status": verdict["status"],
        "gate_compatible": verdict["gate_compatible"],
        "problems": verdict["problems"],
        "metrics": _pick(verdict.get("metrics"), _EVIDENCE_METRICS_FIELDS),
        "feature_schema": _pick(verdict.get("feature_schema"), _EVIDENCE_SCHEMA_FIELDS),
        "provenance": _pick(verdict.get("provenance"), _EVIDENCE_PROVENANCE_FIELDS),
    }


@router.get("/validation/meghalaya/evidence")
def get_meghalaya_evidence():
    """
    Serve the persisted Meghalaya model-evidence bundle, read-only.

    Identical contract to /validation/sikkim/evidence, /validation/assam/evidence and
    /validation/arunachal/evidence: returns HTTP 200 with an explicit `status`
    ("VALID" / "MISSING" / "INVALID") and reads every number from the artifacts written
    by the Meghalaya training run (meghalaya_metrics.json /
    meghalaya_feature_schema.json / meghalaya_provenance.json, resolved by
    model_artifacts.verify_artifact_set(state_name="Meghalaya")). It never computes,
    defaults, or back-fills a value. The ONLY Meghalaya-specific facts here are the
    state label and pilot area; the field projections are shared with Sikkim.

    Like the Assam and Arunachal pilots, Meghalaya's single methodological difference
    from Sikkim -- land_cover_class is REAL ESA WorldCover treated as a categorical
    feature, not an elevation proxy -- is carried through faithfully in the returned
    feature_schema/provenance.
    """
    from app.services import model_artifacts
    verdict = model_artifacts.verify_artifact_set(
        state_name="Meghalaya", require_full_metrics=False
    )
    return {
        "state": "Meghalaya",
        "pilot_area": "East Khasi + Jaintia Hills belt",
        "status": verdict["status"],
        "gate_compatible": verdict["gate_compatible"],
        "problems": verdict["problems"],
        "metrics": _pick(verdict.get("metrics"), _EVIDENCE_METRICS_FIELDS),
        "feature_schema": _pick(verdict.get("feature_schema"), _EVIDENCE_SCHEMA_FIELDS),
        "provenance": _pick(verdict.get("provenance"), _EVIDENCE_PROVENANCE_FIELDS),
    }


@router.get("/validation/sikkim/events")
def get_sikkim_events():
    """
    The real NASA GLC landslide positives inside the canonical East Sikkim pilot
    AOI -- the exact inventory the pilot model was trained on.

    Delegates to app.services.pilot_events, which serves the committed validated
    snapshot (data/models/sikkim_events.json) when present and otherwise filters
    the raw GLC catalog live. Refuses with HTTP 503 DATA_UNAVAILABLE only when
    BOTH real sources are absent, rather than returning an empty or synthesised
    list.
    """
    aoi, events, precise, source_artifact = pilot_events.resolve_pilot_events()
    if events is None:
        raise HTTPException(
            status_code=DATA_UNAVAILABLE_STATUS_CODE,
            detail={
                "status": "DATA_UNAVAILABLE",
                "reason": (
                    "No real Sikkim landslide inventory is available on the "
                    "server: neither the committed events snapshot "
                    "(data/models/sikkim_events.json) nor the raw NASA GLC "
                    "catalog (data/raw/glc_legacy.csv) is present."
                ),
                "aoi": aoi,
            },
        )
    if source_artifact == "validated_snapshot":
        served_from = (
            " Served from the committed validated snapshot "
            "(data/models/sikkim_events.json)."
        )
    else:
        served_from = " Served live from the raw catalog (data/raw/glc_legacy.csv)."
    return {
        "state": "Sikkim",
        "pilot_area": "East Sikkim",
        "aoi": aoi,
        "count": len(events),
        "source": (
            "NASA Global Landslide Catalog (glc_legacy.csv), AOI-filtered; "
            "de-duplicated on (latitude, longitude, event_date)." + served_from
        ),
        "source_artifact": source_artifact,
        "spatial_uncertainty_summary": pilot_events.spatial_uncertainty_summary(
            events, precise
        ),
        "events": events,
    }


@router.get("/validation/assam/events")
def get_assam_events():
    """
    The real NASA GLC landslide positives inside the canonical Assam pilot AOI
    (Guwahati-Kamrup + western Karbi Anglong) -- the exact inventory the Assam
    pilot model was trained on.

    Same delegation and refusal contract as /validation/sikkim/events, only the
    pilot state differs: app.services.pilot_events serves the committed validated
    snapshot (data/models/assam_events.json) when present and otherwise filters the
    raw GLC catalog live with the identical AOI rule. Refuses with HTTP 503
    DATA_UNAVAILABLE only when BOTH real sources are absent, rather than returning
    an empty or synthesised list.
    """
    aoi, events, precise, source_artifact = pilot_events.resolve_pilot_events(
        state_name="Assam"
    )
    if events is None:
        raise HTTPException(
            status_code=DATA_UNAVAILABLE_STATUS_CODE,
            detail={
                "status": "DATA_UNAVAILABLE",
                "reason": (
                    "No real Assam landslide inventory is available on the "
                    "server: neither the committed events snapshot "
                    "(data/models/assam_events.json) nor the raw NASA GLC "
                    "catalog (data/raw/glc_legacy.csv) is present."
                ),
                "aoi": aoi,
            },
        )
    if source_artifact == "validated_snapshot":
        served_from = (
            " Served from the committed validated snapshot "
            "(data/models/assam_events.json)."
        )
    else:
        served_from = " Served live from the raw catalog (data/raw/glc_legacy.csv)."
    return {
        "state": "Assam",
        "pilot_area": "Guwahati-Kamrup + western Karbi Anglong",
        "aoi": aoi,
        "count": len(events),
        "source": (
            "NASA Global Landslide Catalog (glc_legacy.csv), AOI-filtered; "
            "de-duplicated on (latitude, longitude, event_date)." + served_from
        ),
        "source_artifact": source_artifact,
        "spatial_uncertainty_summary": pilot_events.spatial_uncertainty_summary(
            events, precise
        ),
        "events": events,
    }


@router.get("/validation/arunachal/events")
def get_arunachal_events():
    """
    The real NASA GLC landslide positives inside the canonical Arunachal Pradesh
    pilot AOI (central Subansiri-Siang belt) -- the exact inventory the Arunachal
    pilot model was trained on.

    Same real-source-or-refuse contract as /validation/sikkim/events and
    /validation/assam/events: the committed validated snapshot
    (data/models/arunachal_pradesh_events.json) is served when present, otherwise the
    raw GLC catalog is filtered live with the identical AOI rule. Refuses with HTTP
    503 DATA_UNAVAILABLE only when BOTH real sources are absent, never returning an
    empty or synthesised list.

    NOTE: unlike the Sikkim/Assam routes, this composes the pilot_events loaders
    directly instead of calling resolve_pilot_events(). pilot_events derives the
    per-state snapshot filename by lower-casing the state name, which for the
    two-word "Arunachal Pradesh" yields the SPACE form "arunachal pradesh_events.json"
    -- but the committed artifact uses the canonical UNDERSCORE form
    "arunachal_pradesh_events.json". Passing json_path explicitly resolves the
    snapshot correctly WITHOUT modifying the shared pilot_events module, while
    preserving resolve_pilot_events()'s exact snapshot-first / raw-CSV-fallback order
    and semantics (the AOI is still the live canonical one for "Arunachal Pradesh").
    """
    import os
    snapshot_path = os.path.join(
        os.path.dirname(pilot_events.DEFAULT_SNAPSHOT_PATH),
        "arunachal_pradesh_events.json",
    )
    aoi, events, precise = pilot_events.load_events_from_snapshot(
        json_path=snapshot_path, state_name="Arunachal Pradesh"
    )
    if events is not None:
        source_artifact = "validated_snapshot"
    else:
        aoi, events, precise = pilot_events.load_events_from_csv(
            state_name="Arunachal Pradesh"
        )
        source_artifact = "raw_glc_catalog" if events is not None else None
    if events is None:
        raise HTTPException(
            status_code=DATA_UNAVAILABLE_STATUS_CODE,
            detail={
                "status": "DATA_UNAVAILABLE",
                "reason": (
                    "No real Arunachal Pradesh landslide inventory is available on "
                    "the server: neither the committed events snapshot "
                    "(data/models/arunachal_pradesh_events.json) nor the raw NASA GLC "
                    "catalog (data/raw/glc_legacy.csv) is present."
                ),
                "aoi": aoi,
            },
        )
    if source_artifact == "validated_snapshot":
        served_from = (
            " Served from the committed validated snapshot "
            "(data/models/arunachal_pradesh_events.json)."
        )
    else:
        served_from = " Served live from the raw catalog (data/raw/glc_legacy.csv)."
    return {
        "state": "Arunachal Pradesh",
        "pilot_area": "central Subansiri-Siang belt",
        "aoi": aoi,
        "count": len(events),
        "source": (
            "NASA Global Landslide Catalog (glc_legacy.csv), AOI-filtered; "
            "de-duplicated on (latitude, longitude, event_date)." + served_from
        ),
        "source_artifact": source_artifact,
        "spatial_uncertainty_summary": pilot_events.spatial_uncertainty_summary(
            events, precise
        ),
        "events": events,
    }


@router.get("/validation/meghalaya/events")
def get_meghalaya_events():
    """
    The real NASA GLC landslide positives inside the canonical Meghalaya pilot AOI
    (East Khasi + Jaintia Hills belt) -- the exact inventory the Meghalaya pilot
    model was trained on.

    Same delegation and refusal contract as /validation/sikkim/events and
    /validation/assam/events: app.services.pilot_events serves the committed
    validated snapshot (data/models/meghalaya_events.json) when present and otherwise
    filters the raw GLC catalog live with the identical AOI rule. Refuses with HTTP
    503 DATA_UNAVAILABLE only when BOTH real sources are absent, rather than returning
    an empty or synthesised list.

    NOTE: unlike the two-word "Arunachal Pradesh" route, "Meghalaya" is a single word,
    so pilot_events._snapshot_path_for_state lower-cases it to the canonical
    "meghalaya_events.json" that matches the committed artifact. The plain
    resolve_pilot_events() therefore finds the validated snapshot directly and needs
    NO explicit-json_path workaround -- exactly like Sikkim and Assam.
    """
    aoi, events, precise, source_artifact = pilot_events.resolve_pilot_events(
        state_name="Meghalaya"
    )
    if events is None:
        raise HTTPException(
            status_code=DATA_UNAVAILABLE_STATUS_CODE,
            detail={
                "status": "DATA_UNAVAILABLE",
                "reason": (
                    "No real Meghalaya landslide inventory is available on the "
                    "server: neither the committed events snapshot "
                    "(data/models/meghalaya_events.json) nor the raw NASA GLC "
                    "catalog (data/raw/glc_legacy.csv) is present."
                ),
                "aoi": aoi,
            },
        )
    if source_artifact == "validated_snapshot":
        served_from = (
            " Served from the committed validated snapshot "
            "(data/models/meghalaya_events.json)."
        )
    else:
        served_from = " Served live from the raw catalog (data/raw/glc_legacy.csv)."
    return {
        "state": "Meghalaya",
        "pilot_area": "East Khasi + Jaintia Hills belt",
        "aoi": aoi,
        "count": len(events),
        "source": (
            "NASA Global Landslide Catalog (glc_legacy.csv), AOI-filtered; "
            "de-duplicated on (latitude, longitude, event_date)." + served_from
        ),
        "source_artifact": source_artifact,
        "spatial_uncertainty_summary": pilot_events.spatial_uncertainty_summary(
            events, precise
        ),
        "events": events,
    }


@router.get("/predict/sikkim/grid")
def predict_sikkim_grid(date: Optional[str] = None,
                        step: float = sikkim_prediction.DEFAULT_STEP_DEG,
                        run_type: str = "Early"):
    """
    Real per-grid-cell landslide susceptibility for the East Sikkim pilot AOI.

    Runs the persisted 11-feature LightGBM (real terrain + elevation-proxy land
    cover + real IMERG antecedent rainfall) over a coarse grid tiling the AOI and
    returns, per cell, the model's RAW susceptibility probability and the system
    warning class. This is NOT the Option-C fused /risk/current score -- see the
    response `disclosures`. Read-only; it fabricates nothing and back-fills no
    cell: a cell with missing/nodata terrain comes back status UNAVAILABLE with no
    probability.

    Query params:
      * date     -- optional prediction date 'YYYY-MM-DD' (default: today, UTC).
                    Rainfall is the antecedent T-1..T-14 window, so `date` itself
                    need not be in the catalog, only its preceding 14 days.
      * step     -- grid cell size in degrees (default is a coarse grid; the range
                    and a max cell count are enforced by the service).
      * run_type -- IMERG run: Early (default), Late or Final.

    Refuses with HTTP 503 DATA_UNAVAILABLE when the model artifacts or the real
    rainfall cannot be obtained, and HTTP 400 for a bad `date` or `step`.
    """
    if date is None:
        target_date = datetime.utcnow()
    else:
        try:
            target_date = datetime.strptime(date, "%Y-%m-%d")
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail="Invalid 'date' %r; expected format YYYY-MM-DD." % date,
            )
    try:
        return _with_rainfall_provenance(sikkim_prediction.predict_sikkim_grid(
            target_date, step_deg=step, run_type=run_type,
        ))
    except sikkim_prediction.PredictionUnavailable as exc:
        raise HTTPException(
            status_code=DATA_UNAVAILABLE_STATUS_CODE,
            detail={
                "status": "DATA_UNAVAILABLE",
                "reason": exc.reason,
                "details": exc.details,
            },
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.get("/predict/assam/grid")
def predict_assam_grid(date: Optional[str] = None,
                       step: float = assam_prediction.DEFAULT_STEP_DEG,
                       run_type: str = "Early"):
    """
    Real per-grid-cell landslide susceptibility for the Assam pilot AOI
    (Guwahati-Kamrup + western Karbi Anglong).

    Identical contract to /predict/sikkim/grid -- runs the persisted 11-feature
    LightGBM over a coarse grid tiling the AOI and returns, per cell, the model's RAW
    susceptibility probability and the system warning class (NOT the Option-C fused
    /risk/current score; see the response `disclosures`). Read-only; it fabricates
    nothing and back-fills no cell.

    The two Assam-specific differences from the Sikkim endpoint are that land cover is
    REAL ESA WorldCover (not an elevation proxy) and is scored as a CATEGORICAL feature,
    exactly as the Assam model was trained. A cell whose terrain OR WorldCover land
    cover is missing/nodata/out-of-coverage comes back status UNAVAILABLE with no
    probability -- never a substituted class or value.

    Query params:
      * date     -- optional prediction date 'YYYY-MM-DD' (default: today, UTC).
                    Rainfall is the antecedent T-1..T-14 window, so `date` itself
                    need not be in the catalog, only its preceding 14 days.
      * step     -- grid cell size in degrees (default is a coarse grid; the range
                    and a max cell count are enforced by the service).
      * run_type -- IMERG run: Early (default), Late or Final.

    Refuses with HTTP 503 DATA_UNAVAILABLE when the model artifacts or the real
    rainfall cannot be obtained, and HTTP 400 for a bad `date` or `step`.
    """
    if date is None:
        target_date = datetime.utcnow()
    else:
        try:
            target_date = datetime.strptime(date, "%Y-%m-%d")
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail="Invalid 'date' %r; expected format YYYY-MM-DD." % date,
            )
    try:
        return _with_rainfall_provenance(assam_prediction.predict_assam_grid(
            target_date, step_deg=step, run_type=run_type,
        ))
    except assam_prediction.PredictionUnavailable as exc:
        raise HTTPException(
            status_code=DATA_UNAVAILABLE_STATUS_CODE,
            detail={
                "status": "DATA_UNAVAILABLE",
                "reason": exc.reason,
                "details": exc.details,
            },
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.get("/predict/arunachal/grid")
def predict_arunachal_grid(date: Optional[str] = None,
                           step: float = arunachal_prediction.DEFAULT_STEP_DEG,
                           run_type: str = "Early"):
    """
    Real per-grid-cell landslide susceptibility for the Arunachal Pradesh pilot AOI
    (central Subansiri-Siang belt).

    Identical contract to /predict/sikkim/grid and /predict/assam/grid -- runs the
    persisted 11-feature LightGBM over a coarse grid tiling the AOI and returns, per
    cell, the model's RAW susceptibility probability and the system warning class (NOT
    the Option-C fused /risk/current score; see the response `disclosures`). Read-only;
    it fabricates nothing and back-fills no cell.

    Exactly like the Assam endpoint, the two Arunachal-specific differences from the
    Sikkim endpoint are that land cover is REAL ESA WorldCover (not an elevation
    proxy) and is scored as a CATEGORICAL feature, precisely as the Arunachal model
    was trained. A cell whose terrain OR WorldCover land cover is missing / nodata /
    out-of-coverage comes back status UNAVAILABLE with no probability -- never a
    substituted class or value.

    Query params:
      * date     -- optional prediction date 'YYYY-MM-DD' (default: today, UTC).
                    Rainfall is the antecedent T-1..T-14 window, so `date` itself
                    need not be in the catalog, only its preceding 14 days.
      * step     -- grid cell size in degrees (default is a coarse grid; the range
                    and a max cell count are enforced by the service).
      * run_type -- IMERG run: Early (default), Late or Final.

    Refuses with HTTP 503 DATA_UNAVAILABLE when the model artifacts or the real
    rainfall cannot be obtained, and HTTP 400 for a bad `date` or `step`.
    """
    if date is None:
        target_date = datetime.utcnow()
    else:
        try:
            target_date = datetime.strptime(date, "%Y-%m-%d")
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail="Invalid 'date' %r; expected format YYYY-MM-DD." % date,
            )
    try:
        return _with_rainfall_provenance(arunachal_prediction.predict_arunachal_grid(
            target_date, step_deg=step, run_type=run_type,
        ))
    except arunachal_prediction.PredictionUnavailable as exc:
        raise HTTPException(
            status_code=DATA_UNAVAILABLE_STATUS_CODE,
            detail={
                "status": "DATA_UNAVAILABLE",
                "reason": exc.reason,
                "details": exc.details,
            },
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.get("/predict/meghalaya/grid")
def predict_meghalaya_grid(date: Optional[str] = None,
                           step: float = meghalaya_prediction.DEFAULT_STEP_DEG,
                           run_type: str = "Early"):
    """
    Real per-grid-cell landslide susceptibility for the Meghalaya pilot AOI
    (East Khasi + Jaintia Hills belt).

    Identical contract to /predict/sikkim/grid, /predict/assam/grid and
    /predict/arunachal/grid -- runs the persisted 11-feature LightGBM over a coarse
    grid tiling the AOI and returns, per cell, the model's RAW susceptibility
    probability and the system warning class (NOT the Option-C fused /risk/current
    score; see the response `disclosures`). Read-only; it fabricates nothing and
    back-fills no cell.

    Exactly like the Assam and Arunachal endpoints, the two Meghalaya-specific
    differences from the Sikkim endpoint are that land cover is REAL ESA WorldCover
    (not an elevation proxy) and is scored as a CATEGORICAL feature, precisely as the
    Meghalaya model was trained. A cell whose terrain OR WorldCover land cover is
    missing / nodata / out-of-coverage comes back status UNAVAILABLE with no
    probability -- never a substituted class or value.

    Query params:
      * date     -- optional prediction date 'YYYY-MM-DD' (default: today, UTC).
                    Rainfall is the antecedent T-1..T-14 window, so `date` itself
                    need not be in the catalog, only its preceding 14 days.
      * step     -- grid cell size in degrees (default is a coarse grid; the range
                    and a max cell count are enforced by the service).
      * run_type -- IMERG run: Early (default), Late or Final.

    Refuses with HTTP 503 DATA_UNAVAILABLE when the model artifacts or the real
    rainfall cannot be obtained, and HTTP 400 for a bad `date` or `step`.
    """
    if date is None:
        target_date = datetime.utcnow()
    else:
        try:
            target_date = datetime.strptime(date, "%Y-%m-%d")
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail="Invalid 'date' %r; expected format YYYY-MM-DD." % date,
            )
    try:
        return _with_rainfall_provenance(meghalaya_prediction.predict_meghalaya_grid(
            target_date, step_deg=step, run_type=run_type,
        ))
    except meghalaya_prediction.PredictionUnavailable as exc:
        raise HTTPException(
            status_code=DATA_UNAVAILABLE_STATUS_CODE,
            detail={
                "status": "DATA_UNAVAILABLE",
                "reason": exc.reason,
                "details": exc.details,
            },
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


# ---------------------------------------------------------------------------
# Lightweight map views of the four pilot grid predictions.
#
# WHY THESE EXIST
#     /predict/<state>/grid returns, per cell, the full 11-value feature vector,
#     the cell bbox and the per-cell reasons. That payload is the right one for
#     auditing a prediction and its contract is FROZEN -- nothing below changes
#     it. It is the wrong payload for drawing a map, where only a coordinate, a
#     probability and a class are needed.
#
# WHAT THEY COST
#     Exactly what the matching /grid endpoint costs. Each handler runs its
#     prediction service ONCE and hands that one result to
#     pilot_map_view.to_map_geojson, which is a pure transform: it runs no model,
#     fetches no rainfall and opens no raster, so it cannot change a probability
#     because it only ever copies one.
#
# HONESTY
#     Cells are copied, never filtered: a cell the backend could not score stays
#     in the FeatureCollection with probability=None and status="UNAVAILABLE",
#     because a dropped cell would read as "safe". The rainfall provenance block
#     is copied verbatim, so a FALLBACK series can never be presented here as a
#     live IMERG observation.
# ---------------------------------------------------------------------------

def _parse_map_target_date(date):
    """The same 'YYYY-MM-DD' contract (and the same HTTP 400) as /grid."""
    if date is None:
        return datetime.utcnow()
    try:
        return datetime.strptime(date, "%Y-%m-%d")
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail="Invalid 'date' %r; expected format YYYY-MM-DD." % date,
        )


def _pilot_map_response(service, predict, date, step, run_type):
    """
    Run one pilot grid prediction ONCE and project it into the map view.

    `predict` is the service's own predict_<state>_grid callable; the error
    mapping is identical to the /grid handlers (503 DATA_UNAVAILABLE for a
    missing real input, 400 for a bad `date` or `step`).
    """
    target_date = _parse_map_target_date(date)
    try:
        prediction = predict(target_date, step_deg=step, run_type=run_type)
    except service.PredictionUnavailable as exc:
        raise HTTPException(
            status_code=DATA_UNAVAILABLE_STATUS_CODE,
            detail={
                "status": "DATA_UNAVAILABLE",
                "reason": exc.reason,
                "details": exc.details,
            },
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return pilot_map_view.to_map_geojson(_with_rainfall_provenance(prediction))


@router.get("/predict/sikkim/map")
def predict_sikkim_map(date: Optional[str] = None,
                       step: float = sikkim_prediction.DEFAULT_STEP_DEG,
                       run_type: str = "Early"):
    """
    Map-sized projection of /predict/sikkim/grid for the SAME request.

    A GeoJSON FeatureCollection with one Point feature per grid cell at the cell
    CENTER -- the point the model was actually sampled at -- carrying only
    `cell_id`, `status`, `probability`, `risk_class` and
    `exceeds_decision_threshold`. Per-cell feature vectors, cell bboxes and
    per-cell reasons are omitted here and remain unchanged on
    /predict/sikkim/grid.

    Top level also carries `state`, `pilot_area`, `target_date`,
    `generated_from`, `decision_threshold`, `aoi`, `grid`, the rainfall
    provenance subset and `summary` -- all O(1) in the cell count. `aoi` and
    `grid` are present because without per-cell bboxes a client still needs the
    extent and the cell size to draw the cells.

    Query params and error behaviour are identical to /predict/sikkim/grid; the
    prediction runs exactly once.
    """
    return _pilot_map_response(
        sikkim_prediction, sikkim_prediction.predict_sikkim_grid,
        date, step, run_type,
    )


@router.get("/predict/assam/map")
def predict_assam_map(date: Optional[str] = None,
                      step: float = assam_prediction.DEFAULT_STEP_DEG,
                      run_type: str = "Early"):
    """
    Map-sized projection of /predict/assam/grid for the SAME request.

    Identical contract to /predict/sikkim/map: a GeoJSON FeatureCollection of
    cell-center Points whose properties are limited to `cell_id`, `status`,
    `probability`, `risk_class` and `exceeds_decision_threshold`. The full
    per-cell feature vectors (including the REAL categorical WorldCover land
    cover), bboxes and reasons stay on /predict/assam/grid.

    Query params and error behaviour are identical to /predict/assam/grid; the
    prediction runs exactly once.
    """
    return _pilot_map_response(
        assam_prediction, assam_prediction.predict_assam_grid,
        date, step, run_type,
    )


@router.get("/predict/arunachal/map")
def predict_arunachal_map(date: Optional[str] = None,
                          step: float = arunachal_prediction.DEFAULT_STEP_DEG,
                          run_type: str = "Early"):
    """
    Map-sized projection of /predict/arunachal/grid for the SAME request.

    Identical contract to /predict/sikkim/map; the full per-cell feature vectors,
    bboxes and reasons stay on /predict/arunachal/grid.

    Query params and error behaviour are identical to /predict/arunachal/grid;
    the prediction runs exactly once.
    """
    return _pilot_map_response(
        arunachal_prediction, arunachal_prediction.predict_arunachal_grid,
        date, step, run_type,
    )


@router.get("/predict/meghalaya/map")
def predict_meghalaya_map(date: Optional[str] = None,
                          step: float = meghalaya_prediction.DEFAULT_STEP_DEG,
                          run_type: str = "Early"):
    """
    Map-sized projection of /predict/meghalaya/grid for the SAME request.

    Identical contract to /predict/sikkim/map; the full per-cell feature vectors,
    bboxes and reasons stay on /predict/meghalaya/grid.

    Query params and error behaviour are identical to /predict/meghalaya/grid;
    the prediction runs exactly once.
    """
    return _pilot_map_response(
        meghalaya_prediction, meghalaya_prediction.predict_meghalaya_grid,
        date, step, run_type,
    )


@router.get("/rainfall/latest")
def rainfall_latest(state: str):
    """
    Latest AVAILABLE rainfall observation for a pilot AOI (monitoring read).

    This is deliberately NOT called "current rainfall": the returned record
    exposes `observed_at_utc`, `fetched_at_utc`, `age_minutes`,
    `freshness_label` and `is_stale` so a stale observation cannot be mistaken
    for a now-value.

    Source preference is IMERG Early HHR, then IMERG Late HHR, then a clearly
    labelled Open-Meteo FALLBACK; if all fail the record's
    `data_quality_status` is UNAVAILABLE with every numeric field null.

    This endpoint is entirely separate from the antecedent model rainfall
    features (T-1..T-14). It never feeds derive_rainfall_features(), it uses
    its own cache and its own SIH_LIVE_RAINFALL_* environment surface, and it
    does not change any prediction output.

    Because this is a monitoring read rather than a data dependency, an
    UNAVAILABLE record is returned with HTTP 200 (the honest answer is the
    payload, not an error). Only an unknown state is a 400.
    """
    try:
        return live_rainfall.get_latest_rainfall(state)
    except KeyError:
        raise HTTPException(
            status_code=400,
            detail=(
                "Unknown pilot state: %r. Supported: %s"
                % (state, ", ".join(live_rainfall.supported_states()))
            ),
        )
