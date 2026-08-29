import os
import math
import pandas as pd
from typing import Dict, Any
from app.core.config_states import NER_STATES_CONFIG
from app.services.weather_ingestion import get_earthdata_session

def evaluate_landslide_inventory(state_config: Dict[str, Any], glc_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Evaluates the usable historical landslide events for a given state bounding box.
    """
    mask = (
        (glc_df['latitude'] >= state_config['min_lat']) &
        (glc_df['latitude'] <= state_config['max_lat']) &
        (glc_df['longitude'] >= state_config['min_lon']) &
        (glc_df['longitude'] <= state_config['max_lon'])
    )
    state_events = glc_df[mask].copy()
    
    total_events = len(state_events)
    if total_events == 0:
        return {
            "inventory_events": 0,
            "usable_events": 0,
            "spatial_quality": "Poor",
            "temporal_quality": "Poor"
        }
    
    # Assess exact dates (temporal precision)
    if 'event_date' in state_events.columns:
        state_events['parsed_date'] = pd.to_datetime(state_events['event_date'], errors='coerce', format='mixed')
        exact_dates = state_events.dropna(subset=['parsed_date'])
    else:
        exact_dates = pd.DataFrame()
        
    # Remove duplicates (same location and date)
    if not exact_dates.empty and 'latitude' in exact_dates.columns and 'longitude' in exact_dates.columns:
        usable_events_df = exact_dates.drop_duplicates(subset=['latitude', 'longitude', 'parsed_date'])
        usable_count = len(usable_events_df)
    else:
        usable_count = 0
        
    # Assess spatial uncertainty
    high_accuracy_count = 0
    if 'location_accuracy' in state_events.columns:
        accuracy_counts = state_events['location_accuracy'].value_counts().to_dict()
        high_accuracy_count = sum(count for acc, count in accuracy_counts.items() if acc in ['1km', 'exact', '100m'])
        
    spatial_quality = "Good" if high_accuracy_count / max(1, total_events) > 0.5 else "Moderate/Poor"
    temporal_quality = "Good" if usable_count / max(1, total_events) > 0.8 else "Moderate/Poor"
    
    return {
        "inventory_events": total_events,
        "usable_events": usable_count,
        "spatial_quality": spatial_quality,
        "temporal_quality": temporal_quality
    }

def get_dem_tiles_for_bbox(bbox: Dict[str, float]) -> list:
    """
    Calculates the required integer-degree Copernicus tiles from the state's bounding box.
    """
    import math
    min_lat = int(math.floor(bbox["min_lat"]))
    max_lat = int(math.floor(bbox["max_lat"]))
    min_lon = int(math.floor(bbox["min_lon"]))
    max_lon = int(math.floor(bbox["max_lon"]))
    
    tiles = []
    for lat in range(min_lat, max_lat + 1):
        for lon in range(min_lon, max_lon + 1):
            tiles.append((lat, lon))
    return tiles

# Pilot states persist their DEM under a collision-free "<state>_pilot_dem.tif"
# name (see prepare_<state>_terrain.py), NOT the generic "<clean_state>_dem.tif"
# that non-pilot states use. Without this map, the DEM lookups below would probe
# e.g. "arunachal_pradesh_dem.tif" / "assam_dem.tif" / "meghalaya_dem.tif" —
# files that never exist — and always report DEM Missing for those pilots.
# Sikkim is intentionally omitted: it keeps the generic "sikkim_dem.tif" name,
# which acquire_state_dem populates by copying "east_sikkim_dem.tif".
PILOT_DEM_FILENAMES = {
    "arunachal pradesh": "arunachal_pilot_dem.tif",
    "assam": "assam_pilot_dem.tif",
    "meghalaya": "meghalaya_pilot_dem.tif",
}


def resolve_state_dem_filename(state_name: str) -> str:
    """
    Single source of truth for the on-disk DEM basename of a state.

    Arunachal Pradesh / Assam / Meghalaya use their pilot filenames; every other
    state (including Sikkim) keeps the generic "<clean_state>_dem.tif".
    """
    clean_state_name = state_name.lower().replace(' ', '_')
    return PILOT_DEM_FILENAMES.get(state_name.strip().lower(), f"{clean_state_name}_dem.tif")


def acquire_state_dem(state_name: str, state_config: Dict[str, Any]) -> str:
    """
    State-aware DEM downloader, loader, cache manager, mosaic, and cropper.
    Returns path of the compiled state DEM file.

    The target path uses resolve_state_dem_filename, so the three pilot states
    reuse their already-prepared "<state>_pilot_dem.tif" instead of downloading
    a redundant copy. If that file is absent (e.g. a fresh production checkout),
    the Copernicus tile acquisition path below runs unchanged and writes the
    mosaic to that same pilot filename. Availability is never assumed.
    """
    import math
    import urllib.request
    # NOTE: rasterio is imported lazily further down, immediately before the
    # mosaic step. The reuse-existing-DEM and tile-download paths need no raster
    # library, so deferring the import keeps them usable (and unit-testable) in
    # environments where rasterio is not installed. Behaviour is unchanged: the
    # mosaic step still requires rasterio and will raise if it is missing.

    raw_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw"))
    dem_cache_dir = os.path.join(raw_dir, "dem")
    os.makedirs(dem_cache_dir, exist_ok=True)

    state_dem_path = os.path.join(raw_dir, resolve_state_dem_filename(state_name))
    
    # If already compiled and valid, reuse it!
    if os.path.exists(state_dem_path) and os.path.getsize(state_dem_path) > 1000:
        return state_dem_path
        
    # Sikkim pilot already exists locally
    if state_config.get("is_pilot"):
        pilot_path = os.path.join(raw_dir, "east_sikkim_dem.tif")
        if os.path.exists(pilot_path):
            import shutil
            shutil.copy2(pilot_path, state_dem_path)
            return state_dem_path
            
    # Solve state bounding box tiles
    tiles = get_dem_tiles_for_bbox(state_config)
    
    # Limit maximum downloads to a 2x2 grid around the center of the bounding box
    # to avoid huge downloads of 28-40 tiles that exceed timeout/disk limits.
    lat_center = (state_config["min_lat"] + state_config["max_lat"]) / 2.0
    lon_center = (state_config["min_lon"] + state_config["max_lon"]) / 2.0
    c_lat = int(math.floor(lat_center))
    c_lon = int(math.floor(lon_center))
    
    target_tiles = [
        (c_lat, c_lon),
        (c_lat + 1, c_lon),
        (c_lat, c_lon + 1),
        (c_lat + 1, c_lon + 1),
    ]
    
    subset_tiles = [t for t in tiles if t in target_tiles]
    if not subset_tiles:
        subset_tiles = [(c_lat, c_lon)]
        
    downloaded_files = []
    
    for lat, lon in subset_tiles:
        tile_name = f"Copernicus_DSM_COG_10_N{lat:02d}_00_E{lon:03d}_00_DEM"
        tile_file = f"{tile_name}.tif"
        tile_url = f"https://copernicus-dem-30m.s3.amazonaws.com/{tile_name}/{tile_file}"
        dest_file = os.path.join(dem_cache_dir, tile_file)
        
        if os.path.exists(dest_file) and os.path.getsize(dest_file) > 1000:
            downloaded_files.append(dest_file)
            continue
            
        try:
            print(f"[DEM] Downloading tile: {tile_file}...")
            urllib.request.urlretrieve(tile_url, dest_file)
            downloaded_files.append(dest_file)
        except Exception as e:
            print(f"[DEM] Failed to download {tile_file}: {e}")
            
    if not downloaded_files:
        raise RuntimeError("No DEM tiles could be downloaded or resolved for this state.")
        
    # Merge and clip using rasterio
    import rasterio
    from rasterio.merge import merge

    src_files = [rasterio.open(fp) for fp in downloaded_files]
    try:
        min_lon = max(state_config["min_lon"], min(src.bounds.left for src in src_files))
        max_lon = min(state_config["max_lon"], max(src.bounds.right for src in src_files))
        min_lat = max(state_config["min_lat"], min(src.bounds.bottom for src in src_files))
        max_lat = min(state_config["max_lat"], max(src.bounds.top for src in src_files))
        
        mosaic, out_trans = merge(src_files, bounds=(min_lon, min_lat, max_lon, max_lat))
        
        out_meta = src_files[0].meta.copy()
        out_meta.update({
            'driver': 'GTiff',
            'height': mosaic.shape[1],
            'width': mosaic.shape[2],
            'transform': out_trans,
            'crs': src_files[0].crs,
            'dtype': 'float32'
        })
        
        with rasterio.open(state_dem_path, 'w', **out_meta) as dst:
            dst.write(mosaic[0].astype('float32'), 1)
            
    finally:
        for src in src_files:
            src.close()
            
    return state_dem_path

# Pilot states persist their DEM under a collision-free "<state>_pilot_dem.tif"
# name (see PILOT_DEM_FILENAMES / resolve_state_dem_filename above).
def evaluate_terrain_data(state_name: str, state_config: Dict[str, Any]) -> str:
    """
    Checks if raw DEM dataset exists for the state locally.

    For the Arunachal Pradesh / Assam / Meghalaya pilots the on-disk DEM is
    named "<state>_pilot_dem.tif"; all other states fall back to the generic
    "<clean_state>_dem.tif". The size guard (>1000 bytes) is preserved so a
    truncated/placeholder file is never reported as Available.
    """
    dem_path = os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw",
                            resolve_state_dem_filename(state_name))

    if os.path.exists(dem_path) and os.path.getsize(dem_path) > 1000:
        return "Available"
    return "Missing (Requires Download)"

def evaluate_rainfall_status() -> str:
    """
    Checks Earthdata authentication status.

    The wording deliberately says "Unavailable", not "Fallback Active": when
    Earthdata authentication fails, NO substitute satellite rainfall is produced.
    fetch_imerg_precipitation raises, and (since the antecedent-rainfall hardening)
    fetch_historical_rainfall_series raises rather than zero-filling. Calling this
    state a "fallback" implied a working replacement that does not exist.
    """
    try:
        get_earthdata_session()
        return "Authenticated (Satellite IMERG)"
    except PermissionError:
        return "Unavailable (NASA Earthdata auth missing)"
    except Exception:
        return "Unavailable (NASA Earthdata connection error)"

def evaluate_state_rainfall(state_name: str, state_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Handles rainfall authentication for a specific state.

    The failure branch used to report source "Open-Meteo / Fallback Synthetic",
    which was wrong twice over: nothing synthetic is generated anywhere in the
    rainfall path, and Open-Meteo (used by the training pipeline for ERA5
    antecedent series) is a real observational source, not a synthetic one. The
    honest statement is that satellite IMERG is UNAVAILABLE for this state, with
    the reason attached.
    """
    try:
        get_earthdata_session()
        return {
            "source": "NASA IMERG (Satellite)",
            "status": "Authenticated (Satellite IMERG)",
            "is_fallback": False,
            "imerg_available": True,
            "unavailable_reason": None
        }
    except Exception as exc:
        reason = "%s: %s" % (type(exc).__name__, exc)
        print(f"Satellite IMERG rainfall UNAVAILABLE for {state_name} ({reason}); "
              f"no synthetic substitute is generated.")
        return {
            "source": "UNAVAILABLE (NASA IMERG not authenticated; no synthetic substitute)",
            "status": "Unavailable (NASA Earthdata authentication failed)",
            "is_fallback": False,
            "imerg_available": False,
            "unavailable_reason": reason
        }

def acquire_state_osm(state_name: str, state_config: Dict[str, Any]) -> str:
    """
    Acquires OSM exposure data using get_osm_assets, ensuring it is cached to GeoJSON.
    """
    from app.services.exposure import get_osm_assets
    raw_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw"))
    osm_cache_dir = os.path.join(raw_dir, "osm")
    
    clean_state_name = state_name.lower().replace(' ', '_')
    state_osm_path = os.path.join(raw_dir, f"{clean_state_name}_osm.geojson")
    
    if os.path.exists(state_osm_path) and os.path.getsize(state_osm_path) > 100:
        return state_osm_path
        
    cache_path = os.path.join(osm_cache_dir, f"{clean_state_name}_osm.geojson")
    if os.path.exists(cache_path) and os.path.getsize(cache_path) > 100:
        import shutil
        shutil.copy2(cache_path, state_osm_path)
        return state_osm_path
        
    # Query online
    try:
        get_osm_assets(state_name, state_config)
        if os.path.exists(cache_path):
            import shutil
            shutil.copy2(cache_path, state_osm_path)
            return state_osm_path
    except Exception as e:
        print(f"[OSM] Failed to download real OSM for {state_name}: {e}")
        
    return state_osm_path

def evaluate_exposure_data(state_name: str, state_config: Dict[str, Any]) -> str:
    """
    Checks if OSM exposure dataset exists for the state locally.
    """
    clean_state_name = state_name.lower().replace(' ', '_')
    osm_path = os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw", f"{clean_state_name}_osm.geojson")
    
    if os.path.exists(osm_path) and os.path.getsize(osm_path) > 100:
        return "Available"
    return "Missing (Requires Download)"

# ---------------------------------------------------------------------------
# Persisted validation evidence gate
# ---------------------------------------------------------------------------
# A state (including the East Sikkim pilot) may only be reported as
# VALIDATED_PILOT / "Trained & Validated" when REAL, reproducible validation
# evidence has been persisted to disk by an actual training/validation run.
# The minimum evidence contract is three non-empty files in data/models/:
#   * <state>_model.pkl            -- the trained model artifact
#   * <state>_metrics.json         -- metrics produced by a real validation run
#   * <state>_feature_schema.json  -- feature schema / provenance
# Nothing in this repository fabricates these files; until a genuine run writes
# them, the gate returns "incomplete" and the caller must fall back to an honest
# VALIDATION_REQUIRED status. Metrics are ONLY ever read from the persisted
# metrics.json -- they are never hardcoded.
REQUIRED_METRIC_KEYS = ("PR-AUC", "ROC-AUC")


def _is_finite_metric(value) -> bool:
    """
    True only for a real, finite numeric metric value. Rejects bools, None,
    strings and NaN/inf so that a metrics.json which is structurally present but
    degenerate (e.g. {"PR-AUC": null, "ROC-AUC": "n/a"}) cannot be accepted as
    validation evidence. Mirrors model_artifacts._is_real_number without importing
    the scientific stack.
    """
    if isinstance(value, bool) or value is None:
        return False
    if not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _evidence_paths(state_name: str, base_dir: str = None) -> Dict[str, str]:
    clean_state_name = state_name.lower().replace(' ', '_')
    if base_dir is None:
        base_dir = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..", "data", "models")
        )
    return {
        "model": os.path.join(base_dir, f"{clean_state_name}_model.pkl"),
        "metrics": os.path.join(base_dir, f"{clean_state_name}_metrics.json"),
        "schema": os.path.join(base_dir, f"{clean_state_name}_feature_schema.json"),
    }

def load_validation_evidence(state_name: str, base_dir: str = None) -> Dict[str, Any]:
    """
    Loads persisted, reproducible validation evidence for a state, or returns an
    explicit 'incomplete' result. This is the ONLY thing that may justify a
    VALIDATED_PILOT claim. It NEVER fabricates metrics: the numbers can only come
    from a real metrics.json written by an actual validation run.
    """
    import json
    paths = _evidence_paths(state_name, base_dir)
    missing = [
        name for name, p in paths.items()
        if not (os.path.exists(p) and os.path.getsize(p) > 0)
    ]
    if missing:
        return {"complete": False, "missing": missing, "metrics": {},
                "risk_result": None, "paths": paths}

    try:
        with open(paths["metrics"], "r", encoding="utf-8") as fh:
            doc = json.load(fh)
    except Exception as e:
        return {"complete": False, "missing": ["metrics (unreadable)"],
                "metrics": {}, "risk_result": None, "paths": paths, "error": str(e)}

    metrics = doc.get("validation_metrics", doc) if isinstance(doc, dict) else None
    if not isinstance(metrics, dict) or any(k not in metrics for k in REQUIRED_METRIC_KEYS):
        # The metrics file exists but is not a structurally valid validation
        # result. We refuse to treat it as evidence rather than inventing values.
        return {"complete": False, "missing": ["metrics (invalid schema)"],
                "metrics": {}, "risk_result": None, "paths": paths}
    if not all(_is_finite_metric(metrics[k]) for k in REQUIRED_METRIC_KEYS):
        # The required keys are present but at least one value is null, non-numeric
        # or non-finite. Accepting that would surface a degenerate or tampered
        # metrics file as a real VALIDATED_PILOT result, so we refuse it here just
        # as the writer gate (model_artifacts._is_real_number) refuses to persist
        # such values.
        return {"complete": False,
                "missing": ["metrics (non-numeric or non-finite values)"],
                "metrics": {}, "risk_result": None, "paths": paths}

    risk_result = doc.get("risk_result") if isinstance(doc, dict) else None
    return {"complete": True, "missing": [], "metrics": metrics,
            "risk_result": risk_result, "paths": paths}

def determine_overall_status(
    state_name: str, 
    inventory: Dict[str, Any], 
    terrain: str, 
    rainfall: str, 
    exposure: str, 
    is_pilot: bool,
    evidence_dir: str = None
) -> Dict[str, Any]:
    """
    Determines overall state validation status and lists blocking reasons.
    """
    blockers = []
    
    if terrain != "Available" and not is_pilot:
        blockers.append("Missing DEM Data")
    if exposure != "Available" and not is_pilot:
        blockers.append("Missing OSM Exposure Data")
    # Recognise the live "Unavailable (...)" rainfall vocabulary emitted by
    # evaluate_rainfall_status / evaluate_state_rainfall, not only the older
    # "Unauthenticated" / "Missing ..." tokens. Without this, a non-pilot state
    # whose satellite IMERG is genuinely unavailable would silently omit the
    # rainfall blocker from its reasons list.
    rainfall_unavailable = (
        rainfall == "Unauthenticated"
        or rainfall.startswith("Missing")
        or rainfall.startswith("Unavailable")
        or "Missing Earthdata" in rainfall
    )
    if rainfall_unavailable and not is_pilot:
        blockers.append("Missing Earthdata Credentials for IMERG")
    if inventory.get("usable_events", 0) < 50 and not is_pilot:
        blockers.append(f"Insufficient usable landslide events ({inventory.get('usable_events', 0)} < 50)")
        
    if is_pilot:
        # A pilot may ONLY be reported as VALIDATED_PILOT when real, persisted,
        # reproducible validation evidence exists on disk (a trained model
        # artifact, a metrics.json written by an actual validation run, and a
        # feature schema/provenance file). Being flagged is_pilot in config is
        # NOT itself evidence of validation. When the evidence is absent we
        # report an honest VALIDATION_REQUIRED state instead of fabricating a
        # VALIDATED_PILOT claim or hardcoded metrics.
        evidence = load_validation_evidence(state_name, base_dir=evidence_dir)
        if evidence["complete"]:
            overall_status = "VALIDATED_PILOT"
            # Metrics come ONLY from the persisted validation artifact; they are
            # never hardcoded here.
            metrics = evidence["metrics"]
            model_status = "Trained & Validated"
            risk_result = evidence["risk_result"]
        else:
            overall_status = "VALIDATION_REQUIRED"
            metrics = {}
            model_status = "Validation Required (Persisted Model/Metrics Artifacts Absent)"
            risk_result = None
            blockers.append(
                "Missing Persisted Validation Evidence ("
                + ", ".join(evidence["missing"]) + ")"
            )
    elif len(blockers) > 0:
        metrics = {}
        model_status = "Not Trained"
        risk_result = None
        
        if "Missing DEM Data" in blockers or "Missing OSM Exposure Data" in blockers or "Missing Earthdata Credentials for IMERG" in blockers:
            overall_status = "DATA UNAVAILABLE"
        else:
            overall_status = "INSUFFICIENT DATA"
    else:
        overall_status = "VALIDATION IN PROGRESS"
        metrics = {}
        model_status = "Data Ready (Pending Training)"
        risk_result = None
        
    return {
        "overall_status": overall_status,
        "blocking_reasons": blockers,
        "model_status": model_status,
        "validation_metrics": metrics,
        "risk_result": risk_result
    }

def reconcile_reported_status(record: Dict[str, Any], evidence_dir: str = None) -> Dict[str, Any]:
    """
    Reconciles a persisted state-validation record against the validation
    evidence that is ACTUALLY present on disk right now. Earlier runs may have
    written a VALIDATED_PILOT claim into state_validation.json; if the required
    persisted evidence is no longer present (or never was), this returns a copy
    whose reported status is downgraded to VALIDATION_REQUIRED so a runtime
    reader cannot re-present an unbacked claim as current truth.

    This NEVER rewrites the on-disk file -- historical evidence is preserved. It
    only affects what the reader serves.
    """
    if not isinstance(record, dict):
        return record
    claims_validated = (
        record.get("overall_status") == "VALIDATED_PILOT"
        or record.get("validation_status") == "VALIDATED_PILOT"
        or record.get("model_status") == "Trained & Validated"
    )
    if not claims_validated:
        return record

    state_name = record.get("state_name") or record.get("state") or ""
    evidence = load_validation_evidence(state_name, base_dir=evidence_dir)
    if evidence["complete"]:
        return record

    reconciled = dict(record)
    reconciled["overall_status"] = "VALIDATION_REQUIRED"
    reconciled["validation_status"] = "VALIDATION_REQUIRED"
    reconciled["model_status"] = "Validation Required (Persisted Model/Metrics Artifacts Absent)"
    reconciled["validation_metrics"] = {}
    reconciled["risk_result"] = None
    reconciled["reported_status_note"] = (
        "Stored record claimed VALIDATED_PILOT, but required persisted validation "
        "evidence (" + ", ".join(evidence["missing"]) + ") is absent at runtime; "
        "reported status downgraded to VALIDATION_REQUIRED. The on-disk record was "
        "left unchanged."
    )
    return reconciled

def reconcile_validation_report(records: Any, evidence_dir: str = None) -> Any:
    """
    Applies reconcile_reported_status to every record in a loaded
    state_validation report. Non-list payloads are returned unchanged.
    """
    if not isinstance(records, list):
        return records
    return [reconcile_reported_status(r, evidence_dir=evidence_dir) for r in records]

# ---------------------------------------------------------------------------
# Serve-time refresh of the ASSAM record against the real on-disk artifacts
# ---------------------------------------------------------------------------
# data/processed/state_validation.json was written by an early NER sweep, before
# any Assam DEM / terrain / OSM / model artifact existed, so its Assam record
# still reports dem_status/exposure_status "Missing (Requires Download)" and
# model_status "Not Trained". Those artifacts now exist and are real:
#   * terrain  -- the five assam_pilot_* rasters the Assam model was trained on
#                 (NOT "assam_dem.tif", the name the generic evaluate_terrain_data
#                 looks for and which never existed for Assam);
#   * exposure -- data/raw/assam_osm.geojson (real OSM features, correctly named);
#   * model    -- the persisted assam_model.pkl + assam_metrics.json +
#                 assam_feature_schema.json evidence bundle.
# Neither reconcile_validation_report (which only DOWNGRADES an unbacked
# VALIDATED_PILOT claim) nor determine_overall_status (whose model-evidence gate
# is consulted only for is_pilot states -- Assam is is_pilot=False) will lift the
# stale values. This serve-time refresh closes both gaps for ASSAM ONLY. It
# recomputes each field from a real artifact (or leaves it unavailable), never
# rewrites the on-disk file, and returns every other state's record unchanged. It
# fabricates nothing: metrics are read verbatim from assam_metrics.json and a
# field is reported "Available" / "Trained & Validated" only when its real backing
# artifact is present.


def _is_assam_record(record: Any) -> bool:
    """True iff a loaded validation record identifies the state of Assam."""
    if not isinstance(record, dict):
        return False
    for key in ("state_id", "id", "state_name", "state"):
        value = record.get(key)
        if isinstance(value, str) and value.strip().lower().replace(" ", "_") == "assam":
            return True
    return False


def _assam_dem_available(data_dir: str = None):
    """
    True / False when the real Assam terrain rasters can be checked on disk, else
    None (could not check).

    Reuses assam_prediction's OWN terrain-path source of truth -- the five
    assam_pilot_* rasters the model was trained on -- so the dashboard's DEM status
    agrees with what the Assam predictor actually reads, rather than the generic
    evaluate_terrain_data(), which looks for a differently-named "assam_dem.tif"
    that never existed for Assam. Imported lazily so this module still imports
    cleanly in the offline test sandbox (assam_prediction pulls in risk_inputs /
    worldcover / sikkim_prediction). Only existence + non-empty size are checked
    here; the rasters are never opened (no rasterio needed).
    """
    try:
        from app.services import assam_prediction
        return not assam_prediction.missing_assam_terrain_rasters(data_dir)
    except Exception:
        return None


def _recompute_assam_blockers(refreshed: Dict[str, Any], evidence: Dict[str, Any]) -> list:
    """
    Rebuild the Assam blocking_reasons from the refreshed field states: preserve
    any pre-existing reason this refresh does not own (e.g. an inventory note) and
    (re)raise only the DEM / OSM / model-evidence blockers whose backing artifact
    is actually absent right now.
    """
    owned_prefixes = ("Missing DEM Data", "Missing OSM Exposure Data",
                      "Missing Persisted Validation Evidence")
    blockers = [
        b for b in (refreshed.get("blocking_reasons") or [])
        if not any(str(b).startswith(p) for p in owned_prefixes)
    ]
    if refreshed.get("dem_status") != "Available":
        blockers.append("Missing DEM Data")
    if refreshed.get("exposure_status") != "Available":
        blockers.append("Missing OSM Exposure Data")
    if not evidence.get("complete"):
        blockers.append(
            "Missing Persisted Validation Evidence ("
            + ", ".join(evidence.get("missing", [])) + ")"
        )
    return blockers


def _refresh_assam_record(record: Dict[str, Any], evidence_dir: str = None,
                          data_dir: str = None) -> Dict[str, Any]:
    """Return a refreshed COPY of a single Assam record (input left unmutated)."""
    refreshed = dict(record)

    # --- DEM / terrain: real assam_pilot_* rasters ----------------------------
    dem_available = _assam_dem_available(data_dir)
    if dem_available is True:
        refreshed["dem_status"] = "Available"
    elif dem_available is False:
        refreshed["dem_status"] = "Missing (Requires Download)"
    # None -> could not check -> leave the stored value untouched.

    # --- Exposure: real assam_osm.geojson (existing, correctly-named check) ----
    config = NER_STATES_CONFIG.get("Assam")
    if config is not None:
        refreshed["exposure_status"] = evaluate_exposure_data("Assam", config)

    # --- Model evidence: persisted model + metrics + schema -------------------
    # Assam is the project's 2nd pilot; this is exactly the persisted-evidence
    # contract that already justifies Sikkim's VALIDATED_PILOT. Metrics come
    # verbatim from assam_metrics.json -- never hardcoded here.
    evidence = load_validation_evidence("Assam", base_dir=evidence_dir)
    if evidence["complete"]:
        refreshed["model_status"] = "Trained & Validated"
        refreshed["validation_metrics"] = evidence["metrics"]
        if evidence["risk_result"] is not None:
            refreshed["risk_result"] = evidence["risk_result"]
    else:
        refreshed["model_status"] = (
            "Validation Required (Persisted Model/Metrics Artifacts Absent)"
        )
        refreshed["validation_metrics"] = {}
        refreshed["risk_result"] = None

    # --- Overall status, coherent with the refreshed sub-states ----------------
    # Evidence-gated exactly like the pilot contract in determine_overall_status:
    # a real persisted model+metrics is what earns VALIDATED_PILOT. When evidence
    # is absent we fall back to an honest data-availability status instead.
    if evidence["complete"]:
        overall = "VALIDATED_PILOT"
    elif (refreshed.get("dem_status") == "Available"
          and refreshed.get("exposure_status") == "Available"):
        overall = "VALIDATION_REQUIRED"
    else:
        overall = "DATA UNAVAILABLE"
    refreshed["overall_status"] = overall
    refreshed["validation_status"] = overall

    refreshed["blocking_reasons"] = _recompute_assam_blockers(refreshed, evidence)
    return refreshed


def refresh_assam_data_status(records: Any, evidence_dir: str = None,
                              data_dir: str = None) -> Any:
    """
    Return the loaded validation report with ONLY the Assam record refreshed
    against the real Assam artifacts now on disk; every other record (and a
    non-list payload) is returned unchanged. Never rewrites state_validation.json.
    """
    if not isinstance(records, list):
        return records
    return [
        _refresh_assam_record(r, evidence_dir=evidence_dir, data_dir=data_dir)
        if _is_assam_record(r) else r
        for r in records
    ]


# ---------------------------------------------------------------------------
# Serve-time refresh of the ARUNACHAL PRADESH record against real on-disk artifacts
# ---------------------------------------------------------------------------
# Identical situation to Assam (see the block above): data/processed/state_validation.json
# was written by an early NER sweep before any Arunachal artifact existed, so its
# Arunachal Pradesh record still reports dem_status/exposure_status
# "Missing (Requires Download)" and model_status "Not Trained". Those artifacts now
# exist and are real:
#   * terrain  -- the five arunachal_pilot_* rasters the Arunachal model was trained
#                 on (DEM at data/raw/arunachal_pilot_dem.tif + the four derivatives
#                 at data/processed/arunachal_pilot_<name>.tif); NOT "arunachal_dem.tif",
#                 the name the generic evaluate_terrain_data looks for and which never
#                 existed for Arunachal;
#   * exposure -- data/raw/arunachal_pradesh_osm.geojson (real OSM features);
#   * model    -- the persisted arunachal_pradesh_model.pkl + arunachal_pradesh_metrics.json
#                 + arunachal_pradesh_feature_schema.json evidence bundle.
# Neither reconcile_validation_report (downgrade-only) nor determine_overall_status
# (whose model-evidence gate is consulted only for is_pilot states -- Arunachal is
# is_pilot=False) lifts the stale values. This serve-time refresh closes both gaps for
# ARUNACHAL PRADESH ONLY. It recomputes each field from a real artifact (or leaves it
# unavailable), never rewrites the on-disk file, and returns every other state's record
# unchanged. It fabricates nothing: metrics are read verbatim from
# arunachal_pradesh_metrics.json and a field is reported "Available" / "Trained &
# Validated" only when its real backing artifact is present.


def _is_arunachal_record(record: Any) -> bool:
    """True iff a loaded validation record identifies the state of Arunachal Pradesh."""
    if not isinstance(record, dict):
        return False
    for key in ("state_id", "id", "state_name", "state"):
        value = record.get(key)
        if isinstance(value, str) and value.strip().lower().replace(" ", "_") == "arunachal_pradesh":
            return True
    return False


def _arunachal_dem_available(data_dir: str = None):
    """
    True / False when the real Arunachal terrain rasters can be checked on disk, else
    None (could not check).

    Reuses arunachal_prediction's OWN terrain-path source of truth -- the five
    arunachal_pilot_* rasters the model was trained on -- so the dashboard's DEM status
    agrees with what the Arunachal predictor actually reads, rather than the generic
    evaluate_terrain_data(), which looks for a differently-named "arunachal_dem.tif"
    that never existed for Arunachal. Imported lazily so this module still imports
    cleanly in the offline test sandbox (arunachal_prediction pulls in risk_inputs /
    worldcover / sikkim_prediction). Only existence + non-empty size are checked here;
    the rasters are never opened (no rasterio needed).
    """
    try:
        from app.services import arunachal_prediction
        return not arunachal_prediction.missing_arunachal_terrain_rasters(data_dir)
    except Exception:
        return None


def _recompute_arunachal_blockers(refreshed: Dict[str, Any], evidence: Dict[str, Any]) -> list:
    """
    Rebuild the Arunachal blocking_reasons from the refreshed field states: preserve
    any pre-existing reason this refresh does not own (e.g. an inventory note) and
    (re)raise only the DEM / OSM / model-evidence blockers whose backing artifact is
    actually absent right now.
    """
    owned_prefixes = ("Missing DEM Data", "Missing OSM Exposure Data",
                      "Missing Persisted Validation Evidence")
    blockers = [
        b for b in (refreshed.get("blocking_reasons") or [])
        if not any(str(b).startswith(p) for p in owned_prefixes)
    ]
    if refreshed.get("dem_status") != "Available":
        blockers.append("Missing DEM Data")
    if refreshed.get("exposure_status") != "Available":
        blockers.append("Missing OSM Exposure Data")
    if not evidence.get("complete"):
        blockers.append(
            "Missing Persisted Validation Evidence ("
            + ", ".join(evidence.get("missing", [])) + ")"
        )
    return blockers


def _refresh_arunachal_record(record: Dict[str, Any], evidence_dir: str = None,
                              data_dir: str = None) -> Dict[str, Any]:
    """Return a refreshed COPY of a single Arunachal record (input left unmutated)."""
    refreshed = dict(record)

    # --- DEM / terrain: real arunachal_pilot_* rasters -----------------------
    dem_available = _arunachal_dem_available(data_dir)
    if dem_available is True:
        refreshed["dem_status"] = "Available"
    elif dem_available is False:
        refreshed["dem_status"] = "Missing (Requires Download)"
    # None -> could not check -> leave the stored value untouched.

    # --- Exposure: real arunachal_pradesh_osm.geojson (existing check) --------
    config = NER_STATES_CONFIG.get("Arunachal Pradesh")
    if config is not None:
        refreshed["exposure_status"] = evaluate_exposure_data("Arunachal Pradesh", config)

    # --- Model evidence: persisted model + metrics + schema -------------------
    # Arunachal is the project's 3rd pilot; this is exactly the persisted-evidence
    # contract that already justifies Sikkim's and Assam's VALIDATED_PILOT. Metrics
    # come verbatim from arunachal_pradesh_metrics.json -- never hardcoded here.
    evidence = load_validation_evidence("Arunachal Pradesh", base_dir=evidence_dir)
    if evidence["complete"]:
        refreshed["model_status"] = "Trained & Validated"
        refreshed["validation_metrics"] = evidence["metrics"]
        if evidence["risk_result"] is not None:
            refreshed["risk_result"] = evidence["risk_result"]
    else:
        refreshed["model_status"] = (
            "Validation Required (Persisted Model/Metrics Artifacts Absent)"
        )
        refreshed["validation_metrics"] = {}
        refreshed["risk_result"] = None

    # --- Overall status, coherent with the refreshed sub-states ----------------
    # Evidence-gated exactly like the pilot contract in determine_overall_status:
    # a real persisted model+metrics is what earns VALIDATED_PILOT. When evidence
    # is absent we fall back to an honest data-availability status instead.
    if evidence["complete"]:
        overall = "VALIDATED_PILOT"
    elif (refreshed.get("dem_status") == "Available"
          and refreshed.get("exposure_status") == "Available"):
        overall = "VALIDATION_REQUIRED"
    else:
        overall = "DATA UNAVAILABLE"
    refreshed["overall_status"] = overall
    refreshed["validation_status"] = overall

    refreshed["blocking_reasons"] = _recompute_arunachal_blockers(refreshed, evidence)
    return refreshed


def refresh_arunachal_data_status(records: Any, evidence_dir: str = None,
                                  data_dir: str = None) -> Any:
    """
    Return the loaded validation report with ONLY the Arunachal Pradesh record
    refreshed against the real Arunachal artifacts now on disk; every other record
    (and a non-list payload) is returned unchanged. Never rewrites state_validation.json.
    """
    if not isinstance(records, list):
        return records
    return [
        _refresh_arunachal_record(r, evidence_dir=evidence_dir, data_dir=data_dir)
        if _is_arunachal_record(r) else r
        for r in records
    ]


# ---------------------------------------------------------------------------
# Serve-time refresh of the MEGHALAYA record against real on-disk artifacts
# ---------------------------------------------------------------------------
# Identical situation to Assam / Arunachal (see the two blocks above):
# data/processed/state_validation.json was written by an early NER sweep before any
# Meghalaya artifact existed, so its Meghalaya record still reports
# dem_status/exposure_status "Missing (Requires Download)" and model_status
# "Not Trained". Those artifacts now exist and are real:
#   * terrain  -- the five meghalaya_pilot_* rasters the Meghalaya model was trained
#                 on (DEM at data/raw/meghalaya_pilot_dem.tif + the four derivatives
#                 at data/processed/meghalaya_pilot_<name>.tif); NOT "meghalaya_dem.tif",
#                 the name the generic evaluate_terrain_data looks for and which never
#                 existed for Meghalaya;
#   * exposure -- data/raw/meghalaya_osm.geojson (real OSM features);
#   * model    -- the persisted meghalaya_model.pkl + meghalaya_metrics.json
#                 + meghalaya_feature_schema.json evidence bundle.
# Neither reconcile_validation_report (downgrade-only) nor determine_overall_status
# (whose model-evidence gate is consulted only for is_pilot states -- Meghalaya is
# is_pilot=False) lifts the stale values. This serve-time refresh closes both gaps for
# MEGHALAYA ONLY. It recomputes each field from a real artifact (or leaves it
# unavailable), never rewrites the on-disk file, and returns every other state's record
# unchanged. It fabricates nothing: metrics are read verbatim from
# meghalaya_metrics.json and a field is reported "Available" / "Trained & Validated"
# only when its real backing artifact is present.


def _is_meghalaya_record(record: Any) -> bool:
    """True iff a loaded validation record identifies the state of Meghalaya."""
    if not isinstance(record, dict):
        return False
    for key in ("state_id", "id", "state_name", "state"):
        value = record.get(key)
        if isinstance(value, str) and value.strip().lower().replace(" ", "_") == "meghalaya":
            return True
    return False


def _meghalaya_dem_available(data_dir: str = None):
    """
    True / False when the real Meghalaya terrain rasters can be checked on disk, else
    None (could not check).

    Reuses meghalaya_prediction's OWN terrain-path source of truth -- the five
    meghalaya_pilot_* rasters the model was trained on -- so the dashboard's DEM status
    agrees with what the Meghalaya predictor actually reads, rather than the generic
    evaluate_terrain_data(), which looks for a differently-named "meghalaya_dem.tif"
    that never existed for Meghalaya. Imported lazily so this module still imports
    cleanly in the offline test sandbox (meghalaya_prediction pulls in risk_inputs /
    worldcover / sikkim_prediction). Only existence + non-empty size are checked here;
    the rasters are never opened (no rasterio needed).
    """
    try:
        from app.services import meghalaya_prediction
        return not meghalaya_prediction.missing_meghalaya_terrain_rasters(data_dir)
    except Exception:
        return None


def _recompute_meghalaya_blockers(refreshed: Dict[str, Any], evidence: Dict[str, Any]) -> list:
    """
    Rebuild the Meghalaya blocking_reasons from the refreshed field states: preserve
    any pre-existing reason this refresh does not own (e.g. an inventory note) and
    (re)raise only the DEM / OSM / model-evidence blockers whose backing artifact is
    actually absent right now.
    """
    owned_prefixes = ("Missing DEM Data", "Missing OSM Exposure Data",
                      "Missing Persisted Validation Evidence")
    blockers = [
        b for b in (refreshed.get("blocking_reasons") or [])
        if not any(str(b).startswith(p) for p in owned_prefixes)
    ]
    if refreshed.get("dem_status") != "Available":
        blockers.append("Missing DEM Data")
    if refreshed.get("exposure_status") != "Available":
        blockers.append("Missing OSM Exposure Data")
    if not evidence.get("complete"):
        blockers.append(
            "Missing Persisted Validation Evidence ("
            + ", ".join(evidence.get("missing", [])) + ")"
        )
    return blockers


def _refresh_meghalaya_record(record: Dict[str, Any], evidence_dir: str = None,
                              data_dir: str = None) -> Dict[str, Any]:
    """Return a refreshed COPY of a single Meghalaya record (input left unmutated)."""
    refreshed = dict(record)

    # --- DEM / terrain: real meghalaya_pilot_* rasters -----------------------
    dem_available = _meghalaya_dem_available(data_dir)
    if dem_available is True:
        refreshed["dem_status"] = "Available"
    elif dem_available is False:
        refreshed["dem_status"] = "Missing (Requires Download)"
    # None -> could not check -> leave the stored value untouched.

    # --- Exposure: real meghalaya_osm.geojson (existing check) ----------------
    config = NER_STATES_CONFIG.get("Meghalaya")
    if config is not None:
        refreshed["exposure_status"] = evaluate_exposure_data("Meghalaya", config)

    # --- Model evidence: persisted model + metrics + schema -------------------
    # Meghalaya is the project's 4th pilot; this is exactly the persisted-evidence
    # contract that already justifies Sikkim's / Assam's / Arunachal's VALIDATED_PILOT.
    # Metrics come verbatim from meghalaya_metrics.json -- never hardcoded here.
    evidence = load_validation_evidence("Meghalaya", base_dir=evidence_dir)
    if evidence["complete"]:
        refreshed["model_status"] = "Trained & Validated"
        refreshed["validation_metrics"] = evidence["metrics"]
        if evidence["risk_result"] is not None:
            refreshed["risk_result"] = evidence["risk_result"]
    else:
        refreshed["model_status"] = (
            "Validation Required (Persisted Model/Metrics Artifacts Absent)"
        )
        refreshed["validation_metrics"] = {}
        refreshed["risk_result"] = None

    # --- Overall status, coherent with the refreshed sub-states ----------------
    # Evidence-gated exactly like the pilot contract in determine_overall_status:
    # a real persisted model+metrics is what earns VALIDATED_PILOT. When evidence
    # is absent we fall back to an honest data-availability status instead.
    if evidence["complete"]:
        overall = "VALIDATED_PILOT"
    elif (refreshed.get("dem_status") == "Available"
          and refreshed.get("exposure_status") == "Available"):
        overall = "VALIDATION_REQUIRED"
    else:
        overall = "DATA UNAVAILABLE"
    refreshed["overall_status"] = overall
    refreshed["validation_status"] = overall

    refreshed["blocking_reasons"] = _recompute_meghalaya_blockers(refreshed, evidence)
    return refreshed


def refresh_meghalaya_data_status(records: Any, evidence_dir: str = None,
                                  data_dir: str = None) -> Any:
    """
    Return the loaded validation report with ONLY the Meghalaya record refreshed
    against the real Meghalaya artifacts now on disk; every other record (and a
    non-list payload) is returned unchanged. Never rewrites state_validation.json.
    """
    if not isinstance(records, list):
        return records
    return [
        _refresh_meghalaya_record(r, evidence_dir=evidence_dir, data_dir=data_dir)
        if _is_meghalaya_record(r) else r
        for r in records
    ]


def compute_inventory_diagnostics(state_name: str, config: Dict[str, Any], glc_df: pd.DataFrame) -> dict:
    raw_india = glc_df[glc_df['country_name'] == 'India']
    raw_india_count = len(raw_india)
    
    bbox_mask = (
        (glc_df['latitude'] >= config['min_lat']) &
        (glc_df['latitude'] <= config['max_lat']) &
        (glc_df['longitude'] >= config['min_lon']) &
        (glc_df['longitude'] <= config['max_lon'])
    )
    bbox_df = glc_df[bbox_mask]
    bbox_count = len(bbox_df)
    
    state_names = [state_name, state_name.replace(' ', ''), state_name.lower()]
    if state_name == "Arunachal Pradesh":
        state_names.extend(["Arunāchal Pradesh", "Arunachal", "arunachal"])
    elif state_name == "Nagaland":
        state_names.extend(["Nāgāland", "nagaland"])
    elif state_name == "Meghalaya":
        state_names.extend(["Meghālaya", "meghalaya"])
        
    # admin_division_name is NaN for catalog rows with no admin label. Under
    # pandas 3.x, Series.astype(str) preserves missing values as NA rather than
    # coercing them to the literal "nan" string (the pre-3.x behavior), so a raw
    # NaN would reach the lambda as a missing scalar and raise AttributeError on
    # x.lower(). fillna('') makes this NaN-safe across pandas versions; an empty
    # admin string matches no state name, identical to the prior "nan"-string result.
    admin_mask = glc_df['admin_division_name'].fillna('').astype(str).apply(
        lambda x: any(name.lower() in x.lower() for name in state_names)
    )
    admin_df = glc_df[admin_mask & (glc_df['country_name'] == 'India')]
    admin_count = len(admin_df)
    
    valid_coords_df = bbox_df.dropna(subset=['latitude', 'longitude'])
    valid_coords_count = len(valid_coords_df)
    
    if 'event_date' in valid_coords_df.columns:
        valid_coords_df = valid_coords_df.copy()
        valid_coords_df['parsed_date'] = pd.to_datetime(valid_coords_df['event_date'], errors='coerce', format='mixed')
        exact_dates = valid_coords_df.dropna(subset=['parsed_date'])
        dedup_df = exact_dates.drop_duplicates(subset=['latitude', 'longitude', 'parsed_date'])
        dedup_count = len(dedup_df)
    else:
        dedup_count = 0
        
    return {
        "raw_india": raw_india_count,
        "bbox": bbox_count,
        "admin": admin_count,
        "valid_coords": valid_coords_count,
        "deduplicated": dedup_count
    }

def process_state(state_name: str, config: Dict[str, Any], glc_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Independently processes a single NER state through the validation pipeline.
    """
    import json
    from app.services.terrain_processing import process_dem_in_chunks
    
    print(f"[STATE] Processing {state_name}")
    
    # 1. Landslide Inventory
    inventory = evaluate_landslide_inventory(config, glc_df)
    
    # 2. Terrain DEM
    dem_path = ""
    try:
        dem_path = acquire_state_dem(state_name, config)
    except Exception as e:
        print(f"[DEM] Error acquiring DEM for {state_name}: {e}")
        
    terrain = "Available" if (dem_path and os.path.exists(dem_path)) else "Missing (Requires Download)"
    
    if terrain == "Available":
        proc_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed"))
        clean_state_name = state_name.lower().replace(' ', '_')
        try:
            print(f"[DEM] Generating terrain features (slope, aspect, etc.) for {state_name}...")
            process_dem_in_chunks(dem_path, proc_dir, state_prefix=clean_state_name)
        except Exception as e:
            print(f"[DEM] Error calculating terrain features for {state_name}: {e}")
            terrain = "Error in Feature Calculation"
            
    # 3. OSM Exposure
    osm_path = ""
    try:
        osm_path = acquire_state_osm(state_name, config)
    except Exception as e:
        print(f"[OSM] Error acquiring OSM for {state_name}: {e}")
        
    exposure = "Available" if (osm_path and os.path.exists(osm_path)) else "Missing (Requires Download)"
    
    # 4. State-specific Rainfall & Fallback
    rain_info = evaluate_state_rainfall(state_name, config)
    
    # 5. Determine Status & Metrics
    status_info = determine_overall_status(
        state_name,
        inventory,
        terrain,
        rain_info["status"],
        exposure,
        config.get("is_pilot", False)
    )
    
    # 6. Print diagnostics
    diag = compute_inventory_diagnostics(state_name, config, glc_df)
    
    print(f"\n[DEM]")
    print(f"Source: Copernicus 30m S3")
    print(f"Bounding box: {config['min_lat']}N-{config['max_lat']}N, {config['min_lon']}E-{config['max_lon']}E")
    print(f"Files discovered: {os.path.basename(dem_path) if (dem_path and os.path.exists(dem_path)) else 'None'}")
    print(f"Coverage: {state_name} core bounds")
    print(f"Status: {'AVAILABLE' if terrain == 'Available' else 'MISSING'}")
    
    print(f"\n[OSM]")
    print(f"Source: Overpass API")
    print(f"Query region: center box")
    num_features = 0
    if osm_path and os.path.exists(osm_path):
        try:
            with open(osm_path, 'r', encoding='utf-8') as f:
                features_data = json.load(f)
                num_features = len(features_data.get("features", []))
        except Exception:
            pass
    print(f"Features retrieved: {num_features}")
    print(f"Status: {'AVAILABLE' if exposure == 'Available' else 'MISSING'}")
    
    print(f"\n[INVENTORY]")
    print(f"Raw India records: {diag['raw_india']}")
    print(f"Bounding-box records: {diag['bbox']}")
    print(f"Admin/state matched records: {diag['admin']}")
    print(f"Valid coordinates: {diag['valid_coords']}")
    print(f"Deduplicated records: {diag['deduplicated']}")
    
    print(f"\n[RAINFALL]")
    print(f"Primary: {rain_info['source']}")
    print(f"Fallback: {rain_info['is_fallback']}")
    print()
    
    state_id = config.get("id", state_name.lower().replace(" ", "_"))
    
    report = {
        "id": state_id,
        "state_id": state_id,
        "state": state_name,
        "state_name": state_name,
        "processing_status": "COMPLETED",
        "validation_status": status_info["overall_status"],
        "overall_status": status_info["overall_status"],
        "rainfall_source": rain_info["source"],
        "rainfall_status": rain_info["status"],
        "inventory_events": inventory["inventory_events"],
        "usable_events": inventory["usable_events"],
        "spatial_quality": inventory["spatial_quality"],
        "temporal_quality": inventory["temporal_quality"],
        "dem_status": terrain,
        "exposure_status": exposure,
        "model_status": status_info["model_status"],
        "validation_metrics": status_info["validation_metrics"],
        "risk_result": status_info["risk_result"],
        "blocking_reasons": status_info["blocking_reasons"],
        "error": None
    }
    
    print(f"[STATE] {state_name} completed")
    return report
