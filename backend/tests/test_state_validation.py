import os
import sys
import json
import pandas as pd
import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.core.config_states import NER_STATES_CONFIG
from app.services.state_validation import (
    evaluate_landslide_inventory,
    determine_overall_status,
    load_validation_evidence,
    reconcile_validation_report,
)
import app.services.state_validation as sv

# ---------------------------------------------------------------------------
# Test fixtures / helpers
# ---------------------------------------------------------------------------
# NOTE ON HONESTY: the "evidence" files written below are SYNTHETIC unit-test
# fixtures, not real trained models or real validation runs. The metric values
# are deliberately arbitrary placeholders (and intentionally NOT the historical
# 0.7762 / 0.9190 figures) so that any test asserting metrics come from the
# persisted file proves the value actually FLOWED FROM THE FILE rather than from
# a hardcoded literal. A .pkl "model" here is just placeholder bytes; these
# tests never load or execute a model and never train anything.

def _write_evidence(dir_path, clean_state, metrics=None, risk_result=None,
                    write_model=True, write_schema=True, write_metrics=True):
    os.makedirs(dir_path, exist_ok=True)
    if write_model:
        (dir_path / f"{clean_state}_model.pkl").write_bytes(b"UNIT-TEST-PLACEHOLDER-NOT-A-REAL-MODEL")
    if write_schema:
        schema = {"features": ["elevation", "slope"], "provenance": "unit-test fixture"}
        (dir_path / f"{clean_state}_feature_schema.json").write_text(json.dumps(schema))
    if write_metrics:
        if metrics is None:
            metrics = {"PR-AUC": 0.5, "ROC-AUC": 0.6, "F1": 0.42}
        doc = {"validation_metrics": metrics}
        if risk_result is not None:
            doc["risk_result"] = risk_result
        (dir_path / f"{clean_state}_metrics.json").write_text(json.dumps(doc))


@pytest.fixture
def mock_glc_df():
    # 3 events in Assam (2 exact same location/date, 1 different)
    # 1 event in Manipur
    data = {
        'latitude': [26.0, 26.0, 26.5, 24.5],
        'longitude': [91.0, 91.0, 92.0, 93.5],
        'event_date': ['2015-06-01', '2015-06-01', '2016-07-15', '2017-08-10'],
        'location_accuracy': ['1km', '1km', '5km', 'exact']
    }
    return pd.DataFrame(data)

def test_evaluate_landslide_inventory_assam(mock_glc_df):
    config = NER_STATES_CONFIG["Assam"]
    res = evaluate_landslide_inventory(config, mock_glc_df)

    assert res["inventory_events"] == 4
    # Two events are duplicates, so usable should be 3
    assert res["usable_events"] == 3
    # 1km, 1km, 5km, exact -> 3 out of 4 are high accuracy (> 0.5)
    assert res["spatial_quality"] == "Good"

def test_evaluate_landslide_inventory_empty(mock_glc_df):
    config = NER_STATES_CONFIG["Mizoram"]
    res = evaluate_landslide_inventory(config, mock_glc_df)
    assert res["inventory_events"] == 0
    assert res["usable_events"] == 0

# ---------------------------------------------------------------------------
# Validation evidence gate (Phase 2B-4)
# ---------------------------------------------------------------------------

def test_pilot_without_evidence_is_validation_required():
    """A pilot with NO persisted model/metrics/schema artifacts must NOT be
    reported as VALIDATED_PILOT. Being flagged is_pilot (and the descriptive
    STATIC_MODEL_METADATA that lives in code) is not, on its own, validation
    evidence."""
    inventory = {"usable_events": 100}
    status = determine_overall_status(
        "Sikkim", inventory, "Available", "Authenticated", "Available", True,
        evidence_dir="/nonexistent/evidence/dir/for/unit-test"
    )
    assert status["overall_status"] == "VALIDATION_REQUIRED"
    assert status["validation_metrics"] == {}
    assert "PR-AUC" not in status["validation_metrics"]
    assert status["risk_result"] is None
    assert status["model_status"].startswith("Validation Required")
    assert any("Missing Persisted Validation Evidence" in b for b in status["blocking_reasons"])

def test_pilot_with_valid_evidence_is_validated_pilot(tmp_path):
    """When the full three-file evidence contract exists AND metrics.json is
    structurally valid, the pilot may be reported VALIDATED_PILOT, and metrics
    come verbatim from the persisted file (not from any hardcode)."""
    inventory = {"usable_events": 100}
    metrics = {"PR-AUC": 0.5, "ROC-AUC": 0.6, "F1": 0.42}  # arbitrary fixture values, NOT 0.7762/0.9190
    risk = {"susceptibility_score": 0.5, "warning_level": "MEDIUM", "coverage": "unit-test"}
    _write_evidence(tmp_path, "sikkim", metrics=metrics, risk_result=risk)

    status = determine_overall_status(
        "Sikkim", inventory, "Available", "Authenticated", "Available", True,
        evidence_dir=str(tmp_path)
    )
    assert status["overall_status"] == "VALIDATED_PILOT"
    assert status["model_status"] == "Trained & Validated"
    assert status["validation_metrics"] == metrics          # flowed from the file
    assert status["risk_result"] == risk
    assert status["blocking_reasons"] == []

def test_pilot_incomplete_evidence_is_not_validated(tmp_path):
    """Metrics present but the model artifact + schema are missing -> not
    validated (all required evidence must be present)."""
    _write_evidence(tmp_path, "sikkim", write_model=False, write_schema=False)
    status = determine_overall_status(
        "Sikkim", {"usable_events": 100}, "Available", "Authenticated", "Available", True,
        evidence_dir=str(tmp_path)
    )
    assert status["overall_status"] == "VALIDATION_REQUIRED"
    assert status["validation_metrics"] == {}

def test_pilot_invalid_metrics_schema_is_not_validated(tmp_path):
    """All three files exist, but metrics.json lacks the required metric keys.
    The gate refuses it rather than inventing/accepting partial metrics -> no
    fake metrics are ever produced."""
    _write_evidence(tmp_path, "sikkim", metrics={"note": "not real metrics"})
    status = determine_overall_status(
        "Sikkim", {"usable_events": 100}, "Available", "Authenticated", "Available", True,
        evidence_dir=str(tmp_path)
    )
    assert status["overall_status"] == "VALIDATION_REQUIRED"
    assert status["validation_metrics"] == {}

def test_load_validation_evidence_reports_missing(tmp_path):
    ev = load_validation_evidence("Sikkim", base_dir=str(tmp_path))
    assert ev["complete"] is False
    assert ev["metrics"] == {}
    assert ev["risk_result"] is None
    # All three components should be reported missing when the dir is empty.
    assert set(ev["missing"]) == {"model", "metrics", "schema"}

def test_load_validation_evidence_complete(tmp_path):
    metrics = {"PR-AUC": 0.51, "ROC-AUC": 0.63}
    _write_evidence(tmp_path, "sikkim", metrics=metrics)
    ev = load_validation_evidence("Sikkim", base_dir=str(tmp_path))
    assert ev["complete"] is True
    assert ev["metrics"] == metrics
    assert ev["missing"] == []

# ---------------------------------------------------------------------------
# Reader reconciliation of stale state_validation.json claims (Phase 2B-4)
# ---------------------------------------------------------------------------

def test_reconcile_downgrades_stale_validated_pilot(tmp_path):
    """A stored record claiming VALIDATED_PILOT with hardcoded metrics is
    downgraded to VALIDATION_REQUIRED at read time when no evidence exists."""
    stale = [{
        "state_name": "Sikkim",
        "overall_status": "VALIDATED_PILOT",
        "validation_status": "VALIDATED_PILOT",
        "model_status": "Trained & Validated",
        "validation_metrics": {"PR-AUC": 0.7762, "ROC-AUC": 0.9190},
        "risk_result": {"susceptibility_score": 0.72},
    }]
    out = reconcile_validation_report(stale, evidence_dir=str(tmp_path))
    assert out[0]["overall_status"] == "VALIDATION_REQUIRED"
    assert out[0]["validation_status"] == "VALIDATION_REQUIRED"
    assert out[0]["validation_metrics"] == {}
    assert out[0]["risk_result"] is None
    assert "reported_status_note" in out[0]

def test_reconcile_preserves_validated_pilot_when_evidence_present(tmp_path):
    _write_evidence(tmp_path, "sikkim", metrics={"PR-AUC": 0.5, "ROC-AUC": 0.6})
    rec = [{
        "state_name": "Sikkim",
        "overall_status": "VALIDATED_PILOT",
        "validation_status": "VALIDATED_PILOT",
        "model_status": "Trained & Validated",
        "validation_metrics": {"PR-AUC": 0.5, "ROC-AUC": 0.6},
    }]
    out = reconcile_validation_report(rec, evidence_dir=str(tmp_path))
    assert out[0]["overall_status"] == "VALIDATED_PILOT"  # unchanged; evidence present

def test_reconcile_leaves_non_validated_records_untouched(tmp_path):
    rec = [{"state_name": "Assam", "overall_status": "DATA UNAVAILABLE", "validation_metrics": {}}]
    out = reconcile_validation_report(rec, evidence_dir=str(tmp_path))
    assert out[0]["overall_status"] == "DATA UNAVAILABLE"

# ---------------------------------------------------------------------------
# Non-pilot status logic (unchanged behavior; regression guard)
# ---------------------------------------------------------------------------

def test_determine_overall_status_insufficient_data():
    inventory = {"usable_events": 10}
    status = determine_overall_status("Assam", inventory, "Available", "Authenticated", "Available", False)
    assert status["overall_status"] == "INSUFFICIENT DATA"
    assert any("Insufficient" in b for b in status["blocking_reasons"])

def test_determine_overall_status_data_unavailable():
    inventory = {"usable_events": 100}
    status = determine_overall_status("Assam", inventory, "Missing", "Authenticated", "Available", False)
    assert status["overall_status"] == "DATA UNAVAILABLE"
    assert "Missing DEM Data" in status["blocking_reasons"]

    status2 = determine_overall_status("Assam", inventory, "Available", "Unauthenticated", "Available", False)
    assert status2["overall_status"] == "DATA UNAVAILABLE"
    assert "Missing Earthdata Credentials for IMERG" in status2["blocking_reasons"]

def test_determine_overall_status_in_progress():
    inventory = {"usable_events": 100}
    status = determine_overall_status("Assam", inventory, "Available", "Authenticated", "Available", False)
    assert status["overall_status"] == "VALIDATION IN PROGRESS"
    assert len(status["blocking_reasons"]) == 0

# ---------------------------------------------------------------------------
# Phase 2F: reader gate rejects structurally-present but degenerate metrics
# ---------------------------------------------------------------------------
# The three keys can all be present yet carry null / non-numeric / non-finite
# values (a partially-written, hand-edited or otherwise corrupt metrics.json).
# Accepting that would report VALIDATED_PILOT off a meaningless number, so the
# reader gate now refuses such values -- mirroring the writer gate.

def test_pilot_with_null_metric_value_is_not_validated(tmp_path):
    _write_evidence(tmp_path, "sikkim", metrics={"PR-AUC": None, "ROC-AUC": 0.6})
    status = determine_overall_status(
        "Sikkim", {"usable_events": 100}, "Available", "Authenticated", "Available", True,
        evidence_dir=str(tmp_path)
    )
    assert status["overall_status"] == "VALIDATION_REQUIRED"
    assert status["validation_metrics"] == {}

def test_load_validation_evidence_rejects_non_numeric_metric(tmp_path):
    _write_evidence(tmp_path, "sikkim", metrics={"PR-AUC": "n/a", "ROC-AUC": 0.6})
    ev = load_validation_evidence("Sikkim", base_dir=str(tmp_path))
    assert ev["complete"] is False
    assert ev["metrics"] == {}
    assert any(("non-numeric" in m or "non-finite" in m) for m in ev["missing"])

def test_load_validation_evidence_rejects_nonfinite_metric(tmp_path):
    # json writes NaN as the literal `NaN`, which json.load reads back as float
    # nan; a non-finite metric is not real evidence.
    _write_evidence(tmp_path, "sikkim", metrics={"PR-AUC": float("nan"), "ROC-AUC": 0.6})
    ev = load_validation_evidence("Sikkim", base_dir=str(tmp_path))
    assert ev["complete"] is False

def test_load_validation_evidence_rejects_bool_metric(tmp_path):
    # bool is a subclass of int; True must NOT masquerade as a numeric metric.
    _write_evidence(tmp_path, "sikkim", metrics={"PR-AUC": True, "ROC-AUC": 0.6})
    ev = load_validation_evidence("Sikkim", base_dir=str(tmp_path))
    assert ev["complete"] is False

def test_load_validation_evidence_still_accepts_real_floats(tmp_path):
    # Regression guard: legitimate float metrics remain acceptable and flow verbatim.
    metrics = {"PR-AUC": 0.0, "ROC-AUC": 0.6}  # 0.0 is a real, finite value
    _write_evidence(tmp_path, "sikkim", metrics=metrics)
    ev = load_validation_evidence("Sikkim", base_dir=str(tmp_path))
    assert ev["complete"] is True
    assert ev["metrics"] == metrics

# ---------------------------------------------------------------------------
# Phase 2F: the live "Unavailable (...)" rainfall vocabulary is recognised
# ---------------------------------------------------------------------------

def test_unavailable_rainfall_string_flags_the_earthdata_blocker():
    # evaluate_state_rainfall now emits "Unavailable (NASA Earthdata ...)" instead
    # of the old "Unauthenticated"/"Missing ..." tokens. A non-pilot state whose
    # satellite IMERG is genuinely unavailable must still surface the blocker.
    status = determine_overall_status(
        "Assam", {"usable_events": 100}, "Available",
        "Unavailable (NASA Earthdata authentication failed)", "Available", False
    )
    assert "Missing Earthdata Credentials for IMERG" in status["blocking_reasons"]
    assert status["overall_status"] == "DATA UNAVAILABLE"

def test_authenticated_rainfall_string_does_not_flag_the_blocker():
    status = determine_overall_status(
        "Assam", {"usable_events": 100}, "Available",
        "Authenticated (Satellite IMERG)", "Available", False
    )
    assert "Missing Earthdata Credentials for IMERG" not in status["blocking_reasons"]
    assert status["overall_status"] == "VALIDATION IN PROGRESS"

# ---------------------------------------------------------------------------
# Phase 2F: an UNAVAILABLE satellite-IMERG state must not be labeled a fallback
# ---------------------------------------------------------------------------
# evaluate_state_rainfall has NO synthetic/fallback rainfall path: when Earthdata
# auth fails it reports the data UNAVAILABLE. The is_fallback flag -- and the
# "Fallback: ..." line process_state prints from it -- must therefore be False in
# BOTH branches; reporting True implied a working substitute that does not exist.

def _swap_earthdata(session_value, raises):
    """Rebind state_validation.get_earthdata_session; return the original."""
    original = sv.get_earthdata_session
    if raises:
        def _sess():
            raise PermissionError("BLOCKER: Missing NASA Earthdata credentials")
    else:
        def _sess():
            return session_value
    sv.get_earthdata_session = _sess
    return original

def test_state_rainfall_unavailable_is_not_labeled_fallback():
    original = _swap_earthdata(None, raises=True)
    try:
        info = sv.evaluate_state_rainfall("Assam", NER_STATES_CONFIG["Assam"])
    finally:
        sv.get_earthdata_session = original
    assert info["imerg_available"] is False
    assert info["is_fallback"] is False          # nothing is substituted -> not a fallback
    assert info["unavailable_reason"]            # the real reason is carried
    assert "no synthetic substitute" in info["source"]
    assert info["status"].startswith("Unavailable")

def test_state_rainfall_available_is_not_fallback():
    original = _swap_earthdata(object(), raises=False)
    try:
        info = sv.evaluate_state_rainfall("Sikkim", NER_STATES_CONFIG["Sikkim"])
    finally:
        sv.get_earthdata_session = original
    assert info["imerg_available"] is True
    assert info["is_fallback"] is False
    assert info["unavailable_reason"] is None


# ---------------------------------------------------------------------------
# Serve-time Assam refresh (refresh_assam_data_status)
# ---------------------------------------------------------------------------
# state_validation.json was written by an early NER sweep, before any Assam
# artifact existed, so its Assam record reports dem/exposure "Missing" and model
# "Not Trained". reconcile_validation_report only downgrades unbacked
# VALIDATED_PILOT claims, and determine_overall_status only consults the model
# evidence gate for is_pilot states (Assam is is_pilot=False), so neither lifts
# the stale values. refresh_assam_data_status closes that gap for ASSAM ONLY,
# recomputing each field from a real on-disk artifact. These tests drive the DEM
# and model-evidence branches off TEMP directories (so they never depend on the
# checkout's large rasters); exposure is asserted to agree with the existing
# evaluate_exposure_data() check against the committed assam_osm.geojson.

def _write_assam_terrain_rasters(data_dir):
    """Create the five non-empty assam_pilot_* raster files the refresh checks for.
    Placeholder bytes only -- the refresh checks existence + size, never opens them."""
    raw = os.path.join(str(data_dir), "raw")
    proc = os.path.join(str(data_dir), "processed")
    os.makedirs(raw, exist_ok=True)
    os.makedirs(proc, exist_ok=True)
    with open(os.path.join(raw, "assam_pilot_dem.tif"), "wb") as fh:
        fh.write(b"NOT-A-REAL-RASTER-PLACEHOLDER")
    for name in ("slope", "aspect", "roughness", "tpi"):
        with open(os.path.join(proc, "assam_pilot_%s.tif" % name), "wb") as fh:
            fh.write(b"NOT-A-REAL-RASTER-PLACEHOLDER")


def _stale_assam_record():
    """A copy of the stale Assam record shape persisted in state_validation.json."""
    return {
        "id": "assam", "state_id": "assam", "state": "Assam", "state_name": "Assam",
        "processing_status": "COMPLETED",
        "validation_status": "DATA UNAVAILABLE",
        "overall_status": "DATA UNAVAILABLE",
        "rainfall_source": "Open-Meteo / Fallback Synthetic",
        "rainfall_status": "Fallback Active (Open-Meteo / Local)",
        "inventory_events": 401, "usable_events": 401,
        "spatial_quality": "Moderate/Poor", "temporal_quality": "Good",
        "dem_status": "Missing (Requires Download)",
        "exposure_status": "Missing (Requires Download)",
        "model_status": "Not Trained",
        "validation_metrics": {}, "risk_result": None,
        "blocking_reasons": ["Missing DEM Data", "Missing OSM Exposure Data"],
        "error": None,
    }


def test_is_assam_record_matches_by_id_or_name():
    assert sv._is_assam_record({"state_id": "assam"}) is True
    assert sv._is_assam_record({"state_name": "Assam"}) is True
    assert sv._is_assam_record({"id": "ASSAM"}) is True
    assert sv._is_assam_record({"state": "  assam  "}) is True
    assert sv._is_assam_record({"state_name": "Sikkim"}) is False
    assert sv._is_assam_record("not a dict") is False


def test_refresh_non_list_payload_returned_unchanged():
    assert sv.refresh_assam_data_status({"not": "a list"}) == {"not": "a list"}
    assert sv.refresh_assam_data_status(None) is None


def test_refresh_leaves_non_assam_records_untouched(tmp_path):
    import copy
    sikkim = {"id": "sikkim", "state_name": "Sikkim", "overall_status": "VALIDATED_PILOT",
              "dem_status": "Available", "model_status": "Trained & Validated"}
    megh = {"id": "meghalaya", "state_name": "Meghalaya",
            "overall_status": "DATA UNAVAILABLE",
            "dem_status": "Missing (Requires Download)"}
    records = [copy.deepcopy(sikkim), _stale_assam_record(), copy.deepcopy(megh)]
    out = sv.refresh_assam_data_status(
        records, evidence_dir=str(tmp_path / "none"), data_dir=str(tmp_path / "none"))
    assert out[0] == sikkim          # non-Assam record returned byte-identical
    assert out[2] == megh            # non-Assam record returned byte-identical
    assert out[1]["state_name"] == "Assam"   # only the Assam record is refreshed


def test_refresh_assam_upgrades_from_real_artifacts(tmp_path):
    data_dir = tmp_path / "data"
    _write_assam_terrain_rasters(data_dir)
    ev_dir = tmp_path / "models"
    metrics = {"PR-AUC": 0.5878, "ROC-AUC": 0.7456, "F1": 0.566}  # fixture values
    _write_evidence(ev_dir, "assam", metrics=metrics)

    records = [_stale_assam_record()]
    out = sv.refresh_assam_data_status(
        records, evidence_dir=str(ev_dir), data_dir=str(data_dir))
    rec = out[0]

    assert rec["dem_status"] == "Available"
    assert rec["model_status"] == "Trained & Validated"
    assert rec["validation_metrics"] == metrics          # flowed from the file
    assert rec["overall_status"] == "VALIDATED_PILOT"
    assert rec["validation_status"] == "VALIDATED_PILOT"

    # Exposure agrees with the existing check against the committed assam_osm.geojson.
    expected_exposure = sv.evaluate_exposure_data("Assam", NER_STATES_CONFIG["Assam"])
    assert rec["exposure_status"] == expected_exposure

    assert "Missing DEM Data" not in rec["blocking_reasons"]
    assert not any("Persisted Validation Evidence" in b for b in rec["blocking_reasons"])
    assert ("Missing OSM Exposure Data" in rec["blocking_reasons"]) == (
        expected_exposure != "Available")

    # The input record is copied, not mutated in place.
    assert records[0]["dem_status"] == "Missing (Requires Download)"
    assert records[0]["overall_status"] == "DATA UNAVAILABLE"


def test_refresh_assam_incomplete_evidence_is_not_validated(tmp_path):
    data_dir = tmp_path / "data"
    _write_assam_terrain_rasters(data_dir)
    ev_dir = tmp_path / "empty_models"
    os.makedirs(ev_dir, exist_ok=True)          # no assam_* evidence files

    out = sv.refresh_assam_data_status(
        [_stale_assam_record()], evidence_dir=str(ev_dir), data_dir=str(data_dir))
    rec = out[0]

    assert rec["dem_status"] == "Available"
    assert rec["model_status"].startswith("Validation Required")
    assert rec["validation_metrics"] == {}
    assert rec["risk_result"] is None
    assert any("Persisted Validation Evidence" in b for b in rec["blocking_reasons"])

    expected_exposure = sv.evaluate_exposure_data("Assam", NER_STATES_CONFIG["Assam"])
    if expected_exposure == "Available":
        assert rec["overall_status"] == "VALIDATION_REQUIRED"
    else:
        assert rec["overall_status"] == "DATA UNAVAILABLE"


def test_refresh_assam_missing_terrain_is_reported(tmp_path):
    data_dir = tmp_path / "data"
    os.makedirs(data_dir, exist_ok=True)        # no assam_pilot_* rasters
    ev_dir = tmp_path / "models"
    _write_evidence(ev_dir, "assam", metrics={"PR-AUC": 0.5878, "ROC-AUC": 0.7456})

    out = sv.refresh_assam_data_status(
        [_stale_assam_record()], evidence_dir=str(ev_dir), data_dir=str(data_dir))
    rec = out[0]

    assert rec["dem_status"] == "Missing (Requires Download)"
    assert "Missing DEM Data" in rec["blocking_reasons"]
    # Real persisted model evidence still earns "Trained & Validated" (evidence-
    # gated, exactly like the Sikkim pilot contract) even though a live raster is
    # currently absent -- the DEM chip / blocker report that independently.
    assert rec["model_status"] == "Trained & Validated"
    assert rec["overall_status"] == "VALIDATED_PILOT"


# ---------------------------------------------------------------------------
# Serve-time Arunachal Pradesh refresh (refresh_arunachal_data_status)
# ---------------------------------------------------------------------------
# Identical situation to Assam: state_validation.json was written by an early NER
# sweep before any Arunachal artifact existed, so its Arunachal Pradesh record
# reports dem/exposure "Missing" and model "Not Trained". reconcile_validation_report
# only downgrades unbacked VALIDATED_PILOT claims, and determine_overall_status only
# consults the model-evidence gate for is_pilot states (Arunachal is is_pilot=False),
# so neither lifts the stale values. refresh_arunachal_data_status closes that gap for
# ARUNACHAL PRADESH ONLY, recomputing each field from a real on-disk artifact. These
# tests drive the DEM and model-evidence branches off TEMP directories (so they never
# depend on the checkout's large rasters); exposure is asserted to agree with the
# existing evaluate_exposure_data() check against the committed arunachal_pradesh_osm.geojson.

def _write_arunachal_terrain_rasters(data_dir):
    """Create the five non-empty arunachal_pilot_* raster files the refresh checks for.
    Placeholder bytes only -- the refresh checks existence + size, never opens them.
    Layout matches arunachal_prediction.arunachal_terrain_raster_paths: DEM in raw/,
    the four derivatives in processed/."""
    raw = os.path.join(str(data_dir), "raw")
    proc = os.path.join(str(data_dir), "processed")
    os.makedirs(raw, exist_ok=True)
    os.makedirs(proc, exist_ok=True)
    with open(os.path.join(raw, "arunachal_pilot_dem.tif"), "wb") as fh:
        fh.write(b"NOT-A-REAL-RASTER-PLACEHOLDER")
    for name in ("slope", "aspect", "roughness", "tpi"):
        with open(os.path.join(proc, "arunachal_pilot_%s.tif" % name), "wb") as fh:
            fh.write(b"NOT-A-REAL-RASTER-PLACEHOLDER")


def _stale_arunachal_record():
    """A copy of the stale Arunachal record shape persisted in state_validation.json."""
    return {
        "id": "arunachal_pradesh", "state_id": "arunachal_pradesh",
        "state": "Arunachal Pradesh", "state_name": "Arunachal Pradesh",
        "processing_status": "COMPLETED",
        "validation_status": "DATA UNAVAILABLE",
        "overall_status": "DATA UNAVAILABLE",
        "rainfall_source": "Open-Meteo / Fallback Synthetic",
        "rainfall_status": "Fallback Active (Open-Meteo / Local)",
        "inventory_events": 88, "usable_events": 88,
        "spatial_quality": "Moderate/Poor", "temporal_quality": "Good",
        "dem_status": "Missing (Requires Download)",
        "exposure_status": "Missing (Requires Download)",
        "model_status": "Not Trained",
        "validation_metrics": {}, "risk_result": None,
        "blocking_reasons": ["Missing DEM Data", "Missing OSM Exposure Data"],
        "error": None,
    }


def test_is_arunachal_record_matches_by_id_or_name():
    assert sv._is_arunachal_record({"state_id": "arunachal_pradesh"}) is True
    assert sv._is_arunachal_record({"state_name": "Arunachal Pradesh"}) is True
    assert sv._is_arunachal_record({"id": "ARUNACHAL_PRADESH"}) is True
    assert sv._is_arunachal_record({"state": "  Arunachal Pradesh  "}) is True
    assert sv._is_arunachal_record({"state_name": "Assam"}) is False
    assert sv._is_arunachal_record({"state_name": "Sikkim"}) is False
    assert sv._is_arunachal_record("not a dict") is False


def test_refresh_arunachal_non_list_payload_returned_unchanged():
    assert sv.refresh_arunachal_data_status({"not": "a list"}) == {"not": "a list"}
    assert sv.refresh_arunachal_data_status(None) is None


def test_refresh_leaves_non_arunachal_records_untouched(tmp_path):
    import copy
    sikkim = {"id": "sikkim", "state_name": "Sikkim", "overall_status": "VALIDATED_PILOT",
              "dem_status": "Available", "model_status": "Trained & Validated"}
    assam = {"id": "assam", "state_name": "Assam", "overall_status": "DATA UNAVAILABLE",
             "dem_status": "Missing (Requires Download)"}
    records = [copy.deepcopy(sikkim), _stale_arunachal_record(), copy.deepcopy(assam)]
    out = sv.refresh_arunachal_data_status(
        records, evidence_dir=str(tmp_path / "none"), data_dir=str(tmp_path / "none"))
    assert out[0] == sikkim          # non-Arunachal record returned byte-identical
    assert out[2] == assam           # non-Arunachal record returned byte-identical
    assert out[1]["state_name"] == "Arunachal Pradesh"   # only Arunachal is refreshed


def test_refresh_arunachal_upgrades_from_real_artifacts(tmp_path):
    data_dir = tmp_path / "data"
    _write_arunachal_terrain_rasters(data_dir)
    ev_dir = tmp_path / "models"
    metrics = {"PR-AUC": 0.6013, "ROC-AUC": 0.7321, "F1": 0.548}  # fixture values
    _write_evidence(ev_dir, "arunachal_pradesh", metrics=metrics)

    records = [_stale_arunachal_record()]
    out = sv.refresh_arunachal_data_status(
        records, evidence_dir=str(ev_dir), data_dir=str(data_dir))
    rec = out[0]

    assert rec["dem_status"] == "Available"
    assert rec["model_status"] == "Trained & Validated"
    assert rec["validation_metrics"] == metrics          # flowed from the file
    assert rec["overall_status"] == "VALIDATED_PILOT"
    assert rec["validation_status"] == "VALIDATED_PILOT"

    # Exposure agrees with the existing check against the committed arunachal_pradesh_osm.geojson.
    expected_exposure = sv.evaluate_exposure_data(
        "Arunachal Pradesh", NER_STATES_CONFIG["Arunachal Pradesh"])
    assert rec["exposure_status"] == expected_exposure

    assert "Missing DEM Data" not in rec["blocking_reasons"]
    assert not any("Persisted Validation Evidence" in b for b in rec["blocking_reasons"])
    assert ("Missing OSM Exposure Data" in rec["blocking_reasons"]) == (
        expected_exposure != "Available")

    # The input record is copied, not mutated in place.
    assert records[0]["dem_status"] == "Missing (Requires Download)"
    assert records[0]["overall_status"] == "DATA UNAVAILABLE"


def test_refresh_arunachal_incomplete_evidence_is_not_validated(tmp_path):
    data_dir = tmp_path / "data"
    _write_arunachal_terrain_rasters(data_dir)
    ev_dir = tmp_path / "empty_models"
    os.makedirs(ev_dir, exist_ok=True)          # no arunachal_pradesh_* evidence files

    out = sv.refresh_arunachal_data_status(
        [_stale_arunachal_record()], evidence_dir=str(ev_dir), data_dir=str(data_dir))
    rec = out[0]

    assert rec["dem_status"] == "Available"
    assert rec["model_status"].startswith("Validation Required")
    assert rec["validation_metrics"] == {}
    assert rec["risk_result"] is None
    assert any("Persisted Validation Evidence" in b for b in rec["blocking_reasons"])

    expected_exposure = sv.evaluate_exposure_data(
        "Arunachal Pradesh", NER_STATES_CONFIG["Arunachal Pradesh"])
    if expected_exposure == "Available":
        assert rec["overall_status"] == "VALIDATION_REQUIRED"
    else:
        assert rec["overall_status"] == "DATA UNAVAILABLE"


def test_refresh_arunachal_missing_terrain_is_reported(tmp_path):
    data_dir = tmp_path / "data"
    os.makedirs(data_dir, exist_ok=True)        # no arunachal_pilot_* rasters
    ev_dir = tmp_path / "models"
    _write_evidence(ev_dir, "arunachal_pradesh", metrics={"PR-AUC": 0.6013, "ROC-AUC": 0.7321})

    out = sv.refresh_arunachal_data_status(
        [_stale_arunachal_record()], evidence_dir=str(ev_dir), data_dir=str(data_dir))
    rec = out[0]

    assert rec["dem_status"] == "Missing (Requires Download)"
    assert "Missing DEM Data" in rec["blocking_reasons"]
    # Real persisted model evidence still earns "Trained & Validated" (evidence-
    # gated, exactly like the Sikkim pilot contract) even though a live raster is
    # currently absent -- the DEM chip / blocker report that independently.
    assert rec["model_status"] == "Trained & Validated"
    assert rec["overall_status"] == "VALIDATED_PILOT"


# ---------------------------------------------------------------------------
# Serve-time Meghalaya refresh (refresh_meghalaya_data_status)
# ---------------------------------------------------------------------------
# Identical situation to Assam / Arunachal Pradesh: state_validation.json was written
# by an early NER sweep before any Meghalaya artifact existed, so its Meghalaya record
# reports dem/exposure "Missing" and model "Not Trained". reconcile_validation_report
# only downgrades unbacked VALIDATED_PILOT claims, and determine_overall_status only
# consults the model-evidence gate for is_pilot states (Meghalaya is is_pilot=False),
# so neither lifts the stale values. refresh_meghalaya_data_status closes that gap for
# MEGHALAYA ONLY, recomputing each field from a real on-disk artifact. These tests
# drive the DEM and model-evidence branches off TEMP directories (so they never depend
# on the checkout's large rasters); exposure is asserted to agree with the existing
# evaluate_exposure_data() check against the committed meghalaya_osm.geojson.

def _write_meghalaya_terrain_rasters(data_dir):
    """Create the five non-empty meghalaya_pilot_* raster files the refresh checks for.
    Placeholder bytes only -- the refresh checks existence + size, never opens them.
    Layout matches meghalaya_prediction.meghalaya_terrain_raster_paths: DEM in raw/,
    the four derivatives in processed/."""
    raw = os.path.join(str(data_dir), "raw")
    proc = os.path.join(str(data_dir), "processed")
    os.makedirs(raw, exist_ok=True)
    os.makedirs(proc, exist_ok=True)
    with open(os.path.join(raw, "meghalaya_pilot_dem.tif"), "wb") as fh:
        fh.write(b"NOT-A-REAL-RASTER-PLACEHOLDER")
    for name in ("slope", "aspect", "roughness", "tpi"):
        with open(os.path.join(proc, "meghalaya_pilot_%s.tif" % name), "wb") as fh:
            fh.write(b"NOT-A-REAL-RASTER-PLACEHOLDER")


def _stale_meghalaya_record():
    """A copy of the stale Meghalaya record shape persisted in state_validation.json."""
    return {
        "id": "meghalaya", "state_id": "meghalaya",
        "state": "Meghalaya", "state_name": "Meghalaya",
        "processing_status": "COMPLETED",
        "validation_status": "DATA UNAVAILABLE",
        "overall_status": "DATA UNAVAILABLE",
        "rainfall_source": "Open-Meteo / Fallback Synthetic",
        "rainfall_status": "Fallback Active (Open-Meteo / Local)",
        "inventory_events": 34, "usable_events": 34,
        "spatial_quality": "Moderate/Poor", "temporal_quality": "Good",
        "dem_status": "Missing (Requires Download)",
        "exposure_status": "Missing (Requires Download)",
        "model_status": "Not Trained",
        "validation_metrics": {}, "risk_result": None,
        "blocking_reasons": ["Missing DEM Data", "Missing OSM Exposure Data"],
        "error": None,
    }


def test_is_meghalaya_record_matches_by_id_or_name():
    assert sv._is_meghalaya_record({"state_id": "meghalaya"}) is True
    assert sv._is_meghalaya_record({"state_name": "Meghalaya"}) is True
    assert sv._is_meghalaya_record({"id": "MEGHALAYA"}) is True
    assert sv._is_meghalaya_record({"state": "  Meghalaya  "}) is True
    assert sv._is_meghalaya_record({"state_name": "Assam"}) is False
    assert sv._is_meghalaya_record({"state_name": "Arunachal Pradesh"}) is False
    assert sv._is_meghalaya_record({"state_name": "Sikkim"}) is False
    assert sv._is_meghalaya_record("not a dict") is False


def test_refresh_meghalaya_non_list_payload_returned_unchanged():
    assert sv.refresh_meghalaya_data_status({"not": "a list"}) == {"not": "a list"}
    assert sv.refresh_meghalaya_data_status(None) is None


def test_refresh_leaves_non_meghalaya_records_untouched(tmp_path):
    import copy
    sikkim = {"id": "sikkim", "state_name": "Sikkim", "overall_status": "VALIDATED_PILOT",
              "dem_status": "Available", "model_status": "Trained & Validated"}
    arunachal = {"id": "arunachal_pradesh", "state_name": "Arunachal Pradesh",
                 "overall_status": "VALIDATED_PILOT", "dem_status": "Available"}
    records = [copy.deepcopy(sikkim), _stale_meghalaya_record(), copy.deepcopy(arunachal)]
    out = sv.refresh_meghalaya_data_status(
        records, evidence_dir=str(tmp_path / "none"), data_dir=str(tmp_path / "none"))
    assert out[0] == sikkim          # non-Meghalaya record returned byte-identical
    assert out[2] == arunachal       # non-Meghalaya record returned byte-identical
    assert out[1]["state_name"] == "Meghalaya"   # only Meghalaya is refreshed


def test_refresh_meghalaya_upgrades_from_real_artifacts(tmp_path):
    data_dir = tmp_path / "data"
    _write_meghalaya_terrain_rasters(data_dir)
    ev_dir = tmp_path / "models"
    metrics = {"PR-AUC": 0.3129, "ROC-AUC": 0.6188, "F1": 0.401}  # fixture values
    _write_evidence(ev_dir, "meghalaya", metrics=metrics)

    records = [_stale_meghalaya_record()]
    out = sv.refresh_meghalaya_data_status(
        records, evidence_dir=str(ev_dir), data_dir=str(data_dir))
    rec = out[0]

    assert rec["dem_status"] == "Available"
    assert rec["model_status"] == "Trained & Validated"
    assert rec["validation_metrics"] == metrics          # flowed from the file
    assert rec["overall_status"] == "VALIDATED_PILOT"
    assert rec["validation_status"] == "VALIDATED_PILOT"

    # Exposure agrees with the existing check against the committed meghalaya_osm.geojson.
    expected_exposure = sv.evaluate_exposure_data(
        "Meghalaya", NER_STATES_CONFIG["Meghalaya"])
    assert rec["exposure_status"] == expected_exposure

    assert "Missing DEM Data" not in rec["blocking_reasons"]
    assert not any("Persisted Validation Evidence" in b for b in rec["blocking_reasons"])
    assert ("Missing OSM Exposure Data" in rec["blocking_reasons"]) == (
        expected_exposure != "Available")

    # The input record is copied, not mutated in place.
    assert records[0]["dem_status"] == "Missing (Requires Download)"
    assert records[0]["overall_status"] == "DATA UNAVAILABLE"


def test_refresh_meghalaya_incomplete_evidence_is_not_validated(tmp_path):
    data_dir = tmp_path / "data"
    _write_meghalaya_terrain_rasters(data_dir)
    ev_dir = tmp_path / "empty_models"
    os.makedirs(ev_dir, exist_ok=True)          # no meghalaya_* evidence files

    out = sv.refresh_meghalaya_data_status(
        [_stale_meghalaya_record()], evidence_dir=str(ev_dir), data_dir=str(data_dir))
    rec = out[0]

    assert rec["dem_status"] == "Available"
    assert rec["model_status"].startswith("Validation Required")
    assert rec["validation_metrics"] == {}
    assert rec["risk_result"] is None
    assert any("Persisted Validation Evidence" in b for b in rec["blocking_reasons"])

    expected_exposure = sv.evaluate_exposure_data(
        "Meghalaya", NER_STATES_CONFIG["Meghalaya"])
    if expected_exposure == "Available":
        assert rec["overall_status"] == "VALIDATION_REQUIRED"
    else:
        assert rec["overall_status"] == "DATA UNAVAILABLE"


def test_refresh_meghalaya_missing_terrain_is_reported(tmp_path):
    data_dir = tmp_path / "data"
    os.makedirs(data_dir, exist_ok=True)        # no meghalaya_pilot_* rasters
    ev_dir = tmp_path / "models"
    _write_evidence(ev_dir, "meghalaya", metrics={"PR-AUC": 0.3129, "ROC-AUC": 0.6188})

    out = sv.refresh_meghalaya_data_status(
        [_stale_meghalaya_record()], evidence_dir=str(ev_dir), data_dir=str(data_dir))
    rec = out[0]

    assert rec["dem_status"] == "Missing (Requires Download)"
    assert "Missing DEM Data" in rec["blocking_reasons"]
    # Real persisted model evidence still earns "Trained & Validated" (evidence-
    # gated, exactly like the Sikkim pilot contract) even though a live raster is
    # currently absent -- the DEM chip / blocker report that independently.
    assert rec["model_status"] == "Trained & Validated"
    assert rec["overall_status"] == "VALIDATED_PILOT"


# ---------------------------------------------------------------------------
# Regression: evaluate_terrain_data resolves the correct DEM filename per state
# ---------------------------------------------------------------------------
# The pilot DEMs are persisted as "<state>_pilot_dem.tif" (collision-free), while
# the generic evaluator historically looked for "<clean_state>_dem.tif". These
# tests pin the filename that gets probed on disk without requiring any real
# .tif files: os.path.exists / os.path.getsize are stubbed to answer only for the
# expected basename, so a wrong lookup would report "Missing".

def _terrain_status_for(state_name, expected_basename, monkeypatch):
    seen = {}

    def fake_exists(path):
        seen["basename"] = os.path.basename(path)
        return os.path.basename(path) == expected_basename

    def fake_getsize(path):
        return 5000 if os.path.basename(path) == expected_basename else 0

    monkeypatch.setattr(sv.os.path, "exists", fake_exists)
    monkeypatch.setattr(sv.os.path, "getsize", fake_getsize)
    status = sv.evaluate_terrain_data(state_name, {})
    return status, seen.get("basename")


def test_evaluate_terrain_data_uses_pilot_dem_names(monkeypatch):
    for state, expected in [
        ("Arunachal Pradesh", "arunachal_pilot_dem.tif"),
        ("Assam", "assam_pilot_dem.tif"),
        ("Meghalaya", "meghalaya_pilot_dem.tif"),
    ]:
        status, basename = _terrain_status_for(state, expected, monkeypatch)
        assert basename == expected
        assert status == "Available"


def test_evaluate_terrain_data_sikkim_keeps_generic_name(monkeypatch):
    status, basename = _terrain_status_for("Sikkim", "sikkim_dem.tif", monkeypatch)
    assert basename == "sikkim_dem.tif"
    assert status == "Available"


def test_evaluate_terrain_data_nonpilot_uses_generic_name(monkeypatch):
    status, basename = _terrain_status_for("Manipur", "manipur_dem.tif", monkeypatch)
    assert basename == "manipur_dem.tif"
    assert status == "Available"


def test_evaluate_terrain_data_missing_file_is_not_available(monkeypatch):
    # No file matches -> generic + pilot lookups both fail -> honestly "Missing",
    # never hard-coded Available.
    monkeypatch.setattr(sv.os.path, "exists", lambda path: False)
    monkeypatch.setattr(sv.os.path, "getsize", lambda path: 0)
    assert sv.evaluate_terrain_data("Meghalaya", {}) == "Missing (Requires Download)"


def test_evaluate_terrain_data_size_guard_rejects_tiny_file(monkeypatch):
    # A present but truncated (<=1000 byte) placeholder must not read as Available.
    monkeypatch.setattr(sv.os.path, "exists", lambda path: True)
    monkeypatch.setattr(sv.os.path, "getsize", lambda path: 500)
    assert sv.evaluate_terrain_data("Assam", {}) == "Missing (Requires Download)"


