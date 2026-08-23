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
