import os
import json
import sys
import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.models.ml_pipeline import (
    dynamic_risk_module,
    calculate_warning_level,
    STATIC_MODEL_METADATA,
    DOCUMENTARY_REFERENCE_METRICS
)

def test_warning_level_classification():
    assert calculate_warning_level(0.90) == "EXTREME"
    assert calculate_warning_level(0.75) == "HIGH"
    assert calculate_warning_level(0.50) == "MEDIUM"
    assert calculate_warning_level(0.20) == "LOW"

def test_structured_risk_fusion_output():
    risk = dynamic_risk_module(
        susceptibility_score=0.70,
        current_rainfall_mm=60.0,
        forecast_rainfall_mm=120.0,
        slope_deg=38.0,
        exposure_score=0.6,
        has_real_dem=True,
        has_real_rainfall=True
    )
    
    # Verify uncollapsed separate output fields
    assert "susceptibility_score" in risk
    assert "current_trigger_score" in risk
    assert "forecast_trigger_score" in risk
    assert "exposure_score" in risk
    assert "final_risk_score" in risk
    assert "warning_level" in risk
    assert "confidence" in risk
    
    assert risk["susceptibility_score"] == 0.70
    assert risk["exposure_score"] == 0.6
    assert risk["warning_level"] in ["HIGH", "EXTREME"]
    assert risk["confidence"] == "HIGH"

def test_static_model_metadata():
    assert "LightGBMClassifier" in STATIC_MODEL_METADATA["model_types"]
    assert "elevation" in STATIC_MODEL_METADATA["features_used"]
    assert "relative terrain susceptibility index" in STATIC_MODEL_METADATA["calibration_note"]
def test_run_metrics_separated_from_descriptive_metadata():
    # Descriptive metadata must NOT carry run-derived performance numbers, so it
    # can never be mistaken for a current validation result.
    assert "temporal_holdout_metrics" not in STATIC_MODEL_METADATA
    assert "PR-AUC" not in json.dumps(STATIC_MODEL_METADATA)
    # The historical figures are preserved (not deleted) but quarantined as
    # clearly-labelled documentary reference, not current evidence.
    assert "temporal_holdout_metrics" in DOCUMENTARY_REFERENCE_METRICS
    assert "provenance" in DOCUMENTARY_REFERENCE_METRICS
    lgb = DOCUMENTARY_REFERENCE_METRICS["temporal_holdout_metrics"]["LightGBM"]
    assert "PR-AUC" in lgb and "ROC-AUC" in lgb  # numbers not lost
