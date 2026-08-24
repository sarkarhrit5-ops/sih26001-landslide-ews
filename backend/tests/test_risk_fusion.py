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


# ---------------------------------------------------------------------------
# Fail-closed defaults and unmeasured inputs (Phase 2E-4)
# ---------------------------------------------------------------------------
def test_omitted_provenance_flags_cannot_yield_high_confidence():
    # Previously exposure_score defaulted to 0.5 and both provenance flags
    # defaulted to True, so a caller that passed nothing still got a
    # HIGH-confidence answer built on an assumed real DEM and real rainfall.
    risk = dynamic_risk_module(
        susceptibility_score=0.70,
        current_rainfall_mm=60.0,
        forecast_rainfall_mm=120.0,
        slope_deg=38.0,
    )
    assert risk["confidence"] == "LOW"
    assert risk["inputs_are_real"] == {"dem": False, "rainfall": False}
    assert risk["exposure_score"] is None
    assert risk["exposure_score_status"] == "UNAVAILABLE"


def test_unmeasured_exposure_is_null_and_does_not_move_the_score():
    without = dynamic_risk_module(0.70, 60.0, 120.0, 38.0, exposure_score=None)
    with_exposure = dynamic_risk_module(0.70, 60.0, 120.0, 38.0, exposure_score=0.6)
    assert without["exposure_score"] is None
    assert with_exposure["exposure_score"] == 0.6
    assert with_exposure["exposure_score_status"] == "REAL"
    # Exposure is reported alongside the hazard, not folded into it; this is why
    # an UNAVAILABLE exposure is allowed to be non-blocking upstream.
    assert without["final_risk_score"] == with_exposure["final_risk_score"]


def test_unevaluated_forecast_is_null_not_a_measured_zero():
    risk = dynamic_risk_module(
        susceptibility_score=0.70,
        current_rainfall_mm=60.0,
        forecast_rainfall_mm=None,
        slope_deg=38.0,
    )
    assert risk["forecast_evaluated"] is False
    assert risk["forecast_trigger_score"] is None, (
        "a forecast that was never requested must not be scored 0.0"
    )
    assert risk["trigger_details"]["forecast"]["status"] == "missing_data"
    assert risk["trigger_details"]["forecast"]["trigger_exceeded"] is False


def test_passing_no_forecast_is_arithmetically_identical_to_zero_mm():
    """
    /risk/current now passes forecast_rainfall_mm=None instead of the old
    fabricated 0.0. This must change the reporting only, never the hazard.
    """
    unevaluated = dynamic_risk_module(0.70, 60.0, None, 38.0)
    zero_mm = dynamic_risk_module(0.70, 60.0, 0.0, 38.0)
    for key in ("susceptibility_score", "current_trigger_score",
                "final_risk_score", "warning_level", "confidence"):
        assert unevaluated[key] == zero_mm[key]
    assert unevaluated["forecast_trigger_score"] is None
    assert zero_mm["forecast_trigger_score"] == 0.0
    assert zero_mm["forecast_evaluated"] is True
