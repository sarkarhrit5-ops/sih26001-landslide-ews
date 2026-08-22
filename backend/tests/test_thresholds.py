import os
import sys
import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.models.thresholds import (
    calculate_critical_intensity,
    calculate_critical_accumulation,
    evaluate_rainfall_trigger,
    THRESHOLD_METADATA
)

def test_threshold_metadata():
    assert "East Sikkim pilot-derived empirical threshold" in THRESHOLD_METADATA["name"]
    assert THRESHOLD_METADATA["units"]["intensity"] == "mm/hour"
    assert THRESHOLD_METADATA["events_count"] == 82
    assert THRESHOLD_METADATA["is_universal"] is False

def test_critical_intensity_calculation():
    # I = 14.2 * D^(-0.62)
    # For D = 1h: I_crit = 14.2 mm/h
    i1 = calculate_critical_intensity(1.0)
    assert i1 == 14.2
    
    # For D = 24h: I_crit = 14.2 * 24^(-0.62) ≈ 1.977 mm/h
    i24 = calculate_critical_intensity(24.0)
    assert 1.5 < i24 < 2.5

def test_critical_accumulation_calculation():
    p24 = calculate_critical_accumulation(24.0)
    assert p24 > 30.0

def test_evaluate_rainfall_trigger_exceeded():
    res = evaluate_rainfall_trigger(current_rain_mm=100.0, duration_hours=24.0)
    assert res["trigger_exceeded"] is True
    assert res["trigger_score"] > 0.5
    assert res["status"] == "valid"

def test_evaluate_rainfall_trigger_below():
    res = evaluate_rainfall_trigger(current_rain_mm=5.0, duration_hours=24.0)
    assert res["trigger_exceeded"] is False
    assert res["trigger_score"] < 0.3
    assert res["status"] == "valid"

def test_missing_rainfall_data():
    res = evaluate_rainfall_trigger(current_rain_mm=None, duration_hours=24.0)
    assert res["trigger_exceeded"] is False
    assert res["trigger_score"] == 0.0
    assert res["status"] == "missing_data"
