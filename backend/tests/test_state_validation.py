import os
import sys
import pandas as pd
import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.core.config_states import NER_STATES_CONFIG
from app.services.state_validation import (
    evaluate_landslide_inventory,
    determine_overall_status
)

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

def test_determine_overall_status_validated():
    inventory = {"usable_events": 100}
    # East Sikkim is pilot, should be VALIDATED regardless of raw checks here
    status = determine_overall_status("Sikkim", inventory, "Available", "Authenticated", "Available", True)
    assert status["overall_status"] == "VALIDATED"
    assert "PR-AUC" in status["validation_metrics"]

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
