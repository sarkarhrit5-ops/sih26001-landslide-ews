import os
import sys
import pytest
from datetime import datetime
from unittest.mock import patch, MagicMock

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.services.weather_ingestion import (
    get_earthdata_session,
    fetch_imerg_precipitation,
    get_imerg_indices
)
import requests
import xarray as xr
import numpy as np
import tempfile

def test_get_imerg_indices():
    bounds = {"min_lat": 27.0, "max_lat": 28.2, "min_lon": 88.0, "max_lon": 89.0}
    lat_min, lat_max, lon_min, lon_max = get_imerg_indices(bounds)
    
    assert lat_min == 1170
    assert lat_max == 1182
    assert lon_min == 2680
    assert lon_max == 2690

@patch.dict(os.environ, {}, clear=True)
def test_earthdata_missing_credentials():
    with pytest.raises(PermissionError) as excinfo:
        get_earthdata_session()
    assert "Missing NASA Earthdata credentials" in str(excinfo.value)

@patch("requests.Session.get")
@patch.dict(os.environ, {"EARTHDATA_TOKEN": "mock_token"})
def test_earthdata_invalid_credentials(mock_get):
    mock_resp = MagicMock()
    mock_resp.status_code = 401
    mock_get.return_value = mock_resp
    
    with pytest.raises(PermissionError) as excinfo:
        get_earthdata_session()
    assert "AUTHENTICATION FAILED" in str(excinfo.value)

@patch("app.services.weather_ingestion.get_earthdata_session")
@patch("app.services.weather_ingestion._fetch_imerg_day")
def test_fetch_imerg_precipitation_aggregations(mock_fetch_day, mock_session):
    # Mock daily rainfall: day 0: 10mm, day 1: 5mm, day 2: 2mm, day 3-6: 1mm
    def mock_rainfall(session, date, bounds, run_type):
        d = (datetime(2023, 6, 1) - date).days
        if d == 0: return 10.0
        if d == 1: return 5.0
        if d == 2: return 2.0
        return 1.0

    mock_fetch_day.side_effect = mock_rainfall
    
    bounds = {"min_lat": 27.0, "max_lat": 28.2, "min_lon": 88.0, "max_lon": 89.0}
    res = fetch_imerg_precipitation(bounds, datetime(2023, 6, 1), windows=[1, 3, 7])
    
    assert res["status"] == "success"
    # 1d = day0 = 10
    assert res["accumulations"]["accumulation_1d_mm"] == 10.0
    # 3d = day0 + day1 + day2 = 10 + 5 + 2 = 17
    assert res["accumulations"]["accumulation_3d_mm"] == 17.0
    # 7d = 17 + (4 days * 1) = 21
    assert res["accumulations"]["accumulation_7d_mm"] == 21.0

@patch("requests.Session.get")
@patch("app.services.weather_ingestion.get_earthdata_session")
def test_fetch_imerg_day_parsing(mock_session_func, mock_get):
    """
    Test that _fetch_imerg_day correctly parses a valid NetCDF file snippet via xarray.
    """
    # Create a dummy netcdf file in memory using xarray
    ds = xr.Dataset({
        "precipitationCal": (["time", "lon", "lat"], [[[15.0, 25.0], [5.0, 15.0]]])
    })
    
    with tempfile.NamedTemporaryFile(suffix=".nc4", delete=False) as tmp:
        ds.to_netcdf(tmp.name, engine="h5netcdf")
        with open(tmp.name, "rb") as f:
            content = f.read()
    os.remove(tmp.name)
    
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.content = content
    
    mock_session = MagicMock()
    mock_session.get.return_value = mock_resp
    mock_session_func.return_value = mock_session
    
    from app.services.weather_ingestion import _fetch_imerg_day
    bounds = {"min_lat": 27.0, "max_lat": 28.2, "min_lon": 88.0, "max_lon": 89.0}
    
    # average of 15, 25, 5, 15 is 15.0
    val = _fetch_imerg_day(mock_session, datetime(2023, 6, 1), bounds)
    assert val == 15.0

def test_fetch_imerg_precipitation_missing_credentials_raises():
    with patch.dict(os.environ, {}, clear=True):
        bounds = {"min_lat": 27.0, "max_lat": 28.2, "min_lon": 88.0, "max_lon": 89.0}
        with pytest.raises(PermissionError):
            fetch_imerg_precipitation(bounds, datetime(2023, 6, 1))
