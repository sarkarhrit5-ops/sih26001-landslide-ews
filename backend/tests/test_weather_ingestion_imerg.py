import os
import sys
import pytest
from datetime import datetime
from unittest.mock import patch, MagicMock

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.services.weather_ingestion import (
    get_earthdata_session,
    fetch_imerg_precipitation,
    get_imerg_indices,
    _mean_valid_precipitation,
    _accumulate_forecast_precipitation,
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
        "precipitation": (["time", "lon", "lat"], [[[15.0, 25.0], [5.0, 15.0]]])
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


# ---------------------------------------------------------------------------
# Phase 2F: missing IMERG / forecast data must not become a fabricated 0 mm
# ---------------------------------------------------------------------------
# These exercise the pure precipitation-reduction helpers directly, so they need
# neither the network, Earthdata credentials nor a real NetCDF file -- only numpy.
# Previously `_fetch_imerg_day` did `max(0.0, mean_precip)`, which turned an
# all-NaN or all-fill-sentinel subset (and any negative fill anomaly) into a
# confident 0 mm "measurement".

def test_mean_valid_precipitation_ignores_fill_sentinels():
    # Two cells carry the ~-9999.9 fill sentinel; two are real.
    values = np.array([[-9999.9, 2.0], [4.0, -9999.9]])
    # Mean is over the real cells (2.0, 4.0) = 3.0, NOT dragged negative by the
    # sentinels and clamped up to 0.0.
    assert _mean_valid_precipitation(values) == 3.0

def test_mean_valid_precipitation_averages_real_cells():
    assert _mean_valid_precipitation(np.array([10.0, 20.0, 30.0])) == 20.0

def test_mean_valid_precipitation_all_fill_raises_not_zero():
    with pytest.raises(ValueError):
        _mean_valid_precipitation(np.array([-9999.9, -9999.9, -9999.9]))

def test_mean_valid_precipitation_all_nan_raises_not_zero():
    with pytest.raises(ValueError):
        _mean_valid_precipitation(np.array([np.nan, np.nan]))

def test_mean_valid_precipitation_clamps_only_tiny_negative_noise():
    # A genuinely-valid near-zero cell that dipped slightly negative from numerical
    # noise is clamped up to exactly 0.0 -- the one legitimate use of the clamp.
    assert _mean_valid_precipitation(np.array([-1e-9])) == 0.0

def test_mean_valid_precipitation_real_dry_subset_is_preserved():
    # An honest dry day (all real 0.0 cells) is a measurement, not no-data.
    assert _mean_valid_precipitation(np.array([0.0, 0.0, 0.0])) == 0.0

def test_accumulate_forecast_precipitation_sums_finite_hours():
    assert _accumulate_forecast_precipitation([1.0, 2.0, 3.0]) == 6.0

def test_accumulate_forecast_precipitation_empty_window_raises():
    with pytest.raises(ValueError):
        _accumulate_forecast_precipitation([])

def test_accumulate_forecast_precipitation_all_null_raises_not_zero():
    # Open-Meteo may return null for a gap; an ALL-null window must not sum to a
    # confident 0 mm forecast.
    with pytest.raises(ValueError):
        _accumulate_forecast_precipitation([np.nan, np.nan])

def test_accumulate_forecast_precipitation_interior_gap_contributes_zero():
    # A single interior gap is tolerated (contributes nothing) as long as the
    # window still has a real value.
    assert _accumulate_forecast_precipitation([1.0, np.nan, 2.0]) == 3.0

@patch("requests.Session.get")
@patch.dict(os.environ, {"EARTHDATA_TOKEN": "fake_token_for_test_only"})
def test_earthdata_session_sends_bearer_token(mock_get):
    """The token session must carry Authorization: Bearer <token> (env-sourced)."""
    mock_get.return_value = MagicMock(status_code=200)
    session = get_earthdata_session()
    assert session.headers.get("Authorization") == "Bearer fake_token_for_test_only"


def test_earthdata_session_preserves_bearer_across_nasa_redirect():
    """
    GES DISC 302-redirects through urs.earthdata.nasa.gov; requests would strip the
    Authorization header on that cross-host hop and 401. The session must KEEP the
    bearer for NASA Earthdata/EOSDIS hosts and still DROP it for any other host.
    """
    from app.services.weather_ingestion import _EarthdataAuthSession
    session = _EarthdataAuthSession()
    gesdisc = "https://gpm1.gesdisc.eosdis.nasa.gov/opendap/GPM_L3/x.nc4"
    urs = "https://urs.earthdata.nasa.gov/oauth/authorize"
    other = "https://not-nasa.example.com/collect"
    # NASA -> NASA (both directions): bearer preserved
    assert session.should_strip_auth(gesdisc, urs) is False
    assert session.should_strip_auth(urs, gesdisc) is False
    # NASA -> unrelated host: bearer stripped (security default preserved)
    assert session.should_strip_auth(gesdisc, other) is True
    # a look-alike host must NOT be trusted
    assert _EarthdataAuthSession._is_trusted("evil-eosdis.nasa.gov") is False
    assert _EarthdataAuthSession._is_trusted("gpm1.gesdisc.eosdis.nasa.gov") is True
