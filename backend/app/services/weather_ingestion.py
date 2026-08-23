import requests
import pandas as pd
from datetime import datetime, timedelta

import os
import requests
import pandas as pd
from datetime import datetime, timedelta

def get_earthdata_session():
    """
    Creates an authenticated requests.Session for NASA Earthdata (urs.earthdata.nasa.gov / GES DISC).
    Raises PermissionError if env vars are missing or if auth fails.
    """
    username = os.environ.get("EARTHDATA_USERNAME")
    password = os.environ.get("EARTHDATA_PASSWORD")
    token = os.environ.get("EARTHDATA_TOKEN")

    if not (token or (username and password)):
        raise PermissionError(
            "BLOCKER: Missing NASA Earthdata credentials! "
            "Neither EARTHDATA_TOKEN nor (EARTHDATA_USERNAME and EARTHDATA_PASSWORD) environment variables are set. "
            "Cannot authenticate with urs.earthdata.nasa.gov to access real IMERG rainfall data."
        )
    
    session = requests.Session()
    if token:
        session.headers.update({"Authorization": f"Bearer {token}"})
    else:
        session.auth = (username, password)
    
    # Test authentication against NASA Earthdata URS endpoint
    auth_test_url = "https://urs.earthdata.nasa.gov/api/v2/users/tokens"
    try:
        if token:
            resp = session.get(auth_test_url, timeout=10)
        else:
            resp = session.post(auth_test_url, auth=(username, password), timeout=10)
        
        if resp.status_code in [401, 403]:
            raise PermissionError(
                f"EARTHDATA AUTHENTICATION FAILED (HTTP {resp.status_code}): Invalid username or password for urs.earthdata.nasa.gov."
            )
    except requests.RequestException as req_err:
        # If network or endpoint fails, raise exact exception
        if isinstance(req_err, requests.HTTPError) and req_err.response is not None and req_err.response.status_code in [401, 403]:
            raise PermissionError(f"EARTHDATA AUTHENTICATION FAILED: {req_err}")
        # Note: urs.earthdata.nasa.gov token post may return 400 if user doesn't use tokens, try direct GES DISC probe
        try:
            probe_resp = session.get("https://gpm1.gesdisc.eosdis.nasa.gov/data/GPM_L3/GPM_3IMERGDF.07/", timeout=10)
            if probe_resp.status_code in [401, 403]:
                raise PermissionError(f"EARTHDATA AUTHENTICATION FAILED (HTTP {probe_resp.status_code}) on GES DISC.")
        except Exception as probe_err:
            raise PermissionError(f"EARTHDATA ACCESS FAILED: {req_err}")

    return session

def fetch_open_meteo_forecast(lat: float, lon: float, hours: int = 72):
    """
    Fetches hourly precipitation forecast from Open-Meteo.
    """
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "precipitation",
        "forecast_days": min(14, max(1, hours // 24)),
        "timezone": "auto"
    }
    
    response = requests.get(url, params=params, timeout=15)
    response.raise_for_status()
    data = response.json()
    
    df = pd.DataFrame({
        "time": pd.to_datetime(data["hourly"]["time"]),
        "precipitation_mm": data["hourly"]["precipitation"]
    })
    
    # Calculate cumulative forecast
    future_precip = df[df["time"] > pd.Timestamp.now(tz=df["time"].dt.tz)]
    cumulative = future_precip.head(hours)["precipitation_mm"].sum()
    
    return cumulative

import xarray as xr
import tempfile

def get_imerg_indices(bounds: dict):
    """
    Converts lat/lon bounds to IMERG grid indices.
    IMERG grid: lat is -89.95 to 89.95 (1800 cells), lon is -179.95 to 179.95 (3600 cells).
    """
    lat_min_idx = max(0, int(round((bounds['min_lat'] + 89.95) * 10)))
    lat_max_idx = min(1799, int(round((bounds['max_lat'] + 89.95) * 10)))
    lon_min_idx = max(0, int(round((bounds['min_lon'] + 179.95) * 10)))
    lon_max_idx = min(3599, int(round((bounds['max_lon'] + 179.95) * 10)))
    return lat_min_idx, lat_max_idx, lon_min_idx, lon_max_idx

def _fetch_imerg_day(session: requests.Session, date: datetime, bounds: dict, run_type: str = "Early") -> float:
    """
    Fetches a single day's IMERG precipitation for a specific bounding box using OPeNDAP.
    Returns the mean precipitation in mm for the bounding box.
    """
    year = date.strftime("%Y")
    month = date.strftime("%m")
    day = date.strftime("%d")
    
    if run_type.lower() == "final":
        product = "GPM_3IMERGDF.07"
        filename = f"3B-DAY.MS.MRG.3IMERG.{year}{month}{day}-S000000-E235959.V07B.nc4"
    elif run_type.lower() == "late":
        product = "GPM_3IMERGDL.07"
        filename = f"3B-DAY-L.MS.MRG.3IMERG.{year}{month}{day}-S000000-E235959.V07B.nc4"
    else: # Early
        product = "GPM_3IMERGDE.07"
        filename = f"3B-DAY-E.MS.MRG.3IMERG.{year}{month}{day}-S000000-E235959.V07B.nc4"
        
    base_url = f"https://gpm1.gesdisc.eosdis.nasa.gov/opendap/GPM_L3/{product}/{year}/{month}/{filename}"
    
    lat_min, lat_max, lon_min, lon_max = get_imerg_indices(bounds)
    
    # OPeNDAP constraint query: var[time][lon][lat]
    # IMERG dimensions: time (1), lon (3600), lat (1800)
    query = f"?precipitationCal[0:0][{lon_min}:{lon_max}][{lat_min}:{lat_max}]"
    url = f"{base_url}.nc4{query}"
    
    try:
        resp = session.get(url, timeout=30)
        if resp.status_code in [401, 403]:
            raise PermissionError(f"EARTHDATA AUTHENTICATION REJECTED (HTTP {resp.status_code}) for URL {url}")
        resp.raise_for_status()
        
        # Save tiny subset to a temporary file for xarray parsing to keep RAM usage low
        with tempfile.NamedTemporaryFile(suffix=".nc4", delete=False) as tmp:
            tmp.write(resp.content)
            tmp_path = tmp.name
            
        try:
            # Parse NetCDF4 using xarray
            with xr.open_dataset(tmp_path, engine="h5netcdf") as ds:
                # Extract mean precipitation across the requested bounding box
                mean_precip = float(ds['precipitationCal'].mean().values)
        finally:
            os.remove(tmp_path)
            
        return max(0.0, mean_precip) # Ensure no negative values from fill data anomalies
        
    except requests.RequestException as e:
        raise RuntimeError(f"EARTHDATA IMERG FETCH FAILED for {date.strftime('%Y-%m-%d')} ({run_type}): {str(e)}")

def fetch_imerg_precipitation(bounds: dict, date: datetime, run_type="Early", windows=[1, 3, 7]):
    """
    Fetches NASA GPM IMERG satellite rainfall data via Earthdata OPeNDAP subsetting.
    Supports 1, 3, and 7-day accumulations.
    Maintains 8GB RAM constraint by utilizing spatial subsetting on the server side.
    
    Requires Earthdata authentication via environment variables (EARTHDATA_USERNAME & EARTHDATA_PASSWORD or EARTHDATA_TOKEN).
    If authentication fails, reports the exact error and halts (does NOT substitute synthetic data).
    """
    session = get_earthdata_session()
    
    max_days = max(windows)
    daily_precip = {}
    
    # Fetch historical daily data up to the maximum window
    for d in range(max_days):
        target_date = date - timedelta(days=d)
        daily_precip[d] = _fetch_imerg_day(session, target_date, bounds, run_type)
        
    # Aggregate according to requested windows
    results = {}
    for w in windows:
        accum = sum(daily_precip[d] for d in range(w))
        results[f"accumulation_{w}d_mm"] = round(accum, 4)
        
    return {
        "source": f"IMERG_{run_type}",
        "target_date": date.strftime("%Y-%m-%d"),
        "status": "success",
        "accumulations": results,
        "spatial_bounds": bounds
    }

if __name__ == "__main__":
    forecast = fetch_open_meteo_forecast(27.3314, 88.6138, 24)
    print(f"24h Forecast Precipitation: {forecast} mm")

