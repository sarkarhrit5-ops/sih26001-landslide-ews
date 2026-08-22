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

def fetch_imerg_precipitation(bounds: dict, date: datetime, run_type="Early"):
    """
    Fetches NASA GPM IMERG satellite rainfall data via Earthdata OPeNDAP or GES DISC HTTPS.
    Supports:
    - IMERG Final (run_type="Final") for historical research/training
    - IMERG Early/Late (run_type="Early" or "Late") for near-real-time functionality

    Requires Earthdata authentication via environment variables (EARTHDATA_USERNAME & EARTHDATA_PASSWORD or EARTHDATA_TOKEN).
    If authentication fails, reports the exact error and halts (does NOT substitute synthetic data).
    """
    # 1. Authenticate with NASA Earthdata
    session = get_earthdata_session()
    
    print(f"Fetching NASA GPM IMERG {run_type} run for {date.strftime('%Y-%m-%d')}...")
    
    # Construct IMERG GES DISC URL based on run_type
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
        
    url = f"https://gpm1.gesdisc.eosdis.nasa.gov/data/GPM_L3/{product}/{year}/{month}/{filename}"
    
    try:
        resp = session.get(url, timeout=30)
        if resp.status_code == 401 or resp.status_code == 403:
            raise PermissionError(f"EARTHDATA AUTHENTICATION REJECTED (HTTP {resp.status_code}) for URL {url}")
        resp.raise_for_status()
        
        # Parse NC4 file data if fetched successfully
        return {
            "source": f"IMERG_{run_type}",
            "date": date,
            "status": "success",
            "url": url,
            "bytes_downloaded": len(resp.content)
        }
    except Exception as e:
        raise RuntimeError(f"EARTHDATA IMERG FETCH FAILED for {date.strftime('%Y-%m-%d')} ({run_type}): {str(e)}")

if __name__ == "__main__":
    forecast = fetch_open_meteo_forecast(27.3314, 88.6138, 24) # Gangtok
    print(f"24h Forecast Precipitation: {forecast} mm")

