import os
import sys
import psutil
from datetime import datetime, timedelta

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.services.weather_ingestion import fetch_imerg_precipitation
from app.core.config_states import get_pilot_aoi_bounds

def print_ram(stage):
    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss / (1024 * 1024)
    print(f"[{stage}] RAM Usage: {mem:.2f} MB")
    return mem

def run_smoke_test():
    print("==================================================")
    print("      NASA IMERG AUTHENTICATED SMOKE TEST         ")
    print("==================================================")
    
    # 1. Check for credentials silently
    username = os.environ.get("EARTHDATA_USERNAME")
    password = os.environ.get("EARTHDATA_PASSWORD")
    token = os.environ.get("EARTHDATA_TOKEN")
    
    auth_method = "None"
    if token:
        auth_method = "EARTHDATA_TOKEN"
    elif username and password:
        auth_method = "EARTHDATA_USERNAME / PASSWORD"
        
    print(f"Product Selected: IMERG Early (GPM_3IMERGDE.07)")
    print(f"Access Method: OPeNDAP Subsetting (gpm1.gesdisc.eosdis.nasa.gov)")
    print(f"Authentication Detected: {auth_method}")
    
    if auth_method == "None":
        print("\nBLOCKER: No Earthdata credentials found in environment.")
        print("Skipping actual data download to prevent authentication failure.")
        print("Set EARTHDATA_TOKEN or EARTHDATA_USERNAME/PASSWORD to test real data.")
        return
        
    print("\n--- Running Real Earthdata Subsetting ---")
    peak_ram = []
    peak_ram.append(print_ram("Pre-fetch"))
    
    # Target: the canonical East Sikkim pilot AOI (not restated here)
    bounds = get_pilot_aoi_bounds("Sikkim")
    # Pick a recent date
    test_date = datetime.now() - timedelta(days=7) 
    
    try:
        print(f"Initiating fetch for {test_date.strftime('%Y-%m-%d')} with 1/3/7 day accumulations...")
        res = fetch_imerg_precipitation(bounds, test_date, run_type="Early", windows=[1, 3, 7])
        
        peak_ram.append(print_ram("Post-fetch"))
        
        print("\nSUCCESS: Data fetched and parsed successfully.")
        print("Results:")
        for k, v in res["accumulations"].items():
            print(f"  {k}: {v} mm")
            
        print(f"\nPeak RAM Usage: {max(peak_ram):.2f} MB (Constraint: 8000 MB)")
    except Exception as e:
        print(f"\nERROR during smoke test: {e}")

if __name__ == "__main__":
    run_smoke_test()
