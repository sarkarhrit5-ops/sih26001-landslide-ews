import argparse
import os
import sys
import psutil
from datetime import datetime

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.services.weather_ingestion import fetch_imerg_precipitation
from app.core.config_states import get_pilot_aoi_bounds

def print_ram(stage):
    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss / (1024 * 1024)
    print(f"[{stage}] RAM Usage: {mem:.2f} MB")
    return mem

DEFAULT_PROBE_DATE = "2025-09-18"  # known-available in-range IMERG Early (V07) date; override with --date


def _parse_date(value):
    """Parse a YYYY-MM-DD probe date; argparse-friendly error on bad input."""
    try:
        return datetime.strptime(value, "%Y-%m-%d")
    except ValueError:
        raise argparse.ArgumentTypeError(
            "invalid date %r: expected YYYY-MM-DD (e.g. %s)" % (value, DEFAULT_PROBE_DATE)
        )


def run_smoke_test(test_date):
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
        print("(A 404 means no IMERG Early granule exists for that date yet -- pick an "
              "in-range date with --date YYYY-MM-DD, e.g. 2025-09-18.)")

def main(argv=None):
    parser = argparse.ArgumentParser(
        description="NASA IMERG authenticated smoke test (real GES DISC OPeNDAP fetch)."
    )
    parser.add_argument(
        "--date",
        type=_parse_date,
        default=DEFAULT_PROBE_DATE,
        help="probe date YYYY-MM-DD (default %(default)s); must be an available IMERG "
             "Early granule date -- future/too-recent dates return HTTP 404.",
    )
    args = parser.parse_args(argv)
    run_smoke_test(args.date)


if __name__ == "__main__":
    main()
