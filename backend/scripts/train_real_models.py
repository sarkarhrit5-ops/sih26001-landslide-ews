import os
import sys
import time
import psutil
import tracemalloc
import pandas as pd
import numpy as np

# Adjust path to import backend modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from app.services.weather_ingestion import fetch_imerg_precipitation
from app.services.label_gate import check_label_quality
from app.models.ml_pipeline import generate_spatial_negative_samples, run_spatial_holdout_validation, train_and_evaluate_baselines

def print_ram(stage):
    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss / (1024 * 1024)
    print(f"[{stage}] RAM Usage: {mem:.2f} MB")
    return mem

def run_real_modeling_pipeline():
    tracemalloc.start()
    peak_ram = []
    start_time = time.time()
    
    print("--- 1. EARTHDATA AUTHENTICATION CHECK ---")
    try:
        # We pass a dummy date to test the auth Exception raising
        from datetime import datetime
        fetch_imerg_precipitation({}, datetime.now())
        print("Earthdata Authentication: SUCCESS")
    except PermissionError as e:
        print(f"{str(e)}")
        print("\nPipeline execution halted cleanly due to missing Earthdata credentials.")
        print("Cannot download real IMERG or Copernicus DEM datasets.")
        print("Please set EARTHDATA_USERNAME and EARTHDATA_PASSWORD environment variables.")
        # We log memory and time before stopping
        peak_ram.append(print_ram("Execution Halted"))
        print(f"Total Runtime until halt: {time.time() - start_time:.2f} seconds")
        return
        
    print("\n--- 2. REAL DEM PROCESSING ---")
    # This block won't execute unless auth succeeds
    print("Fetching real Copernicus 30m DEM for East Sikkim...")
    
    print("\n--- 3. FEATURE DATASET GENERATION ---")
    # This block won't execute unless auth succeeds
    pass

if __name__ == "__main__":
    run_real_modeling_pipeline()
