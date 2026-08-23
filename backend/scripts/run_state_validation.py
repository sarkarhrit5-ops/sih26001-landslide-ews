import os
import sys
import json
import psutil
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.core.config_states import NER_STATES_CONFIG
from app.services.state_validation import (
    evaluate_landslide_inventory,
    evaluate_terrain_data,
    evaluate_rainfall_status,
    evaluate_exposure_data,
    determine_overall_status
)

def print_ram(stage):
    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss / (1024 * 1024)
    print(f"[{stage}] RAM Usage: {mem:.2f} MB")
    return mem

def run_validation_pipeline():
    peak_ram = []
    peak_ram.append(print_ram("Init"))
    
    glc_path = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "glc_legacy.csv")
    if not os.path.exists(glc_path):
        print(f"Error: GLC Legacy dataset not found at {glc_path}. Cannot evaluate inventory.")
        return
        
    print("Loading GLC Legacy dataset...")
    glc_df = pd.read_csv(glc_path)
    peak_ram.append(print_ram("After GLC Load"))
    
    rainfall_status = evaluate_rainfall_status()
    print(f"Rainfall (IMERG) Status: {rainfall_status}")
    
    report = []
    
    print("\nStarting State Validation Pipeline...\n" + "="*50)
    for state_name, config in NER_STATES_CONFIG.items():
        print(f"Evaluating {state_name}...")
        
        # A. Landslide Inventory
        inventory = evaluate_landslide_inventory(config, glc_df)
        
        # B. Terrain
        terrain = evaluate_terrain_data(state_name, config)
        
        # C. Exposure
        exposure = evaluate_exposure_data(state_name, config)
        
        # E. Determine Status
        status_info = determine_overall_status(
            state_name, inventory, terrain, rainfall_status, exposure, config.get("is_pilot", False)
        )
        
        state_report = {
            "state": state_name,
            "inventory_events": inventory["inventory_events"],
            "usable_events": inventory["usable_events"],
            "spatial_quality": inventory["spatial_quality"],
            "temporal_quality": inventory["temporal_quality"],
            "dem_status": terrain,
            "rainfall_status": rainfall_status,
            "exposure_status": exposure,
            "model_status": status_info["model_status"],
            "validation_metrics": status_info["validation_metrics"],
            "overall_status": status_info["overall_status"],
            "blocking_reasons": status_info["blocking_reasons"]
        }
        
        report.append(state_report)
        print(f"  -> Usable Events: {inventory['usable_events']}")
        print(f"  -> Status: {status_info['overall_status']}")
        if status_info["blocking_reasons"]:
            print(f"  -> Blockers: {', '.join(status_info['blocking_reasons'])}")
            
    # Save Report
    out_dir = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "state_validation.json")
    
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
        
    peak_ram.append(print_ram("Final"))
    max_ram = max(peak_ram)
    
    print("\n" + "="*50 + "\nVALIDATION PIPELINE COMPLETE")
    print(f"Report saved to: {out_path}")
    print("\nSUMMARY:")
    for r in report:
        print(f"{r['state']:20s} : {r['overall_status']}")
        
    print(f"\nPeak RAM Usage: {max_ram:.2f} MB (Constraint: 8000 MB)")
    print("8 GB RAM Constraint Met: YES" if max_ram < 8000 else "8 GB RAM Constraint Met: NO")
    
if __name__ == "__main__":
    run_validation_pipeline()
