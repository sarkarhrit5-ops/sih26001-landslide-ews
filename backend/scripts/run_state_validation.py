import os
import sys
import json
import psutil
import pandas as pd

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.core.config_states import NER_STATES_CONFIG
from app.services.state_validation import (
    process_state,
    evaluate_landslide_inventory,
    evaluate_terrain_data,
    evaluate_state_rainfall,
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
    
    report = []
    
    print("\n[PIPELINE] Starting Northeast India processing\n")
    processed_count = 0
    for state_name, config in NER_STATES_CONFIG.items():
        try:
            state_report = process_state(state_name, config, glc_df)
        except Exception as exc:
            print(f"[STATE] Error processing {state_name}: {exc}")
            state_id = config.get("id", state_name.lower().replace(" ", "_"))
            state_report = {
                "id": state_id,
                "state_id": state_id,
                "state": state_name,
                "state_name": state_name,
                "processing_status": "ERROR",
                "validation_status": "ERROR",
                "overall_status": "ERROR",
                "rainfall_source": "None",
                "rainfall_status": "Error",
                "inventory_events": 0,
                "usable_events": 0,
                "spatial_quality": "Poor",
                "temporal_quality": "Poor",
                "dem_status": "Missing",
                "exposure_status": "Missing",
                "model_status": "Error",
                "validation_metrics": {},
                "risk_result": None,
                "blocking_reasons": [f"Processing failed: {str(exc)}"],
                "error": str(exc)
            }
            print(f"[STATE] {state_name} completed")
            
        report.append(state_report)
        processed_count += 1
            
    # Save Report
    out_dir = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "state_validation.json")
    
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
        
    peak_ram.append(print_ram("Final"))
    
    print(f"\n[PIPELINE] {processed_count}/8 states processed\n")
    
    print("[SUMMARY]")
    for item in report:
        st_name = item.get("state_name", item.get("state"))
        st_status = item.get("overall_status", item.get("validation_status"))
        print(f"{st_name:<20} {st_status}")
    
if __name__ == "__main__":
    run_validation_pipeline()

