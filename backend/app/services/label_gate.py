import pandas as pd
import numpy as np

def check_label_quality(csv_path: str, bounds: dict) -> dict:
    """
    Reads the GLC CSV and evaluates the temporal and spatial quality of labels 
    within the pilot bounding box to determine the ML approach.
    """
    df = pd.read_csv(csv_path)
    
    # Check if necessary columns exist
    if not all(col in df.columns for col in ['latitude', 'longitude', 'event_date']):
        raise ValueError("Dataset missing required columns.")
    
    # Spatially filter for Pilot AOI (East Sikkim)
    mask = (
        (df['latitude'] >= bounds['min_lat']) &
        (df['latitude'] <= bounds['max_lat']) &
        (df['longitude'] >= bounds['min_lon']) &
        (df['longitude'] <= bounds['max_lon'])
    )
    pilot_df = df[mask].copy()
    
    total_usable = len(pilot_df)
    
    # Assess date precision
    # In GLC, exact dates are often strings. We will try to parse them and look for complete day precision.
    pilot_df['parsed_date'] = pd.to_datetime(pilot_df['event_date'], errors='coerce')
    exact_day_events = pilot_df.dropna(subset=['parsed_date'])
    exact_day_count = len(exact_day_events)
    
    # Calculate independent events (events on different days in different locations to avoid duplicates)
    # Simple proxy: unique dates
    independent_events_count = exact_day_events['parsed_date'].nunique()
    
    # Gating Logic
    # We require at least 50 independent day-level events for a minimally defensible temporal model
    use_temporal_ml = independent_events_count >= 50
    
    return {
        "total_usable_events": total_usable,
        "exact_day_events": exact_day_count,
        "independent_events": independent_events_count,
        "use_temporal_ml": use_temporal_ml,
        "recommended_approach": "Option A: Temporal ML" if use_temporal_ml else "Option C: Static ML + IMERG Thresholds"
    }

if __name__ == "__main__":
    import urllib.request
    import os
    
    url = "https://data.nasa.gov/docs/legacy/Global_Landslide_Catalog_Export/Global_Landslide_Catalog_Export_rows.csv"
    file_path = "../../data/raw/glc_legacy.csv"
    
    if not os.path.exists(file_path):
        print("Downloading GLC dataset...")
        urllib.request.urlretrieve(url, file_path)
        print("Download complete.")
        
    bounds = {
        "min_lat": 27.0,
        "max_lat": 28.1,
        "min_lon": 88.0,
        "max_lon": 88.9
    }
    
    results = check_label_quality(file_path, bounds)
    print("Label Gate Results:")
    for k, v in results.items():
        print(f"{k}: {v}")
