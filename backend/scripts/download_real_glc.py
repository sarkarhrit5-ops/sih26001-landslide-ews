import os
import urllib.request
import pandas as pd

def download_and_prepare_glc(url, dest_path):
    print(f"Downloading real NASA GLC dataset from {url}...")
    urllib.request.urlretrieve(url, dest_path)
    print("Download complete.")
    
    # Just loading to verify columns
    df = pd.read_csv(dest_path)
    print(f"Total global events in dataset: {len(df)}")
    print(f"Columns: {list(df.columns)}")

if __name__ == "__main__":
    url = "https://data.nasa.gov/docs/legacy/Global_Landslide_Catalog_Export/Global_Landslide_Catalog_Export_rows.csv"
    
    # Ensure directory exists
    raw_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "raw"))
    os.makedirs(raw_dir, exist_ok=True)
    
    dest_path = os.path.join(raw_dir, "glc_legacy.csv")
    download_and_prepare_glc(url, dest_path)
