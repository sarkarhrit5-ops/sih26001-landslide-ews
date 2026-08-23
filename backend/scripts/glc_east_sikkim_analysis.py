"""
Step 1 (REAL LANDSLIDES) — East Sikkim GLC analysis.
Read-only. Uses ONLY the real, already-present NASA GLC CSV + pandas (installed).
No network, no synthetic data, no modification of any project file.
"""
import sys, json
import pandas as pd
import numpy as np

GLC = "/sessions/relaxed-stoic-mayer/mnt/landslide/sih26001-landslide-ews/backend/data/raw/glc_legacy.csv"

# AOIs
PILOT_AOI = {"name": "East Sikkim pilot AOI (train_real_models.py)",
             "min_lat": 27.0, "max_lat": 28.1, "min_lon": 88.0, "max_lon": 88.9}
CFG_SIKKIM = {"name": "Sikkim bbox (config_states.py NER_STATES_CONFIG)",
              "min_lat": 27.0, "max_lat": 28.2, "min_lon": 88.0, "max_lon": 89.0}

df = pd.read_csv(GLC, low_memory=False)
print("=== GLC FILE ===")
print("total rows in catalog:", len(df))
print("columns:", list(df.columns))

def analyze(aoi):
    print("\n" + "=" * 70)
    print("AOI:", aoi["name"])
    print("bounds: lat[%s,%s] lon[%s,%s]" % (aoi["min_lat"], aoi["max_lat"], aoi["min_lon"], aoi["max_lon"]))
    m = ((df["latitude"] >= aoi["min_lat"]) & (df["latitude"] <= aoi["max_lat"]) &
         (df["longitude"] >= aoi["min_lon"]) & (df["longitude"] <= aoi["max_lon"]))
    sub = df[m].copy()
    raw = len(sub)
    print("total events in AOI (raw):", raw)
    if raw == 0:
        return
    # date parsing / precision
    sub["event_date_parsed"] = pd.to_datetime(sub["event_date"], errors="coerce")
    n_bad_date = int(sub["event_date_parsed"].isna().sum())
    print("events with UNPARSEABLE/missing event_date:", n_bad_date)
    with_date = sub.dropna(subset=["event_date_parsed"])
    print("events with usable date:", len(with_date))
    if len(with_date):
        print("date range:", with_date["event_date_parsed"].min(), "->", with_date["event_date_parsed"].max())
    # dedup (pipeline: dropna(event_date) then drop_duplicates lat/lon/event_date)
    dedup = with_date.drop_duplicates(subset=["latitude", "longitude", "event_date_parsed"])
    print("USABLE events after dedup (dropna+drop_duplicates lat/lon/date):", len(dedup))
    print("duplicate/overlap rows removed:", len(with_date) - len(dedup))
    print("independent event-dates (unique dates):", dedup["event_date_parsed"].dt.date.nunique())
    # spatial uncertainty
    if "location_accuracy" in sub.columns:
        acc = dedup["location_accuracy"].value_counts(dropna=False).to_dict()
        print("location_accuracy breakdown (deduped):", acc)
        # pipeline heuristic: not in {1km, exact, 100m} counts as >=5km uncertainty
        good = {"1km", "exact", "100m"}
        low = sum(c for a, c in acc.items() if (a not in good))
        pct = 100.0 * low / len(dedup) if len(dedup) else 0.0
        print("pct events with spatial uncertainty >= ~5km (pipeline heuristic): %.1f%%" % pct)
    # other useful context
    for col in ["landslide_category", "landslide_trigger", "landslide_size"]:
        if col in dedup.columns:
            print("%s:" % col, dedup[col].value_counts(dropna=False).head(8).to_dict())

for aoi in (PILOT_AOI, CFG_SIKKIM):
    analyze(aoi)

print("\n=== BACKEND LIBRARY CAPABILITY CHECK ===")
for p in ["fastapi", "uvicorn", "pydantic", "sklearn", "lightgbm", "rasterio",
          "geopandas", "pyarrow", "shap", "joblib", "scipy", "xarray", "httpx", "pytest"]:
    try:
        mod = __import__(p)
        print("PRESENT ", p, getattr(mod, "__version__", "?"))
    except Exception as e:
        print("MISSING ", p)
