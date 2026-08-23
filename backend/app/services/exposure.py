import os
import time
import json
import requests
import geopandas as gpd
from shapely.geometry import Point, LineString

def analyze_exposure(hazard_grid: gpd.GeoDataFrame, osm_assets: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Intersects the high-hazard grid cells with OpenStreetMap infrastructure assets.
    """
    # 1. Filter grid cells that are at least HIGH hazard (e.g. hazard > 0.6)
    high_hazard_cells = hazard_grid[hazard_grid['current_hazard'] > 0.6]
    
    if high_hazard_cells.empty:
        return gpd.GeoDataFrame()
        
    # 2. Perform spatial join (ST_Intersects equivalent)
    # Using 'inner' to find assets inside high hazard zones
    exposed_assets = gpd.sjoin(osm_assets, high_hazard_cells, how="inner", predicate="intersects")
    
    return exposed_assets

def mock_get_osm_assets():
    """
    TEST / DEMO FIXTURE ONLY -- returns two hardcoded points (a road and a
    hospital). This is NOT real exposure data and MUST NOT be used to decide
    production exposure availability. It is kept solely because existing tests
    and the demo /exposure/alerts route depend on it; the real production path
    is get_osm_assets(), which never fabricates assets.
    """
    return gpd.GeoDataFrame({
        "asset_name": ["NH-10", "STNM Hospital"],
        "asset_type": ["road", "hospital"],
        "geometry": [Point(88.61, 27.33), Point(88.62, 27.34)]
    })

def get_osm_assets(state_name: str, bbox: dict) -> gpd.GeoDataFrame:
    """
    Fetches real OSM infrastructure assets for the state's bounding box using Overpass API,
    converts them to GeoDataFrame, and caches them to disk.
    """
    osm_dir = os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw", "osm")
    os.makedirs(osm_dir, exist_ok=True)
    
    clean_state_name = state_name.lower().replace(' ', '_')
    cache_path = os.path.join(osm_dir, f"{clean_state_name}_osm.geojson")
    
    # Check cache
    if os.path.exists(cache_path) and os.path.getsize(cache_path) > 100:
        try:
            gdf = gpd.read_file(cache_path)
            if not gdf.empty:
                return gdf
        except Exception as e:
            print(f"[OSM] Failed to read cached file for {state_name}: {e}. Retrying download...")
            
    # Overpass query: major highways and medical facilities
    # Use a slightly restricted bounding box centered on the state's central region to prevent huge payloads
    min_lat, max_lat = bbox["min_lat"], bbox["max_lat"]
    min_lon, max_lon = bbox["min_lon"], bbox["max_lon"]
    
    # Tighter query bounding box (centered on capital / populated core area) to stay under Overpass limits
    lat_center = (min_lat + max_lat) / 2.0
    lon_center = (min_lon + max_lon) / 2.0
    
    # 0.4 degree query bounding box (approx 44km x 44km) centered on the state core
    q_min_lat = max(min_lat, lat_center - 0.2)
    q_max_lat = min(max_lat, lat_center + 0.2)
    q_min_lon = max(min_lon, lon_center - 0.2)
    q_max_lon = min(max_lon, lon_center + 0.2)
    
    query = f"""
    [out:json][timeout:30];
    (
      way["highway"~"motorway|trunk|primary|secondary"]({q_min_lat},{q_min_lon},{q_max_lat},{q_max_lon});
      node["amenity"~"hospital|clinic"]({q_min_lat},{q_min_lon},{q_max_lat},{q_max_lon});
      node["amenity"="school"]({q_min_lat},{q_min_lon},{q_max_lat},{q_max_lon});
    );
    out geom;
    """
    
    headers = {"User-Agent": "SIH-EWS-Bot/1.0 (contact: gaura@test.com)"}
    url = "https://overpass-api.de/api/interpreter"
    
    retries = 3
    data = None
    for attempt in range(retries):
        try:
            resp = requests.post(url, data={"data": query}, headers=headers, timeout=30)
            if resp.status_code == 200:
                data = resp.json()
                break
            elif resp.status_code == 429:
                print(f"[OSM] Overpass rate limited for {state_name}, retrying in 5s...")
                time.sleep(5)
            else:
                print(f"[OSM] Overpass returned HTTP {resp.status_code} for {state_name}, retrying...")
                time.sleep(2)
        except Exception as e:
            print(f"[OSM] Request failed for {state_name} (attempt {attempt+1}): {e}")
            time.sleep(2)
            
    if not data or "elements" not in data:
        raise RuntimeError(f"Failed to query OSM assets from Overpass for {state_name}")
        
    features = []
    for elem in data["elements"]:
        elem_type = elem["type"]
        tags = elem.get("tags", {})
        name = tags.get("name", f"OSM {elem_type.capitalize()} {elem['id']}")
        
        # Determine asset type
        asset_type = "road"
        if "amenity" in tags:
            asset_type = tags["amenity"]
        elif "highway" in tags:
            asset_type = "road"
            
        if elem_type == "node" and "lat" in elem and "lon" in elem:
            geom = Point(elem["lon"], elem["lat"])
            features.append({
                "type": "Feature",
                "properties": {"asset_name": name, "asset_type": asset_type, "osm_id": elem["id"]},
                "geometry": geom
            })
        elif elem_type == "way" and "geometry" in elem:
            coords = [(pt["lon"], pt["lat"]) for pt in elem["geometry"]]
            if len(coords) >= 2:
                geom = LineString(coords)
                features.append({
                    "type": "Feature",
                    "properties": {"asset_name": name, "asset_type": asset_type, "osm_id": elem["id"]},
                    "geometry": geom
                })
                
    if not features:
        # Overpass returned a VALID response but ZERO matching real assets for
        # this bounding box. That is a legitimate "no exposure available"
        # result -- it is NOT an error and NOT a licence to fabricate. We must
        # not invent a state-centre point, dummy hospital, dummy road, zero-id
        # asset, or any other synthetic feature. Return an explicitly EMPTY
        # GeoDataFrame and DO NOT write a cache file, so that a downstream
        # availability check (which treats any >100-byte cached GeoJSON as
        # "Available") cannot be misled into reporting exposure that does not
        # exist.
        print(
            f"[OSM] Overpass returned zero real assets for {state_name}; "
            f"reporting EMPTY exposure (no synthetic asset created, no cache written)."
        )
        return gpd.GeoDataFrame(
            columns=["asset_name", "asset_type", "osm_id", "geometry"],
            geometry="geometry",
            crs="EPSG:4326",
        )

    gdf = gpd.GeoDataFrame.from_features(features, crs="EPSG:4326")

    # Persist cache as GeoJSON ONLY when we have REAL assets. An empty or
    # synthetic result is never written to disk, precisely so that a stale or
    # empty file cannot later be interpreted as "exposure Available".
    try:
        gdf.to_file(cache_path, driver="GeoJSON")
    except Exception as e:
        print(f"[OSM] Failed to write cache for {state_name}: {e}")

    return gdf

