import geopandas as gpd
from shapely.geometry import Point

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
    Mock fetching OSM assets (highways, hospitals, schools) from Geofabrik/OSMnx.
    """
    return gpd.GeoDataFrame({
        "asset_name": ["NH-10", "STNM Hospital"],
        "asset_type": ["road", "hospital"],
        "geometry": [Point(88.61, 27.33), Point(88.62, 27.34)]
    })
