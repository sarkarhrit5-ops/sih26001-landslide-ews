import numpy as np
import rasterio
from rasterio.windows import Window
import math

def calculate_gradients(dem_array, cell_size=30.0):
    """
    Calculates dz/dx and dz/dy gradients using Horn's 3x3 window method.
    """
    z = dem_array.astype(np.float32)
    dz_dx = ((z[2:, 2:] + 2 * z[1:-1, 2:] + z[:-2, 2:]) - 
             (z[2:, :-2] + 2 * z[1:-1, :-2] + z[:-2, :-2])) / (8 * cell_size)
    
    dz_dy = ((z[2:, 2:] + 2 * z[2:, 1:-1] + z[2:, :-2]) - 
             (z[:-2, 2:] + 2 * z[:-2, 1:-1] + z[:-2, :-2])) / (8 * cell_size)
    return dz_dx, dz_dy

def calculate_slope(dem_array, cell_size=30.0):
    """
    Calculates slope in degrees using Horn's 3x3 method.
    """
    dz_dx, dz_dy = calculate_gradients(dem_array, cell_size)
    slope_rad = np.arctan(np.sqrt(dz_dx**2 + dz_dy**2))
    slope_deg = np.degrees(slope_rad)
    return slope_deg

def calculate_aspect(dem_array, cell_size=30.0):
    """
    Calculates aspect in degrees (0 to 360, 0 = North, clockwise).
    """
    dz_dx, dz_dy = calculate_gradients(dem_array, cell_size)
    aspect_rad = np.arctan2(dz_dy, -dz_dx)
    aspect_deg = np.degrees(aspect_rad)
    # Convert to 0 - 360 degrees clockwise from North
    aspect_deg = (90.0 - aspect_deg) % 360.0
    return aspect_deg

def calculate_roughness(dem_array):
    """
    Calculates Terrain Ruggedness / Roughness (max - min elevation in 3x3 neighborhood).
    """
    z = dem_array.astype(np.float32)
    neighborhoods = np.stack([
        z[:-2, :-2], z[:-2, 1:-1], z[:-2, 2:],
        z[1:-1, :-2], z[1:-1, 1:-1], z[1:-1, 2:],
        z[2:, :-2], z[2:, 1:-1], z[2:, 2:]
    ], axis=0)
    roughness = np.max(neighborhoods, axis=0) - np.min(neighborhoods, axis=0)
    return roughness

def calculate_tpi(dem_array):
    """
    Calculates Topographic Position Index (TPI = center elevation - mean elevation of 3x3 neighborhood).
    """
    z = dem_array.astype(np.float32)
    neighborhoods = np.stack([
        z[:-2, :-2], z[:-2, 1:-1], z[:-2, 2:],
        z[1:-1, :-2], z[1:-1, 1:-1], z[1:-1, 2:],
        z[2:, :-2], z[2:, 1:-1], z[2:, 2:]
    ], axis=0)
    tpi = z[1:-1, 1:-1] - np.mean(neighborhoods, axis=0)
    return tpi

def process_dem_in_chunks(dem_path, out_dir, chunk_size=512, state_prefix=None):
    """
    Processes real DEM in chunked windows with 1-pixel overlap padding to compute terrain features:
    - slope (degrees)
    - aspect (degrees)
    - roughness (m)
    - tpi (m)
    
    Guarantees zero chunk-boundary artifacts across all chunk edges.
    Returns paths to generated rasters.
    """
    with rasterio.open(dem_path) as src:
        cell_size = abs(src.res[0])
        # Convert degree res to approx meters if geographic CRS (1 deg approx 111,000m * cos(lat))
        if src.crs and src.crs.to_epsg() == 4326:
            mean_lat = (src.bounds.bottom + src.bounds.top) / 2.0
            cell_size = cell_size * 111000.0 * np.cos(np.radians(mean_lat))

        meta = src.meta.copy()
        meta.update(dtype=rasterio.float32, nodata=-9999.0)

        prefix = f"{state_prefix}_" if state_prefix else "real_"
        out_paths = {
            "slope": f"{out_dir}/{prefix}slope.tif",
            "aspect": f"{out_dir}/{prefix}aspect.tif",
            "roughness": f"{out_dir}/{prefix}roughness.tif",
            "tpi": f"{out_dir}/{prefix}tpi.tif"
        }

        # Initialize output rasters
        writers = {}
        for name, path in out_paths.items():
            writers[name] = rasterio.open(path, 'w', **meta)

        try:
            for i in range(0, src.height, chunk_size):
                for j in range(0, src.width, chunk_size):
                    r_stop = min(src.height, i + chunk_size)
                    c_stop = min(src.width, j + chunk_size)
                    
                    # Read chunk with 1-pixel buffer for 3x3 window
                    read_r0 = max(0, i - 1)
                    read_r1 = min(src.height, r_stop + 1)
                    read_c0 = max(0, j - 1)
                    read_c1 = min(src.width, c_stop + 1)

                    window = Window.from_slices((read_r0, read_r1), (read_c0, read_c1))
                    raw_chunk = src.read(1, window=window).astype(np.float32)

                    # Determine required padding if at raster boundary
                    pad_top = 1 if i == 0 else 0
                    pad_bottom = 1 if r_stop == src.height else 0
                    pad_left = 1 if j == 0 else 0
                    pad_right = 1 if c_stop == src.width else 0

                    if pad_top or pad_bottom or pad_left or pad_right:
                        padded_chunk = np.pad(
                            raw_chunk, 
                            ((pad_top, pad_bottom), (pad_left, pad_right)), 
                            mode='edge'
                        )
                    else:
                        padded_chunk = raw_chunk

                    # Calculate derivatives (3x3 reduces shape by 2 in each dim, matching exact r_stop - i, c_stop - j)
                    slope_c = calculate_slope(padded_chunk, cell_size=cell_size)
                    aspect_c = calculate_aspect(padded_chunk, cell_size=cell_size)
                    rough_c = calculate_roughness(padded_chunk)
                    tpi_c = calculate_tpi(padded_chunk)

                    # Write exact chunk output
                    write_window = Window(j, i, c_stop - j, r_stop - i)

                    writers["slope"].write(slope_c, 1, window=write_window)
                    writers["aspect"].write(aspect_c, 1, window=write_window)
                    writers["roughness"].write(rough_c, 1, window=write_window)
                    writers["tpi"].write(tpi_c, 1, window=write_window)
        finally:
            for w in writers.values():
                w.close()

    return out_paths

def verify_dem_terrain_features(dem_path, terrain_paths):
    """
    Verifies CRS, elevation, slope, aspect, terrain derivatives, and checks for chunk boundary artifacts
    by comparing chunked calculation against reference full calculation.
    """
    verifications = {}
    
    with rasterio.open(dem_path) as dem_src:
        dem_arr = dem_src.read(1)
        verifications["CRS"] = str(dem_src.crs)
        verifications["Elevation_Min"] = float(np.min(dem_arr))
        verifications["Elevation_Max"] = float(np.max(dem_arr))
        verifications["Elevation_Mean"] = float(np.mean(dem_arr))
        mean_lat = (dem_src.bounds.bottom + dem_src.bounds.top) / 2.0
        cell_size = abs(dem_src.res[0]) * 111000.0 * np.cos(np.radians(mean_lat))

    for name, path in terrain_paths.items():
        with rasterio.open(path) as src:
            arr = src.read(1)
            valid_arr = arr[arr != src.nodata]
            verifications[f"{name}_min"] = float(np.min(valid_arr))
            verifications[f"{name}_max"] = float(np.max(valid_arr))
            verifications[f"{name}_mean"] = float(np.mean(valid_arr))

    # Verify chunk boundary artifacts by comparing chunked slope against direct full calculation
    padded_full = np.pad(dem_arr.astype(np.float32), ((1, 1), (1, 1)), mode='edge')
    full_slope = calculate_slope(padded_full, cell_size=cell_size)

    with rasterio.open(terrain_paths["slope"]) as slope_src:
        chunked_slope = slope_src.read(1)
        max_chunk_diff = float(np.max(np.abs(full_slope - chunked_slope)))

    verifications["max_chunk_vs_full_diff"] = max_chunk_diff
    verifications["has_boundary_artifacts"] = bool(max_chunk_diff > 1e-4)

    return verifications

if __name__ == "__main__":
    print("Terrain processing module with chunking, aspect, roughness, TPI, and artifact verification ready.")

