import numpy as np
import rasterio
from rasterio.windows import Window
import math

def calculate_slope(dem_array, cell_size=30.0):
    """
    Calculates slope in degrees using a 3x3 window (Horn's method).
    """
    z = dem_array
    dz_dx = ((z[2:, 2:] + 2*z[1:-1, 2:] + z[:-2, 2:]) - 
             (z[2:, :-2] + 2*z[1:-1, :-2] + z[:-2, :-2])) / (8 * cell_size)
    
    dz_dy = ((z[2:, 2:] + 2*z[2:, 1:-1] + z[2:, :-2]) - 
             (z[:-2, 2:] + 2*z[:-2, 1:-1] + z[:-2, :-2])) / (8 * cell_size)
    
    slope_rad = np.arctan(np.sqrt(dz_dx**2 + dz_dy**2))
    slope_deg = np.degrees(slope_rad)
    return slope_deg

def process_dem_in_chunks(dem_path, out_slope_path, chunk_size=1024, overlap=1):
    """
    Processes DEM in chunks with overlaps to prevent boundary artifacts.
    """
    with rasterio.open(dem_path) as src:
        meta = src.meta.copy()
        meta.update(dtype=rasterio.float32)
        
        with rasterio.open(out_slope_path, 'w', **meta) as dst:
            for i in range(0, src.height, chunk_size):
                for j in range(0, src.width, chunk_size):
                    # Define window with overlap
                    row_start = max(0, i - overlap)
                    row_stop = min(src.height, i + chunk_size + overlap)
                    col_start = max(0, j - overlap)
                    col_stop = min(src.width, j + chunk_size + overlap)
                    
                    window = Window.from_slices((row_start, row_stop), (col_start, col_stop))
                    dem_chunk = src.read(1, window=window).astype(np.float32)
                    
                    # Calculate slope (returns smaller array due to 3x3 window)
                    if dem_chunk.shape[0] >= 3 and dem_chunk.shape[1] >= 3:
                        slope_chunk = calculate_slope(dem_chunk, cell_size=src.res[0])
                        
                        # Calculate the offset for writing back to the output raster
                        # The slope_chunk is 2 pixels smaller in each dimension than dem_chunk
                        write_row_start = row_start + 1
                        write_col_start = col_start + 1
                        
                        write_window = Window.from_slices(
                            (write_row_start, write_row_start + slope_chunk.shape[0]),
                            (write_col_start, write_col_start + slope_chunk.shape[1])
                        )
                        
                        dst.write(slope_chunk, 1, window=write_window)

if __name__ == "__main__":
    # Mock usage
    print("Terrain processing initialized with chunking and overlap protection.")
