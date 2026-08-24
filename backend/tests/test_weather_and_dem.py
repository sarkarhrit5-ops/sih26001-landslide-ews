import os
import sys
import pytest
from datetime import datetime

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.services.weather_ingestion import fetch_imerg_precipitation, get_earthdata_session
from app.services.terrain_processing import calculate_slope, calculate_aspect, calculate_roughness, calculate_tpi
import numpy as np

def test_earthdata_missing_credentials_raises_permission_error():
    # Save original env vars
    orig_user = os.environ.pop("EARTHDATA_USERNAME", None)
    orig_pass = os.environ.pop("EARTHDATA_PASSWORD", None)
    orig_tok = os.environ.pop("EARTHDATA_TOKEN", None)
    
    try:
        with pytest.raises(PermissionError) as exc_info:
            fetch_imerg_precipitation({}, datetime.now(), run_type="Final")
        assert "BLOCKER: Missing NASA Earthdata credentials" in str(exc_info.value)
    finally:
        # Restore env vars if present
        if orig_user: os.environ["EARTHDATA_USERNAME"] = orig_user
        if orig_pass: os.environ["EARTHDATA_PASSWORD"] = orig_pass
        if orig_tok: os.environ["EARTHDATA_TOKEN"] = orig_tok

def test_terrain_derivative_calculations():
    # Create a 5x5 test elevation array
    dem = np.array([
        [100, 105, 110, 115, 120],
        [102, 107, 112, 117, 122],
        [105, 110, 115, 120, 125],
        [108, 113, 118, 123, 128],
        [110, 115, 120, 125, 130]
    ], dtype=np.float32)
    
    slope = calculate_slope(dem, cell_size=30.0)
    aspect = calculate_aspect(dem, cell_size=30.0)
    rough = calculate_roughness(dem)
    tpi = calculate_tpi(dem)
    
    assert slope.shape == (3, 3)
    assert aspect.shape == (3, 3)
    assert rough.shape == (3, 3)
    assert tpi.shape == (3, 3)
    assert np.all(slope >= 0.0)
    assert np.all((aspect >= 0.0) & (aspect <= 360.0))

# ---------------------------------------------------------------------------
# Phase 2F-1: DEM cell size must be supplied explicitly, never silently defaulted
# ---------------------------------------------------------------------------
# calculate_slope/aspect/gradients previously defaulted to cell_size=30.0, so a
# caller that forgot the DEM's real ground resolution silently received terrain
# derivatives scaled to a fabricated 30 m grid (wrong for any non-30 m DEM). The
# cell size is now REQUIRED and validated; these tests need only numpy.

def test_calculate_slope_requires_explicit_cell_size():
    dem = np.zeros((3, 3), dtype=np.float32)
    with pytest.raises(TypeError):
        calculate_slope(dem)  # no silent 30 m assumption

def test_calculate_aspect_requires_explicit_cell_size():
    dem = np.zeros((3, 3), dtype=np.float32)
    with pytest.raises(TypeError):
        calculate_aspect(dem)

def test_calculate_slope_rejects_nonpositive_cell_size():
    dem = np.zeros((3, 3), dtype=np.float32)
    for bad in (0.0, -30.0):
        with pytest.raises(ValueError):
            calculate_slope(dem, cell_size=bad)

def test_calculate_slope_rejects_nonfinite_cell_size():
    dem = np.zeros((3, 3), dtype=np.float32)
    for bad in (float("nan"), float("inf")):
        with pytest.raises(ValueError):
            calculate_slope(dem, cell_size=bad)

def test_calculate_slope_rejects_non_numeric_cell_size():
    dem = np.zeros((3, 3), dtype=np.float32)
    for bad in (None, "abc"):
        with pytest.raises(ValueError):
            calculate_slope(dem, cell_size=bad)

def test_calculate_slope_uses_the_supplied_cell_size():
    # A plane rising 10 m per column and flat along rows has gradient 1.0 with a
    # 10 m cell -> exactly 45 degrees. The removed 30 m default would have produced
    # ~18.43 degrees, so this pins the result to the cell size actually supplied.
    dem = np.array([
        [0.0, 10.0, 20.0],
        [0.0, 10.0, 20.0],
        [0.0, 10.0, 20.0],
    ], dtype=np.float32)
    slope = calculate_slope(dem, cell_size=10.0)
    assert slope.shape == (1, 1)
    assert abs(float(slope[0, 0]) - 45.0) < 1e-3
