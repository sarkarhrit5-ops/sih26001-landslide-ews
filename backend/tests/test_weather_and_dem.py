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
