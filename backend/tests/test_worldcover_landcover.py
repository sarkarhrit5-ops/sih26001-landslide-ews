"""
Focused tests for app.services.worldcover -- the REAL ESA WorldCover land-cover
feature for the Assam pilot that replaces the degenerate elevation proxy.

DEPENDENCY BUDGET: stdlib + numpy/pandas only. No rasterio, no network, no GDAL. The
raster read in sample_worldcover_at_points is the ONLY host-only step; every test
here injects a fake `reader` (or exercises the pure grouping/tile logic directly),
so the whole file runs offline. rasterio is never imported.

What these tests protect:
  * WorldCover codes are handled as NOMINAL, never ordinal (explicit lookup).
  * nodata / unknown / non-finite samples surface as UNAVAILABLE and are never
    filled with a real class.
  * the tile geometry is DERIVED from the AOI (Assam -> N24E090+N24E093).
  * the categorical contract constants are what an Assam trainer will rely on.
  * the Sikkim elevation proxy in risk_inputs is left completely unchanged, and this
    module does not reach into it.
"""
import math
import os
import sys

import numpy as np
import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.core.config_states import get_pilot_aoi_bounds
from app.services import worldcover as wc

# The authoritative expectation, restated independently of the module so a silent
# edit to the mapping is caught here.
EXPECTED_CODE_TO_GROUP = {
    10: 1, 20: 2, 30: 2, 40: 3, 50: 4, 60: 5, 70: 5, 80: 6, 90: 6, 95: 6, 100: 2,
}


# --------------------------------------------------------------------------- #
# Scalar grouping: exact mapping + strict UNAVAILABLE behaviour
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("code,expected", sorted(EXPECTED_CODE_TO_GROUP.items()))
def test_group_mapping_exact(code, expected):
    assert wc.group_worldcover_code(code) == expected


def test_group_nodata_raises():
    # ESA WorldCover nodata is 0 -> must be UNAVAILABLE, never a class.
    with pytest.raises(wc.LandCoverUnavailable):
        wc.group_worldcover_code(wc.WORLDCOVER_NODATA)


@pytest.mark.parametrize("bad", [5, 11, 45, 55, 99, 101, 999])
def test_group_unknown_code_raises(bad):
    # A code that is not one of the 11 official classes must not be guessed.
    with pytest.raises(wc.LandCoverUnavailable):
        wc.group_worldcover_code(bad)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), None, "tree"])
def test_group_non_finite_raises(bad):
    with pytest.raises(wc.LandCoverUnavailable):
        wc.group_worldcover_code(bad)


def test_group_accepts_float_codes_from_a_raster():
    # A sampler may hand back float pixel values; 40.0 must round-trip to cropland.
    assert wc.group_worldcover_code(40.0) == 3
    assert wc.group_worldcover_code(10.0) == 1


def test_grouping_is_nominal_not_ordinal():
    # 1) The mapping is exactly the explicit lookup table.
    assert dict(wc.ASSAM_LANDCOVER_GROUPS) == EXPECTED_CODE_TO_GROUP
    # 2) It is NOT monotonic in the raw code: Moss(100) -> 2 sits below Cropland(40)
    #    -> 3, so no threshold on the code could reproduce it. This is the proof the
    #    codes are treated as nominal labels, not an ordered scale.
    groups_in_code_order = [EXPECTED_CODE_TO_GROUP[c] for c in sorted(EXPECTED_CODE_TO_GROUP)]
    assert groups_in_code_order != sorted(groups_in_code_order)
    assert wc.ASSAM_LANDCOVER_GROUPS[100] < wc.ASSAM_LANDCOVER_GROUPS[40]


# --------------------------------------------------------------------------- #
# Vectorised grouping: sentinel + mask, correct dtype
# --------------------------------------------------------------------------- #
def test_group_array_vectorized():
    codes = np.array([10, 40, 0, 80, 999, 100, 20], dtype="int64")
    groups, mask = wc.group_worldcover_array(codes)

    assert groups.dtype == np.int32
    # 0 (nodata) and 999 (unknown) are UNAVAILABLE -> sentinel + mask True.
    assert groups.tolist() == [1, 3, wc.UNAVAILABLE_SENTINEL, 6, wc.UNAVAILABLE_SENTINEL, 2, 2]
    assert mask.tolist() == [False, False, True, False, True, False, False]
    # Sentinel is never a valid group code.
    assert wc.UNAVAILABLE_SENTINEL not in wc.ASSAM_LANDCOVER_GROUP_CODES


def test_group_array_handles_float_and_nan():
    codes = np.array([10.0, np.nan, 30.0], dtype="float64")
    groups, mask = wc.group_worldcover_array(codes)
    assert groups.tolist() == [1, wc.UNAVAILABLE_SENTINEL, 2]
    assert mask.tolist() == [False, True, False]


# --------------------------------------------------------------------------- #
# Tile geometry derived from the AOI
# --------------------------------------------------------------------------- #
def test_worldcover_tiles_assam():
    tiles = wc.worldcover_tiles_for_bbox(get_pilot_aoi_bounds("Assam"))
    assert tiles == ["N24E090", "N24E093"]


def test_worldcover_tiles_sikkim():
    tiles = wc.worldcover_tiles_for_bbox(get_pilot_aoi_bounds("Sikkim"))
    assert tiles == ["N27E087"]


def test_tile_name_filename_and_url():
    assert wc.worldcover_tile_name(24, 90) == "N24E090"
    assert wc.worldcover_map_filename("N24E090") == "ESA_WorldCover_10m_2021_v200_N24E090_Map.tif"
    url = wc.worldcover_https_url("N24E093")
    assert url.endswith("v200/2021/map/ESA_WorldCover_10m_2021_v200_N24E093_Map.tif")
    assert url.startswith("https://esa-worldcover.s3.eu-central-1.amazonaws.com/")


def test_floor_to_tile_is_multiple_of_three():
    # Sanity on the SW-corner snapping used to name tiles.
    assert wc._floor_to_tile(91.3) == 90
    assert wc._floor_to_tile(93.7) == 93
    assert wc._floor_to_tile(27.0) == 27
    assert wc._floor_to_tile(28.1) == 27


# --------------------------------------------------------------------------- #
# Categorical contract an Assam trainer relies on
# --------------------------------------------------------------------------- #
def test_categorical_contract():
    assert wc.LANDCOVER_FEATURE_NAME == "land_cover_class"
    assert wc.LANDCOVER_IS_CATEGORICAL is True
    assert wc.landcover_categorical_feature() == ["land_cover_class"]
    # Small, sorted, contiguous group set for the 59-event dataset.
    assert wc.ASSAM_LANDCOVER_GROUP_CODES == (1, 2, 3, 4, 5, 6)
    # Every group code has a human label.
    for code in wc.ASSAM_LANDCOVER_GROUP_CODES:
        assert code in wc.ASSAM_LANDCOVER_GROUP_LABELS


# --------------------------------------------------------------------------- #
# Point sampler + DataFrame integration via an injected (non-rasterio) reader
# --------------------------------------------------------------------------- #
def _fake_reader(codes_by_point):
    """Build a reader that returns preset raw codes regardless of the path."""
    def reader(raster_path, coords):
        assert len(coords) == len(codes_by_point)
        # coords must be (lon, lat) tuples.
        for lon, lat in coords:
            assert isinstance(lon, float) or isinstance(lon, int)
            assert isinstance(lat, float) or isinstance(lat, int)
        return list(codes_by_point)
    return reader


def test_sampler_injected_reader_flags_nodata():
    lats = [26.0, 26.1, 26.2, 26.3, 26.4]
    lons = [91.5, 92.0, 92.5, 93.0, 93.5]
    reader = _fake_reader([10, 40, 0, 80, 999])
    groups, mask = wc.sample_worldcover_at_points("unused.tif", lats, lons, reader=reader)
    assert groups.tolist() == [1, 3, wc.UNAVAILABLE_SENTINEL, 6, wc.UNAVAILABLE_SENTINEL]
    assert mask.tolist() == [False, False, True, False, True]


def test_sampler_length_mismatch_raises():
    reader = lambda path, coords: [10, 20]  # too few
    with pytest.raises(ValueError):
        wc.sample_worldcover_at_points("unused.tif", [1.0, 2.0, 3.0], [4.0, 5.0, 6.0], reader=reader)


def test_sampler_rejects_ragged_inputs():
    with pytest.raises(ValueError):
        wc.sample_worldcover_at_points("unused.tif", [1.0, 2.0], [3.0], reader=lambda p, c: [])


def test_assign_assam_land_cover_from_raster_sets_categorical_column():
    import pandas as pd

    df = pd.DataFrame({
        "latitude": [26.0, 26.1, 26.2, 26.3],
        "longitude": [91.5, 92.0, 92.5, 93.0],
    })
    reader = _fake_reader([10, 40, 0, 50])  # third point is nodata
    out, mask = wc.assign_assam_land_cover_from_raster(df, "unused.tif", reader=reader)

    assert out is df  # modified in place and returned
    assert str(df["land_cover_class"].dtype) == "int32"
    assert df["land_cover_class"].tolist() == [1, 3, wc.UNAVAILABLE_SENTINEL, 4]
    # The nodata row is flagged, NOT filled with a real class.
    assert mask.tolist() == [False, False, True, False]
    assert df.loc[2, "land_cover_class"] == wc.UNAVAILABLE_SENTINEL


# --------------------------------------------------------------------------- #
# Sikkim elevation proxy must remain completely unchanged and untouched
# --------------------------------------------------------------------------- #
def test_sikkim_elevation_proxy_unchanged():
    from app.services import risk_inputs as ri

    # Constants exactly as the Sikkim pilot defines them.
    assert ri.LAND_COVER_ELEVATION_BREAKS_M == (3000.0, 4200.0)
    assert ri.LAND_COVER_PROXY_CLASSES == (1, 2, 3)
    # Behaviour at representative elevations is intact.
    assert ri.land_cover_class_from_elevation(1000.0) == 1
    assert ri.land_cover_class_from_elevation(3500.0) == 2
    assert ri.land_cover_class_from_elevation(5000.0) == 3


def test_worldcover_module_is_independent_of_the_elevation_proxy():
    # The WorldCover module must not import or restate the elevation-proxy constants;
    # keeping them separate is what prevents the Sikkim proxy from being touched.
    assert not hasattr(wc, "LAND_COVER_ELEVATION_BREAKS_M")
    assert not hasattr(wc, "land_cover_class_from_elevation")
