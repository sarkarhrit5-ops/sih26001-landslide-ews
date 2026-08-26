"""
UNIT TESTS FOR THE CANONICAL AOI CONTRACT (Phase 2E-3).

These tests verify AOI BOOKKEEPING ONLY:
  * that exactly ONE definition of the East Sikkim pilot AOI exists,
  * that every consumer in the training/reproduction path reads from it instead
    of restating the numbers,
  * that the pilot AOI's relationship to Sikkim's administrative bounding box is
    explicit and enforced.

They make NO assertion about scientific model performance, and they deliberately
require no DEM, no rainfall, no network, no LightGBM, no rasterio and no
geopandas -- only app.core.config_states (pure stdlib) and app.models.thresholds
(numpy).
"""

import os
import re

import pytest

from app.core import config_states
from app.core.config_states import (
    ARUNACHAL_PILOT_AOI,
    ASSAM_PILOT_AOI,
    BBOX_KEYS,
    EAST_SIKKIM_PILOT_AOI,
    NER_STATES_CONFIG,
    PILOT_AOIS,
    aoi_bounds_tuple,
    assert_pilot_aoi_consistency,
    bbox_contains,
    check_pilot_aoi_consistency,
    get_pilot_aoi,
    get_pilot_aoi_bounds,
    get_state_bbox,
)

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# The canonical values, pinned in exactly one test file. If someone edits the AOI
# these assertions fail loudly, which is the point: changing the AOI changes the
# trained model, its negative samples and its rainfall thresholds, so it must be a
# deliberate, reviewed act rather than a tidy-up.
CANONICAL_MIN_LAT = 27.0
CANONICAL_MAX_LAT = 28.1
CANONICAL_MIN_LON = 88.0
CANONICAL_MAX_LON = 88.9

# Files that participate in the training / reproduction path and must therefore
# contain no AOI literals of their own.
CONSUMER_FILES = [
    os.path.join("app", "models", "thresholds.py"),
    os.path.join("app", "services", "label_gate.py"),
    os.path.join("scripts", "train_real_models.py"),
    os.path.join("scripts", "run_imerg_smoke_test.py"),
]

# The AOI edge values that uniquely identify the pilot box. min_lat (27.0) and
# min_lon (88.0) are shared with the administrative box and with unrelated state
# bboxes, so they are not reliable drift markers; the maxima are.
DRIFT_MARKERS = ("28.1", "88.9")


def _strip_comments(source: str) -> str:
    """
    Removes whole-line and trailing '#' comments so that a prose mention of the
    AOI in a comment is not mistaken for a hardcoded value in code.
    """
    out_lines = []
    for line in source.splitlines():
        stripped = line.lstrip()
        if stripped.startswith("#"):
            continue
        out_lines.append(line.split("#", 1)[0])
    return "\n".join(out_lines)


def _read_backend_file(relative_path: str) -> str:
    with open(os.path.join(BACKEND_DIR, relative_path), "r", encoding="utf-8") as handle:
        return handle.read()


# ---------------------------------------------------------------------------
# 1. The canonical definition itself
# ---------------------------------------------------------------------------

def test_canonical_aoi_has_expected_bounds():
    assert EAST_SIKKIM_PILOT_AOI["min_lat"] == CANONICAL_MIN_LAT
    assert EAST_SIKKIM_PILOT_AOI["max_lat"] == CANONICAL_MAX_LAT
    assert EAST_SIKKIM_PILOT_AOI["min_lon"] == CANONICAL_MIN_LON
    assert EAST_SIKKIM_PILOT_AOI["max_lon"] == CANONICAL_MAX_LON


def test_canonical_aoi_is_a_plausible_sikkim_rectangle():
    """Guards against a typo (e.g. 8.9 instead of 88.9) silently passing."""
    assert 26.0 < EAST_SIKKIM_PILOT_AOI["min_lat"] < EAST_SIKKIM_PILOT_AOI["max_lat"] < 29.0
    assert 87.0 < EAST_SIKKIM_PILOT_AOI["min_lon"] < EAST_SIKKIM_PILOT_AOI["max_lon"] < 90.0


def test_pilot_aoi_registry_points_at_the_canonical_object():
    assert PILOT_AOIS["Sikkim"] is EAST_SIKKIM_PILOT_AOI


def test_state_config_shape_is_unchanged_by_the_aoi_work():
    """
    The pilot AOI is registered OUTSIDE NER_STATES_CONFIG on purpose, because the
    state config dicts are iterated and partially serialised by the state sweep.
    """
    sikkim = NER_STATES_CONFIG["Sikkim"]
    assert set(sikkim) == {"id", "min_lat", "max_lat", "min_lon", "max_lon", "is_pilot", "pilot_area"}
    assert "pilot_aoi" not in sikkim


# ---------------------------------------------------------------------------
# 2. Accessors
# ---------------------------------------------------------------------------

def test_get_pilot_aoi_returns_a_defensive_copy():
    first = get_pilot_aoi("Sikkim")
    first["max_lat"] = 99.0
    assert EAST_SIKKIM_PILOT_AOI["max_lat"] == CANONICAL_MAX_LAT
    assert get_pilot_aoi("Sikkim")["max_lat"] == CANONICAL_MAX_LAT


def test_get_pilot_aoi_bounds_returns_only_bbox_keys_in_order():
    bounds = get_pilot_aoi_bounds("Sikkim")
    assert tuple(bounds.keys()) == BBOX_KEYS
    assert all(isinstance(value, float) for value in bounds.values())
    assert "name" not in bounds


def test_unknown_state_raises_for_both_accessors():
    with pytest.raises(KeyError):
        get_pilot_aoi("Atlantis")
    with pytest.raises(KeyError):
        get_pilot_aoi_bounds("Atlantis")
    with pytest.raises(KeyError):
        get_state_bbox("Atlantis")


def test_state_without_a_pilot_aoi_raises():
    """Manipur is configured but is not a pilot, so it has no canonical pilot AOI."""
    assert "Manipur" in NER_STATES_CONFIG
    assert "Manipur" not in PILOT_AOIS
    with pytest.raises(KeyError):
        get_pilot_aoi("Manipur")


def test_assam_pilot_aoi_is_registered_and_points_at_the_canonical_object():
    """Assam is the second pilot: it now has a canonical AOI registered."""
    assert "Assam" in PILOT_AOIS
    assert PILOT_AOIS["Assam"] is ASSAM_PILOT_AOI


def test_assam_pilot_aoi_has_expected_bounds():
    """
    The Assam pilot AOI is pinned here (the single place these numbers are
    asserted). It was chosen data-driven to cover the Guwahati/Kamrup cluster
    plus the western Karbi Anglong hills; changing it changes the Assam
    positives and the DEM/terrain rasters, so it must be a deliberate act.
    """
    assert get_pilot_aoi_bounds("Assam") == {
        "min_lat": 25.6,
        "max_lat": 26.6,
        "min_lon": 91.3,
        "max_lon": 93.7,
    }


def test_assam_pilot_aoi_is_contained_in_its_administrative_bbox():
    report = check_pilot_aoi_consistency("Assam")
    assert report["pilot_within_state"] is True
    assert report["state_bbox"] == get_state_bbox("Assam")
    assert report["pilot_aoi"] == get_pilot_aoi_bounds("Assam")
    assert assert_pilot_aoi_consistency("Assam")["pilot_within_state"] is True


def test_arunachal_pilot_aoi_is_registered_and_points_at_the_canonical_object():
    """Arunachal Pradesh is the third pilot: it now has a canonical AOI registered."""
    assert "Arunachal Pradesh" in PILOT_AOIS
    assert PILOT_AOIS["Arunachal Pradesh"] is ARUNACHAL_PILOT_AOI


def test_arunachal_pilot_aoi_has_expected_bounds():
    """
    The Arunachal pilot AOI is pinned here (the single place these numbers are
    asserted). It was chosen data-driven to cover the central Subansiri/Siang
    landslide cluster; changing it changes the Arunachal positives and the
    DEM/terrain rasters, so it must be a deliberate act. Note max_lat is 27.99
    (not 28.0) on purpose: an integer 28.0 floors to a third DEM tile row (see
    the get_dem_tiles_for_bbox test below and the comment in config_states.py).
    """
    assert get_pilot_aoi_bounds("Arunachal Pradesh") == {
        "min_lat": 26.5,
        "max_lat": 27.99,
        "min_lon": 92.0,
        "max_lon": 94.5,
    }


def test_arunachal_pilot_aoi_is_contained_in_its_administrative_bbox():
    report = check_pilot_aoi_consistency("Arunachal Pradesh")
    assert report["pilot_within_state"] is True
    assert report["state_bbox"] == get_state_bbox("Arunachal Pradesh")
    assert report["pilot_aoi"] == get_pilot_aoi_bounds("Arunachal Pradesh")
    assert assert_pilot_aoi_consistency("Arunachal Pradesh")["pilot_within_state"] is True


def test_aoi_bounds_tuple_uses_rasterio_west_south_east_north_order():
    assert aoi_bounds_tuple(get_pilot_aoi_bounds("Sikkim")) == (
        CANONICAL_MIN_LON,
        CANONICAL_MIN_LAT,
        CANONICAL_MAX_LON,
        CANONICAL_MAX_LAT,
    )


# ---------------------------------------------------------------------------
# 3. Containment helper
# ---------------------------------------------------------------------------

def test_bbox_contains_accepts_inner_and_identical_boxes():
    outer = {"min_lat": 27.0, "max_lat": 28.2, "min_lon": 88.0, "max_lon": 89.0}
    inner = {"min_lat": 27.5, "max_lat": 28.0, "min_lon": 88.2, "max_lon": 88.8}
    assert bbox_contains(outer, inner) is True
    assert bbox_contains(outer, dict(outer)) is True


@pytest.mark.parametrize("key,value", [
    ("min_lat", 26.9),
    ("max_lat", 28.3),
    ("min_lon", 87.9),
    ("max_lon", 89.1),
])
def test_bbox_contains_rejects_escape_on_every_side(key, value):
    outer = {"min_lat": 27.0, "max_lat": 28.2, "min_lon": 88.0, "max_lon": 89.0}
    inner = dict(outer)
    inner[key] = value
    assert bbox_contains(outer, inner) is False


def test_bbox_contains_tolerance_is_applied():
    outer = {"min_lat": 27.0, "max_lat": 28.0, "min_lon": 88.0, "max_lon": 89.0}
    inner = {"min_lat": 27.0, "max_lat": 28.05, "min_lon": 88.0, "max_lon": 89.0}
    assert bbox_contains(outer, inner) is False
    assert bbox_contains(outer, inner, tol=0.1) is True


# ---------------------------------------------------------------------------
# 4. The pilot-vs-administrative relationship
# ---------------------------------------------------------------------------

def test_pilot_aoi_is_contained_in_the_administrative_bbox():
    report = check_pilot_aoi_consistency("Sikkim")
    assert report["pilot_within_state"] is True
    assert report["state_bbox"] == get_state_bbox("Sikkim")
    assert report["pilot_aoi"] == get_pilot_aoi_bounds("Sikkim")


def test_the_two_boxes_are_documented_as_different_and_differ_only_at_the_maxima():
    report = check_pilot_aoi_consistency("Sikkim")
    assert report["boxes_identical"] is False
    assert report["differing_keys"] == ["max_lat", "max_lon"]


def test_assert_pilot_aoi_consistency_passes_and_returns_the_report():
    report = assert_pilot_aoi_consistency("Sikkim")
    assert report["pilot_within_state"] is True


def test_assert_pilot_aoi_consistency_raises_when_the_pilot_escapes_the_state_box():
    original = config_states.PILOT_AOIS["Sikkim"]
    config_states.PILOT_AOIS["Sikkim"] = {
        "name": "deliberately oversized AOI",
        "min_lat": 27.0,
        "max_lat": 30.0,
        "min_lon": 88.0,
        "max_lon": 88.9,
    }
    try:
        with pytest.raises(ValueError):
            assert_pilot_aoi_consistency("Sikkim")
    finally:
        config_states.PILOT_AOIS["Sikkim"] = original
    # The invariant holds again once the canonical AOI is restored.
    assert assert_pilot_aoi_consistency("Sikkim")["pilot_within_state"] is True


# ---------------------------------------------------------------------------
# 5. Consumers read from the canonical definition
# ---------------------------------------------------------------------------

def test_threshold_metadata_spatial_bounds_is_the_canonical_aoi():
    from app.models.thresholds import THRESHOLD_METADATA

    spatial_bounds = THRESHOLD_METADATA["spatial_bounds"]
    assert spatial_bounds == get_pilot_aoi_bounds("Sikkim")
    assert tuple(spatial_bounds.keys()) == BBOX_KEYS


def test_threshold_metadata_does_not_alias_the_canonical_definition():
    """
    The threshold must read the AOI, not hold a reference that could mutate it.
    """
    from app.models.thresholds import THRESHOLD_METADATA

    assert THRESHOLD_METADATA["spatial_bounds"] is not EAST_SIKKIM_PILOT_AOI
    THRESHOLD_METADATA["spatial_bounds"]["max_lat"] = 99.0
    try:
        assert EAST_SIKKIM_PILOT_AOI["max_lat"] == CANONICAL_MAX_LAT
    finally:
        THRESHOLD_METADATA["spatial_bounds"]["max_lat"] = CANONICAL_MAX_LAT


def test_threshold_events_count_still_matches_the_pilot_aoi_catalog():
    """
    82 is the number of NASA GLC events inside the canonical AOI. It is asserted
    here as a documented, AOI-dependent figure: if the AOI ever changes, this
    count must be recomputed from the catalog rather than carried over.
    """
    from app.models.thresholds import THRESHOLD_METADATA

    assert THRESHOLD_METADATA["events_count"] == 82


@pytest.mark.parametrize("relative_path", CONSUMER_FILES)
def test_no_consumer_restates_the_aoi_numbers(relative_path):
    code = _strip_comments(_read_backend_file(relative_path))
    for marker in DRIFT_MARKERS:
        assert marker not in code, (
            f"{relative_path} contains the AOI literal {marker}; it must read the "
            "AOI from app.core.config_states instead."
        )


def test_the_aoi_numbers_appear_exactly_once_in_the_definition_site():
    code = _strip_comments(_read_backend_file(os.path.join("app", "core", "config_states.py")))
    for marker in DRIFT_MARKERS:
        occurrences = len(re.findall(re.escape(marker), code))
        assert occurrences == 1, (
            f"AOI literal {marker} appears {occurrences} times in config_states.py; "
            "it must be defined exactly once."
        )


def test_training_script_reads_the_canonical_aoi():
    """
    Source-level check (the script itself cannot be imported without rasterio /
    psutil / requests): its AOI constant must be assigned from the canonical
    accessor rather than built as a literal dict.
    """
    source = _read_backend_file(os.path.join("scripts", "train_real_models.py"))
    assert 'EAST_SIKKIM_AOI = get_pilot_aoi_bounds("Sikkim")' in source
    assert "aoi_bounds_tuple(EAST_SIKKIM_AOI)" in source
    assert "get_dem_tiles_for_bbox(EAST_SIKKIM_AOI)" in source


# ---------------------------------------------------------------------------
# 6. Agreement with the DEM tile helper used by the state sweep
# ---------------------------------------------------------------------------

def test_dem_tiles_derived_from_the_canonical_aoi():
    """
    Skipped when state_validation's own import chain is unavailable offline. The
    expected tiles are the two Copernicus tiles the pilot has always used.
    """
    state_validation = pytest.importorskip("app.services.state_validation")

    tiles = state_validation.get_dem_tiles_for_bbox(get_pilot_aoi_bounds("Sikkim"))
    assert tiles == [(27, 88), (28, 88)]


def test_assam_dem_tiles_derived_from_the_canonical_aoi():
    """
    The Assam pilot AOI must resolve to exactly the six Copernicus GLO-30 tiles
    its DEM/terrain prep downloads (scripts/prepare_assam_terrain.py). Skipped
    offline when state_validation's import chain is unavailable.
    """
    state_validation = pytest.importorskip("app.services.state_validation")

    tiles = state_validation.get_dem_tiles_for_bbox(get_pilot_aoi_bounds("Assam"))
    assert tiles == [(25, 91), (25, 92), (25, 93), (26, 91), (26, 92), (26, 93)]


def test_arunachal_dem_tiles_derived_from_the_canonical_aoi():
    """
    The Arunachal pilot AOI must resolve to exactly six Copernicus GLO-30 tiles.
    This is the reason max_lat is 27.99 rather than 28.0: get_dem_tiles_for_bbox
    floors max_lat, so 28.0 would floor to 28 and pull in a third tile row (N28*)
    that lies outside the AOI, inflating the mosaic to nine tiles. Skipped offline
    when state_validation's import chain is unavailable.
    """
    state_validation = pytest.importorskip("app.services.state_validation")

    tiles = state_validation.get_dem_tiles_for_bbox(get_pilot_aoi_bounds("Arunachal Pradesh"))
    assert tiles == [(26, 92), (26, 93), (26, 94), (27, 92), (27, 93), (27, 94)]
