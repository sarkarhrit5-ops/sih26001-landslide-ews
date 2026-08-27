"""
Configuration for the 8 North Eastern Region (NER) states.
Includes bounding boxes and base metadata required for the validation pipeline.

TWO DISTINCT KINDS OF BOUNDING BOX ARE DEFINED IN THIS MODULE. They are not
interchangeable and must not be conflated:

1. NER_STATES_CONFIG[<state>]  --  the ADMINISTRATIVE bounding box of a state.
   A deliberately loose over-approximation of the state's extent, used for the
   8-state comparison sweep: landslide-inventory counts, DEM tile selection,
   exposure query windows and rainfall subsetting.

2. EAST_SIKKIM_PILOT_AOI  --  the MODELLED area of the Sikkim pilot, and the
   single canonical AOI for the training / reproduction pipeline. The DEM mosaic
   extent, the GLC positive filter, the buffered-negative sampling domain and the
   empirical rainfall-threshold derivation all use these exact numbers.

Anything that trains, evaluates, reproduces or describes the pilot model MUST
read the AOI from here via get_pilot_aoi() / get_pilot_aoi_bounds() and MUST NOT
restate the numbers inline. The pilot AOI is required to be contained within its
state's administrative box; check_pilot_aoi_consistency() and
assert_pilot_aoi_consistency() enforce that, so the two definitions can no longer
drift apart silently.
"""

# The four keys that make up a bounding box anywhere in this project, in the
# order they are conventionally written.
BBOX_KEYS = ("min_lat", "max_lat", "min_lon", "max_lon")

# ---------------------------------------------------------------------------
# CANONICAL PILOT AOI -- single source of truth for the training/reproduction path
# ---------------------------------------------------------------------------
# These numbers describe the area the pilot's REAL data products actually cover:
#   * backend/data/raw/east_sikkim_dem.tif is mosaicked and cropped to this extent
#   * the NASA GLC positives used by the pilot are selected with this filter
#   * buffered negatives are drawn uniformly from this rectangle (seed 42)
#   * the empirical rainfall threshold in app.models.thresholds was fitted on the
#     events inside this rectangle
# Editing them changes the trained model, its negative samples and its rainfall
# thresholds. They must therefore NOT be edited merely to "tidy up" a mismatch
# against the wider administrative Sikkim box below.
#
# NAMING CAVEAT: the label "East Sikkim" is historical. This rectangle spans most
# of the state of Sikkim, not only the former East Sikkim district. The name is
# retained because every existing artifact, filename and report uses it; only the
# name is imprecise -- the numbers are the ones actually used.
EAST_SIKKIM_PILOT_AOI = {
    "name": "East Sikkim pilot AOI",
    "min_lat": 27.0,
    "max_lat": 28.1,
    "min_lon": 88.0,
    "max_lon": 88.9
}

NER_STATES_CONFIG = {
    # ADMINISTRATIVE extent of Sikkim, intentionally slightly wider than the
    # modelled EAST_SIKKIM_PILOT_AOI above (max_lat 28.2 vs 28.1, max_lon 89.0 vs
    # 88.9). This box drives the state-level sweep, NOT the pilot model. It is
    # left as-is on purpose: narrowing it would silently change the exposure
    # query window and the rainfall subsetting for Sikkim.
    "Sikkim": {
        "id": "sikkim",
        "min_lat": 27.0,
        "max_lat": 28.2,
        "min_lon": 88.0,
        "max_lon": 89.0,
        "is_pilot": True,
        "pilot_area": "East Sikkim"
    },
    "Arunachal Pradesh": {
        "id": "arunachal_pradesh",
        "min_lat": 26.5,
        "max_lat": 29.5,
        "min_lon": 91.5,
        "max_lon": 97.5,
        "is_pilot": False
    },
    "Assam": {
        "id": "assam",
        "min_lat": 24.0,
        "max_lat": 28.0,
        "min_lon": 89.5,
        "max_lon": 96.0,
        "is_pilot": False
    },
    "Manipur": {
        "id": "manipur",
        "min_lat": 23.8,
        "max_lat": 25.7,
        "min_lon": 93.0,
        "max_lon": 94.8,
        "is_pilot": False
    },
    "Meghalaya": {
        "id": "meghalaya",
        "min_lat": 25.0,
        "max_lat": 26.1,
        "min_lon": 89.8,
        "max_lon": 92.8,
        "is_pilot": False
    },
    "Mizoram": {
        "id": "mizoram",
        "min_lat": 21.9,
        "max_lat": 24.5,
        "min_lon": 92.2,
        "max_lon": 93.4,
        "is_pilot": False
    },
    "Nagaland": {
        "id": "nagaland",
        "min_lat": 25.2,
        "max_lat": 27.0,
        "min_lon": 93.3,
        "max_lon": 95.3,
        "is_pilot": False
    },
    "Tripura": {
        "id": "tripura",
        "min_lat": 22.9,
        "max_lat": 24.5,
        "min_lon": 91.1,
        "max_lon": 92.4,
        "is_pilot": False
    }
}

# ---------------------------------------------------------------------------
# ASSAM PILOT AOI -- second pilot, added to replicate the Sikkim data pipeline
# ---------------------------------------------------------------------------
# Unlike Sikkim (a small, jurisdiction-clean mountain state), Assam's
# administrative box is a loose ~4-by-6.5 degree over-approximation dominated by
# the flat Brahmaputra floodplain, and it overlaps six neighbouring states plus
# parts of Bhutan, Bangladesh and Myanmar. A whole-state 30 m DEM is neither
# feasible on the 8 GB target nor meaningful (most of it is not landslide
# terrain). This AOI was therefore chosen DATA-DRIVEN from the real NASA GLC
# catalog: it covers the dominant Guwahati / Kamrup landslide cluster and the
# western Karbi Anglong hills.
#
# Real GLC positives inside this box (de-duplicated on lat/lon/date): 59 total,
# of which 38 carry admin_division_name == "Assam"; the remainder are genuine
# landslides inside the box attributed to neighbouring jurisdictions. Positives
# are kept on the SAME pure-bbox rule used by the Sikkim pilot, and the
# jurisdiction breakdown is recorded in the events snapshot
# (backend/data/models/assam_events.json) rather than silently dropped. The DEM
# extent is 6 Copernicus GLO-30 tiles.
#
# Exactly like EAST_SIKKIM_PILOT_AOI, editing these numbers changes which
# positives the Assam pilot would train on and the DEM / terrain rasters it
# derives, so change them only as a deliberate, reviewed act. The AOI is required
# to sit inside Assam's administrative box; assert_pilot_aoi_consistency("Assam")
# enforces that.
ASSAM_PILOT_AOI = {
    "name": "Assam pilot AOI (Guwahati-Kamrup + western Karbi Anglong)",
    "min_lat": 25.6,
    "max_lat": 26.6,
    "min_lon": 91.3,
    "max_lon": 93.7
}

# ---------------------------------------------------------------------------
# ARUNACHAL PRADESH PILOT AOI -- third pilot, same data pipeline as Assam
# ---------------------------------------------------------------------------
# Arunachal's administrative box (26.5/29.5 lat, 91.5/97.5 lon) is the largest in
# the NER sweep -- a ~3-by-6 degree over-approximation of rugged Eastern Himalaya
# that also overlaps Assam, Nagaland and the Tibet Autonomous Region. A whole-state
# 30 m DEM is infeasible on the 8 GB target and mostly not the terrain of interest.
# This AOI was therefore chosen DATA-DRIVEN from the real NASA GLC catalog: it
# covers the dominant central-Arunachal landslide cluster (the Subansiri / Siang
# belt around 27.0-27.5 N, 93.5-94.0 E) plus its western/central surroundings.
#
# Real GLC positives inside this box (de-duplicated on lat/lon/date): 50, spanning
# 40 independent event-dates. As with Sikkim (37) and Assam (37), this is well
# below the >=100 independent dates the Option-A gate needs -- even the ENTIRE
# administrative box yields only 76 -- so the Arunachal pilot is Option-C by
# construction, and the AOI was sized to balance training positives against the
# 8 GB DEM/terrain budget (6 Copernicus GLO-30 tiles, ~1.56x the Assam raster).
# Positives are kept on the SAME pure-bbox rule as the other pilots; the mixed
# jurisdiction breakdown is recorded in the events snapshot
# (backend/data/models/arunachal_pradesh_events.json), never silently dropped.
#
# Exactly like the other pilot AOIs, editing these numbers changes which positives
# the pilot trains on and the DEM / terrain rasters it derives. The AOI is required
# to sit inside Arunachal's administrative box;
# assert_pilot_aoi_consistency("Arunachal Pradesh") enforces that.
# NOTE ON max_lat = 27.99 (not 28.0): get_dem_tiles_for_bbox() selects tiles by
# floor(min_lat)..floor(max_lat) inclusive, so an integer max_lat of 28.0 would
# floor to 28 and pull in a whole third tile row (N28*, covering 28-29 N lying
# outside the AOI) -- 9 GLO-30 tiles instead of 6. Capping at 27.99 keeps the tile
# set at the intended 6 (rows N26/N27) with no loss of DEM coverage (the N27 tile
# spans up to 28.0 N) and no loss of positives (the two northern-most events sit at
# 27.98 N, still inside).
ARUNACHAL_PILOT_AOI = {
    "name": "Arunachal Pradesh pilot AOI (central Subansiri-Siang belt)",
    "min_lat": 26.5,
    "max_lat": 27.99,
    "min_lon": 92.0,
    "max_lon": 94.5
}

# ---------------------------------------------------------------------------
# MEGHALAYA PILOT AOI -- fourth pilot, same data pipeline as Assam / Arunachal
# ---------------------------------------------------------------------------
# Meghalaya's administrative box (25.0/26.1 lat, 89.8/92.8 lon) is a loose
# ~1.1-by-3 degree over-approximation that spans two physiographically distinct
# landslide clusters separated by a sparse gap: the western Garo Hills (~90.0 E)
# and the dominant eastern Khasi + Jaintia Hills belt (~91.7-92.3 E). A
# whole-admin-box 30 m DEM would be 8 Copernicus GLO-30 tiles for two disjoint
# clusters -- most of the intervening terrain carries no catalogued events. This
# AOI was therefore chosen DATA-DRIVEN from the real NASA GLC catalog: it covers
# the single densest, most physiographically coherent belt -- the Shillong /
# Cherrapunji southern scarp (India's highest-rainfall monsoon terrain) plus the
# Jaintia Hills to its east.
#
# Real GLC positives inside this box (de-duplicated on lat/lon/date): 34, spanning
# 29 independent event-dates. As with Sikkim (37), Assam (37) and Arunachal (40),
# this is far below the >=100 independent dates the Option-A gate needs -- even the
# ENTIRE administrative box yields only 38 -- so the Meghalaya pilot is Option-C by
# construction, and the AOI was sized to maximise the coherent-belt positive count
# against the 8 GB DEM/terrain budget (just 2 Copernicus GLO-30 tiles -- the leanest
# of the four pilots). Positives are kept on the SAME pure-bbox rule as the other
# pilots; the mixed jurisdiction breakdown (the box clips small parts of Assam and
# Bangladesh's Sylhet division) is recorded in the events snapshot
# (backend/data/models/meghalaya_events.json), never silently dropped.
#
# Exactly like the other pilot AOIs, editing these numbers changes which positives
# the pilot trains on and the DEM / terrain rasters it derives. The AOI is required
# to sit inside Meghalaya's administrative box;
# assert_pilot_aoi_consistency("Meghalaya") enforces that.
# NOTE ON max_lat = 25.99 (not 26.1): get_dem_tiles_for_bbox() selects tiles by
# floor(min_lat)..floor(max_lat) inclusive, so a max_lat of 26.0+ floors to 26 and
# pulls in a whole second tile row (N26*, covering 26-27 N) -- 4 GLO-30 tiles
# instead of 2. Capping at 25.99 keeps the tile set at the intended 2 (row N25,
# columns E091/E092); the only positives this excludes are the two northern-most
# events near 26.0-26.1 N, which sit in the Guwahati fringe rather than the Khasi
# belt this pilot models.
MEGHALAYA_PILOT_AOI = {
    "name": "Meghalaya pilot AOI (East Khasi + Jaintia Hills belt)",
    "min_lat": 25.0,
    "max_lat": 25.99,
    "min_lon": 91.0,
    "max_lon": 92.8
}

# Canonical pilot AOI per state. Kept OUTSIDE NER_STATES_CONFIG on purpose so the
# shape of the state config dictionaries (which are iterated and partially
# serialised by the validation sweep) is unchanged.
PILOT_AOIS = {
    "Sikkim": EAST_SIKKIM_PILOT_AOI,
    "Assam": ASSAM_PILOT_AOI,
    "Arunachal Pradesh": ARUNACHAL_PILOT_AOI,
    "Meghalaya": MEGHALAYA_PILOT_AOI
}


def get_pilot_aoi(state_name: str = "Sikkim") -> dict:
    """
    Returns the canonical pilot AOI for a state, including its descriptive
    "name" key, as a fresh copy (so callers cannot mutate the definition).
    """
    if state_name not in PILOT_AOIS:
        raise KeyError(
            f"No canonical pilot AOI is defined for state '{state_name}'. "
            f"Defined pilot AOIs: {sorted(PILOT_AOIS)}"
        )
    return dict(PILOT_AOIS[state_name])


def get_pilot_aoi_bounds(state_name: str = "Sikkim") -> dict:
    """
    Returns ONLY the four numeric bbox keys of the canonical pilot AOI, in
    BBOX_KEYS order. This is the shape expected by the bbox-consuming helpers
    (GLC filters, negative sampling, IMERG subsetting, threshold metadata).
    """
    aoi = get_pilot_aoi(state_name)
    return {key: float(aoi[key]) for key in BBOX_KEYS}


def get_state_bbox(state_name: str) -> dict:
    """
    Returns ONLY the four numeric bbox keys of a state's ADMINISTRATIVE box.
    """
    if state_name not in NER_STATES_CONFIG:
        raise KeyError(
            f"Unknown state '{state_name}'. Configured states: {sorted(NER_STATES_CONFIG)}"
        )
    config = NER_STATES_CONFIG[state_name]
    return {key: float(config[key]) for key in BBOX_KEYS}


def aoi_bounds_tuple(bbox: dict) -> tuple:
    """
    Converts a bbox dict into the (west, south, east, north) tuple ordering used
    by rasterio (e.g. rasterio.merge.merge(..., bounds=...)), i.e.
    (min_lon, min_lat, max_lon, max_lat).
    """
    return (
        float(bbox["min_lon"]),
        float(bbox["min_lat"]),
        float(bbox["max_lon"]),
        float(bbox["max_lat"])
    )


def bbox_contains(outer: dict, inner: dict, tol: float = 1e-9) -> bool:
    """
    True if `inner` lies entirely within `outer` (edges allowed, with a tiny
    floating-point tolerance).
    """
    return (
        float(inner["min_lat"]) >= float(outer["min_lat"]) - tol and
        float(inner["max_lat"]) <= float(outer["max_lat"]) + tol and
        float(inner["min_lon"]) >= float(outer["min_lon"]) - tol and
        float(inner["max_lon"]) <= float(outer["max_lon"]) + tol
    )


def check_pilot_aoi_consistency(state_name: str = "Sikkim") -> dict:
    """
    Compares a state's administrative box against its canonical pilot AOI and
    reports the relationship. Does NOT raise -- use this when you want to record
    the relationship (e.g. in provenance) rather than enforce it.

    Returns keys: state_name, state_bbox, pilot_aoi, pilot_within_state,
    boxes_identical, differing_keys.
    """
    state_bbox = get_state_bbox(state_name)
    pilot_bounds = get_pilot_aoi_bounds(state_name)
    differing = [
        key for key in BBOX_KEYS
        if abs(state_bbox[key] - pilot_bounds[key]) > 1e-9
    ]
    return {
        "state_name": state_name,
        "state_bbox": state_bbox,
        "pilot_aoi": pilot_bounds,
        "pilot_within_state": bbox_contains(state_bbox, pilot_bounds),
        "boxes_identical": len(differing) == 0,
        "differing_keys": differing
    }


def assert_pilot_aoi_consistency(state_name: str = "Sikkim") -> dict:
    """
    Enforces the invariant that the canonical pilot AOI is contained within the
    state's administrative box. Raises ValueError if it is not, because in that
    case the pilot would be modelling terrain the state sweep does not cover.

    Returns the same report as check_pilot_aoi_consistency() on success.
    """
    report = check_pilot_aoi_consistency(state_name)
    if not report["pilot_within_state"]:
        raise ValueError(
            f"Pilot AOI for '{state_name}' is not contained within its "
            f"administrative bounding box. "
            f"pilot_aoi={report['pilot_aoi']} state_bbox={report['state_bbox']}"
        )
    return report
