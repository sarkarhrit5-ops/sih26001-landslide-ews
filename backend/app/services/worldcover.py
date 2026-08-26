"""
Real ESA WorldCover land-cover feature for the ASSAM pilot -- SINGLE definition site.

WHY THIS MODULE EXISTS
----------------------
The Sikkim pilot's `land_cover_class` feature is an ELEVATION-derived proxy
(app.services.risk_inputs.land_cover_class_from_elevation /
scripts/train_real_models.assign_land_cover_proxy, breaks 3000/4200 m). That proxy
is degenerate for Assam: the Assam pilot AOI is 43-1658 m, so the proxy is CONSTANT
= 1 across the whole 236-row matrix (a zero-variance feature). This module replaces
it, for ASSAM ONLY, with a REAL satellite-derived land cover sampled from ESA
WorldCover 2021 v200 (10 m, Sentinel-1/2, EPSG:4326 -- the same CRS as our DEM and
terrain rasters, so NO reprojection is needed).

It is deliberately a NEW, additive module. It does NOT import, read or mutate the
Sikkim elevation proxy in app.services.risk_inputs, and nothing here is wired into
the Sikkim training or serving path. Sikkim keeps its proxy exactly as-is; only the
Assam feature-preparation path (and a future Assam serving path) imports from here,
so the two pilots' land-cover definitions cannot drift through this file.

TWO CONTRACTS THAT MUST NOT BE VIOLATED
---------------------------------------
1. WorldCover codes are NOMINAL, never ordinal. 10=Tree, 20=Shrub, ... 100=Moss are
   category labels, not a ranked scale. We collapse the 11 raw codes into a small
   set of nominal GROUP codes with an explicit lookup table (never an arithmetic
   threshold), and the model must treat the resulting `land_cover_class` column as
   CATEGORICAL (LightGBM `categorical_feature=[...]`), never as a number to compare
   with `<`/`>`. See LANDCOVER_IS_CATEGORICAL / landcover_categorical_feature().

2. Nodata is UNAVAILABLE, never filled. ESA WorldCover uses 0 for no-data / outside
   swath. A nodata sample -- or a point outside the raster's coverage, or an
   unrecognised code -- is surfaced as UNAVAILABLE and NEVER silently mapped to a
   real class. The scalar path raises LandCoverUnavailable; the bulk (array /
   DataFrame) path returns an explicit boolean mask plus the UNAVAILABLE_SENTINEL so
   the caller can drop or abort, exactly as the rainfall path aborts rather than
   zero-filling.

The small class grouping below is intentionally coarse (6 groups) because the Assam
positives number only 59: too many rare land-cover levels would overfit. The
grouping is exposed as a single dict so it is a one-line change to merge groups once
the real class distribution is known from scripts/prepare_assam_landcover.py (which
reports per-group counts at the 59 events and AOI-wide). No grouping decision is
hardcoded anywhere else.
"""
import math

# ---------------------------------------------------------------------------
# Product identity (ESA WorldCover 2021 v200)
# ---------------------------------------------------------------------------
WORLDCOVER_VERSION = "v200"
WORLDCOVER_YEAR = 2021
WORLDCOVER_EPSG = 4326            # matches the DEM/terrain rasters -> no reprojection
WORLDCOVER_NODATA = 0            # ESA WorldCover: 0 = no-data / outside swath
WORLDCOVER_TILE_DEG = 3          # tiles are 3 deg x 3 deg, named by SW corner

# The 11 official nominal classes (see the ESA WorldCover product user manual).
WORLDCOVER_CLASS_LABELS = {
    10: "Tree cover",
    20: "Shrubland",
    30: "Grassland",
    40: "Cropland",
    50: "Built-up",
    60: "Bare / sparse vegetation",
    70: "Snow and ice",
    80: "Permanent water bodies",
    90: "Herbaceous wetland",
    95: "Mangroves",
    100: "Moss and lichen",
}

# ---------------------------------------------------------------------------
# Assam class grouping -- SINGLE definition site (NOMINAL, not ordinal)
# ---------------------------------------------------------------------------
# Explicit code -> group lookup. The group integers are LABELS, not a scale; the
# fact that Cropland(40)->3 while Moss(100)->2 (a higher raw code mapping to a lower
# group) is deliberate proof that this is a lookup and never a magnitude threshold.
# Classes that physically cannot occur in the 43-1658 m Brahmaputra/Karbi AOI
# (Snow 70, Mangroves 95, Moss 100) are still mapped, for completeness, to the
# nearest sensible group; scripts/prepare_assam_landcover.py reports whether they
# actually appear (they should be ~0).
ASSAM_LANDCOVER_GROUPS = {
    10: 1,    # Tree cover              -> forest
    20: 2,    # Shrubland              -> shrub/grass/herbaceous
    30: 2,    # Grassland              -> shrub/grass/herbaceous
    40: 3,    # Cropland               -> cropland
    50: 4,    # Built-up               -> built-up
    60: 5,    # Bare / sparse veg      -> bare / sparse
    70: 5,    # Snow and ice           -> bare / sparse (not expected in Assam)
    80: 6,    # Permanent water        -> water / wetland
    90: 6,    # Herbaceous wetland     -> water / wetland
    95: 6,    # Mangroves              -> water / wetland (not expected inland)
    100: 2,   # Moss and lichen        -> shrub/grass/herbaceous (not expected)
}

ASSAM_LANDCOVER_GROUP_LABELS = {
    1: "forest (tree cover)",
    2: "shrub / grass / herbaceous",
    3: "cropland",
    4: "built-up",
    5: "bare / sparse / snow",
    6: "water / wetland",
}

# The complete, sorted set of valid Assam group codes (used as the LightGBM
# categorical level set and asserted by the tests).
ASSAM_LANDCOVER_GROUP_CODES = tuple(sorted(set(ASSAM_LANDCOVER_GROUPS.values())))

# ---------------------------------------------------------------------------
# Categorical contract for the model
# ---------------------------------------------------------------------------
# The grouped feature is kept as a SINGLE integer column named exactly like the
# Sikkim schema's column, so the Assam feature matrix is a shape-compatible analogue
# (same 11-feature layout) -- but it MUST be declared categorical to LightGBM so the
# nominal group codes are never split on as if ordered.
LANDCOVER_FEATURE_NAME = "land_cover_class"
LANDCOVER_IS_CATEGORICAL = True

# Bulk paths flag UNAVAILABLE rows with this sentinel (never a valid group code, so
# it cannot be confused with a class) AND a parallel boolean mask. The sentinel row
# must be dropped or aborted on by the caller -- never used as data.
UNAVAILABLE_SENTINEL = -1


class LandCoverUnavailable(Exception):
    """
    Raised when a WorldCover sample cannot yield a real land-cover class -- because
    it is nodata (0), non-finite, outside raster coverage, or an unrecognised code.
    Callers MUST surface this as UNAVAILABLE and never substitute a class.
    """


# ---------------------------------------------------------------------------
# Grouping -- scalar (strict, raises) and vectorised (mask, for bulk reporting)
# ---------------------------------------------------------------------------
def group_worldcover_code(code):
    """
    Map ONE raw ESA WorldCover class code to its Assam nominal group code.

    Strict, serving-style contract: nodata / non-finite / unknown code RAISES
    LandCoverUnavailable rather than returning anything, because a single unusable
    sample at inference time must surface as UNAVAILABLE, not as a guessed class.
    (This mirrors risk_inputs.land_cover_class_from_elevation, which raises on a
    non-finite elevation instead of silently returning class 1.)
    """
    try:
        value = float(code)
    except (TypeError, ValueError):
        raise LandCoverUnavailable(
            "WorldCover code %r is not numeric; treating as UNAVAILABLE." % (code,)
        )
    if not math.isfinite(value):
        raise LandCoverUnavailable(
            "WorldCover code %r is non-finite; treating as UNAVAILABLE." % (code,)
        )
    raw = int(round(value))
    if raw == WORLDCOVER_NODATA:
        raise LandCoverUnavailable(
            "WorldCover sample is nodata (%d); treating as UNAVAILABLE, not filling."
            % WORLDCOVER_NODATA
        )
    if raw not in ASSAM_LANDCOVER_GROUPS:
        raise LandCoverUnavailable(
            "WorldCover code %r is not one of the 11 known classes %s; refusing to "
            "guess a group." % (raw, sorted(ASSAM_LANDCOVER_GROUPS))
        )
    return ASSAM_LANDCOVER_GROUPS[raw]


def group_worldcover_array(codes):
    """
    Vectorised grouping for bulk sampling / whole-raster histograms.

    Returns (groups, unavailable_mask):
      * groups          -- int32 numpy array; UNAVAILABLE entries hold
                           UNAVAILABLE_SENTINEL (-1), never a real group code.
      * unavailable_mask -- bool numpy array, True where the raw code was nodata,
                           non-finite, or unrecognised.

    Implemented as an explicit lookup table indexed by raw code (0..100), so the
    mapping is nominal by construction -- there is no arithmetic on the codes.
    """
    import numpy as np

    arr = np.asarray(codes)
    # Round to nearest integer code (WorldCover rasters are uint8, but a sampler may
    # hand back float); non-finite -> a value the LUT will treat as unavailable.
    with np.errstate(invalid="ignore"):
        rounded = np.rint(np.asarray(arr, dtype="float64"))
    finite = np.isfinite(rounded)

    max_code = max(ASSAM_LANDCOVER_GROUPS)
    lut = np.full(max_code + 1, UNAVAILABLE_SENTINEL, dtype=np.int32)
    for raw, group in ASSAM_LANDCOVER_GROUPS.items():
        lut[raw] = group  # nodata (0) is intentionally left at the sentinel

    codes_int = np.zeros(rounded.shape, dtype=np.int64)
    in_range = finite & (rounded >= 0) & (rounded <= max_code)
    codes_int[in_range] = rounded[in_range].astype(np.int64)

    groups = np.full(rounded.shape, UNAVAILABLE_SENTINEL, dtype=np.int32)
    groups[in_range] = lut[codes_int[in_range]]
    unavailable_mask = groups == UNAVAILABLE_SENTINEL
    return groups, unavailable_mask


def landcover_categorical_feature():
    """
    The LightGBM `categorical_feature` list an Assam trainer must pass so
    land_cover_class is split categorically, never as an ordered number.
    """
    return [LANDCOVER_FEATURE_NAME]


# ---------------------------------------------------------------------------
# WorldCover tile geometry (3 deg x 3 deg, named by SW corner at multiples of 3)
# ---------------------------------------------------------------------------
def _floor_to_tile(value):
    """Largest multiple of WORLDCOVER_TILE_DEG that is <= value (works for negatives)."""
    return int(math.floor(value / WORLDCOVER_TILE_DEG)) * WORLDCOVER_TILE_DEG


def worldcover_tile_name(lat_sw, lon_sw):
    """
    Format a WorldCover tile name from the SW-corner degrees, e.g. (24, 90) ->
    'N24E090'. lat is zero-padded to 2 digits, lon to 3, with N/S and E/W.
    """
    lat_hemi = "N" if lat_sw >= 0 else "S"
    lon_hemi = "E" if lon_sw >= 0 else "W"
    return "%s%02d%s%03d" % (lat_hemi, abs(int(lat_sw)), lon_hemi, abs(int(lon_sw)))


def worldcover_tiles_for_bbox(bbox):
    """
    The WorldCover tiles a bbox needs, DERIVED from the bbox (never hardcoded), in a
    stable (lat, lon) ascending order. `bbox` is the standard
    {min_lat,max_lat,min_lon,max_lon} dict (e.g. config_states.get_pilot_aoi_bounds).

    For the canonical Assam pilot AOI this returns ['N24E090', 'N24E093']; for the
    Sikkim pilot AOI it returns ['N27E087'] -- both asserted by the tests.
    """
    lat0 = _floor_to_tile(float(bbox["min_lat"]))
    lat1 = _floor_to_tile(float(bbox["max_lat"]))
    lon0 = _floor_to_tile(float(bbox["min_lon"]))
    lon1 = _floor_to_tile(float(bbox["max_lon"]))

    tiles = []
    lat = lat0
    while lat <= lat1:
        lon = lon0
        while lon <= lon1:
            tiles.append(worldcover_tile_name(lat, lon))
            lon += WORLDCOVER_TILE_DEG
        lat += WORLDCOVER_TILE_DEG
    return tiles


def worldcover_map_filename(tile_name):
    """The Map GeoTIFF filename for a tile, e.g. ESA_WorldCover_10m_2021_v200_N24E090_Map.tif."""
    return "ESA_WorldCover_10m_%d_%s_%s_Map.tif" % (
        WORLDCOVER_YEAR, WORLDCOVER_VERSION, tile_name
    )


def worldcover_s3_key(tile_name):
    """Key of a tile's Map GeoTIFF in the public AWS Open Data bucket 'esa-worldcover'."""
    return "%s/%d/map/%s" % (
        WORLDCOVER_VERSION, WORLDCOVER_YEAR, worldcover_map_filename(tile_name)
    )


def worldcover_https_url(tile_name):
    """
    Public HTTPS URL of a tile's Map GeoTIFF (AWS Open Data, region eu-central-1, no
    credentials). HOST-ONLY: the offline sandbox cannot reach it.
    """
    return "https://esa-worldcover.s3.eu-central-1.amazonaws.com/%s" % worldcover_s3_key(tile_name)


# ---------------------------------------------------------------------------
# Point sampling (raster read is HOST-ONLY; grouping/nodata handling is testable)
# ---------------------------------------------------------------------------
def _rasterio_reader(raster_path, coords):
    """
    Default reader: sample the WorldCover raster at (lon, lat) points with rasterio.

    HOST-ONLY -- rasterio is absent in the offline sandbox, which is why this is a
    lazily-imported, injectable hook: the tests pass their own `reader` and never
    touch rasterio. Uses .sample(), i.e. NEAREST pixel (no interpolation), which is
    the only correct sampling for a categorical raster.
    """
    import rasterio  # host-only

    with rasterio.open(raster_path) as src:
        return [int(vals[0]) for vals in src.sample(coords)]


def sample_worldcover_at_points(raster_path, lats, lons, reader=None):
    """
    Sample the REAL WorldCover raster at the given points and return the Assam
    nominal group codes.

    Returns (groups, unavailable_mask) exactly like group_worldcover_array:
    UNAVAILABLE points (nodata / outside coverage / unknown code) are flagged in the
    mask and set to UNAVAILABLE_SENTINEL -- never filled with a class.

    `reader(raster_path, coords)` must return an iterable of raw WorldCover codes,
    one per (lon, lat) coordinate; it defaults to the rasterio-based host reader.
    """
    lats = list(lats)
    lons = list(lons)
    if len(lats) != len(lons):
        raise ValueError("lats and lons must be the same length (%d != %d)."
                         % (len(lats), len(lons)))
    coords = list(zip(lons, lats))  # rasterio.sample expects (x=lon, y=lat)
    read = reader or _rasterio_reader
    raw_codes = list(read(raster_path, coords))
    if len(raw_codes) != len(coords):
        raise ValueError(
            "reader returned %d codes for %d points." % (len(raw_codes), len(coords))
        )
    return group_worldcover_array(raw_codes)


def assign_assam_land_cover_from_raster(df, raster_path, reader=None):
    """
    Assam feature-preparation counterpart of
    scripts/train_real_models.assign_land_cover_proxy, but sourced from the REAL
    WorldCover raster instead of elevation. Intended to be called on the Assam
    training frame in place of the elevation proxy; Sikkim's path is untouched.

    Sets df['land_cover_class'] to the grouped, nominal, int32 group code sampled at
    each row's (latitude, longitude). Rows whose sample is UNAVAILABLE are set to
    UNAVAILABLE_SENTINEL and reported in the returned mask -- they are NOT filled
    with a real class. The caller (the Assam trainer) must drop or abort on them,
    the same way the rainfall stage aborts rather than zero-filling; this function
    refuses to invent a land-cover class.

    Returns (df, unavailable_mask). df is modified in place and also returned.
    """
    groups, unavailable_mask = sample_worldcover_at_points(
        raster_path, df["latitude"], df["longitude"], reader=reader
    )
    # Kept as int32 to match the existing land_cover_class dtype; declared CATEGORICAL
    # to the model via landcover_categorical_feature().
    df[LANDCOVER_FEATURE_NAME] = groups.astype("int32")
    return df, unavailable_mask
