"""
Prepare the REAL ESA WorldCover 2021 v200 land-cover raster for the canonical
Meghalaya pilot AOI, and report its class distribution. This is the Meghalaya
analogue of scripts/prepare_assam_landcover.py and
scripts/prepare_arunachal_landcover.py: it produces the data product that lets the
Meghalaya feature-preparation use REAL satellite-derived land cover (a categorical
feature) in place of the Sikkim elevation land_cover_class proxy, WITHOUT touching
the Sikkim, Assam or Arunachal pilots, the state-comparison sweep, IMERG or the
frontend.

WHAT IT PRODUCES (under backend/data/, none committed):
    data/raw/meghalaya_pilot_landcover.tif                  -- 1-tile crop to AOI
    data/models/meghalaya_pilot_landcover_classdist.json    -- machine-readable class
                                                               distribution + grouping used

WHY THIS NAME (do not "simplify" to meghalaya_landcover.tif):
    Same reason as meghalaya_pilot_dem.tif -- the sweep (app.services.state_validation)
    owns the "meghalaya_" prefix (clean_state_name = state_name.lower().
    replace(' ', '_')). "meghalaya_pilot_" keeps the pilot product cleanly separate
    from the sweep and from the other pilots. See scripts/prepare_meghalaya_terrain.py
    for the full rationale.

REUSE OF THE WORLDCOVER SERVICE (app.services.worldcover) -- NOT MODIFIED:
    Everything land-cover-specific -- the product identity, the 3-deg tile geometry,
    the tile URLs, the categorical NODATA contract, and the nominal class grouping --
    lives in app.services.worldcover and is imported here verbatim. That module is the
    SINGLE definition site and is NOT edited by this driver, so the Assam and Arunachal
    pilots that also import it are byte-for-byte unaffected.

    The grouping constants are named ASSAM_LANDCOVER_* in that module for HISTORICAL
    reasons (that is where the general 6-group collapse of WorldCover's 11 nominal
    classes was first defined for the Assam pilot). Despite the name, the collapse is
    GENERAL, not Assam-specific: 10->forest, 20/30->shrub/grass/herbaceous,
    40->cropland, 50->built-up, 60/70->bare/sparse/snow, 80/90/95->water/wetland,
    100->shrub/grass/herbaceous. It is reused verbatim for Meghalaya so the pilots'
    land-cover grouping cannot drift. Note built-up (50) maps to group 4 -- relevant
    for Meghalaya because the AOI includes the Shillong urban area on the Khasi
    plateau; the class report surfaces how much actually appears rather than assuming.

CATEGORICAL + NODATA CONTRACT (enforced via app.services.worldcover):
    * WorldCover's 11 codes are NOMINAL; they are collapsed into the small nominal
      GROUP set above with an explicit lookup (never a threshold), and the model must
      treat land_cover_class as categorical.
    * Mosaicking uses NEAREST resampling only (never averaging) -- averaging category
      codes is meaningless.
    * nodata (0) / outside-coverage / unknown codes are reported as UNAVAILABLE and
      NEVER filled with a class. The class-distribution report counts them explicitly.

REUSE, NOT REINVENTION: the AOI comes from
app.core.config_states.get_pilot_aoi_bounds("Meghalaya") (single source of truth);
the tiles are DERIVED from that AOI by
app.services.worldcover.worldcover_tiles_for_bbox (so the download/crop cannot cover
a different area than the positives were filtered to) and resolve to the SINGLE 3-deg
tile N24E090 -- the leanest of the four pilots. That tile is the same one the Assam
and Arunachal pilots use, so the shared data/raw/worldcover tile cache is reused
rather than re-downloaded.

RAM NOTE (8 GB target): the Meghalaya AOI (~1 deg lat x 1.8 deg lon) is the smallest
of the four pilots, so its 10 m read is the least memory-intensive. The whole-AOI
histogram is nonetheless computed in row strips (--chunk-rows, default 1024) and
_print_ram() reports RSS at each stage so a run that approaches the limit is visible.

HOST-ONLY: this must run on a machine that has rasterio + numpy installed and
outbound network access to the public ESA WorldCover AWS Open Data bucket
(esa-worldcover, eu-central-1, no credentials). It does NOT run in the offline
sandbox (rasterio and network are absent there); the file still compiles under
py_compile so it can be static-checked offline.

Usage (from the backend/ directory, on the host):
    python scripts/prepare_meghalaya_landcover.py            # build if missing
    python scripts/prepare_meghalaya_landcover.py --force    # rebuild even if present
"""
import argparse
import datetime as _dt
import json
import os
import sys

# Only pure-stdlib config/service helpers are imported at module load, so this file
# can be imported/compiled offline. The heavy, host-only dependencies (rasterio,
# numpy, urllib) are imported lazily inside the functions that need them.
_BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from app.core.config_states import (  # noqa: E402  (path set up above)
    aoi_bounds_tuple,
    assert_pilot_aoi_consistency,
    bbox_contains,
    get_pilot_aoi_bounds,
)
from app.services import worldcover as wc  # noqa: E402  (single definition site; NOT modified)

STATE_NAME = "Meghalaya"
PILOT_AREA = "East Khasi + Jaintia Hills belt"

LANDCOVER_FILENAME = "meghalaya_pilot_landcover.tif"
CLASSDIST_FILENAME = "meghalaya_pilot_landcover_classdist.json"
# Canonical events snapshot (see build_meghalaya_events_snapshot.py). "Meghalaya" is a
# single word, so this underscore filename is ALSO the serving resolver's derived path
# -- no space-vs-underscore reconciliation is needed here (unlike Arunachal). Read by
# this exact name, the same way the Assam driver reads assam_events.json.
EVENTS_FILENAME = "meghalaya_events.json"


def _print_ram(stage):
    """Best-effort RAM print for the 8 GB target; silently skipped if psutil absent."""
    try:
        import psutil  # noqa: WPS433 (optional dependency)
        rss_mb = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
        print("    [ram] %-32s %8.1f MB RSS" % (stage, rss_mb))
    except Exception:
        pass


def build_meghalaya_landcover(raw_dir, tile_cache_dir, force=False):
    """
    Download the ESA WorldCover tiles the Meghalaya pilot AOI needs and mosaic+crop
    them to exactly the AOI, writing data/raw/meghalaya_pilot_landcover.tif.

    Categorical-safe: NEAREST resampling, nodata preserved as 0, dtype uint8, CRS and
    10 m resolution left untouched (EPSG:4326, so no reprojection). Returns
    (landcover_path, aoi, tiles). Idempotent unless force=True.
    """
    import urllib.request

    import rasterio
    from rasterio.enums import Resampling
    from rasterio.merge import merge

    aoi = get_pilot_aoi_bounds(STATE_NAME)
    tiles = wc.worldcover_tiles_for_bbox(aoi)
    landcover_path = os.path.join(raw_dir, LANDCOVER_FILENAME)

    if os.path.exists(landcover_path) and os.path.getsize(landcover_path) > 1000 and not force:
        print("Land cover already present, reusing (use --force to rebuild): %s" % landcover_path)
        return landcover_path, aoi, tiles

    print("ESA WorldCover %s %d tiles required by the canonical Meghalaya AOI: %s"
          % (wc.WORLDCOVER_VERSION, wc.WORLDCOVER_YEAR, tiles))
    os.makedirs(tile_cache_dir, exist_ok=True)

    tile_paths = []
    for tile_name in tiles:
        tile_file = wc.worldcover_map_filename(tile_name)
        tile_url = wc.worldcover_https_url(tile_name)
        tile_path = os.path.join(tile_cache_dir, tile_file)
        if not (os.path.exists(tile_path) and os.path.getsize(tile_path) > 1000):
            print("  fetching %s" % tile_file)
            urllib.request.urlretrieve(tile_url, tile_path)
        else:
            print("  cached   %s" % tile_file)
        tile_paths.append(tile_path)

    if not tile_paths:
        raise RuntimeError("No WorldCover tiles resolved for the Meghalaya pilot AOI.")

    src_files = [rasterio.open(p) for p in tile_paths]
    try:
        # NEAREST is mandatory for a categorical raster; merge's default is already
        # nearest, but we pass it explicitly so the intent is unambiguous. nodata is
        # kept as WorldCover's 0 so out-of-coverage pixels stay UNAVAILABLE.
        mosaic, out_trans = merge(
            src_files,
            bounds=aoi_bounds_tuple(aoi),
            nodata=wc.WORLDCOVER_NODATA,
            resampling=Resampling.nearest,
        )
        out_meta = src_files[0].meta.copy()
        out_meta.update({
            "driver": "GTiff",
            "height": mosaic.shape[1],
            "width": mosaic.shape[2],
            "transform": out_trans,
            "crs": src_files[0].crs,
            "dtype": "uint8",
            "nodata": wc.WORLDCOVER_NODATA,
            "count": 1,
        })
        with rasterio.open(landcover_path, "w", **out_meta) as dst:
            dst.write(mosaic[0].astype("uint8"), 1)
    finally:
        for src in src_files:
            src.close()

    print("Wrote Meghalaya pilot land cover: %s" % landcover_path)
    return landcover_path, aoi, tiles


def measure_landcover_coverage(landcover_path, aoi):
    """
    Measure what the raster on disk ACTUALLY covers (from the file, not assumed) and
    compare against the canonical AOI with one pixel of tolerance. Also confirms the
    CRS is EPSG:4326 (the invariant that lets us skip reprojection).
    """
    import rasterio

    with rasterio.open(landcover_path) as src:
        rows = int(src.height)
        cols = int(src.width)
        pixel_deg = float(abs(src.res[0]))
        epsg = src.crs.to_epsg() if src.crs else None
        dtype = str(src.dtypes[0])
        nodata = src.nodata
        bounds = {
            "min_lat": float(src.bounds.bottom),
            "max_lat": float(src.bounds.top),
            "min_lon": float(src.bounds.left),
            "max_lon": float(src.bounds.right),
        }
    return {
        "rows": rows,
        "cols": cols,
        "pixel_size_deg": pixel_deg,
        "epsg": epsg,
        "dtype": dtype,
        "nodata": nodata,
        "bounds_measured_from_file": bounds,
        "fully_covers_canonical_aoi": bbox_contains(bounds, aoi, tol=pixel_deg),
        "crs_is_4326": epsg == wc.WORLDCOVER_EPSG,
    }


def aoi_class_histogram(landcover_path, chunk_rows=1024):
    """
    Whole-AOI histogram of the raw 11 WorldCover codes, computed in row strips to
    stay within the 8 GB target. Returns {raw_code: pixel_count} including the nodata
    code 0.
    """
    import numpy as np
    import rasterio
    from rasterio.windows import Window

    counts = np.zeros(max(wc.ASSAM_LANDCOVER_GROUPS) + 1, dtype=np.int64)
    with rasterio.open(landcover_path) as src:
        width = src.width
        height = src.height
        for r0 in range(0, height, chunk_rows):
            rows = min(chunk_rows, height - r0)
            band = src.read(1, window=Window(0, r0, width, rows))
            flat = band.reshape(-1).astype(np.int64)
            flat = flat[(flat >= 0) & (flat <= max(wc.ASSAM_LANDCOVER_GROUPS))]
            counts += np.bincount(flat, minlength=counts.shape[0])
    return {int(code): int(counts[code]) for code in range(counts.shape[0]) if counts[code] > 0}


def _grouped_from_raw_hist(raw_hist):
    """Collapse a {raw_code: count} histogram into {group_code: count} (nodata excluded)."""
    grouped = {code: 0 for code in wc.ASSAM_LANDCOVER_GROUP_CODES}
    unavailable = 0
    for raw_code, count in raw_hist.items():
        try:
            group = wc.group_worldcover_code(raw_code)
        except wc.LandCoverUnavailable:
            unavailable += count
            continue
        grouped[group] += count
    return grouped, unavailable


def load_event_points(models_dir):
    """
    Load the Meghalaya pilot positives' (lat, lon) from the committed events snapshot.
    Raises if the snapshot is missing so we never silently report a distribution over
    zero events.
    """
    path = os.path.join(models_dir, EVENTS_FILENAME)
    if not os.path.exists(path):
        raise SystemExit("Meghalaya events snapshot not found at %s." % path)
    with open(path, "r", encoding="utf-8") as fh:
        doc = json.load(fh)
    events = doc.get("events", [])
    lats = [float(ev["latitude"]) for ev in events]
    lons = [float(ev["longitude"]) for ev in events]
    return lats, lons


def event_class_distribution(landcover_path, lats, lons):
    """
    Grouped land-cover distribution at the event points, plus how many events fall on
    UNAVAILABLE (nodata / outside-coverage) pixels. Returns (grouped_counts,
    n_unavailable, n_events).
    """
    groups, unavailable_mask = wc.sample_worldcover_at_points(landcover_path, lats, lons)
    grouped = {code: 0 for code in wc.ASSAM_LANDCOVER_GROUP_CODES}
    for group in groups.tolist():
        if group in grouped:
            grouped[group] += 1
    return grouped, int(unavailable_mask.sum()), len(lats)


def _labelled(grouped_counts):
    """Attach human labels to a {group_code: count} dict for the report/JSON."""
    return {
        str(code): {"label": wc.ASSAM_LANDCOVER_GROUP_LABELS[code], "count": grouped_counts[code]}
        for code in wc.ASSAM_LANDCOVER_GROUP_CODES
    }


def write_classdist_json(out_path, document):
    """Write the class-distribution artifact as CRLF/2-space JSON, matching data/models/."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="\r\n") as fh:
        json.dump(document, fh, indent=2, ensure_ascii=False)
        fh.write("\n")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare the real ESA WorldCover land cover for the Meghalaya pilot AOI and report its class distribution."
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Rebuild the land-cover mosaic even if data/raw/meghalaya_pilot_landcover.tif exists.",
    )
    parser.add_argument(
        "--chunk-rows", type=int, default=1024,
        help="Row-strip height for the whole-AOI histogram pass (default 1024).",
    )
    args = parser.parse_args()

    # Fail fast if the pilot AOI is not inside Meghalaya's administrative box.
    report = assert_pilot_aoi_consistency(STATE_NAME)

    raw_dir = os.path.join(_BACKEND_DIR, "data", "raw")
    models_dir = os.path.join(_BACKEND_DIR, "data", "models")
    tile_cache_dir = os.path.join(raw_dir, "worldcover")
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    print("=" * 70)
    print(" MEGHALAYA PILOT LAND COVER (ESA WorldCover %s %d)" % (wc.WORLDCOVER_VERSION, wc.WORLDCOVER_YEAR))
    print(" state=%s  pilot_area=%s" % (STATE_NAME, PILOT_AREA))
    print(" canonical AOI      : %s" % report["pilot_aoi"])
    print(" administrative bbox: %s" % report["state_bbox"])
    print(" pilot within state : %s" % report["pilot_within_state"])
    print("=" * 70)
    _print_ram("start")

    # 1. Real WorldCover mosaic cropped to the AOI (host-only: network + rasterio).
    print("\n--- 1. REAL LAND COVER (ESA WorldCover) ---")
    landcover_path, aoi, tiles = build_meghalaya_landcover(raw_dir, tile_cache_dir, force=args.force)
    coverage = measure_landcover_coverage(landcover_path, aoi)
    print("Land-cover grid measured on disk : %d x %d cells @ %.9f deg/px"
          % (coverage["rows"], coverage["cols"], coverage["pixel_size_deg"]))
    print("CRS EPSG                          : %s (is 4326: %s)"
          % (coverage["epsg"], coverage["crs_is_4326"]))
    print("dtype / nodata                    : %s / %s" % (coverage["dtype"], coverage["nodata"]))
    print("Bounds measured on disk           : %s" % coverage["bounds_measured_from_file"])
    print("Fully covers canonical AOI        : %s" % coverage["fully_covers_canonical_aoi"])
    if not coverage["crs_is_4326"]:
        raise RuntimeError(
            "WorldCover mosaic CRS is not EPSG:4326 (got %s); refusing to proceed, as "
            "the pipeline assumes land cover shares the DEM/terrain CRS." % coverage["epsg"]
        )
    if not coverage["fully_covers_canonical_aoi"]:
        print("WARNING: the land-cover raster does NOT fully cover the canonical "
              "Meghalaya AOI. Samples outside coverage will be UNAVAILABLE (nodata) and "
              "must not be reported as covered.")
    _print_ram("after land-cover mosaic")

    # 2. Class distribution -- AOI-wide and at the event points.
    print("\n--- 2. CLASS DISTRIBUTION ---")
    raw_hist = aoi_class_histogram(landcover_path, chunk_rows=args.chunk_rows)
    aoi_nodata_px = raw_hist.get(wc.WORLDCOVER_NODATA, 0)
    aoi_grouped, aoi_unavailable_px = _grouped_from_raw_hist(raw_hist)
    total_px = sum(raw_hist.values())

    print("AOI raw WorldCover classes (pixels):")
    for raw_code in sorted(raw_hist):
        label = "NODATA" if raw_code == wc.WORLDCOVER_NODATA else wc.WORLDCOVER_CLASS_LABELS.get(raw_code, "UNKNOWN")
        pct = (100.0 * raw_hist[raw_code] / total_px) if total_px else 0.0
        print("    %3d  %-26s %12d  (%5.1f%%)" % (raw_code, label, raw_hist[raw_code], pct))

    print("AOI grouped classes (pixels, nodata excluded):")
    for code in wc.ASSAM_LANDCOVER_GROUP_CODES:
        print("    %d  %-26s %12d" % (code, wc.ASSAM_LANDCOVER_GROUP_LABELS[code], aoi_grouped[code]))
    print("    nodata/UNAVAILABLE pixels        : %d" % aoi_nodata_px)

    event_lats, event_lons = load_event_points(models_dir)
    ev_grouped, ev_unavailable, ev_total = event_class_distribution(
        landcover_path, event_lats, event_lons
    )
    print("\nGrouped land cover at the %d Meghalaya event points:" % ev_total)
    for code in wc.ASSAM_LANDCOVER_GROUP_CODES:
        print("    %d  %-26s %4d" % (code, wc.ASSAM_LANDCOVER_GROUP_LABELS[code], ev_grouped[code]))
    print("    UNAVAILABLE (nodata) events      : %d" % ev_unavailable)
    if ev_unavailable:
        print("    NOTE: %d event(s) sample nodata land cover. They must be surfaced "
              "as UNAVAILABLE at training time (dropped/aborted), NOT filled." % ev_unavailable)
    _print_ram("after class distribution")

    # 3. Machine-readable artifact so the distribution and the grouping used are
    #    captured, not just printed.
    generated_at = _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    document = {
        "schema_version": "1.0.0",
        "state": STATE_NAME,
        "pilot_area": PILOT_AREA,
        "aoi": aoi,
        "source": {
            "product": "ESA WorldCover",
            "version": wc.WORLDCOVER_VERSION,
            "year": wc.WORLDCOVER_YEAR,
            "resolution": "10 m",
            "tiles": tiles,
            "bucket": "s3://esa-worldcover (AWS Open Data, eu-central-1, no credentials)",
            "license": "CC-BY 4.0",
        },
        "raster": {
            "path": "backend/data/raw/%s" % LANDCOVER_FILENAME,
            "rows": coverage["rows"],
            "cols": coverage["cols"],
            "pixel_size_deg": coverage["pixel_size_deg"],
            "epsg": coverage["epsg"],
            "dtype": coverage["dtype"],
            "nodata": coverage["nodata"],
            "bounds_measured_from_file": coverage["bounds_measured_from_file"],
            "fully_covers_canonical_aoi": coverage["fully_covers_canonical_aoi"],
        },
        "grouping": {
            "note": (
                "WorldCover codes are NOMINAL; grouped into a small set (the general "
                "collapse defined once in app.services.worldcover, named ASSAM_LANDCOVER_* "
                "there for historical reasons and reused verbatim for Meghalaya). "
                "land_cover_class must be treated as CATEGORICAL."
            ),
            "raw_code_to_group": {str(k): v for k, v in wc.ASSAM_LANDCOVER_GROUPS.items()},
            "group_labels": {str(k): v for k, v in wc.ASSAM_LANDCOVER_GROUP_LABELS.items()},
        },
        "aoi_distribution": {
            "raw_class_pixels": {str(k): v for k, v in raw_hist.items()},
            "grouped_pixels": _labelled(aoi_grouped),
            "nodata_pixels": aoi_nodata_px,
            "total_pixels": total_px,
        },
        "event_distribution": {
            "n_events": ev_total,
            "grouped": _labelled(ev_grouped),
            "unavailable_nodata_events": ev_unavailable,
        },
        "generated_at": generated_at,
    }
    classdist_path = os.path.join(models_dir, CLASSDIST_FILENAME)
    write_classdist_json(classdist_path, document)

    print("\n--- SUMMARY: FILES PRODUCED ---")
    print("Land cover : %s" % landcover_path)
    print("Class dist : %s" % classdist_path)
    print("WorldCover tiles     : %s" % tiles)
    print("Covers full AOI      : %s" % coverage["fully_covers_canonical_aoi"])
    print("Events on nodata     : %d / %d" % (ev_unavailable, ev_total))
    print("Done.")


if __name__ == "__main__":
    main()
