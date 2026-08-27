"""
Prepare the REAL Copernicus GLO-30 DEM and its Horn terrain derivatives for the
canonical Meghalaya pilot AOI. This is the Meghalaya analogue of the DEM/terrain
stage of scripts/train_real_models.py (and of the Assam / Arunachal drivers
scripts/prepare_assam_terrain.py and scripts/prepare_arunachal_terrain.py),
extracted into its own driver so the Meghalaya data can be prepared WITHOUT
touching the Sikkim, Assam or Arunachal pilots, the state-comparison sweep, IMERG
authentication or the frontend.

WHAT IT PRODUCES (all under backend/data/, none committed):
    data/raw/meghalaya_pilot_dem.tif             -- 2-tile mosaic, cropped to the AOI
    data/processed/meghalaya_pilot_slope.tif
    data/processed/meghalaya_pilot_aspect.tif
    data/processed/meghalaya_pilot_roughness.tif
    data/processed/meghalaya_pilot_tpi.tif

WHY THESE NAMES (do not "simplify" to meghalaya_dem.tif):
    The state-comparison sweep (app.services.state_validation) writes its files
    under clean_state_name = state_name.lower().replace(' ', '_'), i.e.
    "meghalaya_dem.tif" and process_dem_in_chunks(state_prefix="meghalaya") ->
    "meghalaya_*.tif". If this pilot driver reused those bare names,
    acquire_state_dem's size-guard would then serve this pilot-AOI DEM in place of
    the sweep's admin-bbox DEM, silently changing the 8-state sweep. The
    "meghalaya_pilot_" prefix keeps the pilot cleanly separate -- exactly as
    "east_sikkim_dem" / "real_" do for Sikkim and "assam_pilot_" / "arunachal_pilot_"
    do for the other two pilots -- and clashes with neither the sweep nor the other
    pilots. (It also differs from the events snapshot meghalaya_events.json only by
    design; that file is JSON, these are rasters.)

WHY 2 TILES, NOT 4: the canonical AOI's max_lat is 25.99 (not 26.1) precisely so
    get_dem_tiles_for_bbox -- which floors min/max lat & lon -- resolves to the two
    tiles N25 x E091/E092. An integer 26.0+ would floor to 26 and pull in a whole
    second tile row (N26*, ~26-27 N) that lies mostly OUTSIDE the AOI, doubling the
    mosaic to four tiles for no coverage gain. See the comment on MEGHALAYA_PILOT_AOI
    in app/core/config_states.py. This makes Meghalaya the LEANEST of the four pilots
    (Sikkim 2, Assam 6, Arunachal 6, Meghalaya 2).

REUSE, NOT REINVENTION: every AOI-dependent step reads the AOI from
app.core.config_states.get_pilot_aoi_bounds("Meghalaya") (the single source of
truth), selects Copernicus tiles with the same app.services.state_validation
get_dem_tiles_for_bbox helper the sweep uses, and computes terrain with the same
app.services.terrain_processing.process_dem_in_chunks / verify_dem_terrain_features
used by the pilot. The AOI numbers are never restated here. Unlike the sweep's
acquire_state_dem (which caps downloads to a 2x2 tile grid), this driver downloads
ALL tiles the AOI needs (2), exactly like the uncapped pilot path.

RAM NOTE (8 GB target): the Meghalaya AOI raster is the SMALLEST of the four pilots
(2 tiles, ~1 deg lat x 1.8 deg lon), so it is the least memory-intensive to build.
The DEM mosaic and terrain pass are both streamed/chunked (chunk_size default 512),
and _print_ram() reports RSS at each stage so a run that approaches the limit is
visible rather than silent.

HOST-ONLY: this must run on a machine that has rasterio + numpy installed and
outbound network access to the Copernicus GLO-30 S3 bucket. It does NOT run in the
offline sandbox (rasterio and network are absent there); the file still compiles
under py_compile so it can be static-checked offline.

Usage (from the backend/ directory, on the host):
    python scripts/prepare_meghalaya_terrain.py            # build if missing
    python scripts/prepare_meghalaya_terrain.py --force    # rebuild even if present
"""
import argparse
import os
import sys

# Only pure-stdlib config helpers are imported at module load, so this file can be
# imported/compiled offline. The heavy, host-only dependencies (rasterio, numpy,
# urllib, and the pandas-importing state_validation module) are imported lazily
# inside main() so module import never requires them.
_BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from app.core.config_states import (  # noqa: E402  (path set up above)
    aoi_bounds_tuple,
    assert_pilot_aoi_consistency,
    bbox_contains,
    check_pilot_aoi_consistency,
    get_pilot_aoi_bounds,
)

STATE_NAME = "Meghalaya"
PILOT_AREA = "East Khasi + Jaintia Hills belt"

# Pilot-specific output names (see module docstring for why they are NOT "meghalaya_").
DEM_FILENAME = "meghalaya_pilot_dem.tif"
TERRAIN_STATE_PREFIX = "meghalaya_pilot"  # process_dem_in_chunks appends "_" -> meghalaya_pilot_*.tif


def _print_ram(stage):
    """Best-effort RAM print for the 8 GB target; silently skipped if psutil absent."""
    try:
        import psutil  # noqa: WPS433 (optional dependency)
        rss_mb = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
        print("    [ram] %-32s %8.1f MB RSS" % (stage, rss_mb))
    except Exception:
        pass


def build_meghalaya_dem(raw_dir, tile_cache_dir, force=False):
    """
    Download the Copernicus GLO-30 tiles the Meghalaya pilot AOI needs (UNCAPPED)
    and mosaic+crop them to exactly the AOI, writing data/raw/meghalaya_pilot_dem.tif.

    Returns (dem_path, aoi, tiles). Idempotent: skips the download/mosaic when the
    DEM already exists unless force=True.
    """
    import urllib.request

    import rasterio
    from rasterio.merge import merge

    # Tiles are DERIVED from the canonical AOI with the same helper the state sweep
    # uses, so the download and mosaic crop cannot cover a different area than the
    # AOI the positives were filtered to.
    from app.services.state_validation import get_dem_tiles_for_bbox

    aoi = get_pilot_aoi_bounds(STATE_NAME)
    tiles = get_dem_tiles_for_bbox(aoi)
    dem_path = os.path.join(raw_dir, DEM_FILENAME)

    if os.path.exists(dem_path) and os.path.getsize(dem_path) > 1000 and not force:
        print("DEM already present, reusing (use --force to rebuild): %s" % dem_path)
        return dem_path, aoi, tiles

    print("Copernicus GLO-30 tiles required by the canonical Meghalaya AOI: %s" % tiles)
    os.makedirs(tile_cache_dir, exist_ok=True)

    tile_paths = []
    for tile_lat, tile_lon in tiles:
        tile_name = "Copernicus_DSM_COG_10_N%02d_00_E%03d_00_DEM" % (tile_lat, tile_lon)
        tile_url = "https://copernicus-dem-30m.s3.amazonaws.com/%s/%s.tif" % (tile_name, tile_name)
        tile_path = os.path.join(tile_cache_dir, "%s.tif" % tile_name)
        if not (os.path.exists(tile_path) and os.path.getsize(tile_path) > 1000):
            print("  fetching %s" % os.path.basename(tile_path))
            urllib.request.urlretrieve(tile_url, tile_path)
        else:
            print("  cached   %s" % os.path.basename(tile_path))
        tile_paths.append(tile_path)

    if not tile_paths:
        raise RuntimeError("No Copernicus tiles resolved for the Meghalaya pilot AOI.")

    src_files = [rasterio.open(p) for p in tile_paths]
    try:
        mosaic, out_trans = merge(src_files, bounds=aoi_bounds_tuple(aoi))
        out_meta = src_files[0].meta.copy()
        out_meta.update({
            "driver": "GTiff",
            "height": mosaic.shape[1],
            "width": mosaic.shape[2],
            "transform": out_trans,
            "crs": src_files[0].crs,
            "dtype": "float32",
        })
        with rasterio.open(dem_path, "w", **out_meta) as dst:
            dst.write(mosaic[0].astype("float32"), 1)
    finally:
        for src in src_files:
            src.close()

    print("Wrote Meghalaya pilot DEM: %s" % dem_path)
    return dem_path, aoi, tiles


def measure_dem_coverage(dem_path, aoi):
    """
    Measure what the DEM on disk ACTUALLY covers (from the file, not assumed) and
    compare it against the canonical AOI, allowing one pixel of tolerance because
    the mosaic snaps to the source raster grid. Returns a dict of measured facts.
    """
    import rasterio

    with rasterio.open(dem_path) as dem_src:
        rows = int(dem_src.height)
        cols = int(dem_src.width)
        pixel_deg = float(abs(dem_src.res[0]))
        bounds = {
            "min_lat": float(dem_src.bounds.bottom),
            "max_lat": float(dem_src.bounds.top),
            "min_lon": float(dem_src.bounds.left),
            "max_lon": float(dem_src.bounds.right),
        }
    covers = bbox_contains(bounds, aoi, tol=pixel_deg)
    return {
        "rows": rows,
        "cols": cols,
        "pixel_size_deg": pixel_deg,
        "bounds_measured_from_file": bounds,
        "fully_covers_canonical_aoi": covers,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Prepare the real Copernicus DEM + terrain derivatives for the Meghalaya pilot AOI."
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Rebuild the DEM mosaic even if data/raw/meghalaya_pilot_dem.tif already exists.",
    )
    parser.add_argument(
        "--chunk-size", type=int, default=512,
        help="Chunk window size (px) for the terrain derivative pass (default 512).",
    )
    args = parser.parse_args()

    # Fail fast if the pilot AOI is not inside Meghalaya's administrative box; in
    # that case the pilot would be modelling terrain the state sweep does not cover.
    report = assert_pilot_aoi_consistency(STATE_NAME)

    raw_dir = os.path.join(_BACKEND_DIR, "data", "raw")
    proc_dir = os.path.join(_BACKEND_DIR, "data", "processed")
    tile_cache_dir = os.path.join(raw_dir, "dem")
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(proc_dir, exist_ok=True)

    print("=" * 70)
    print(" MEGHALAYA PILOT DEM + TERRAIN PREPARATION")
    print(" state=%s  pilot_area=%s" % (STATE_NAME, PILOT_AREA))
    print(" canonical AOI      : %s" % report["pilot_aoi"])
    print(" administrative bbox: %s" % report["state_bbox"])
    print(" pilot within state : %s" % report["pilot_within_state"])
    print("=" * 70)
    _print_ram("start")

    # 1. Real DEM mosaic cropped to the AOI (host-only: network + rasterio).
    print("\n--- 1. REAL DEM (COPERNICUS GLO-30) ---")
    dem_path, aoi, tiles = build_meghalaya_dem(raw_dir, tile_cache_dir, force=args.force)
    coverage = measure_dem_coverage(dem_path, aoi)
    print("DEM grid measured on disk : %d x %d cells @ %.9f deg/px"
          % (coverage["rows"], coverage["cols"], coverage["pixel_size_deg"]))
    print("DEM bounds measured on disk: %s" % coverage["bounds_measured_from_file"])
    print("DEM fully covers AOI       : %s" % coverage["fully_covers_canonical_aoi"])
    if not coverage["fully_covers_canonical_aoi"]:
        print("WARNING: the DEM on disk does NOT fully cover the canonical Meghalaya AOI. "
              "Terrain features for samples outside DEM coverage will be nodata; this "
              "must be recorded and not reported as full-AOI coverage.")
    _print_ram("after DEM mosaic")

    # 2. Terrain derivatives (host-only: rasterio + numpy). Same Horn 3x3 chunked
    #    implementation the Sikkim pilot uses, only the output prefix differs.
    print("\n--- 2. TERRAIN DERIVATIVES (Horn 3x3, chunked) ---")
    from app.services.terrain_processing import (
        process_dem_in_chunks,
        verify_dem_terrain_features,
    )

    terrain_paths = process_dem_in_chunks(
        dem_path, proc_dir, chunk_size=args.chunk_size, state_prefix=TERRAIN_STATE_PREFIX
    )
    _print_ram("after terrain derivatives")

    verifications = verify_dem_terrain_features(dem_path, terrain_paths)
    print("Terrain verification:")
    for key, value in verifications.items():
        print("    %-24s %s" % (key, value))
    if verifications.get("has_boundary_artifacts"):
        raise RuntimeError(
            "Chunk-boundary artifacts detected in the Meghalaya terrain rasters "
            "(max_chunk_vs_full_diff=%s); refusing to report success."
            % verifications.get("max_chunk_vs_full_diff")
        )
    _print_ram("done")

    # Machine-readable summary of exactly what was produced.
    print("\n--- SUMMARY: FILES PRODUCED ---")
    print("DEM      : %s" % dem_path)
    for name in ("slope", "aspect", "roughness", "tpi"):
        print("%-9s: %s" % (name, terrain_paths[name]))
    print("\nAOI (canonical)      : %s" % aoi)
    print("Copernicus tiles     : %s" % tiles)
    print("DEM covers full AOI  : %s" % coverage["fully_covers_canonical_aoi"])
    print("Boundary artifacts   : %s" % verifications.get("has_boundary_artifacts"))
    print("Done.")


if __name__ == "__main__":
    main()
