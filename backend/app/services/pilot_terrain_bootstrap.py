"""
PILOT TERRAIN BOOTSTRAP (production initialization)

WHY THIS EXISTS
    The Assam / Arunachal Pradesh / Meghalaya dashboards read their DEM status from
    state_validation._{assam,arunachal,meghalaya}_dem_available(), which ask each
    prediction module whether its FIVE real pilot terrain rasters are on disk:

        data/raw/<state>_pilot_dem.tif            (elevation)
        data/processed/<state>_pilot_slope.tif
        data/processed/<state>_pilot_aspect.tif
        data/processed/<state>_pilot_roughness.tif
        data/processed/<state>_pilot_tpi.tif

    Those rasters are deliberately NOT committed to Git (they are hundreds of MB and
    are .gitignored). On the host they are produced by the one-off drivers
    scripts/prepare_<state>_terrain.py. A freshly deployed instance (e.g. Render) has
    never run those drivers, so the rasters are absent and the dashboards honestly --
    but unhelpfully -- report "Missing (Requires Download)". process_state() does call
    acquire_state_dem(), but GET /api/v1/validation/status never runs process_state().

    This module closes that gap at STARTUP (once), not per request.

WHAT IT DOES NOT DO
    * It does not touch _assam_dem_available / _arunachal_dem_available /
      _meghalaya_dem_available. Those stay read-only status readers; this module is
      the only thing that ever *creates* the artifacts they read.
    * It never hard-codes or fakes availability. If acquisition fails, the real error
      is logged and the rasters simply stay absent, so DEM status stays unavailable.
    * It does not touch Sikkim. Sikkim's terrain comes from the committed
      east_sikkim_dem.tif / "real_" derivative path and its behaviour is unchanged.
    * It does not touch model or prediction logic, and it never writes a .tif into Git
      (every output lands in the .gitignored data/raw and data/processed trees).
    * It does not fetch ESA WorldCover. Land cover is a separate artifact with its own
      host driver (scripts/prepare_<state>_landcover.py); DEM status does not depend
      on it, and grid prediction still refuses honestly when it is absent.

IDEMPOTENCE
    The decision to act is taken from each prediction module's own
    missing_<state>_terrain_rasters() -- the same predicate the dashboard uses. If
    nothing is missing, the state is SKIPPED: no tile listing, no download, no
    reprocessing. A run where all three pilots are already prepared performs zero
    network I/O.

REUSE, NOT REINVENTION
    DEM acquisition is state_validation.acquire_state_dem(..., limit_tiles=False) --
    the existing downloader/mosaic/cropper, in its uncapped mode so the mosaic covers
    the whole pilot AOI (the capped 2x2 mode exists for the wide administrative boxes
    of the 8-state sweep). Terrain derivatives are
    terrain_processing.process_dem_in_chunks() -- the same chunked Horn 3x3 pass the
    pilot drivers use. The AOI comes from config_states.get_pilot_aoi_bounds(), the
    single source of truth. Output names are DERIVED from each prediction module's
    own filename constants, so the writer cannot drift from the reader.

HEAVY DEPENDENCIES / THREADING
    rasterio + numpy + network are needed only when work is actually required; they
    are imported lazily inside the execution path, so this module imports cleanly in
    the offline sandbox. Because a cold start may need to download several hundred MB,
    the default startup mode runs the work on a BACKGROUND daemon thread: the API
    binds its port and serves /health immediately, and DEM status flips from missing
    to available once the work completes.

ENVIRONMENT
    SIH_PILOT_TERRAIN_BOOTSTRAP=1     enable this regeneration path (DEFAULT: OFF --
                                      the uncapped mosaic merge OOM-killed the Render
                                      instance with exit status 137; production instead
                                      DOWNLOADS the prebuilt rasters, see
                                      app/services/pilot_artifact_store.py)
    SIH_PILOT_TERRAIN_BOOTSTRAP_BLOCKING=1
                                      run inline during startup instead of on a
                                      background thread (useful for CLI/one-shot jobs)
"""

import logging
import os
import threading

logger = logging.getLogger(__name__)

# The three pilots whose terrain artifacts are absent from a fresh deployment.
# Sikkim is intentionally NOT here -- see the module docstring.
PILOT_TERRAIN_STATES = ("Assam", "Arunachal Pradesh", "Meghalaya")

# Per-state wiring into each prediction module's OWN source of truth. Nothing about
# the raster names is restated here; it is read back off these modules.
_PILOT_MODULES = {
    "Assam": ("app.services.assam_prediction",
              "missing_assam_terrain_rasters",
              "assam_terrain_raster_paths",
              "ASSAM_TERRAIN_DERIVATIVE_FILENAMES"),
    "Arunachal Pradesh": ("app.services.arunachal_prediction",
                          "missing_arunachal_terrain_rasters",
                          "arunachal_terrain_raster_paths",
                          "ARUNACHAL_TERRAIN_DERIVATIVE_FILENAMES"),
    "Meghalaya": ("app.services.meghalaya_prediction",
                  "missing_meghalaya_terrain_rasters",
                  "meghalaya_terrain_raster_paths",
                  "MEGHALAYA_TERRAIN_DERIVATIVE_FILENAMES"),
}

ELEVATION_FEATURE = "elevation"

# Plan actions
ACTION_SKIP = "skip"
ACTION_DERIVATIVES_ONLY = "derivatives_only"
ACTION_ACQUIRE_DEM_AND_DERIVATIVES = "acquire_dem_and_derivatives"


def _load_pilot_module(state_name):
    """Lazily import the prediction module that owns a pilot's raster contract."""
    module_path, missing_fn, paths_fn, derivative_map = _PILOT_MODULES[state_name]
    import importlib
    module = importlib.import_module(module_path)
    return {
        "module": module,
        "missing": getattr(module, missing_fn),
        "paths": getattr(module, paths_fn),
        "derivative_filenames": getattr(module, derivative_map),
    }


def derive_terrain_state_prefix(derivative_filenames):
    """
    Recover the process_dem_in_chunks(state_prefix=...) value implied by a pilot's
    derivative filenames, e.g. {"slope": "assam_pilot_slope.tif", ...} -> "assam_pilot".

    Derived rather than hard-coded so the files this module WRITES are exactly the
    files the prediction module READS. Raises ValueError if the names disagree.
    """
    prefixes = set()
    for feature, filename in derivative_filenames.items():
        suffix = "_%s.tif" % feature
        if not filename.endswith(suffix):
            raise ValueError(
                "Cannot derive terrain prefix: %r does not end with %r" % (filename, suffix)
            )
        prefixes.add(filename[: -len(suffix)])
    if len(prefixes) != 1:
        raise ValueError("Inconsistent terrain filename prefixes: %s" % sorted(prefixes))
    return prefixes.pop()


def pilot_terrain_plan(data_dir=None, states=PILOT_TERRAIN_STATES):
    """
    Read-only decision pass: what (if anything) each pilot needs.

    Returns a list of dicts:
        state              state name
        action             "skip" | "derivatives_only" | "acquire_dem_and_derivatives"
        missing            [(feature, path)] exactly as the prediction module reports
        dem_missing        bool -- the elevation raster itself is absent
        error              str, only when the module's own check could not be run

    Touches no network and opens no raster; safe to call on every startup.
    """
    plan = []
    for state_name in states:
        entry = {"state": state_name, "missing": [], "dem_missing": None,
                 "action": ACTION_SKIP, "error": None}
        try:
            wiring = _load_pilot_module(state_name)
            missing = list(wiring["missing"](data_dir))
        except Exception as exc:  # pragma: no cover - defensive
            entry["error"] = "%s: %s" % (type(exc).__name__, exc)
            entry["action"] = ACTION_SKIP
            plan.append(entry)
            continue

        entry["missing"] = missing
        missing_features = {feature for feature, _path in missing}
        entry["dem_missing"] = ELEVATION_FEATURE in missing_features
        if not missing:
            entry["action"] = ACTION_SKIP
        elif entry["dem_missing"]:
            entry["action"] = ACTION_ACQUIRE_DEM_AND_DERIVATIVES
        else:
            entry["action"] = ACTION_DERIVATIVES_ONLY
        plan.append(entry)
    return plan


def _default_dem_acquirer(state_name, aoi_bounds):
    """
    Real DEM acquisition: the EXISTING state_validation.acquire_state_dem, in its
    uncapped mode, cropped to the canonical pilot AOI. Returns the DEM path.
    """
    from app.services.state_validation import acquire_state_dem
    return acquire_state_dem(state_name, dict(aoi_bounds), limit_tiles=False)


def _default_terrain_processor(dem_path, out_dir, state_prefix, chunk_size):
    """Real terrain derivatives: the EXISTING chunked Horn 3x3 pass."""
    from app.services.terrain_processing import process_dem_in_chunks
    return process_dem_in_chunks(dem_path, out_dir, chunk_size=chunk_size,
                                 state_prefix=state_prefix)


def ensure_pilot_terrain(data_dir=None, states=PILOT_TERRAIN_STATES, chunk_size=512,
                         dem_acquirer=None, terrain_processor=None):
    """
    Ensure the three pilots' DEM + terrain derivative rasters exist. Idempotent.

    For each state whose prediction module reports nothing missing, this is a no-op.
    Otherwise the DEM is acquired (if absent) with the existing acquire_state_dem and
    the four derivatives are (re)generated with the existing process_dem_in_chunks.

    Failures are CONTAINED and HONEST: the real exception is logged (with traceback)
    and the state is recorded as failed; nothing is written to make the artifact look
    present, so the dashboard keeps reporting DEM unavailable. Never raises.

    dem_acquirer / terrain_processor are injectable purely so the decision path can be
    unit-tested without network or rasterio; production uses the real defaults.

    Returns a report dict: {"results": [...], "acted": int, "skipped": int,
    "failed": int}.
    """
    from app.core.config_states import get_pilot_aoi_bounds

    acquire = dem_acquirer or _default_dem_acquirer
    process = terrain_processor or _default_terrain_processor

    results = []
    for entry in pilot_terrain_plan(data_dir=data_dir, states=states):
        state_name = entry["state"]
        result = {
            "state": state_name,
            "action": entry["action"],
            "missing_before": [feature for feature, _p in entry["missing"]],
            "dem_acquired": False,
            "derivatives_generated": False,
            "status": None,
            "error": entry["error"],
        }

        if entry["action"] == ACTION_SKIP:
            if entry["error"]:
                result["status"] = "check_failed"
                logger.warning("[pilot-bootstrap] %s: could not check terrain artifacts: %s",
                               state_name, entry["error"])
            else:
                result["status"] = "already_present"
                logger.info("[pilot-bootstrap] %s: terrain artifacts already present, skipping.",
                            state_name)
            results.append(result)
            continue

        logger.info("[pilot-bootstrap] %s: %s (missing: %s)",
                    state_name, entry["action"], ", ".join(result["missing_before"]) or "-")
        try:
            wiring = _load_pilot_module(state_name)
            paths = wiring["paths"](data_dir)
            dem_path = paths[ELEVATION_FEATURE]
            prefix = derive_terrain_state_prefix(wiring["derivative_filenames"])
            # Derivatives live beside the ones the prediction module reads.
            out_dir = os.path.dirname(paths["slope"])
            os.makedirs(os.path.dirname(dem_path), exist_ok=True)
            os.makedirs(out_dir, exist_ok=True)

            if entry["dem_missing"]:
                aoi = get_pilot_aoi_bounds(state_name)
                acquired = acquire(state_name, aoi)
                result["dem_acquired"] = True
                if acquired and os.path.abspath(acquired) != os.path.abspath(dem_path):
                    # acquire_state_dem resolves its own filename; if it ever stops
                    # agreeing with the prediction module, say so instead of silently
                    # leaving the predictor pointed at a file nobody wrote.
                    raise RuntimeError(
                        "DEM filename disagreement for %s: acquired %r but the predictor "
                        "reads %r" % (state_name, acquired, dem_path)
                    )

            process(dem_path, out_dir, prefix, chunk_size)
            result["derivatives_generated"] = True

            still_missing = [feature for feature, _p in wiring["missing"](data_dir)]
            if still_missing:
                result["status"] = "incomplete"
                result["error"] = "still missing after bootstrap: %s" % ", ".join(still_missing)
                logger.error("[pilot-bootstrap] %s: %s", state_name, result["error"])
            else:
                result["status"] = "prepared"
                logger.info("[pilot-bootstrap] %s: terrain artifacts prepared.", state_name)
        except Exception as exc:
            result["status"] = "failed"
            result["error"] = "%s: %s" % (type(exc).__name__, exc)
            # Log the REAL error (with traceback) and leave the artifacts absent, so
            # DEM status stays unavailable rather than being fabricated.
            logger.exception("[pilot-bootstrap] %s: terrain acquisition failed; DEM status "
                             "will remain unavailable.", state_name)

        results.append(result)

    report = {
        "results": results,
        "acted": sum(1 for r in results if r["status"] == "prepared"),
        "skipped": sum(1 for r in results if r["status"] == "already_present"),
        "failed": sum(1 for r in results
                      if r["status"] in ("failed", "incomplete", "check_failed")),
    }
    return report


def bootstrap_enabled(env=None):
    """
    True only when SIH_PILOT_TERRAIN_BOOTSTRAP is explicitly switched ON.

    DEFAULT IS OFF. Regeneration runs acquire_state_dem(limit_tiles=False), whose
    rasterio.merge.merge materialises whole mosaics in memory; on a small instance that
    OOM-kills the process (Render exit status 137). Production populates these rasters by
    DOWNLOADING them instead -- see app/services/pilot_artifact_store.py. This path stays
    available for a host or a large worker, but can no longer be reached by accident.
    """
    env = os.environ if env is None else env
    raw = str(env.get("SIH_PILOT_TERRAIN_BOOTSTRAP", "0")).strip().lower()
    return raw in ("1", "true", "yes", "on")


def bootstrap_blocking(env=None):
    """True when initialization should run inline instead of on a background thread."""
    env = os.environ if env is None else env
    raw = str(env.get("SIH_PILOT_TERRAIN_BOOTSTRAP_BLOCKING", "0")).strip().lower()
    return raw in ("1", "true", "yes", "on")


def start_pilot_terrain_bootstrap(env=None, **kwargs):
    """
    Startup entry point. Returns the report dict when run inline, the Thread when run
    in the background, or None when disabled. Never raises: a deployment must come up
    even if terrain preparation cannot.
    """
    env = os.environ if env is None else env
    if not bootstrap_enabled(env):
        logger.info("[pilot-bootstrap] not enabled (set SIH_PILOT_TERRAIN_BOOTSTRAP=1 "
                    "to regenerate terrain locally); pilot DEM status will reflect "
                    "whatever is already on disk.")
        return None

    if bootstrap_blocking(env):
        return ensure_pilot_terrain(**kwargs)

    def _run():
        try:
            ensure_pilot_terrain(**kwargs)
        except Exception:  # pragma: no cover - ensure_pilot_terrain already contains
            logger.exception("[pilot-bootstrap] unexpected failure in background thread.")

    thread = threading.Thread(target=_run, name="pilot-terrain-bootstrap", daemon=True)
    thread.start()
    return thread
