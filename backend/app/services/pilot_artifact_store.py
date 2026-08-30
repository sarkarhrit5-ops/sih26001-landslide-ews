"""
PILOT ARTIFACT STORE (production initialization, download path)

WHY THIS EXISTS
    The Assam / Arunachal Pradesh / Meghalaya dashboards read DEM status from
    state_validation._{assam,arunachal,meghalaya}_dem_available(), which ask each
    prediction module whether its FIVE real pilot terrain rasters are on disk:

        data/raw/<state>_pilot_dem.tif            (elevation)
        data/processed/<state>_pilot_slope.tif
        data/processed/<state>_pilot_aspect.tif
        data/processed/<state>_pilot_roughness.tif
        data/processed/<state>_pilot_tpi.tif

    Those rasters are ~1.95 GB of float32 and are deliberately .gitignored, so a fresh
    deployment does not have them. The sibling module pilot_terrain_bootstrap can
    REGENERATE them from Copernicus GLO-30, but acquire_state_dem(limit_tiles=False)
    feeds rasterio.merge.merge, which materialises whole mosaics in memory and got the
    Render instance OOM-killed (exit status 137). That path is therefore disabled by
    default and this module is the production route: DOWNLOAD the already-built
    artifacts from object storage instead of rebuilding them.

    A streamed download holds one fixed-size buffer, so it cannot reproduce the OOM.

SIKKIM HAS TWO NAME FAMILIES FOR THE SAME FIVE RASTERS
    Sikkim was originally excluded here because its DEM was believed to be committed.
    It is not: `git ls-files '*.tif'` is empty, so a fresh deployment has no Sikkim
    terrain either. Sikkim's five rasters are now published as well, but Sikkim reads
    them under TWO different sets of filenames:

        published / 8-state sweep      Sikkim pilot serving path
        (state_validation)             (risk_inputs)
        data/raw/sikkim_dem.tif        data/raw/east_sikkim_dem.tif
        data/processed/sikkim_slope.tif    data/processed/real_slope.tif
        ... aspect / roughness / tpi   ... aspect / roughness / tpi

    Locally each pair is BYTE-IDENTICAL (verified by SHA-256; the sweep copy is what
    acquire_state_dem produces by copying east_sikkim_dem.tif, and the derivatives come
    from the same Horn pass). So only the five sweep-named objects are uploaded and
    carry manifest entries; the serving-named twins are placed as ADDITIONAL LINKS to
    the same verified bytes. That keeps the manifest honest -- one entry per uploaded
    object -- without a second 245 MB upload, and satisfies both readers.

    The alias filenames are read off risk_inputs' own constants via
    risk_inputs.terrain_raster_paths(), so they cannot drift from what the serving path
    actually opens. That import is for FILENAMES only; default_data_dir() is not called
    and the data root is never overridden (asserted by a test).

WHAT IT DOES NOT DO
    * It does not touch _assam_dem_available / _arunachal_dem_available /
      _meghalaya_dem_available, nor any prediction logic. Those stay read-only readers;
      this module only ever *creates* the files they look for.
    * It never fabricates availability. Every failure (missing manifest entry, HTTP
      error, short read, digest mismatch, no disk) is logged with the real exception and
      the canonical path is left ABSENT, so DEM status stays honestly unavailable.
    * It does not change any Sikkim READER, model, threshold or prediction path. Sikkim
      IS now in PILOT_ARTIFACT_STATES (its five rasters are published too, see below),
      but all this module ever does for Sikkim -- exactly as for the other three -- is
      put the bytes the existing readers already look for on disk.
    * It does not change risk_inputs.default_data_dir(). backend/data holds 38 COMMITTED
      files (all four model pickles, state_validation.json, the OSM GeoJSONs, the events
      snapshots); repointing the data root at an empty persistent disk would strand them.
      Instead SIH_PILOT_ARTIFACT_CACHE_DIR names a cache directory for the downloaded
      rasters and a symlink is placed at the canonical path. exists() / getsize() /
      rasterio.open() all follow symlinks, so no reader changes.
    * It does not run from a request handler. /api/v1/validation/status never downloads.
    * It writes no .tif into Git (data/raw and data/processed are .gitignored).

TRUNCATION IS THE REAL RISK
    missing_<state>_terrain_rasters() accepts any file with getsize(path) > 0, so a
    download interrupted at 3 MB would read as "Available" and the predictor would fail
    later on a corrupt raster. Hence: bytes always land in a sibling ".part" file, and
    are promoted with os.replace ONLY after the byte count and SHA-256 both match a
    manifest entry. An unverifiable artifact is never promoted.

IDEMPOTENCE
    The work list is each prediction module's own missing_<state>_terrain_rasters() --
    the same predicate the dashboard reads. A state with nothing missing is skipped
    before the manifest is even fetched: zero network I/O. With a cache directory, a
    cached file that still matches the manifest is re-linked without re-downloading.

MEMORY (the instance has 512 MB; the dataset is 2.2 GB)
    Nothing is ever held whole. Response bodies are consumed with iter_content in
    DOWNLOAD_CHUNK_BYTES blocks written straight to the ".part" file, with the SHA-256 and
    the byte count accumulated in flight -- so verification needs no second pass over the
    file, and response.content / .text / unbounded read() are never used. Written bytes
    are fsync'd and dropped from the page cache every FLUSH_EVERY_BYTES, because dirty
    page cache counts against a container memory limit even when process RSS is flat. The
    manifest is a small JSON document and is refused past MAX_MANIFEST_BYTES rather than
    accumulated. No raster is opened -- rasterio is never imported here.

ENVIRONMENT (all configuration, no code changes needed to deploy)
    SIH_PILOT_ARTIFACT_BASE_URL   master switch. Unset/blank => mechanism disabled.
                                  Public-read HTTPS prefix; filenames are appended.
    SIH_PILOT_ARTIFACT_MANIFEST   manifest filename resolved against the base URL, or an
                                  absolute http(s) URL. Default "pilot_manifest.json",
                                  which is the name the publisher writes and uploads.
    SIH_PILOT_ARTIFACT_STATES     comma-separated subset, for staging one pilot at a
                                  time. Default "Assam,Arunachal Pradesh,Meghalaya,
                                  Sikkim".
    SIH_PILOT_ARTIFACT_CACHE_DIR  persistent-disk cache; canonical paths become symlinks
                                  into it. Unset => write straight to backend/data.
    SIH_PILOT_ARTIFACT_FETCH=0    disable without unsetting the base URL.
    SIH_PILOT_ARTIFACT_BLOCKING=1 run inline instead of on a background thread.
    SIH_PILOT_ARTIFACT_TIMEOUT    per-request timeout in seconds. Default 120.
    SIH_PILOT_ARTIFACT_MAX_TOTAL_MB
                                  refuse a run whose planned bytes exceed this, and
                                  refuse when free disk is short. Default 2600.

LOGGING HYGIENE
    Object URLs are logged at DEBUG only. INFO/WARNING/ERROR mention filenames, so a
    base URL carrying a query-string credential is not printed into deployment logs.
"""

import hashlib
import logging
import os
import shutil
import threading

from app.services.pilot_terrain_bootstrap import (
    ELEVATION_FEATURE,
    _PILOT_MODULES,
    _load_pilot_module,
)

logger = logging.getLogger(__name__)

# The four states whose terrain artifacts are absent from a fresh deployment. Sikkim is
# here now because no .tif is tracked in Git either -- see the module docstring.
PILOT_ARTIFACT_STATES = ("Assam", "Arunachal Pradesh", "Meghalaya", "Sikkim")

SIKKIM_STATE_NAME = "Sikkim"
# The published Sikkim basenames: data/raw/sikkim_dem.tif plus data/processed/sikkim_*.
# state_validation.process_state derives exactly these from the state name
# (resolve_state_dem_filename + process_dem_in_chunks(state_prefix="sikkim")).
SIKKIM_DEM_FILENAME = "sikkim_dem.tif"
SIKKIM_DERIVATIVE_PREFIX = "sikkim_"
SIKKIM_DERIVATIVE_FEATURES = ("slope", "aspect", "roughness", "tpi")

# The name the publisher (scripts/publish_pilot_artifacts.py) actually writes and that
# was uploaded alongside the rasters. A bare "manifest.json" default made a fresh
# deployment request an object that does not exist in the store, so every artifact was
# refused with manifest_unavailable. Overridable via SIH_PILOT_ARTIFACT_MANIFEST.
DEFAULT_MANIFEST_NAME = "pilot_manifest.json"
DEFAULT_TIMEOUT_SECONDS = 120.0
DEFAULT_MAX_TOTAL_MB = 2600
# Streamed in fixed-size blocks so peak memory is independent of artifact size. Small on
# purpose: the instance this runs on has 512 MB, and the largest raster is 184 MB.
DOWNLOAD_CHUNK_BYTES = 256 * 1024
# Written bytes are flushed and dropped from the page cache this often. Dirty page cache
# counts against a container memory limit, so streaming 2.2 GB without this can walk the
# cgroup into an OOM kill even while the process itself holds one small buffer.
FLUSH_EVERY_BYTES = 8 * 1024 * 1024
# The manifest is a small JSON document; anything larger is not ours and is refused
# rather than accumulated.
MANIFEST_CHUNK_BYTES = 64 * 1024
MAX_MANIFEST_BYTES = 4 * 1024 * 1024

_TRUE_VALUES = ("1", "true", "yes", "on")
_FALSE_VALUES = ("0", "false", "no", "off", "")

# ---------------------------------------------------------------------------
# Environment configuration (requirement 7: configurable entirely through env vars)
# ---------------------------------------------------------------------------
def _env(env):
    return os.environ if env is None else env


def _flag(env, name, default):
    raw = str(_env(env).get(name, "1" if default else "0")).strip().lower()
    if raw in _TRUE_VALUES:
        return True
    if raw in _FALSE_VALUES:
        return False
    return default


def artifact_base_url(env=None):
    """
    Public-read HTTPS prefix the artifacts live under, or None when unset/blank.

    Never hard-coded: with no value configured the whole mechanism stays off, which is
    the correct default for a checkout that has its rasters locally already.
    """
    raw = str(_env(env).get("SIH_PILOT_ARTIFACT_BASE_URL", "")).strip()
    return raw.rstrip("/") or None


def manifest_url(env=None):
    """
    URL of the size+digest manifest, or None when no base URL is configured.

    SIH_PILOT_ARTIFACT_MANIFEST may be a bare filename (resolved against the base URL)
    or an absolute http(s) URL, so the manifest can live outside the object prefix.
    """
    base = artifact_base_url(env)
    if base is None:
        return None
    raw = str(_env(env).get("SIH_PILOT_ARTIFACT_MANIFEST", "")).strip() \
        or DEFAULT_MANIFEST_NAME
    if raw.lower().startswith(("http://", "https://")):
        return raw
    return "%s/%s" % (base, raw.lstrip("/"))


def artifact_url(base_url, filename):
    """Object URL for one raster filename under the configured prefix."""
    return "%s/%s" % (base_url.rstrip("/"), filename)


def artifact_states(env=None, states=PILOT_ARTIFACT_STATES):
    """
    Which states to fetch (requirement 2: state-specific), so one can be staged at a
    time. Unknown names are dropped with a warning rather than crashing startup.
    """
    raw = str(_env(env).get("SIH_PILOT_ARTIFACT_STATES", "")).strip()
    if not raw:
        return tuple(states)
    known = known_artifact_states()
    selected = []
    for name in (part.strip() for part in raw.split(",")):
        if not name:
            continue
        if name not in known:
            logger.warning("[pilot-artifacts] ignoring unknown state %r in "
                           "SIH_PILOT_ARTIFACT_STATES.", name)
            continue
        if name not in selected:
            selected.append(name)
    return tuple(selected)


def artifact_cache_dir(env=None):
    """
    Directory on a persistent disk to hold the downloaded rasters, or None to write
    them straight into backend/data (ephemeral, re-downloaded on each cold start).

    Deliberately NOT a data-root override: risk_inputs.default_data_dir() keeps
    resolving backend/data so the 38 committed files under it stay visible.
    """
    raw = str(_env(env).get("SIH_PILOT_ARTIFACT_CACHE_DIR", "")).strip()
    return os.path.abspath(raw) if raw else None


def fetch_enabled(env=None):
    """True only when a base URL is configured AND the fetch flag is not switched off."""
    if artifact_base_url(env) is None:
        return False
    return _flag(env, "SIH_PILOT_ARTIFACT_FETCH", True)


def fetch_blocking(env=None):
    """True when the fetch should run inline instead of on a background thread."""
    return _flag(env, "SIH_PILOT_ARTIFACT_BLOCKING", False)


def _positive_number(env, name, default, cast):
    raw = str(_env(env).get(name, "")).strip()
    if not raw:
        return default
    try:
        value = cast(raw)
    except (TypeError, ValueError):
        logger.warning("[pilot-artifacts] %s=%r is not a number; using %s.",
                       name, raw, default)
        return default
    if value <= 0:
        logger.warning("[pilot-artifacts] %s=%r is not positive; using %s.",
                       name, raw, default)
        return default
    return value


def fetch_timeout(env=None):
    """Per-request timeout in seconds."""
    return _positive_number(env, "SIH_PILOT_ARTIFACT_TIMEOUT",
                            DEFAULT_TIMEOUT_SECONDS, float)


def max_total_mb(env=None):
    """Upper bound on the bytes one run may download."""
    return _positive_number(env, "SIH_PILOT_ARTIFACT_MAX_TOTAL_MB",
                            DEFAULT_MAX_TOTAL_MB, int)


# ---------------------------------------------------------------------------
# Per-state wiring: where each artifact goes, and what else must point at it
# ---------------------------------------------------------------------------
def _data_root(data_dir):
    """
    Resolve the data root WITHOUT importing risk_inputs' default_data_dir.

    data_dir is passed explicitly by every caller in tests; when it is None the layout
    is derived from this file's own location, which is the same backend/data the
    prediction modules resolve. The data root is never overridden by env.
    """
    if data_dir:
        return os.path.abspath(data_dir)
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data"))


def sikkim_artifact_paths(data_dir=None):
    """
    Absolute paths of the five PUBLISHED Sikkim rasters, keyed by feature name.

    These are the sweep-family names (sikkim_dem.tif / sikkim_<name>.tif) -- the ones
    that exist as objects in storage and therefore the ones with manifest entries.
    """
    root = _data_root(data_dir)
    paths = {ELEVATION_FEATURE: os.path.join(root, "raw", SIKKIM_DEM_FILENAME)}
    for feature in SIKKIM_DERIVATIVE_FEATURES:
        paths[feature] = os.path.join(
            root, "processed", "%s%s.tif" % (SIKKIM_DERIVATIVE_PREFIX, feature))
    return paths


def sikkim_alias_paths(data_dir=None):
    """
    Absolute paths of the Sikkim SERVING twins (east_sikkim_dem.tif / real_<name>.tif),
    keyed by the same feature names, or {} if they cannot be resolved.

    Read straight off risk_inputs.terrain_raster_paths() so the alias set cannot drift
    from the paths the serving path actually opens. Only the filenames are used; the
    data root comes from _data_root and default_data_dir() is never called.

    A resolution failure is not fatal: the published names are still placed, so the
    8-state sweep view works and the serving path simply stays honestly unavailable.
    """
    try:
        from app.services import risk_inputs
        published = risk_inputs.terrain_raster_paths(_data_root(data_dir))
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("[pilot-artifacts] Sikkim serving aliases unresolved (%s: %s); "
                       "only the published names will be placed.",
                       type(exc).__name__, exc)
        return {}
    return {feature: path for feature, path in published.items()
            if feature in sikkim_artifact_paths(data_dir)}


def _missing_among(paths):
    """[(feature, path)] for every raster that is absent or empty -- the same predicate
    missing_<state>_terrain_rasters() and _<state>_dem_available() use."""
    return [(feature, path) for feature, path in sorted(paths.items())
            if not (os.path.exists(path) and os.path.getsize(path) > 0)]


def _sikkim_wiring():
    """Sikkim's wiring, kept LOCAL to this module.

    Deliberately not added to pilot_terrain_bootstrap._PILOT_MODULES: the regeneration
    path stays Sikkim-free, so nothing about Sikkim's terrain can be rebuilt on startup.
    """
    return {
        "paths": sikkim_artifact_paths,
        "missing": lambda data_dir=None: _missing_among(sikkim_artifact_paths(data_dir)),
        "aliases": sikkim_alias_paths,
    }


def artifact_wiring(state_name):
    """
    {"paths", "missing", "aliases"} for one state.

    The three pilots delegate to their own prediction module (so the files written are
    exactly the files read); Sikkim uses the local wiring above. Raises KeyError for an
    unknown state.
    """
    if state_name == SIKKIM_STATE_NAME:
        return _sikkim_wiring()
    wiring = _load_pilot_module(state_name)
    return {"paths": wiring["paths"], "missing": wiring["missing"],
            "aliases": lambda data_dir=None: {}}


def known_artifact_states():
    """Every state this module can place artifacts for."""
    return tuple(_PILOT_MODULES) + (SIKKIM_STATE_NAME,)


# ---------------------------------------------------------------------------
# Manifest (what makes verification -- and therefore honesty -- possible)
# ---------------------------------------------------------------------------
def normalize_manifest(payload):
    """
    Normalise a manifest document to {filename: {"bytes": int, "sha256": str}}.

    Accepts either {"artifacts": {...}} or a flat {filename: {...}} mapping, and either
    "bytes" or "size" for the length. Entries that do not carry BOTH a positive length
    and a 64-hex digest are dropped with a warning: an artifact with no verifiable
    entry is left missing rather than downloaded unchecked.
    """
    if not isinstance(payload, dict):
        raise ValueError("manifest must be a JSON object, got %s" % type(payload).__name__)
    raw = payload.get("artifacts", payload)
    if not isinstance(raw, dict):
        raise ValueError("manifest 'artifacts' must be a JSON object")

    entries = {}
    for filename, entry in raw.items():
        if not isinstance(entry, dict):
            logger.warning("[pilot-artifacts] manifest entry for %s is not an object; "
                           "ignoring.", filename)
            continue
        size = entry.get("bytes", entry.get("size"))
        digest = str(entry.get("sha256", "")).strip().lower()
        try:
            size = int(size)
        except (TypeError, ValueError):
            size = -1
        if size <= 0 or len(digest) != 64:
            logger.warning("[pilot-artifacts] manifest entry for %s lacks a usable "
                           "bytes/sha256 pair; ignoring.", filename)
            continue
        entries[str(filename)] = {"bytes": size, "sha256": digest}
    return entries


def _default_manifest_loader(url, timeout):
    """
    Real manifest fetch, read in bounded blocks.

    Small as this document is (20 entries), it is fetched under the same discipline as the
    rasters: never response.content, never response.text, never an unbounded read(). A
    body over MAX_MANIFEST_BYTES is refused instead of buffered -- that is not our
    manifest, and accumulating it would be the exact failure this module exists to avoid.
    """
    import json
    import requests
    blocks = []
    total = 0
    with requests.get(url, stream=True, timeout=timeout) as response:
        response.raise_for_status()
        for chunk in response.iter_content(chunk_size=MANIFEST_CHUNK_BYTES):
            if not chunk:
                continue
            total += len(chunk)
            if total > MAX_MANIFEST_BYTES:
                raise ValueError("manifest body exceeds %d bytes; refusing to buffer it"
                                 % MAX_MANIFEST_BYTES)
            blocks.append(chunk)
    return json.loads(b"".join(blocks).decode("utf-8"))


def _flush_and_release(handle, offset, length):
    """
    Push a written block to disk and ask the kernel to drop it from the page cache.

    Best-effort by design: on a platform without posix_fadvise this degrades to a plain
    flush, which still bounds the dirty-page total. A failure here must never fail a
    download -- the bytes are already correct on disk.
    """
    handle.flush()
    if length <= 0:
        return
    try:
        fileno = handle.fileno()
        os.fsync(fileno)
        if hasattr(os, "posix_fadvise"):
            os.posix_fadvise(fileno, offset, length, os.POSIX_FADV_DONTNEED)
    except (OSError, ValueError, AttributeError):  # pragma: no cover - platform dependent
        logger.debug("[pilot-artifacts] could not release page cache for a written block.")


def _default_fetcher(url, dest_path, timeout):
    """
    Real object fetch: stream the body to dest_path in small fixed-size blocks, hashing as
    it goes, and return {"bytes": int, "sha256": str}.

    Peak RSS is ONE DOWNLOAD_CHUNK_BYTES buffer regardless of artifact size. No whole-file
    bytes object exists anywhere on this path -- response.content, response.text or an
    unbounded read() would each materialise up to 184 MB, which a 512 MB instance cannot
    afford on top of the loaded application.

    The digest is computed from the chunks in flight, so the finished artifact never has
    to be read back a second time merely to hash it, and each block is flushed and
    released from the page cache so the cgroup's memory total stays flat across 2.2 GB.
    """
    import requests
    digest = hashlib.sha256()
    written = 0
    pending = 0
    with requests.get(url, stream=True, timeout=timeout) as response:
        response.raise_for_status()
        with open(dest_path, "wb") as handle:
            for chunk in response.iter_content(chunk_size=DOWNLOAD_CHUNK_BYTES):
                if not chunk:
                    continue
                handle.write(chunk)
                digest.update(chunk)
                written += len(chunk)
                pending += len(chunk)
                # Drop the reference before asking for the next block, so at most one
                # chunk is alive at a time.
                del chunk
                if pending >= FLUSH_EVERY_BYTES:
                    _flush_and_release(handle, written - pending, pending)
                    pending = 0
            _flush_and_release(handle, written - pending, pending)
    return {"bytes": written, "sha256": digest.hexdigest()}


def sha256_of_file(path, chunk_bytes=DOWNLOAD_CHUNK_BYTES):
    """SHA-256 of a file, read incrementally (never loads the raster into memory)."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        offset = 0
        while True:
            block = handle.read(chunk_bytes)
            if not block:
                break
            digest.update(block)
            read = len(block)
            del block
            # Hashing a cached 184 MB raster would otherwise pull all of it into the page
            # cache, which counts against the container limit.
            try:
                if hasattr(os, "posix_fadvise"):
                    os.posix_fadvise(handle.fileno(), offset, read,
                                     os.POSIX_FADV_DONTNEED)
            except (OSError, ValueError):  # pragma: no cover - platform dependent
                pass
            offset += read
    return digest.hexdigest()


def _observed_from_fetch(result):
    """
    Normalise what a fetcher reported: (bytes or None, sha256 or None).

    A fetcher may return a mapping carrying "bytes"/"sha256" (the streaming default, so
    the artifact never has to be re-read to be hashed), a plain int of bytes written, or
    nothing at all. Anything without a usable 64-hex digest falls back to hashing the
    file on disk -- the same check by a slower route, never a weaker one.
    """
    if isinstance(result, dict):
        digest = str(result.get("sha256", "")).strip().lower()
        try:
            size = int(result.get("bytes"))
        except (TypeError, ValueError):
            size = None
        return size, (digest if len(digest) == 64 else None)
    try:
        return int(result), None
    except (TypeError, ValueError):
        return None, None


def verify_artifact(path, entry):
    """
    Return None when path matches the manifest entry, else a human-readable reason.

    Size is checked first because it is O(1) and catches the common truncation case
    without hashing hundreds of MB.
    """
    if not os.path.exists(path):
        return "file was not written"
    actual_size = os.path.getsize(path)
    if actual_size != entry["bytes"]:
        return "size mismatch: got %d bytes, manifest says %d" % (actual_size,
                                                                 entry["bytes"])
    actual_digest = sha256_of_file(path)
    if actual_digest != entry["sha256"]:
        return "sha256 mismatch: got %s, manifest says %s" % (actual_digest,
                                                             entry["sha256"])
    return None


def verify_streamed(path, entry, observed_bytes=None, observed_digest=None):
    """
    Same contract as verify_artifact -- None when path matches the manifest entry, else a
    reason -- but able to use a digest computed during the download.

    The on-disk size is ALWAYS re-measured, so a short write cannot pass on the strength
    of a byte count the fetcher merely claimed. With no observed digest this is exactly
    verify_artifact, which is what keeps injected test fetchers on the identical path.
    """
    if observed_digest is None:
        return verify_artifact(path, entry)
    if not os.path.exists(path):
        return "file was not written"
    actual_size = os.path.getsize(path)
    if observed_bytes is not None and actual_size != observed_bytes:
        return "short write: %d bytes on disk, %d streamed" % (actual_size,
                                                              observed_bytes)
    if actual_size != entry["bytes"]:
        return "size mismatch: got %d bytes, manifest says %d" % (actual_size,
                                                                 entry["bytes"])
    if observed_digest != entry["sha256"]:
        return "sha256 mismatch: got %s, manifest says %s" % (observed_digest,
                                                             entry["sha256"])
    return None


# ---------------------------------------------------------------------------
# Local placement (cache dir + symlink, so default_data_dir() never changes)
# ---------------------------------------------------------------------------
def cache_path_for(cache_dir, final_path):
    """Where the bytes physically live: the cache dir when set, else the final path."""
    if cache_dir is None:
        return final_path
    return os.path.join(cache_dir, os.path.basename(final_path))


def link_into_place(target_path, final_path):
    """
    Make final_path resolve to target_path, preferring a symlink.

    exists() / getsize() / rasterio.open() all follow symlinks, so the prediction
    modules find their raster with no change to their path helpers. Falls back to a hard
    link and then to a copy for filesystems that refuse symlinks (e.g. Windows without
    developer mode), because correctness matters more than saving the bytes.
    """
    if target_path == final_path:
        return "in_place"
    os.makedirs(os.path.dirname(final_path), exist_ok=True)
    if os.path.islink(final_path) or os.path.exists(final_path):
        if os.path.islink(final_path) and \
                os.path.realpath(final_path) == os.path.realpath(target_path):
            return "already_linked"
        # A stale or broken link would read as missing forever; clear it.
        os.remove(final_path)
    try:
        os.symlink(target_path, final_path)
        return "symlink"
    except OSError:
        logger.debug("[pilot-artifacts] symlink unavailable for %s; trying hard link.",
                     os.path.basename(final_path))
    try:
        os.link(target_path, final_path)
        return "hardlink"
    except OSError:
        logger.debug("[pilot-artifacts] hard link unavailable for %s; copying.",
                     os.path.basename(final_path))
    shutil.copyfile(target_path, final_path)
    return "copy"


def _discard(path):
    """Remove a partial/rejected download; never let it reach a canonical path."""
    try:
        if os.path.exists(path):
            os.remove(path)
    except OSError:  # pragma: no cover - defensive
        logger.debug("[pilot-artifacts] could not remove %s", os.path.basename(path))


# ---------------------------------------------------------------------------
# Decision pass (requirements 1 and 3: idempotent, downloads only when required)
# ---------------------------------------------------------------------------
def pilot_artifact_plan(data_dir=None, states=PILOT_ARTIFACT_STATES):
    """
    Read-only: what each state is missing, per its OWN missing-raster predicate.

    Returns [{state, missing: [(feature, path)], dem_missing, error}]. Opens no raster
    and touches no network, so it is safe on every startup.
    """
    plan = []
    for state_name in states:
        entry = {"state": state_name, "missing": [], "dem_missing": None, "error": None}
        try:
            entry["missing"] = list(artifact_wiring(state_name)["missing"](data_dir))
        except Exception as exc:  # pragma: no cover - defensive
            entry["error"] = "%s: %s" % (type(exc).__name__, exc)
            plan.append(entry)
            continue
        entry["dem_missing"] = ELEVATION_FEATURE in {f for f, _p in entry["missing"]}
        plan.append(entry)
    return plan


def _fetch_one(base_url, cache_dir, final_path, entry, fetcher, timeout, aliases=()):
    """
    Materialise ONE raster at final_path. Returns (status, bytes) where status is
    "cached" (already on the persistent disk and still matching the manifest) or
    "downloaded". Raises RuntimeError with the real reason on any failure.

    Bytes never touch final_path until they have been verified: they are streamed to a
    sibling ".part", checked against the manifest, then promoted with os.replace. This
    is what stops a truncated download from satisfying the getsize(path) > 0 gate.

    aliases are ADDITIONAL canonical paths that must resolve to the same verified bytes
    (Sikkim's serving-name twins). They are linked only after verification succeeds, so
    an unverifiable download cannot make an alias look available either.
    """
    filename = os.path.basename(final_path)
    target = cache_path_for(cache_dir, final_path)
    os.makedirs(os.path.dirname(target), exist_ok=True)

    if target != final_path and os.path.exists(target):
        reason = verify_artifact(target, entry)
        if reason is None:
            placement = link_into_place(target, final_path)
            _link_aliases(target, aliases)
            logger.info("[pilot-artifacts] %s: reused cached copy (%s).",
                        filename, placement)
            return "cached", os.path.getsize(target)
        logger.warning("[pilot-artifacts] %s: cached copy rejected (%s); re-downloading.",
                       filename, reason)
        _discard(target)

    part = target + ".part"
    _discard(part)
    url = artifact_url(base_url, filename)
    logger.debug("[pilot-artifacts] %s: GET %s", filename, url)
    try:
        observed = fetcher(url, part, timeout)
        observed_bytes, observed_digest = _observed_from_fetch(observed)
        reason = verify_streamed(part, entry, observed_bytes, observed_digest)
        if reason is not None:
            raise RuntimeError("%s failed verification: %s" % (filename, reason))
        size = os.path.getsize(part)
        os.replace(part, target)
    finally:
        # Whatever happened, no partial file survives anywhere.
        _discard(part)

    placement = link_into_place(target, final_path)
    _link_aliases(target, aliases)
    logger.info("[pilot-artifacts] %s: downloaded %d bytes (%s).",
                filename, size, placement)
    return "downloaded", size


def _link_aliases(target_path, aliases):
    """
    Point every alias path at the already-verified bytes.

    A failure here is logged and swallowed: the published name is already in place, so
    downgrading it because a secondary name could not be linked would help nobody. The
    alias simply stays absent and whatever reads it reports unavailable honestly.
    """
    for alias_path in aliases:
        try:
            placement = link_into_place(target_path, alias_path)
        except OSError as exc:
            logger.warning("[pilot-artifacts] could not place alias %s (%s: %s).",
                           os.path.basename(alias_path), type(exc).__name__, exc)
            continue
        logger.info("[pilot-artifacts] %s: alias of %s (%s).",
                    os.path.basename(alias_path), os.path.basename(target_path),
                    placement)


def _capacity_problem(needed_bytes, probe_dir, limit_mb):
    """
    Reason to refuse the run up front, or None. Better to say "not enough disk" once
    than to half-populate a pilot and leave the dashboard in a mixed state.
    """
    limit_bytes = limit_mb * 1024 * 1024
    if needed_bytes > limit_bytes:
        return ("planned download is %.0f MB, above SIH_PILOT_ARTIFACT_MAX_TOTAL_MB=%d"
                % (needed_bytes / (1024.0 * 1024.0), limit_mb))
    try:
        os.makedirs(probe_dir, exist_ok=True)
        free = shutil.disk_usage(probe_dir).free
    except OSError as exc:
        logger.debug("[pilot-artifacts] could not measure free space: %s", exc)
        return None
    if free < needed_bytes * 1.05:
        return ("insufficient free disk: %.0f MB available, %.0f MB required"
                % (free / (1024.0 * 1024.0), needed_bytes / (1024.0 * 1024.0)))
    return None


def _result(state_name, missing, status=None, error=None):
    return {
        "state": state_name,
        "status": status,
        "missing_before": [feature for feature, _p in missing],
        "downloaded": [],
        "cached": [],
        "bytes": 0,
        "error": error,
    }


def _summarize(results, disabled=False, reason=None):
    return {
        "results": results,
        "disabled": disabled,
        "reason": reason,
        "acted": sum(1 for r in results if r["status"] == "prepared"),
        "skipped": sum(1 for r in results if r["status"] == "already_present"),
        "failed": sum(1 for r in results if r["status"] in (
            "failed", "incomplete", "check_failed", "manifest_unavailable", "refused")),
        "bytes": sum(r["bytes"] for r in results),
    }


def ensure_pilot_artifacts(data_dir=None, states=None, env=None,
                           fetcher=None, manifest_loader=None):
    """
    Download the missing pilot terrain rasters from object storage. Idempotent, honest,
    and never raises -- a deployment must come up even when storage is unreachable.

    Returns a report dict (see _summarize). fetcher/manifest_loader are injectable so
    the whole decision path is testable with no network and no storage credentials
    (requirement 8); production uses the real streamed requests implementations.
    """
    env = _env(env)
    base_url = artifact_base_url(env)
    if not fetch_enabled(env):
        reason = ("SIH_PILOT_ARTIFACT_BASE_URL is not set" if base_url is None
                  else "SIH_PILOT_ARTIFACT_FETCH is off")
        logger.info("[pilot-artifacts] disabled (%s); pilot DEM status will reflect "
                    "whatever is already on disk.", reason)
        return _summarize([], disabled=True, reason=reason)

    selected = artifact_states(env) if states is None else tuple(states)
    cache_dir = artifact_cache_dir(env)
    timeout = fetch_timeout(env)
    limit_mb = max_total_mb(env)
    fetch = fetcher or _default_fetcher
    load_manifest = manifest_loader or _default_manifest_loader

    results = []
    pending = []
    for entry in pilot_artifact_plan(data_dir=data_dir, states=selected):
        state_name = entry["state"]
        if entry["error"]:
            logger.warning("[pilot-artifacts] %s: could not check terrain artifacts: %s",
                           state_name, entry["error"])
            results.append(_result(state_name, [], "check_failed", entry["error"]))
        elif not entry["missing"]:
            logger.info("[pilot-artifacts] %s: terrain artifacts already present, "
                        "skipping.", state_name)
            results.append(_result(state_name, [], "already_present"))
        else:
            pending.append(entry)

    if not pending:
        # Requirement 3: nothing missing anywhere means zero network I/O -- the manifest
        # is not even fetched.
        return _summarize(results)

    try:
        manifest = normalize_manifest(load_manifest(manifest_url(env), timeout))
    except Exception as exc:
        # No manifest means no way to verify a download. Refuse rather than write
        # unchecked bytes to a path the dashboard treats as proof of availability.
        detail = "%s: %s" % (type(exc).__name__, exc)
        logger.exception("[pilot-artifacts] manifest unavailable; no artifact will be "
                         "downloaded and pilot DEM status stays unavailable.")
        for entry in pending:
            results.append(_result(entry["state"], entry["missing"],
                                   "manifest_unavailable", detail))
        return _summarize(results)

    needed_bytes = 0
    for entry in pending:
        for _feature, path in entry["missing"]:
            record = manifest.get(os.path.basename(path))
            if record:
                needed_bytes += record["bytes"]
    probe_dir = cache_dir or os.path.dirname(pending[0]["missing"][0][1])
    problem = _capacity_problem(needed_bytes, probe_dir, limit_mb)
    if problem:
        logger.error("[pilot-artifacts] refusing to fetch: %s.", problem)
        for entry in pending:
            results.append(_result(entry["state"], entry["missing"], "refused", problem))
        return _summarize(results)

    logger.info("[pilot-artifacts] fetching %d artifact(s), %.0f MB, for: %s",
                sum(len(e["missing"]) for e in pending),
                needed_bytes / (1024.0 * 1024.0),
                ", ".join(e["state"] for e in pending))

    for entry in pending:
        state_name = entry["state"]
        result = _result(state_name, entry["missing"])
        failures = []
        try:
            wiring = artifact_wiring(state_name)
            alias_paths = wiring["aliases"](data_dir)
        except Exception as exc:  # pragma: no cover - defensive
            wiring, alias_paths = None, {}
            logger.warning("[pilot-artifacts] %s: alias resolution failed (%s: %s).",
                           state_name, type(exc).__name__, exc)
        for feature, final_path in entry["missing"]:
            filename = os.path.basename(final_path)
            record = manifest.get(filename)
            if record is None:
                failures.append("%s: no verifiable manifest entry" % filename)
                logger.error("[pilot-artifacts] %s: %s is absent from the manifest; "
                             "leaving it missing.", state_name, filename)
                continue
            aliases = [alias_paths[feature]] if feature in alias_paths else []
            try:
                status, size = _fetch_one(base_url, cache_dir, final_path, record,
                                          fetch, timeout, aliases=aliases)
            except Exception as exc:
                failures.append("%s: %s: %s" % (filename, type(exc).__name__, exc))
                # Real error, traceback and all; the canonical path stays absent.
                logger.exception("[pilot-artifacts] %s: could not retrieve %s; DEM "
                                 "status will remain unavailable.", state_name, filename)
                continue
            result["bytes"] += size
            result["downloaded" if status == "downloaded" else "cached"].append(feature)

        # The verdict comes from the state's own predicate, not from what this loop
        # believes it wrote.
        try:
            still_missing = [f for f, _p in artifact_wiring(state_name)["missing"](data_dir)]
        except Exception as exc:  # pragma: no cover - defensive
            still_missing = ["<recheck failed: %s>" % exc]
        if still_missing:
            result["status"] = "failed" if len(still_missing) == len(
                result["missing_before"]) else "incomplete"
            result["error"] = "; ".join(failures) or ("still missing: %s"
                                                     % ", ".join(still_missing))
            logger.error("[pilot-artifacts] %s: %s after fetch (still missing: %s)",
                         state_name, result["status"], ", ".join(still_missing))
        else:
            result["status"] = "prepared"
            logger.info("[pilot-artifacts] %s: terrain artifacts complete.", state_name)
        results.append(result)

    return _summarize(results)


# ---------------------------------------------------------------------------
# Startup entry point (requirement 6: off the request path entirely)
# ---------------------------------------------------------------------------
def start_pilot_artifact_fetch(env=None, **kwargs):
    """
    Called once from the FastAPI lifespan hook -- never from a route handler, so
    /api/v1/validation/status latency is unaffected.

    Returns the report dict when run inline (SIH_PILOT_ARTIFACT_BLOCKING=1), the Thread
    when run in the background (default: the app binds its port and serves /health
    immediately while ~2 GB streams in), or None when the mechanism is disabled.
    Never raises.
    """
    env = _env(env)
    if not fetch_enabled(env):
        base_url = artifact_base_url(env)
        logger.info("[pilot-artifacts] not starting: %s.",
                    "SIH_PILOT_ARTIFACT_BASE_URL is not set" if base_url is None
                    else "SIH_PILOT_ARTIFACT_FETCH is off")
        return None

    if fetch_blocking(env):
        return ensure_pilot_artifacts(env=env, **kwargs)

    def _run():
        try:
            ensure_pilot_artifacts(env=env, **kwargs)
        except Exception:  # pragma: no cover - ensure_pilot_artifacts already contains
            logger.exception("[pilot-artifacts] unexpected failure in background thread.")

    thread = threading.Thread(target=_run, name="pilot-artifact-fetch", daemon=True)
    thread.start()
    return thread
