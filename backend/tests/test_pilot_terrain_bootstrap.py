"""
Focused offline tests for app.services.pilot_terrain_bootstrap -- the STARTUP
initialization that makes the three pilots' DEM + terrain rasters exist on a freshly
deployed instance, so /api/v1/validation/status stops reporting DEM missing.

These tests pin the DECISION path, not a real acquisition (which needs network +
rasterio and is therefore host-only). The DEM acquirer and the terrain processor are
injected as fakes, so what is exercised here is exactly what production integrity
depends on:

  * IDEMPOTENCE -- a state whose prediction module reports nothing missing is skipped
    with ZERO acquirer/processor calls (no download, no reprocessing).
  * NO FABRICATION -- when acquisition raises, the failure is contained and reported,
    and nothing is written that would make DEM look available.
  * WRITER/READER AGREEMENT -- the state_prefix handed to process_dem_in_chunks is
    DERIVED from each prediction module's own derivative filenames, and a DEM written
    somewhere other than where the predictor reads is an error, not a silent no-op.
  * SIKKIM IS UNTOUCHED -- it is not in PILOT_TERRAIN_STATES at all.
  * The env gates behave (disabled / blocking-inline).

Dependency budget: stdlib only. pilot_terrain_bootstrap imports rasterio, numpy and
state_validation lazily, so the module imports cleanly in the offline sandbox.
"""

import os
import sys

import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.services import pilot_terrain_bootstrap as ptb


# --------------------------------------------------------------------------------
# Fake prediction-module wiring
# --------------------------------------------------------------------------------

def _paths_for(prefix, root):
    """The same five-raster layout every pilot prediction module exposes."""
    paths = {"elevation": os.path.join(root, "raw", "%s_dem.tif" % prefix)}
    for feature in ("slope", "aspect", "roughness", "tpi"):
        paths[feature] = os.path.join(root, "processed", "%s_%s.tif" % (prefix, feature))
    return paths


def _derivative_filenames(prefix):
    return {feature: "%s_%s.tif" % (prefix, feature)
            for feature in ("slope", "aspect", "roughness", "tpi")}


class _FakePilot:
    """
    Stands in for assam_prediction / arunachal_prediction / meghalaya_prediction.

    `missing_features` is the set the module reports as absent; it is recomputed from
    `prepared` on every call so a post-bootstrap re-check can observe the change.
    """

    def __init__(self, prefix, missing_features, root="/data"):
        self.prefix = prefix
        self.root = root
        self.missing_features = set(missing_features)
        self.paths_calls = 0

    def paths(self, data_dir=None):
        self.paths_calls += 1
        return _paths_for(self.prefix, data_dir or self.root)

    def missing(self, data_dir=None):
        all_paths = _paths_for(self.prefix, data_dir or self.root)
        return sorted((f, all_paths[f]) for f in self.missing_features)

    @property
    def derivative_filenames(self):
        return _derivative_filenames(self.prefix)


def _install_fakes(monkeypatch, fakes):
    """Route _load_pilot_module at the given _FakePilot objects, keyed by state."""
    def _loader(state_name):
        fake = fakes[state_name]
        return {
            "module": fake,
            "missing": fake.missing,
            "paths": fake.paths,
            "derivative_filenames": fake.derivative_filenames,
        }

    monkeypatch.setattr(ptb, "_load_pilot_module", _loader)


class _Recorder:
    """Records acquirer / processor calls so 'skip really skipped' is checkable."""

    def __init__(self, acquire_error=None, process_error=None, acquired_path=None):
        self.acquired = []
        self.processed = []
        self.acquire_error = acquire_error
        self.process_error = process_error
        self.acquired_path = acquired_path

    def acquire(self, state_name, aoi_bounds):
        self.acquired.append((state_name, dict(aoi_bounds)))
        if self.acquire_error:
            raise self.acquire_error
        return self.acquired_path

    def process(self, dem_path, out_dir, state_prefix, chunk_size):
        self.processed.append((dem_path, out_dir, state_prefix, chunk_size))
        if self.process_error:
            raise self.process_error
        return {}


def _sandbox(monkeypatch):
    """
    ensure_pilot_terrain creates the raw/processed directories and reads the canonical
    AOI. Neither should touch the real filesystem or config in these tests. Called
    explicitly (not as an autouse fixture) so it works under the offline shim too.
    """
    monkeypatch.setattr(ptb.os, "makedirs", lambda *a, **k: None)
    import app.core.config_states as cs
    monkeypatch.setattr(
        cs, "get_pilot_aoi_bounds",
        lambda state: {"min_lat": 25.0, "max_lat": 26.0, "min_lon": 91.0, "max_lon": 92.0},
    )


# --------------------------------------------------------------------------------
# Scope
# --------------------------------------------------------------------------------

def test_sikkim_is_not_a_bootstrap_state():
    """Sikkim's terrain comes from the committed east_sikkim_dem.tif path; untouched."""
    assert ptb.PILOT_TERRAIN_STATES == ("Assam", "Arunachal Pradesh", "Meghalaya")
    assert not any("sikkim" in s.lower() for s in ptb.PILOT_TERRAIN_STATES)


def test_pilot_modules_cover_exactly_the_bootstrap_states():
    assert set(ptb._PILOT_MODULES) == set(ptb.PILOT_TERRAIN_STATES)


def test_real_wiring_resolves_against_the_prediction_modules():
    """
    The names in _PILOT_MODULES must actually exist on the real modules -- otherwise
    the bootstrap would silently degrade to 'check_failed' in production.
    """
    for state_name in ptb.PILOT_TERRAIN_STATES:
        wiring = ptb._load_pilot_module(state_name)
        assert callable(wiring["missing"])
        assert callable(wiring["paths"])
        assert set(wiring["derivative_filenames"]) == {"slope", "aspect", "roughness", "tpi"}


# --------------------------------------------------------------------------------
# Prefix derivation (writer/reader agreement)
# --------------------------------------------------------------------------------

def test_derive_terrain_state_prefix_matches_each_real_pilot():
    expected = {
        "Assam": "assam_pilot",
        "Arunachal Pradesh": "arunachal_pilot",
        "Meghalaya": "meghalaya_pilot",
    }
    for state_name, prefix in expected.items():
        wiring = ptb._load_pilot_module(state_name)
        assert ptb.derive_terrain_state_prefix(wiring["derivative_filenames"]) == prefix


def test_derive_terrain_state_prefix_rejects_inconsistent_names():
    with pytest.raises(ValueError):
        ptb.derive_terrain_state_prefix({"slope": "a_pilot_slope.tif",
                                         "aspect": "b_pilot_aspect.tif"})


def test_derive_terrain_state_prefix_rejects_unexpected_suffix():
    with pytest.raises(ValueError):
        ptb.derive_terrain_state_prefix({"slope": "assam_pilot_slope.tiff"})


# --------------------------------------------------------------------------------
# The read-only plan
# --------------------------------------------------------------------------------

def test_plan_skips_when_nothing_missing(monkeypatch):
    _install_fakes(monkeypatch, {"Meghalaya": _FakePilot("meghalaya_pilot", [])})
    entry, = ptb.pilot_terrain_plan(states=("Meghalaya",))
    assert entry["action"] == ptb.ACTION_SKIP
    assert entry["dem_missing"] is False
    assert entry["missing"] == []
    assert entry["error"] is None


def test_plan_requires_acquisition_when_dem_missing(monkeypatch):
    _install_fakes(monkeypatch, {
        "Assam": _FakePilot("assam_pilot", ["elevation", "slope", "aspect",
                                            "roughness", "tpi"]),
    })
    entry, = ptb.pilot_terrain_plan(states=("Assam",))
    assert entry["action"] == ptb.ACTION_ACQUIRE_DEM_AND_DERIVATIVES
    assert entry["dem_missing"] is True


def test_plan_derivatives_only_when_dem_present(monkeypatch):
    _install_fakes(monkeypatch, {"Assam": _FakePilot("assam_pilot", ["slope", "tpi"])})
    entry, = ptb.pilot_terrain_plan(states=("Assam",))
    assert entry["action"] == ptb.ACTION_DERIVATIVES_ONLY
    assert entry["dem_missing"] is False


def test_plan_records_check_failure_without_raising(monkeypatch):
    def _boom(state_name):
        raise ImportError("prediction module unavailable")

    monkeypatch.setattr(ptb, "_load_pilot_module", _boom)
    entry, = ptb.pilot_terrain_plan(states=("Assam",))
    assert entry["action"] == ptb.ACTION_SKIP
    assert "ImportError" in entry["error"]


# --------------------------------------------------------------------------------
# Execution: idempotence
# --------------------------------------------------------------------------------

def test_already_present_performs_no_acquisition_or_processing(monkeypatch):
    _sandbox(monkeypatch)
    fakes = {state: _FakePilot("x_pilot", []) for state in ptb.PILOT_TERRAIN_STATES}
    _install_fakes(monkeypatch, fakes)
    rec = _Recorder()

    report = ptb.ensure_pilot_terrain(dem_acquirer=rec.acquire, terrain_processor=rec.process)

    assert rec.acquired == []
    assert rec.processed == []
    assert report["skipped"] == 3
    assert report["acted"] == 0
    assert report["failed"] == 0
    assert all(r["status"] == "already_present" for r in report["results"])


def test_derivatives_only_skips_the_download(monkeypatch):
    _sandbox(monkeypatch)
    fake = _FakePilot("meghalaya_pilot", ["slope"])
    _install_fakes(monkeypatch, {"Meghalaya": fake})

    def _process(dem_path, out_dir, state_prefix, chunk_size):
        fake.missing_features.clear()

    rec = _Recorder()

    def _process_and_record(dem_path, out_dir, state_prefix, chunk_size):
        rec.process(dem_path, out_dir, state_prefix, chunk_size)
        _process(dem_path, out_dir, state_prefix, chunk_size)

    report = ptb.ensure_pilot_terrain(states=("Meghalaya",), dem_acquirer=rec.acquire,
                                      terrain_processor=_process_and_record)

    assert rec.acquired == []                      # DEM was present: no download
    assert len(rec.processed) == 1                 # derivatives regenerated
    result, = report["results"]
    assert result["dem_acquired"] is False
    assert result["derivatives_generated"] is True
    assert result["status"] == "prepared"


# --------------------------------------------------------------------------------
# Execution: the full acquire path
# --------------------------------------------------------------------------------

def test_full_acquisition_uses_derived_prefix_and_predictor_paths(monkeypatch):
    _sandbox(monkeypatch)
    root = os.path.join("/tmp", "fake-data")
    fake = _FakePilot("assam_pilot", ["elevation", "slope", "aspect", "roughness", "tpi"],
                      root=root)
    _install_fakes(monkeypatch, {"Assam": fake})
    expected_paths = _paths_for("assam_pilot", root)
    rec = _Recorder(acquired_path=expected_paths["elevation"])

    def _process_and_clear(dem_path, out_dir, state_prefix, chunk_size):
        rec.process(dem_path, out_dir, state_prefix, chunk_size)
        fake.missing_features.clear()

    report = ptb.ensure_pilot_terrain(states=("Assam",), chunk_size=256,
                                      dem_acquirer=rec.acquire,
                                      terrain_processor=_process_and_clear)

    assert [s for s, _aoi in rec.acquired] == ["Assam"]
    dem_path, out_dir, prefix, chunk_size = rec.processed[0]
    assert dem_path == expected_paths["elevation"]
    assert out_dir == os.path.dirname(expected_paths["slope"])
    assert prefix == "assam_pilot"
    assert chunk_size == 256
    result, = report["results"]
    assert result["status"] == "prepared"
    assert result["dem_acquired"] is True
    assert report["acted"] == 1


def test_dem_written_somewhere_else_is_an_error_not_a_silent_pass(monkeypatch):
    _sandbox(monkeypatch)
    fake = _FakePilot("assam_pilot", ["elevation", "slope"])
    _install_fakes(monkeypatch, {"Assam": fake})
    rec = _Recorder(acquired_path="/somewhere/else/assam_dem.tif")

    report = ptb.ensure_pilot_terrain(states=("Assam",), dem_acquirer=rec.acquire,
                                      terrain_processor=rec.process)

    result, = report["results"]
    assert result["status"] == "failed"
    assert "disagreement" in result["error"]
    assert rec.processed == []      # never processed a DEM nobody wrote
    assert report["failed"] == 1


# --------------------------------------------------------------------------------
# Execution: no fabrication on failure
# --------------------------------------------------------------------------------

def test_acquisition_failure_is_contained_and_leaves_state_unavailable(monkeypatch):
    _sandbox(monkeypatch)
    fake = _FakePilot("assam_pilot", ["elevation", "slope", "aspect", "roughness", "tpi"])
    _install_fakes(monkeypatch, {"Assam": fake})
    rec = _Recorder(acquire_error=RuntimeError("no DEM tiles could be downloaded"))

    report = ptb.ensure_pilot_terrain(states=("Assam",), dem_acquirer=rec.acquire,
                                      terrain_processor=rec.process)

    result, = report["results"]
    assert result["status"] == "failed"
    assert "no DEM tiles" in result["error"]
    assert result["derivatives_generated"] is False
    assert report["acted"] == 0 and report["failed"] == 1
    # The predictor still reports the rasters missing -> DEM status stays unavailable.
    assert len(fake.missing("/data")) == 5


def test_one_state_failing_does_not_stop_the_others(monkeypatch):
    _sandbox(monkeypatch)
    fakes = {
        "Assam": _FakePilot("assam_pilot", ["elevation"]),
        "Arunachal Pradesh": _FakePilot("arunachal_pilot", []),
        "Meghalaya": _FakePilot("meghalaya_pilot", ["slope"]),
    }
    _install_fakes(monkeypatch, fakes)
    calls = []

    def _acquire(state_name, aoi):
        calls.append(state_name)
        raise OSError("network unreachable")

    def _process(dem_path, out_dir, state_prefix, chunk_size):
        fakes["Meghalaya"].missing_features.clear()

    report = ptb.ensure_pilot_terrain(dem_acquirer=_acquire, terrain_processor=_process)
    by_state = {r["state"]: r for r in report["results"]}
    assert by_state["Assam"]["status"] == "failed"
    assert by_state["Arunachal Pradesh"]["status"] == "already_present"
    assert by_state["Meghalaya"]["status"] == "prepared"
    assert calls == ["Assam"]


def test_still_missing_after_processing_is_reported_incomplete(monkeypatch):
    _sandbox(monkeypatch)
    fake = _FakePilot("meghalaya_pilot", ["slope", "tpi"])
    _install_fakes(monkeypatch, {"Meghalaya": fake})
    rec = _Recorder()

    report = ptb.ensure_pilot_terrain(states=("Meghalaya",), dem_acquirer=rec.acquire,
                                      terrain_processor=rec.process)

    result, = report["results"]
    assert result["status"] == "incomplete"
    assert "slope" in result["error"] and "tpi" in result["error"]
    assert report["acted"] == 0 and report["failed"] == 1


def test_check_failure_is_reported_without_acquiring(monkeypatch):
    _sandbox(monkeypatch)
    def _boom(state_name):
        raise ImportError("prediction module unavailable")

    monkeypatch.setattr(ptb, "_load_pilot_module", _boom)
    rec = _Recorder()

    report = ptb.ensure_pilot_terrain(states=("Assam",), dem_acquirer=rec.acquire,
                                      terrain_processor=rec.process)

    result, = report["results"]
    assert result["status"] == "check_failed"
    assert rec.acquired == [] and rec.processed == []
    assert report["failed"] == 1


# --------------------------------------------------------------------------------
# Env gates / startup entry point
# --------------------------------------------------------------------------------

@pytest.mark.parametrize("raw,expected", [
    (None, False), ("", False),
    ("0", False), ("false", False), ("no", False), ("off", False),
    ("1", True), ("true", True), ("on", True), ("yes", True),
])
def test_bootstrap_enabled_gate(raw, expected):
    """
    DEFAULT MUST BE OFF. Regenerating the mosaics OOM-killed the Render instance
    (exit 137); production downloads the prebuilt rasters instead. Opting in must be
    explicit, so an unset variable can never start a multi-hundred-MB rebuild.
    """
    env = {} if raw is None else {"SIH_PILOT_TERRAIN_BOOTSTRAP": raw}
    assert ptb.bootstrap_enabled(env) is expected


def test_start_does_nothing_when_the_variable_is_unset():
    """An unset SIH_PILOT_TERRAIN_BOOTSTRAP must not regenerate anything."""
    assert ptb.start_pilot_terrain_bootstrap(env={}) is None


@pytest.mark.parametrize("raw,expected", [
    (None, False), ("0", False), ("1", True), ("yes", True), ("on", True),
])
def test_bootstrap_blocking_gate(raw, expected):
    env = {} if raw is None else {"SIH_PILOT_TERRAIN_BOOTSTRAP_BLOCKING": raw}
    assert ptb.bootstrap_blocking(env) is expected


def test_start_returns_none_and_does_nothing_when_disabled(monkeypatch):
    called = []
    monkeypatch.setattr(ptb, "ensure_pilot_terrain",
                        lambda **kw: called.append(kw))
    out = ptb.start_pilot_terrain_bootstrap(env={"SIH_PILOT_TERRAIN_BOOTSTRAP": "0"})
    assert out is None
    assert called == []


def test_start_runs_inline_when_blocking(monkeypatch):
    monkeypatch.setattr(ptb, "ensure_pilot_terrain",
                        lambda **kw: {"results": [], "acted": 0, "skipped": 0, "failed": 0})
    out = ptb.start_pilot_terrain_bootstrap(
        env={"SIH_PILOT_TERRAIN_BOOTSTRAP": "1",
             "SIH_PILOT_TERRAIN_BOOTSTRAP_BLOCKING": "1"})
    assert isinstance(out, dict)
    assert out["acted"] == 0


def test_start_uses_a_background_daemon_thread_by_default(monkeypatch):
    done = []
    monkeypatch.setattr(ptb, "ensure_pilot_terrain", lambda **kw: done.append(True))
    thread = ptb.start_pilot_terrain_bootstrap(
        env={"SIH_PILOT_TERRAIN_BOOTSTRAP": "1"})
    assert isinstance(thread, ptb.threading.Thread)
    assert thread.daemon is True
    thread.join(timeout=5)
    assert done == [True]


def test_background_thread_swallows_unexpected_failures(monkeypatch):
    def _boom(**kwargs):
        raise RuntimeError("unexpected")

    monkeypatch.setattr(ptb, "ensure_pilot_terrain", _boom)
    thread = ptb.start_pilot_terrain_bootstrap(
        env={"SIH_PILOT_TERRAIN_BOOTSTRAP": "1"})
    thread.join(timeout=5)
    assert not thread.is_alive()


# --------------------------------------------------------------------------------
# The real acquirer must use the UNCAPPED tile mode
# --------------------------------------------------------------------------------

def test_default_acquirer_disables_the_2x2_tile_cap(monkeypatch):
    """
    The pilot AOIs need up to 6 Copernicus tiles; the sweep's 2x2 cap would clip the
    mosaic and leave real AOI cells with no terrain. The bootstrap must therefore call
    acquire_state_dem with limit_tiles=False.
    """
    import app.services.state_validation as sv
    seen = {}

    def _fake_acquire(state_name, state_config, limit_tiles=True):
        seen["state"] = state_name
        seen["limit_tiles"] = limit_tiles
        seen["config"] = state_config
        return "/data/raw/assam_pilot_dem.tif"

    monkeypatch.setattr(sv, "acquire_state_dem", _fake_acquire)
    out = ptb._default_dem_acquirer("Assam", {"min_lat": 25.6, "max_lat": 26.6,
                                              "min_lon": 91.3, "max_lon": 93.7})
    assert out == "/data/raw/assam_pilot_dem.tif"
    assert seen["limit_tiles"] is False
    assert seen["state"] == "Assam"
    assert seen["config"]["max_lon"] == 93.7


# --------------------------------------------------------------------------------
# The startup wiring in app/main.py
# --------------------------------------------------------------------------------

def _main_source():
    path = os.path.join(os.path.dirname(__file__), "..", "app", "main.py")
    with open(os.path.abspath(path), "r", encoding="utf-8") as handle:
        return handle.read()


def test_main_registers_the_bootstrap_on_the_app_lifespan():
    """
    fastapi is absent in the offline sandbox, so importing app.main is not possible
    here; this pins the wiring at the SOURCE level instead. Without it the bootstrap
    module would exist but never run, and a fresh deployment would still report DEM
    missing.
    """
    source = _main_source()
    assert "start_pilot_terrain_bootstrap" in source
    assert "lifespan=lifespan" in source
    assert "async def lifespan" in source
    # The call must be inside the lifespan hook, i.e. after its definition.
    assert source.index("async def lifespan") < source.index("start_pilot_terrain_bootstrap()")


def test_main_does_not_bootstrap_per_request():
    """The work must not be hung off a route handler or middleware."""
    source = _main_source()
    for forbidden in ("@app.get", "@app.post", "@app.middleware"):
        before, _sep, after = source.partition("start_pilot_terrain_bootstrap()")
        assert forbidden not in before or "async def lifespan" in before
        assert "start_pilot_terrain_bootstrap" not in after
