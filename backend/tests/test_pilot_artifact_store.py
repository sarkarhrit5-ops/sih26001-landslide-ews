"""
Focused offline tests for app.services.pilot_artifact_store -- the PRODUCTION path that
makes each pilot state's five terrain rasters exist on a freshly deployed instance by
DOWNLOADING the prebuilt artifacts, instead of regenerating Copernicus mosaics (which
OOM-killed the Render instance with exit status 137).

No network and NO STORAGE CREDENTIALS are required: the manifest loader and the object
fetcher are injected collaborators, and a small in-memory _Store stands in for the
bucket. What is pinned here is exactly what production integrity depends on:

  * IDEMPOTENCE -- a state whose prediction module reports nothing missing performs zero
    network I/O; the manifest is not even fetched.
  * ONLY WHAT IS MISSING -- an already-present raster is never re-downloaded.
  * NO FABRICATION -- a truncated body, a digest mismatch, a 404, an unreachable
    manifest or a filename absent from the manifest all leave the canonical path ABSENT,
    so DEM status stays honestly unavailable. Nothing partial is ever promoted.
  * STATE-SPECIFIC -- selecting one pilot fetches only that pilot's artifacts, and one
    state failing does not stop the others.
  * CACHING -- with SIH_PILOT_ARTIFACT_CACHE_DIR the bytes live on the persistent disk
    and the canonical path becomes a symlink, so a redeploy re-links instead of
    re-downloading; risk_inputs.default_data_dir() is never touched.
  * SIKKIM'S TWO NAME FAMILIES -- Sikkim IS an artifact state now (no .tif is tracked in
    Git, so a fresh deployment has no Sikkim terrain either). The five UPLOADED objects
    are the sweep names (sikkim_dem.tif / sikkim_<name>.tif) and only they carry manifest
    entries; the serving twins (east_sikkim_dem.tif / real_<name>.tif) are placed as
    additional links to the SAME verified bytes, only after verification passes. Sikkim
    still has no entry in _PILOT_MODULES, so the regeneration path stays Sikkim-free.
  * Env gates behave, and object URLs never reach INFO-level logs.

Dependency budget: stdlib only (requests is imported lazily, inside the real fetcher).
"""

import ast
import hashlib
import json
import logging
import os
import sys

import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.services import pilot_artifact_store as pas


FEATURES = ("elevation", "slope", "aspect", "roughness", "tpi")


def _paths_for(prefix, root):
    """The same five-raster layout every pilot prediction module exposes."""
    paths = {"elevation": os.path.join(root, "raw", "%s_dem.tif" % prefix)}
    for feature in ("slope", "aspect", "roughness", "tpi"):
        paths[feature] = os.path.join(root, "processed", "%s_%s.tif" % (prefix, feature))
    return paths


class _FsPilot:
    """
    Stands in for assam_prediction / arunachal_prediction / meghalaya_prediction, backed
    by the REAL filesystem and replicating the production predicate byte-for-byte
    (exists and getsize > 0) -- including its weakness, which is the point: these tests
    prove a partial download never satisfies it.
    """

    def __init__(self, prefix, root):
        self.prefix = prefix
        self.root = root

    def paths(self, data_dir=None):
        return _paths_for(self.prefix, data_dir or self.root)

    def missing(self, data_dir=None):
        return [(name, path)
                for name, path in sorted(self.paths(data_dir).items())
                if not (os.path.exists(path) and os.path.getsize(path) > 0)]


class _Store:
    """
    In-memory stand-in for public-read object storage. `payloads` maps filename -> bytes;
    a filename that is absent behaves like a 404. Records every URL it is asked for so
    tests can assert on what was NOT fetched.
    """

    def __init__(self, payloads, truncate=None, corrupt=()):
        self.payloads = dict(payloads)
        self.truncate = truncate or {}
        self.corrupt = set(corrupt)
        self.gets = []
        self.manifest_loads = []

    def manifest(self):
        return {"artifacts": {
            name: {"bytes": len(body), "sha256": hashlib.sha256(body).hexdigest()}
            for name, body in self.payloads.items()
        }}

    def loader(self, url, timeout):
        self.manifest_loads.append(url)
        return json.loads(json.dumps(self.manifest()))

    def fetcher(self, url, dest_path, timeout):
        self.gets.append(url)
        name = url.rsplit("/", 1)[-1]
        if name not in self.payloads:
            raise RuntimeError("HTTP 404 for %s" % name)
        body = self.payloads[name]
        if name in self.truncate:
            body = body[:self.truncate[name]]
        elif name in self.corrupt:
            body = bytes(bytearray(len(body)))  # right length, wrong content
        with open(dest_path, "wb") as handle:
            handle.write(body)
        return len(body)


def _fake_root(tmp_path, name="data"):
    root = os.path.join(str(tmp_path), name)
    os.makedirs(os.path.join(root, "raw"), exist_ok=True)
    os.makedirs(os.path.join(root, "processed"), exist_ok=True)
    return root


def _install_pilots(monkeypatch, pilots):
    """Point _load_pilot_module at the filesystem-backed fakes."""
    def _load(state_name):
        pilot = pilots[state_name]
        return {"module": None, "missing": pilot.missing, "paths": pilot.paths,
                "derivative_filenames": {}}

    monkeypatch.setattr(pas, "_load_pilot_module", _load)


def _payloads_for(prefix, size=64):
    """Distinct bodies per filename so a mixed-up file cannot pass verification."""
    payloads = {}
    for name, path in _paths_for(prefix, "/root").items():
        filename = os.path.basename(path)
        payloads[filename] = (("%s:%s:" % (prefix, name)).encode("utf-8") * size)
    return payloads


def _write(path, body=b"already here"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as handle:
        handle.write(body)


def _all_files(root):
    found = []
    for dirpath, _dirnames, filenames in os.walk(root):
        for filename in filenames:
            found.append(os.path.join(dirpath, filename))
    return found


# --------------------------------------------------------------------------------
# Scope: which states this mechanism covers
# --------------------------------------------------------------------------------

def test_sikkim_is_an_artifact_state_but_not_a_regeneration_state():
    """
    Sikkim's rasters are published too -- `git ls-files '*.tif'` is empty, so nothing
    about Sikkim's terrain is committed and a fresh deployment must download it. What
    must NOT happen is Sikkim joining the Copernicus REGENERATION path, which is the
    thing that OOM-killed Render: that path is driven by _PILOT_MODULES.
    """
    assert "Sikkim" in pas.PILOT_ARTIFACT_STATES
    assert pas.SIKKIM_STATE_NAME == "Sikkim"
    assert pas.artifact_states({}) == pas.PILOT_ARTIFACT_STATES
    assert pas.artifact_states({"SIH_PILOT_ARTIFACT_STATES": "Sikkim"}) == ("Sikkim",)
    assert "Sikkim" not in pas._PILOT_MODULES
    assert not any("sikkim" in name.lower() for name in pas._PILOT_MODULES)


def test_every_artifact_state_has_real_wiring():
    """A typo here would silently skip a state forever."""
    assert set(pas.PILOT_ARTIFACT_STATES) == set(pas.known_artifact_states())
    assert set(pas.known_artifact_states()) == set(pas._PILOT_MODULES) | {"Sikkim"}
    for state_name in pas.PILOT_ARTIFACT_STATES:
        wiring = pas.artifact_wiring(state_name)
        assert callable(wiring["missing"]) and callable(wiring["paths"])
        assert callable(wiring["aliases"])
    with pytest.raises(KeyError):
        pas.artifact_wiring("Kerala")


def test_the_store_never_overrides_the_data_root():
    """
    backend/data holds 38 COMMITTED files (four model pickles, state_validation.json,
    the OSM GeoJSONs, the events snapshots). This module must cache elsewhere and
    symlink, never repoint risk_inputs.default_data_dir().
    """
    path = os.path.join(os.path.dirname(__file__), "..", "app", "services",
                        "pilot_artifact_store.py")
    with open(os.path.abspath(path), "r", encoding="utf-8") as handle:
        source = handle.read()
    assert "SIH_DATA_DIR" not in source
    # risk_inputs is consulted for the Sikkim alias FILENAMES only, and lazily -- never
    # for the data root, which would let this module redefine where the rest of the
    # backend looks for its committed files. Checked on the parsed tree so the
    # docstring's prose about default_data_dir() cannot mask a real call.
    names = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Attribute):
            names.add(node.attr)
        elif isinstance(node, ast.Name):
            names.add(node.id)
    assert "default_data_dir" not in names
    assert not any("risk_inputs" in line for line in source.splitlines()
                   if line.startswith(("import ", "from ")))


# --------------------------------------------------------------------------------
# Environment configuration (requirement 7)
# --------------------------------------------------------------------------------

@pytest.mark.parametrize("raw,expected", [
    (None, None), ("", None), ("   ", None),
    ("https://cdn.example/pilots", "https://cdn.example/pilots"),
    ("https://cdn.example/pilots/", "https://cdn.example/pilots"),
])
def test_base_url_is_never_hard_coded(raw, expected):
    env = {} if raw is None else {"SIH_PILOT_ARTIFACT_BASE_URL": raw}
    assert pas.artifact_base_url(env) == expected


def test_manifest_url_defaults_resolves_and_accepts_absolute():
    base = {"SIH_PILOT_ARTIFACT_BASE_URL": "https://cdn.example/pilots/"}
    assert pas.manifest_url(base) == "https://cdn.example/pilots/pilot_manifest.json"
    named = dict(base, SIH_PILOT_ARTIFACT_MANIFEST="v2/terrain.json")
    assert pas.manifest_url(named) == "https://cdn.example/pilots/v2/terrain.json"
    absolute = dict(base, SIH_PILOT_ARTIFACT_MANIFEST="https://other.example/m.json")
    assert pas.manifest_url(absolute) == "https://other.example/m.json"
    assert pas.manifest_url({}) is None


def test_default_manifest_name_matches_the_published_object():
    """
    Regression: the default was "manifest.json" while the publisher writes -- and the
    Hugging Face dataset holds -- "pilot_manifest.json". A fresh deployment therefore
    requested an object that does not exist and refused every artifact with
    manifest_unavailable. The default must be the name that was actually uploaded.
    """
    assert pas.DEFAULT_MANIFEST_NAME == "pilot_manifest.json"
    # The exact URL a deployment with only the base URL set will request.
    env = {"SIH_PILOT_ARTIFACT_BASE_URL":
           "https://huggingface.co/datasets/USICT-LazyCoders-ai/sih26001-terrain/"
           "resolve/main"}
    assert pas.manifest_url(env) == (
        "https://huggingface.co/datasets/USICT-LazyCoders-ai/sih26001-terrain/"
        "resolve/main/pilot_manifest.json")
    # The base URL itself is untouched by the manifest name, and object URLs keep using
    # the raster basenames -- this fix must not move any artifact.
    assert pas.artifact_base_url(env) == env["SIH_PILOT_ARTIFACT_BASE_URL"]
    assert pas.artifact_url(pas.artifact_base_url(env), "sikkim_dem.tif") == (
        "https://huggingface.co/datasets/USICT-LazyCoders-ai/sih26001-terrain/"
        "resolve/main/sikkim_dem.tif")
    # An explicit override still wins, so a differently named store needs no code change.
    assert pas.manifest_url(dict(env, SIH_PILOT_ARTIFACT_MANIFEST="manifest.json")) == (
        "https://huggingface.co/datasets/USICT-LazyCoders-ai/sih26001-terrain/"
        "resolve/main/manifest.json")


def test_a_run_requests_the_default_manifest_name(monkeypatch, tmp_path):
    """The URL the loader is handed during a real run, not just what manifest_url says."""
    root, _pilots, store = _wire(monkeypatch, tmp_path, ["Assam"])
    report = _run(root, ["Assam"], store)
    assert store.manifest_loads == ["https://cdn.example/pilots/pilot_manifest.json"]
    # Behaviour unchanged: the objects themselves are still fetched by raster basename.
    assert sorted(url.rsplit("/", 1)[-1] for url in store.gets) == [
        "assam_pilot_aspect.tif", "assam_pilot_dem.tif", "assam_pilot_roughness.tif",
        "assam_pilot_slope.tif", "assam_pilot_tpi.tif"]
    assert report["failed"] == 0


def test_artifact_url_joins_without_double_slash():
    assert pas.artifact_url("https://cdn.example/p/", "assam_pilot_dem.tif") == \
        "https://cdn.example/p/assam_pilot_dem.tif"


def test_states_can_be_narrowed_and_unknown_names_are_dropped():
    env = {"SIH_PILOT_ARTIFACT_STATES": "Meghalaya, Assam"}
    assert pas.artifact_states(env) == ("Meghalaya", "Assam")
    # Sikkim is selectable now; a name with no wiring at all still is not.
    assert pas.artifact_states({"SIH_PILOT_ARTIFACT_STATES": "Sikkim,Assam"}) == \
        ("Sikkim", "Assam")
    assert pas.artifact_states({"SIH_PILOT_ARTIFACT_STATES": "Kerala,Assam"}) == ("Assam",)
    # A value naming nothing valid selects nothing -- it must not silently fall back to
    # all four, since that would re-enable a state the operator tried to exclude.
    assert pas.artifact_states({"SIH_PILOT_ARTIFACT_STATES": " , "}) == ()
    assert pas.artifact_states({"SIH_PILOT_ARTIFACT_STATES": ""}) == \
        pas.PILOT_ARTIFACT_STATES


def test_cache_dir_is_absolute_or_none():
    assert pas.artifact_cache_dir({}) is None
    assert pas.artifact_cache_dir({"SIH_PILOT_ARTIFACT_CACHE_DIR": "  "}) is None
    resolved = pas.artifact_cache_dir({"SIH_PILOT_ARTIFACT_CACHE_DIR": "/var/data/pilots"})
    assert resolved == os.path.abspath("/var/data/pilots")


def test_fetch_requires_a_base_url_and_honours_the_off_switch():
    assert pas.fetch_enabled({}) is False
    # An explicit FETCH=1 must not be enough on its own -- there is nothing to fetch from.
    assert pas.fetch_enabled({"SIH_PILOT_ARTIFACT_FETCH": "1"}) is False
    base = {"SIH_PILOT_ARTIFACT_BASE_URL": "https://cdn.example/p"}
    assert pas.fetch_enabled(base) is True
    assert pas.fetch_enabled(dict(base, SIH_PILOT_ARTIFACT_FETCH="0")) is False
    assert pas.fetch_enabled(dict(base, SIH_PILOT_ARTIFACT_FETCH="off")) is False


@pytest.mark.parametrize("raw,expected", [
    (None, False), ("0", False), ("", False), ("1", True), ("yes", True), ("on", True),
])
def test_blocking_gate(raw, expected):
    env = {} if raw is None else {"SIH_PILOT_ARTIFACT_BLOCKING": raw}
    assert pas.fetch_blocking(env) is expected


def test_numeric_settings_fall_back_on_junk():
    assert pas.fetch_timeout({}) == pas.DEFAULT_TIMEOUT_SECONDS
    assert pas.fetch_timeout({"SIH_PILOT_ARTIFACT_TIMEOUT": "45"}) == 45.0
    assert pas.fetch_timeout({"SIH_PILOT_ARTIFACT_TIMEOUT": "soon"}) == \
        pas.DEFAULT_TIMEOUT_SECONDS
    assert pas.fetch_timeout({"SIH_PILOT_ARTIFACT_TIMEOUT": "-5"}) == \
        pas.DEFAULT_TIMEOUT_SECONDS
    assert pas.max_total_mb({}) == pas.DEFAULT_MAX_TOTAL_MB
    assert pas.max_total_mb({"SIH_PILOT_ARTIFACT_MAX_TOTAL_MB": "3000"}) == 3000


# --------------------------------------------------------------------------------
# Manifest + verification: the only thing standing between a partial download and a
# dashboard that claims the DEM is available
# --------------------------------------------------------------------------------

def test_manifest_accepts_both_shapes_and_the_size_alias():
    digest = "a" * 64
    wrapped = pas.normalize_manifest({"artifacts": {"a.tif": {"bytes": 5,
                                                              "sha256": digest}}})
    assert wrapped == {"a.tif": {"bytes": 5, "sha256": digest}}
    flat = pas.normalize_manifest({"a.tif": {"size": 5, "sha256": digest.upper()}})
    assert flat["a.tif"] == {"bytes": 5, "sha256": digest}


def test_manifest_drops_unverifiable_entries_and_rejects_non_objects():
    entries = pas.normalize_manifest({"artifacts": {
        "good.tif": {"bytes": 3, "sha256": "b" * 64},
        "no_digest.tif": {"bytes": 3},
        "short_digest.tif": {"bytes": 3, "sha256": "b" * 10},
        "no_size.tif": {"sha256": "b" * 64},
        "zero.tif": {"bytes": 0, "sha256": "b" * 64},
        "not_an_object.tif": "whatever",
    }})
    assert list(entries) == ["good.tif"]
    with pytest.raises(ValueError):
        pas.normalize_manifest([1, 2, 3])
    with pytest.raises(ValueError):
        pas.normalize_manifest({"artifacts": "nope"})


def test_verify_detects_absence_truncation_and_corruption(tmp_path):
    body = b"terrain bytes"
    path = os.path.join(str(tmp_path), "a.tif")
    entry = {"bytes": len(body), "sha256": hashlib.sha256(body).hexdigest()}
    assert pas.verify_artifact(path, entry) == "file was not written"
    _write(path, body[:4])
    assert "size mismatch" in pas.verify_artifact(path, entry)
    _write(path, bytes(bytearray(len(body))))
    assert "sha256 mismatch" in pas.verify_artifact(path, entry)
    _write(path, body)
    assert pas.verify_artifact(path, entry) is None
    assert pas.sha256_of_file(path) == entry["sha256"]


# --------------------------------------------------------------------------------
# ensure_pilot_artifacts: the decision path
# --------------------------------------------------------------------------------

PREFIXES = {"Assam": "assam_pilot",
            "Arunachal Pradesh": "arunachal_pilot",
            "Meghalaya": "meghalaya_pilot"}

BASE = "https://cdn.example/pilots"


def _env(**extra):
    return dict({"SIH_PILOT_ARTIFACT_BASE_URL": BASE}, **extra)


def _wire(monkeypatch, tmp_path, states):
    """Filesystem-backed pilots plus a store that holds exactly their artifacts."""
    root = _fake_root(tmp_path)
    pilots = {state: _FsPilot(PREFIXES[state], root) for state in states}
    _install_pilots(monkeypatch, pilots)
    payloads = {}
    for state in states:
        payloads.update(_payloads_for(PREFIXES[state]))
    return root, pilots, _Store(payloads)


def _run(root, states, store, env=None, **kwargs):
    return pas.ensure_pilot_artifacts(
        data_dir=root, states=tuple(states),
        env=_env() if env is None else env,
        fetcher=store.fetcher, manifest_loader=store.loader, **kwargs)


def test_disabled_without_a_base_url_touches_nothing(monkeypatch, tmp_path):
    root, _pilots, store = _wire(monkeypatch, tmp_path, ["Assam"])
    report = _run(root, ["Assam"], store, env={})
    assert report["disabled"] is True
    assert store.gets == [] and store.manifest_loads == []
    assert _all_files(root) == []


def test_nothing_missing_means_zero_network_io(monkeypatch, tmp_path):
    """Requirements 1 and 3: the manifest is not even fetched."""
    root, pilots, store = _wire(monkeypatch, tmp_path, ["Assam"])
    for _feature, path in _paths_for("assam_pilot", root).items():
        _write(path)
    report = _run(root, ["Assam"], store)
    assert store.manifest_loads == [] and store.gets == []
    assert [r["status"] for r in report["results"]] == ["already_present"]
    assert report["skipped"] == 1 and report["bytes"] == 0


def test_only_the_missing_rasters_are_downloaded(monkeypatch, tmp_path):
    root, pilots, store = _wire(monkeypatch, tmp_path, ["Assam"])
    paths = _paths_for("assam_pilot", root)
    _write(paths["elevation"])
    _write(paths["slope"])
    report = _run(root, ["Assam"], store)
    fetched = sorted(url.rsplit("/", 1)[-1] for url in store.gets)
    assert fetched == ["assam_pilot_aspect.tif", "assam_pilot_roughness.tif",
                       "assam_pilot_tpi.tif"]
    assert report["results"][0]["status"] == "prepared"
    assert sorted(report["results"][0]["downloaded"]) == ["aspect", "roughness", "tpi"]


def test_full_fetch_writes_verified_bytes_to_the_canonical_paths(monkeypatch, tmp_path):
    root, pilots, store = _wire(monkeypatch, tmp_path, ["Assam"])
    report = _run(root, ["Assam"], store)
    assert report["acted"] == 1 and report["failed"] == 0
    assert pilots["Assam"].missing(root) == []
    for _feature, path in _paths_for("assam_pilot", root).items():
        with open(path, "rb") as handle:
            assert handle.read() == store.payloads[os.path.basename(path)]
    assert report["bytes"] == sum(len(b) for b in store.payloads.values())
    assert store.manifest_loads == ["%s/pilot_manifest.json" % BASE]


def test_state_specific_selection_leaves_other_pilots_alone(monkeypatch, tmp_path):
    """Requirement 2, and the staging path for a one-state rollout."""
    states = list(PREFIXES)
    root, pilots, store = _wire(monkeypatch, tmp_path, states)
    report = _run(root, ["Meghalaya"], store)
    assert [r["state"] for r in report["results"]] == ["Meghalaya"]
    assert all("meghalaya_pilot" in url for url in store.gets)
    assert pilots["Assam"].missing(root) != []
    assert pilots["Arunachal Pradesh"].missing(root) != []


# --------------------------------------------------------------------------------
# No fabrication: every failure mode must leave the canonical path ABSENT
# --------------------------------------------------------------------------------

def test_truncated_download_is_never_promoted(monkeypatch, tmp_path):
    """
    The production predicate accepts any file with getsize > 0, so a short read that
    landed at the canonical path would read as Available and break the predictor later.
    """
    root, pilots, store = _wire(monkeypatch, tmp_path, ["Assam"])
    store.truncate = {"assam_pilot_dem.tif": 10}
    report = _run(root, ["Assam"], store)
    dem = _paths_for("assam_pilot", root)["elevation"]
    assert not os.path.exists(dem)
    assert ("elevation", dem) in pilots["Assam"].missing(root)
    assert report["results"][0]["status"] == "incomplete"
    assert "size mismatch" in report["results"][0]["error"]
    assert [p for p in _all_files(root) if p.endswith(".part")] == []


def test_corrupt_body_of_the_right_length_is_rejected(monkeypatch, tmp_path):
    root, pilots, store = _wire(monkeypatch, tmp_path, ["Assam"])
    store.corrupt = {"assam_pilot_slope.tif"}
    report = _run(root, ["Assam"], store)
    slope = _paths_for("assam_pilot", root)["slope"]
    assert not os.path.exists(slope)
    assert "sha256 mismatch" in report["results"][0]["error"]
    assert report["failed"] == 1


def test_a_404_leaves_that_raster_missing(monkeypatch, tmp_path):
    root, pilots, store = _wire(monkeypatch, tmp_path, ["Assam"])
    del store.payloads["assam_pilot_tpi.tif"]  # also absent from the manifest
    report = _run(root, ["Assam"], store)
    tpi = _paths_for("assam_pilot", root)["tpi"]
    assert not os.path.exists(tpi)
    assert report["results"][0]["status"] == "incomplete"
    assert "no verifiable manifest entry" in report["results"][0]["error"]
    # Not in the manifest means not even attempted.
    assert not any(url.endswith("assam_pilot_tpi.tif") for url in store.gets)
    # The other four still landed -- a partial failure is not an excuse to do nothing.
    assert sorted(report["results"][0]["downloaded"]) == ["aspect", "elevation",
                                                          "roughness", "slope"]


def test_manifest_lists_it_but_storage_404s(monkeypatch, tmp_path):
    root, pilots, store = _wire(monkeypatch, tmp_path, ["Assam"])
    manifest = store.manifest()

    def _loader(url, timeout):
        store.manifest_loads.append(url)
        return manifest

    del store.payloads["assam_pilot_aspect.tif"]
    report = pas.ensure_pilot_artifacts(data_dir=root, states=("Assam",), env=_env(),
                                        fetcher=store.fetcher, manifest_loader=_loader)
    assert not os.path.exists(_paths_for("assam_pilot", root)["aspect"])
    assert "HTTP 404" in report["results"][0]["error"]
    assert report["results"][0]["status"] == "incomplete"


def test_unreachable_manifest_downloads_nothing(monkeypatch, tmp_path):
    root, pilots, store = _wire(monkeypatch, tmp_path, ["Assam", "Meghalaya"])

    def _loader(url, timeout):
        raise RuntimeError("connection refused")

    report = pas.ensure_pilot_artifacts(data_dir=root, states=("Assam", "Meghalaya"),
                                        env=_env(), fetcher=store.fetcher,
                                        manifest_loader=_loader)
    assert store.gets == []
    assert {r["status"] for r in report["results"]} == {"manifest_unavailable"}
    assert report["failed"] == 2
    assert _all_files(root) == []


def test_one_state_failing_does_not_stop_the_others(monkeypatch, tmp_path):
    states = list(PREFIXES)
    root, pilots, store = _wire(monkeypatch, tmp_path, states)
    for filename in list(store.payloads):
        if filename.startswith("arunachal_pilot"):
            del store.payloads[filename]
    report = _run(root, states, store)
    by_state = {r["state"]: r for r in report["results"]}
    assert by_state["Assam"]["status"] == "prepared"
    assert by_state["Meghalaya"]["status"] == "prepared"
    assert by_state["Arunachal Pradesh"]["status"] == "failed"
    assert pilots["Arunachal Pradesh"].missing(root) != []
    assert report["acted"] == 2 and report["failed"] == 1


def test_capacity_problem_reports_budget_and_disk(tmp_path):
    probe = os.path.join(str(tmp_path), "probe")
    assert pas._capacity_problem(10, probe, 1) is None
    over = pas._capacity_problem(5 * 1024 * 1024, probe, 1)
    assert over is not None and "above SIH_PILOT_ARTIFACT_MAX_TOTAL_MB" in over
    huge = pas._capacity_problem(1 << 60, probe, 1 << 40)
    assert huge is not None and "insufficient free disk" in huge


def test_refuses_up_front_when_the_manifest_exceeds_the_budget(monkeypatch, tmp_path):
    """A refusal must happen BEFORE any bytes move, not halfway through a pilot."""
    root, pilots, store = _wire(monkeypatch, tmp_path, ["Assam"])
    inflated = {"artifacts": {name: {"bytes": 50 * 1024 * 1024,
                                     "sha256": hashlib.sha256(body).hexdigest()}
                              for name, body in store.payloads.items()}}

    def _loader(url, timeout):
        return inflated

    report = pas.ensure_pilot_artifacts(
        data_dir=root, states=("Assam",),
        env=_env(SIH_PILOT_ARTIFACT_MAX_TOTAL_MB="10"),
        fetcher=store.fetcher, manifest_loader=_loader)
    assert store.gets == []
    assert report["results"][0]["status"] == "refused"
    assert "above SIH_PILOT_ARTIFACT_MAX_TOTAL_MB" in report["results"][0]["error"]
    assert pilots["Assam"].missing(root) != []
    assert _all_files(root) == []


# --------------------------------------------------------------------------------
# Requirement 4: cached locally -- persistent disk + symlink, data root untouched
# --------------------------------------------------------------------------------

def test_cache_dir_holds_the_bytes_and_the_canonical_path_is_a_link(monkeypatch,
                                                                   tmp_path):
    root, pilots, store = _wire(monkeypatch, tmp_path, ["Assam"])
    cache = os.path.join(str(tmp_path), "persistent")
    report = _run(root, ["Assam"], store,
                  env=_env(SIH_PILOT_ARTIFACT_CACHE_DIR=cache))
    assert report["acted"] == 1
    for _feature, path in _paths_for("assam_pilot", root).items():
        cached = os.path.join(cache, os.path.basename(path))
        assert os.path.isfile(cached)
        assert os.path.islink(path)
        assert os.path.realpath(path) == os.path.realpath(cached)
    # The predicate the dashboard reads follows the link, so status flips to available.
    assert pilots["Assam"].missing(root) == []


def test_a_redeploy_relinks_the_cache_without_downloading_again(monkeypatch, tmp_path):
    """
    The point of the persistent disk: the container filesystem is new, the disk is not.
    """
    root, pilots, store = _wire(monkeypatch, tmp_path, ["Assam"])
    cache = os.path.join(str(tmp_path), "persistent")
    env = _env(SIH_PILOT_ARTIFACT_CACHE_DIR=cache)
    _run(root, ["Assam"], store, env=env)
    first = len(store.gets)
    assert first == 5

    # Simulate the new container: canonical paths (symlinks) are gone, cache survives.
    for _feature, path in _paths_for("assam_pilot", root).items():
        os.remove(path)
    assert pilots["Assam"].missing(root) != []

    report = _run(root, ["Assam"], store, env=env)
    assert len(store.gets) == first, "cached artifacts must not be re-downloaded"
    assert report["acted"] == 1
    assert sorted(report["results"][0]["cached"]) == sorted(FEATURES)
    assert pilots["Assam"].missing(root) == []


def test_a_corrupt_cached_copy_is_replaced_not_trusted(monkeypatch, tmp_path):
    root, pilots, store = _wire(monkeypatch, tmp_path, ["Assam"])
    cache = os.path.join(str(tmp_path), "persistent")
    env = _env(SIH_PILOT_ARTIFACT_CACHE_DIR=cache)
    os.makedirs(cache, exist_ok=True)
    _write(os.path.join(cache, "assam_pilot_dem.tif"), b"truncated leftover")

    report = _run(root, ["Assam"], store, env=env)
    assert any(url.endswith("assam_pilot_dem.tif") for url in store.gets)
    dem = _paths_for("assam_pilot", root)["elevation"]
    with open(dem, "rb") as handle:
        assert handle.read() == store.payloads["assam_pilot_dem.tif"]
    assert report["acted"] == 1


def test_a_stale_broken_symlink_is_cleared(monkeypatch, tmp_path):
    """A dangling link reads as missing forever unless it is removed first."""
    root, pilots, store = _wire(monkeypatch, tmp_path, ["Assam"])
    cache = os.path.join(str(tmp_path), "persistent")
    dem = _paths_for("assam_pilot", root)["elevation"]
    os.makedirs(os.path.dirname(dem), exist_ok=True)
    os.symlink(os.path.join(cache, "gone.tif"), dem)
    assert not os.path.exists(dem) and os.path.islink(dem)

    report = _run(root, ["Assam"], store,
                  env=_env(SIH_PILOT_ARTIFACT_CACHE_DIR=cache))
    assert report["acted"] == 1
    assert os.path.exists(dem)
    assert os.path.realpath(dem) == os.path.realpath(
        os.path.join(cache, "assam_pilot_dem.tif"))


def test_without_a_cache_dir_the_files_are_real_not_links(monkeypatch, tmp_path):
    root, pilots, store = _wire(monkeypatch, tmp_path, ["Assam"])
    _run(root, ["Assam"], store)
    for _feature, path in _paths_for("assam_pilot", root).items():
        assert os.path.isfile(path) and not os.path.islink(path)


# --------------------------------------------------------------------------------
# Requirement 6: startup only, never on the request path
# --------------------------------------------------------------------------------

def test_start_is_a_no_op_when_unconfigured(monkeypatch):
    called = []
    monkeypatch.setattr(pas, "ensure_pilot_artifacts", lambda **kw: called.append(kw))
    assert pas.start_pilot_artifact_fetch(env={}) is None
    assert pas.start_pilot_artifact_fetch(
        env=_env(SIH_PILOT_ARTIFACT_FETCH="0")) is None
    assert called == []


def test_start_runs_inline_when_blocking(monkeypatch):
    monkeypatch.setattr(pas, "ensure_pilot_artifacts",
                        lambda **kw: {"acted": 0, "results": []})
    out = pas.start_pilot_artifact_fetch(env=_env(SIH_PILOT_ARTIFACT_BLOCKING="1"))
    assert isinstance(out, dict) and out["acted"] == 0


def test_start_uses_a_background_daemon_thread_by_default(monkeypatch):
    """~2 GB must not delay port binding, /health, or /validation/status."""
    done = []
    monkeypatch.setattr(pas, "ensure_pilot_artifacts", lambda **kw: done.append(kw))
    thread = pas.start_pilot_artifact_fetch(env=_env())
    assert isinstance(thread, pas.threading.Thread)
    assert thread.daemon is True
    thread.join(timeout=5)
    assert len(done) == 1


def test_background_thread_swallows_unexpected_failures(monkeypatch):
    def _boom(**kwargs):
        raise RuntimeError("unexpected")

    monkeypatch.setattr(pas, "ensure_pilot_artifacts", _boom)
    thread = pas.start_pilot_artifact_fetch(env=_env())
    thread.join(timeout=5)
    assert not thread.is_alive()


def test_object_urls_are_not_logged_above_debug(monkeypatch, tmp_path):
    """
    A base URL can carry a query-string credential; deployment logs must not leak it.
    """
    root, pilots, store = _wire(monkeypatch, tmp_path, ["Assam"])
    records = []

    class _Capture(logging.Handler):
        def emit(self, record):
            records.append(record)

    handler = _Capture(level=logging.INFO)
    pas.logger.addHandler(handler)
    previous = pas.logger.level
    pas.logger.setLevel(logging.INFO)
    try:
        _run(root, ["Assam"], store)
    finally:
        pas.logger.removeHandler(handler)
        pas.logger.setLevel(previous)

    assert records, "the fetch should say something at INFO"
    for record in records:
        assert BASE not in record.getMessage()
    # Filenames are still reported, so operators can see what landed.
    assert any("assam_pilot_dem.tif" in r.getMessage() for r in records)


def _main_source():
    path = os.path.join(os.path.dirname(__file__), "..", "app", "main.py")
    with open(os.path.abspath(path), "r", encoding="utf-8") as handle:
        return handle.read()


def test_main_starts_the_fetch_from_the_lifespan_hook():
    """
    fastapi is absent offline, so importing app.main is impossible here; pin the wiring
    at the SOURCE level instead. Without it the module would exist but never run.
    """
    source = _main_source()
    assert "start_pilot_artifact_fetch" in source
    assert "lifespan=lifespan" in source and "async def lifespan" in source
    assert source.index("async def lifespan") < \
        source.index("start_pilot_artifact_fetch()")


def test_main_never_fetches_per_request():
    """The download must not hang off a route handler or middleware."""
    source = _main_source()
    before, _sep, after = source.partition("start_pilot_artifact_fetch()")
    assert "async def lifespan" in before
    for forbidden in ("@app.get", "@app.post", "@app.middleware"):
        assert forbidden not in before
    assert "start_pilot_artifact_fetch" not in after


def test_validation_status_route_does_not_download():
    """/api/v1/validation/status must stay a pure read of on-disk state."""
    path = os.path.join(os.path.dirname(__file__), "..", "app", "api", "routes.py")
    with open(os.path.abspath(path), "r", encoding="utf-8") as handle:
        source = handle.read()
    for forbidden in ("pilot_artifact_store", "ensure_pilot_artifacts",
                      "start_pilot_artifact_fetch"):
        assert forbidden not in source


# --------------------------------------------------------------------------------
# Sikkim: five uploaded objects, TWO name families, one set of verified bytes
# --------------------------------------------------------------------------------

SIKKIM = "Sikkim"
SIKKIM_PAYLOADS = _payloads_for("sikkim")


def _run_sikkim(root, store, env=None, **kwargs):
    return pas.ensure_pilot_artifacts(
        data_dir=root, states=(SIKKIM,), env=_env() if env is None else env,
        fetcher=store.fetcher, manifest_loader=store.loader, **kwargs)


def test_sikkim_published_paths_are_the_sweep_family(tmp_path):
    """These are the names that exist as objects in storage and carry manifest entries."""
    root = _fake_root(tmp_path)
    paths = pas.sikkim_artifact_paths(root)
    assert sorted(paths) == sorted(FEATURES)
    assert paths == _paths_for("sikkim", root)
    assert os.path.basename(paths["elevation"]) == "sikkim_dem.tif"
    assert sorted(os.path.basename(p) for p in paths.values()) == sorted(SIKKIM_PAYLOADS)


def test_sikkim_alias_paths_come_from_risk_inputs_not_a_second_hard_coding(tmp_path):
    """
    The serving path opens east_sikkim_dem.tif / real_<name>.tif. Reading the alias set
    off risk_inputs.terrain_raster_paths() is what stops it drifting from those readers.
    """
    from app.services import risk_inputs
    root = _fake_root(tmp_path)
    aliases = pas.sikkim_alias_paths(root)
    published = pas.sikkim_artifact_paths(root)
    assert sorted(aliases) == sorted(FEATURES)
    assert aliases == risk_inputs.terrain_raster_paths(root)
    assert os.path.basename(aliases["elevation"]) == "east_sikkim_dem.tif"
    for feature in ("slope", "aspect", "roughness", "tpi"):
        assert os.path.basename(aliases[feature]) == "real_%s.tif" % feature
    # Distinct paths under the same root: a twin is never the published object itself.
    for feature in FEATURES:
        assert aliases[feature] != published[feature]
        assert aliases[feature].startswith(os.path.abspath(root))


def test_only_sikkim_contributes_aliases(monkeypatch, tmp_path):
    root, _pilots, _store = _wire(monkeypatch, tmp_path, list(PREFIXES))
    for state_name in PREFIXES:
        assert pas.artifact_wiring(state_name)["aliases"](root) == {}
    assert pas.artifact_wiring(SIKKIM)["aliases"](root) != {}


def test_sikkim_missing_predicate_matches_the_published_paths(tmp_path):
    root = _fake_root(tmp_path)
    wiring = pas.artifact_wiring(SIKKIM)
    assert sorted(path for _f, path in wiring["missing"](root)) == \
        sorted(pas.sikkim_artifact_paths(root).values())
    assert [feature for feature, _p in wiring["missing"](root)] == sorted(FEATURES)
    for path in pas.sikkim_artifact_paths(root).values():
        _write(path)
    assert wiring["missing"](root) == []


def test_sikkim_fetch_places_both_name_families_from_the_same_five_objects(tmp_path):
    """
    The whole point of the alias mechanism: 245 MB is uploaded once under the sweep
    names, and both the sweep reader (state_validation) and the serving reader
    (risk_inputs) find their rasters.
    """
    from app.services import risk_inputs
    root = _fake_root(tmp_path)
    store = _Store(SIKKIM_PAYLOADS)
    report = _run_sikkim(root, store)
    assert report["acted"] == 1 and report["failed"] == 0

    # Exactly the five uploaded objects were requested -- no twin was fetched separately.
    assert sorted(url.rsplit("/", 1)[-1] for url in store.gets) == sorted(SIKKIM_PAYLOADS)
    assert report["bytes"] == sum(len(b) for b in SIKKIM_PAYLOADS.values())

    published = pas.sikkim_artifact_paths(root)
    aliases = pas.sikkim_alias_paths(root)
    for feature in FEATURES:
        body = store.payloads[os.path.basename(published[feature])]
        for path in (published[feature], aliases[feature]):
            with open(path, "rb") as handle:
                assert handle.read() == body
        assert os.path.realpath(aliases[feature]) == \
            os.path.realpath(published[feature])

    # Both readers' own predicates now report available.
    assert pas.artifact_wiring(SIKKIM)["missing"](root) == []
    assert risk_inputs.missing_terrain_rasters(root) == []


def test_sikkim_aliases_are_not_linked_when_verification_fails(tmp_path):
    """
    A twin must never look available off unverifiable bytes -- it is linked only after
    the size+sha256 check passes.
    """
    root = _fake_root(tmp_path)
    store = _Store(SIKKIM_PAYLOADS, truncate={"sikkim_dem.tif": 8},
                   corrupt=["sikkim_slope.tif"])
    report = _run_sikkim(root, store)
    published = pas.sikkim_artifact_paths(root)
    aliases = pas.sikkim_alias_paths(root)
    for feature in ("elevation", "slope"):
        assert not os.path.exists(published[feature])
        assert not os.path.exists(aliases[feature])
        assert not os.path.islink(aliases[feature])
    for feature in ("aspect", "roughness", "tpi"):
        assert os.path.exists(published[feature]) and os.path.exists(aliases[feature])
    assert report["results"][0]["status"] == "incomplete"
    assert [p for p in _all_files(root) if p.endswith(".part")] == []


def test_a_sikkim_object_absent_from_the_manifest_leaves_both_names_missing(tmp_path):
    root = _fake_root(tmp_path)
    payloads = dict(SIKKIM_PAYLOADS)
    del payloads["sikkim_tpi.tif"]
    store = _Store(payloads)
    report = _run_sikkim(root, store)
    assert not os.path.exists(pas.sikkim_artifact_paths(root)["tpi"])
    assert not os.path.exists(pas.sikkim_alias_paths(root)["tpi"])
    assert "no verifiable manifest entry" in report["results"][0]["error"]


def test_a_redeploy_relinks_both_sikkim_families_from_the_cache(tmp_path):
    """The cached-reuse branch must place the twins too, not only the published names."""
    root = _fake_root(tmp_path)
    cache = os.path.join(str(tmp_path), "persistent")
    env = _env(SIH_PILOT_ARTIFACT_CACHE_DIR=cache)
    store = _Store(SIKKIM_PAYLOADS)
    _run_sikkim(root, store, env=env)
    assert len(store.gets) == 5

    published = pas.sikkim_artifact_paths(root)
    aliases = pas.sikkim_alias_paths(root)
    # New container: every canonical path (a symlink) is gone; the disk survives.
    for feature in FEATURES:
        os.remove(published[feature])
        os.remove(aliases[feature])

    report = _run_sikkim(root, store, env=env)
    assert len(store.gets) == 5, "cached artifacts must not be re-downloaded"
    assert sorted(report["results"][0]["cached"]) == sorted(FEATURES)
    for feature in FEATURES:
        cached = os.path.join(cache, os.path.basename(published[feature]))
        assert os.path.realpath(published[feature]) == os.path.realpath(cached)
        assert os.path.realpath(aliases[feature]) == os.path.realpath(cached)


def test_an_alias_that_cannot_be_linked_does_not_downgrade_the_published_name(tmp_path,
                                                                             monkeypatch):
    """
    Refusing to promote sikkim_dem.tif because real_slope.tif could not be linked would
    help nobody: the sweep view would lose a raster it can serve.
    """
    root = _fake_root(tmp_path)
    real_link = pas.link_into_place

    def _link(target_path, final_path):
        if os.path.basename(final_path).startswith(("east_", "real_")):
            raise OSError("alias filesystem refused the link")
        return real_link(target_path, final_path)

    monkeypatch.setattr(pas, "link_into_place", _link)
    store = _Store(SIKKIM_PAYLOADS)
    report = _run_sikkim(root, store)
    assert report["acted"] == 1 and report["failed"] == 0
    for _feature, path in pas.sikkim_artifact_paths(root).items():
        assert os.path.isfile(path)
    for path in pas.sikkim_alias_paths(root).values():
        assert not os.path.exists(path)


def test_selecting_one_state_never_places_another_states_rasters(monkeypatch, tmp_path):
    """Sikkim joining the state list must not widen what a one-pilot rollout writes."""
    root, _pilots, store = _wire(monkeypatch, tmp_path, ["Assam"])
    store.payloads.update(SIKKIM_PAYLOADS)
    _run(root, ["Assam"], store)
    assert all("sikkim" not in url for url in store.gets)
    assert pas.artifact_wiring(SIKKIM)["missing"](root) != []


# --------------------------------------------------------------------------------
# scripts/publish_pilot_artifacts.py -- the manifest must describe uploaded OBJECTS
# --------------------------------------------------------------------------------

def _publish_module():
    """Import the host-only publisher by path; it uploads nothing on import."""
    import importlib.util
    path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts",
                                        "publish_pilot_artifacts.py"))
    spec = importlib.util.spec_from_file_location("publish_pilot_artifacts_undertest",
                                                  path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_publisher_collects_five_objects_per_state_and_no_twins(monkeypatch, tmp_path):
    """
    20 objects, not 25: listing east_sikkim_dem.tif / real_*.tif would claim objects
    exist in storage that were never uploaded.
    """
    publish = _publish_module()
    root, _pilots, _store = _wire(monkeypatch, tmp_path, list(PREFIXES))
    items = publish.collect(pas.PILOT_ARTIFACT_STATES, root)
    assert len(items) == 20
    filenames = sorted(os.path.basename(path) for _s, _f, path in items)
    assert len(set(filenames)) == 20
    assert sorted(n for n in filenames if n.startswith("sikkim_")) == \
        sorted(SIKKIM_PAYLOADS)
    for name in filenames:
        assert not name.startswith(("east_", "real_"))


def test_publisher_reports_the_runtime_alias_links(monkeypatch, tmp_path):
    publish = _publish_module()
    root, _pilots, _store = _wire(monkeypatch, tmp_path, list(PREFIXES))
    monkeypatch.setattr(publish, "artifact_wiring", pas.artifact_wiring)
    assert publish.alias_report(list(PREFIXES), root) == []
    rows = publish.alias_report([SIKKIM], root)
    mapping = {os.path.basename(alias): os.path.basename(published)
               for alias, published in rows}
    assert mapping == {"east_sikkim_dem.tif": "sikkim_dem.tif",
                       "real_slope.tif": "sikkim_slope.tif",
                       "real_aspect.tif": "sikkim_aspect.tif",
                       "real_roughness.tif": "sikkim_roughness.tif",
                       "real_tpi.tif": "sikkim_tpi.tif"}


def _sikkim_items(root):
    return [(SIKKIM, feature, path)
            for feature, path in sorted(pas.sikkim_artifact_paths(root).items())]


def test_publisher_hashes_what_is_present_and_reports_what_is_not(tmp_path):
    publish = _publish_module()
    root = _fake_root(tmp_path)
    published = pas.sikkim_artifact_paths(root)
    for feature, path in published.items():
        if feature != "tpi":
            _write(path, SIKKIM_PAYLOADS[os.path.basename(path)])
    entries, missing = publish.build_manifest(_sikkim_items(root))
    assert sorted(entries) == sorted(n for n in SIKKIM_PAYLOADS if n != "sikkim_tpi.tif")
    assert [feature for _s, feature, _p in missing] == ["tpi"]
    for filename, entry in entries.items():
        body = SIKKIM_PAYLOADS[filename]
        assert entry == {"bytes": len(body),
                         "sha256": hashlib.sha256(body).hexdigest()}


def test_publisher_verify_matches_only_when_every_entry_matches(tmp_path):
    """
    --verify runs the same size-then-sha256 check the runtime performs after a download,
    so a manifest can never claim a digest the local raster does not have.
    """
    publish = _publish_module()
    root = _fake_root(tmp_path)
    published = pas.sikkim_artifact_paths(root)
    for path in published.values():
        _write(path, SIKKIM_PAYLOADS[os.path.basename(path)])
    items = _sikkim_items(root)
    entries, missing = publish.build_manifest(items)
    assert missing == []

    ok, rows = publish.verify_manifest(entries, items)
    assert ok is True
    assert [reason for _f, reason in rows] == [None] * 5

    # A truncated local raster is caught by size.
    _write(published["slope"], b"short")
    ok, rows = publish.verify_manifest(entries, items)
    assert ok is False
    reasons = dict(rows)
    assert "size mismatch" in reasons["sikkim_slope.tif"]


def test_publisher_verify_flags_entries_and_files_that_do_not_correspond(tmp_path):
    publish = _publish_module()
    root = _fake_root(tmp_path)
    published = pas.sikkim_artifact_paths(root)
    for path in published.values():
        _write(path, SIKKIM_PAYLOADS[os.path.basename(path)])
    items = _sikkim_items(root)
    entries, _missing = publish.build_manifest(items)

    stray = dict(entries, **{"assam_pilot_dem.tif": {"bytes": 3, "sha256": "c" * 64}})
    ok, rows = publish.verify_manifest(stray, items)
    assert ok is False
    assert dict(rows)["assam_pilot_dem.tif"] == \
        "no local artifact corresponds to this entry"

    incomplete = {k: v for k, v in entries.items() if k != "sikkim_tpi.tif"}
    ok, rows = publish.verify_manifest(incomplete, items)
    assert ok is False
    assert dict(rows)["sikkim_tpi.tif"] == "local artifact has no manifest entry"


# ---------------------------------------------------------------------------
# THE REAL HTTP PATH IS MEMORY-FLAT
#
# The tests above inject a fetcher, so they say nothing about how the SHIPPED
# _default_fetcher / _default_manifest_loader actually consume a response. That is
# precisely what OOM-killed a 512 MB Render instance, so it is pinned here directly: a
# fake "requests" is placed in sys.modules and its response can ONLY be read
# incrementally -- .content, .text and .read() raise. No credentials, no network, and
# neither the fake nor the test ever holds the whole body.
# ---------------------------------------------------------------------------

LARGE_BODY_BYTES = 32 * 1024 * 1024


class _StreamOnlyResponse:
    """A response body that can only be consumed block by block."""

    def __init__(self, total_bytes, watch_path=None, body_blocks=None):
        self.total_bytes = total_bytes
        self.watch_path = watch_path
        self.body_blocks = body_blocks
        self.chunk_sizes = []
        self.blocks_yielded = 0
        self.digest = hashlib.sha256()
        self.bytes_on_disk_before_last_block = None
        self.closed = False
        self.raised_for_status = False

    @property
    def content(self):
        raise AssertionError("response.content was read: the whole body was buffered")

    @property
    def text(self):
        raise AssertionError("response.text was read: the whole body was buffered")

    def read(self, *_args, **_kwargs):
        raise AssertionError("response.read() was called: unbounded whole-body read")

    def raise_for_status(self):
        self.raised_for_status = True

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        self.closed = True
        return False

    def iter_content(self, chunk_size=None):
        assert chunk_size, "iter_content must be given a bounded block size"
        self.chunk_sizes.append(chunk_size)
        if self.body_blocks is not None:
            for block in self.body_blocks:
                self.blocks_yielded += 1
                self.digest.update(block)
                yield block
            return
        # Generated on demand: the payload never exists as one object anywhere.
        pattern = (b"\xde\xad\xbe\xef" * (chunk_size // 4 + 1))[:chunk_size]
        remaining = self.total_bytes
        while remaining > 0:
            block = pattern if remaining >= chunk_size else pattern[:remaining]
            remaining -= len(block)
            self.blocks_yielded += 1
            self.digest.update(block)
            if remaining == 0 and self.watch_path is not None:
                self.bytes_on_disk_before_last_block = (
                    os.path.getsize(self.watch_path)
                    if os.path.exists(self.watch_path) else 0)
            yield block


class _FakeRequests:
    """Stands in for the requests module that _default_* import lazily."""

    def __init__(self, response):
        self.response = response
        self.calls = []

    def get(self, url, stream=False, timeout=None):
        assert stream is True, "the response must be opened in streaming mode"
        self.calls.append((url, timeout))
        return self.response

def test_stream_only_response_guards_are_live():
    """
    Test integrity: the .content / .text / .read() guards must really fire, otherwise the
    fetcher test below would pass by accident rather than by never touching them.
    """
    response = _StreamOnlyResponse(16)
    with pytest.raises(AssertionError):
        response.content
    with pytest.raises(AssertionError):
        response.text
    with pytest.raises(AssertionError):
        response.read()


def test_default_fetcher_streams_a_large_body_in_bounded_blocks(monkeypatch, tmp_path):
    """
    Regression for the 512 MB OOM: a 32 MiB body is consumed block by block straight into
    <target>.part, with the SHA-256 and the byte count accumulated in flight, and the
    whole-body accessors never touched.
    """
    part = os.path.join(str(tmp_path), "assam_pilot_dem.tif.part")
    response = _StreamOnlyResponse(LARGE_BODY_BYTES, watch_path=part)
    fake = _FakeRequests(response)
    monkeypatch.setitem(sys.modules, "requests", fake)
    url = "https://cdn.example/pilots/assam_pilot_dem.tif"

    observed = pas._default_fetcher(url, part, 30.0)

    assert fake.calls == [(url, 30.0)]
    assert response.raised_for_status is True
    assert response.closed is True
    # Bounded blocks, small enough that peak memory is independent of artifact size.
    assert response.chunk_sizes == [pas.DOWNLOAD_CHUNK_BYTES]
    assert pas.DOWNLOAD_CHUNK_BYTES <= 1024 * 1024
    assert response.blocks_yielded == LARGE_BODY_BYTES // pas.DOWNLOAD_CHUNK_BYTES
    # INCREMENTAL: bytes had already reached the disk before the body ran out, which a
    # single .content read could not produce.
    assert response.bytes_on_disk_before_last_block >= pas.FLUSH_EVERY_BYTES
    # Hashed and counted in flight, so verification needs no second pass over the file.
    assert observed == {"bytes": LARGE_BODY_BYTES, "sha256": response.digest.hexdigest()}
    assert os.path.getsize(part) == LARGE_BODY_BYTES
    assert pas.sha256_of_file(part) == observed["sha256"]


def test_default_manifest_loader_streams_bounded_blocks(monkeypatch):
    """The manifest is small, but it is still never read through .content."""
    payload = {"artifacts": {"sikkim_dem.tif": {"bytes": 4, "sha256": "a" * 64}}}
    body = json.dumps(payload).encode("utf-8")
    blocks = [body[i:i + 7] for i in range(0, len(body), 7)]
    response = _StreamOnlyResponse(len(body), body_blocks=blocks)
    monkeypatch.setitem(sys.modules, "requests", _FakeRequests(response))

    url = "https://cdn.example/pilots/pilot_manifest.json"
    assert pas._default_manifest_loader(url, 5.0) == payload
    assert response.chunk_sizes == [pas.MANIFEST_CHUNK_BYTES]
    assert response.blocks_yielded == len(blocks)
    assert response.closed is True


def test_default_manifest_loader_refuses_an_oversized_body(monkeypatch):
    """
    A wrong URL that answers with something enormous must be refused, not accumulated --
    the only place this module joins blocks at all.
    """
    oversized = pas.MAX_MANIFEST_BYTES + pas.MANIFEST_CHUNK_BYTES
    response = _StreamOnlyResponse(oversized)
    monkeypatch.setitem(sys.modules, "requests", _FakeRequests(response))

    with pytest.raises(ValueError):
        pas._default_manifest_loader("https://cdn.example/pilots/pilot_manifest.json", 5.0)
    # Stopped at the bound rather than after reading everything on offer.
    assert response.blocks_yielded <= (pas.MAX_MANIFEST_BYTES // pas.MANIFEST_CHUNK_BYTES) + 1
    assert response.blocks_yielded * pas.MANIFEST_CHUNK_BYTES <= \
        pas.MAX_MANIFEST_BYTES + pas.MANIFEST_CHUNK_BYTES


def test_observed_from_fetch_normalises_what_a_fetcher_may_return():
    digest = "b" * 64
    assert pas._observed_from_fetch({"bytes": 12, "sha256": digest.upper()}) == (12, digest)
    assert pas._observed_from_fetch({"bytes": "12", "sha256": digest}) == (12, digest)
    # A malformed digest is not trusted; verification re-reads the file instead.
    assert pas._observed_from_fetch({"bytes": 12, "sha256": "abc"}) == (12, None)
    assert pas._observed_from_fetch({"bytes": None, "sha256": digest}) == (None, digest)
    # The historical contract -- a bare byte count -- still works unchanged.
    assert pas._observed_from_fetch(9) == (9, None)
    assert pas._observed_from_fetch(None) == (None, None)
    assert pas._observed_from_fetch("nonsense") == (None, None)


def test_verify_streamed_uses_the_in_flight_digest_and_catches_short_writes(tmp_path):
    body = b"terrain-bytes" * 11
    path = os.path.join(str(tmp_path), "artifact.tif")
    _write(path, body)
    entry = {"bytes": len(body), "sha256": hashlib.sha256(body).hexdigest()}

    assert pas.verify_streamed(path, entry, len(body), entry["sha256"]) is None
    assert "sha256 mismatch" in pas.verify_streamed(path, entry, len(body), "c" * 64)
    # More bytes were streamed than reached the disk: the file is short.
    assert "short write" in pas.verify_streamed(path, entry, len(body) + 5,
                                                entry["sha256"])
    bigger = {"bytes": len(body) + 1, "sha256": entry["sha256"]}
    assert "size mismatch" in pas.verify_streamed(path, bigger, len(body),
                                                  entry["sha256"])
    absent = os.path.join(str(tmp_path), "absent.tif")
    assert pas.verify_streamed(absent, entry, 0, entry["sha256"]) == \
        "file was not written"


def test_verify_streamed_falls_back_to_verify_artifact_without_a_digest(monkeypatch,
                                                                       tmp_path):
    """
    An injected fetcher that returns only a byte count (as every test above does, and as
    the pre-change contract allowed) still goes through the unchanged full-read check.
    """
    body = b"x" * 40
    path = os.path.join(str(tmp_path), "artifact.tif")
    _write(path, body)
    entry = {"bytes": len(body), "sha256": hashlib.sha256(body).hexdigest()}
    calls = []
    real = pas.verify_artifact

    def spy(candidate, manifest_entry):
        calls.append(candidate)
        return real(candidate, manifest_entry)

    monkeypatch.setattr(pas, "verify_artifact", spy)
    assert pas.verify_streamed(path, entry, len(body), None) is None
    assert calls == [path]


def test_no_whole_body_read_survives_in_the_source():
    """
    Static guard: response.content, response.text, response.read() or an unbounded
    iter_content() would each buffer a 184 MB raster whole. None may reappear.
    """
    with open(pas.__file__, "r", encoding="utf-8") as handle:
        tree = ast.parse(handle.read())
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) \
                and node.value.id == "response":
            assert node.attr in ("raise_for_status", "iter_content"), \
                "response.%s reintroduces a whole-body read" % node.attr
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) \
                and node.func.attr == "iter_content":
            assert [kw.arg for kw in node.keywords] == ["chunk_size"], \
                "iter_content must always be given a bounded chunk_size"
