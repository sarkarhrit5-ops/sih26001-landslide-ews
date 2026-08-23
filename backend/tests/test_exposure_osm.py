"""
Focused unit tests for the OSM/exposure fallback behavior of
backend/app/services/exposure.py::get_osm_assets.

SCOPE / HONESTY NOTES
---------------------
* These tests use MOCKED HTTP responses (monkeypatched requests.post). They do
  NOT contact the Overpass API and are NOT real OSM integration tests. The JSON
  payloads below are synthetic FIXTURES used only to drive the function's
  control flow.
* Their purpose is to prove that get_osm_assets:
    - preserves REAL assets when Overpass returns them,
    - returns an EXPLICITLY EMPTY result (never a synthetic "State Center Point"
      / dummy hospital / dummy road / osm_id=0 asset) when Overpass returns zero
      assets,
    - raises on Overpass request failure without fabricating assets, and
    - does NOT write a cache file for an empty result (so a downstream
      availability check cannot report "Available" off a synthetic file).

The module imports geopandas + shapely at import time. Where those are
unavailable the module cannot be imported, so these tests are SKIPPED (via
importorskip) rather than reported as passing.
"""
import os
import sys

import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Skip cleanly if the geospatial stack is unavailable.
gpd = pytest.importorskip("geopandas")
pytest.importorskip("shapely")

from app.services import exposure  # noqa: E402

BBOX = {"min_lat": 27.0, "max_lat": 27.6, "min_lon": 88.2, "max_lon": 88.8}
TEST_STATE = "Zzz Test State"  # unlikely to collide with any real cache file


class _FakeResp:
    def __init__(self, status_code=200, payload=None):
        self.status_code = status_code
        self._payload = payload

    def json(self):
        if self._payload is None:
            raise ValueError("no JSON payload")
        return self._payload


@pytest.fixture(autouse=True)
def _no_sleep_no_disk(monkeypatch):
    # Keep retries instant and never touch the real filesystem cache.
    monkeypatch.setattr(exposure.time, "sleep", lambda *a, **k: None)
    calls = {"to_file": []}
    monkeypatch.setattr(
        gpd.GeoDataFrame, "to_file",
        lambda self, *a, **k: calls["to_file"].append((a, k)),
        raising=True,
    )
    return calls


def _set_post(monkeypatch, resp=None, exc=None):
    def fake_post(url, data=None, headers=None, timeout=None):
        if exc is not None:
            raise exc
        return resp
    monkeypatch.setattr(exposure.requests, "post", fake_post)


def test_real_features_are_preserved(monkeypatch, _no_sleep_no_disk):
    payload = {"elements": [
        {"type": "node", "id": 111, "lat": 27.33, "lon": 88.61,
         "tags": {"amenity": "hospital", "name": "Test Hospital"}},
        {"type": "way", "id": 222,
         "geometry": [{"lat": 27.30, "lon": 88.60}, {"lat": 27.31, "lon": 88.61}],
         "tags": {"highway": "primary", "name": "Test Road"}},
    ]}
    _set_post(monkeypatch, resp=_FakeResp(200, payload))

    gdf = exposure.get_osm_assets(TEST_STATE, BBOX)

    assert len(gdf) == 2
    names = set(gdf["asset_name"])
    assert names == {"Test Hospital", "Test Road"}
    ids = set(int(x) for x in gdf["osm_id"])
    assert ids == {111, 222}
    # No fabricated asset, and a real result IS cached.
    assert "State Center Point" not in " ".join(names)
    assert 0 not in ids
    assert len(_no_sleep_no_disk["to_file"]) == 1


def test_zero_features_yields_no_synthetic_state_center_point(monkeypatch, _no_sleep_no_disk):
    # Valid 200 response, but zero matching OSM elements.
    _set_post(monkeypatch, resp=_FakeResp(200, {"elements": []}))

    gdf = exposure.get_osm_assets(TEST_STATE, BBOX)

    # Explicitly empty -- NOT a fabricated centre point.
    assert len(gdf) == 0
    assert "asset_name" in gdf.columns  # explicit empty schema, not a crash
    if len(gdf) == 0:
        joined = " ".join(str(v) for v in gdf.get("asset_name", []))
        assert "State Center Point" not in joined
    # And crucially: NO cache file is written for an empty result, so a
    # downstream availability check cannot flip to "Available" off a synthetic
    # file.
    assert _no_sleep_no_disk["to_file"] == []


def test_overpass_failure_does_not_fabricate_assets(monkeypatch, _no_sleep_no_disk):
    # Every attempt returns a non-200; the function must raise, not fabricate.
    _set_post(monkeypatch, resp=_FakeResp(500, None))

    with pytest.raises(RuntimeError):
        exposure.get_osm_assets(TEST_STATE, BBOX)

    # No synthetic assets, no cache written on failure.
    assert _no_sleep_no_disk["to_file"] == []


def test_overpass_network_exception_does_not_fabricate(monkeypatch, _no_sleep_no_disk):
    import requests
    _set_post(monkeypatch, exc=requests.ConnectionError("DNS failure"))

    with pytest.raises(RuntimeError):
        exposure.get_osm_assets(TEST_STATE, BBOX)

    assert _no_sleep_no_disk["to_file"] == []
