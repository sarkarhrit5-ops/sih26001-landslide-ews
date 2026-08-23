"""
Focused unit tests for the rainfall-failure behavior of
backend/scripts/train_real_models.py::fetch_historical_rainfall_series.

SCOPE / HONESTY NOTES
---------------------
* These tests use MOCKED HTTP responses (unittest via monkeypatch). They do NOT
  contact Open-Meteo and are NOT a test of real weather data or real network
  behavior. The numeric arrays below are synthetic FIXTURES used only to assert
  the function's control flow.
* Their sole purpose is to prove that a failed/incomplete rainfall retrieval
  becomes an explicit RainfallUnavailableError and is NEVER silently converted
  into zero-filled (or otherwise fabricated) rainfall features.

The production module imports the heavy scientific stack (rasterio, lightgbm,
shap, sklearn) at import time. Where those are unavailable the module cannot be
imported, so these tests are SKIPPED (via importorskip) rather than reported as
passing.
"""
import os
import sys

import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Skip cleanly if the module's heavy import chain cannot be satisfied.
trm = pytest.importorskip("scripts.train_real_models")

RAIN_KEYS = {
    "rain_1d",
    "rain_3d",
    "rain_7d",
    "antecedent_rain_14d",
    "rain_intensity_max_3d",
}


class _FakeResp:
    """Minimal stand-in for a requests.Response."""

    def __init__(self, status_code=200, payload=None, text=""):
        self.status_code = status_code
        self._payload = payload
        self.text = text

    def json(self):
        if self._payload is None:
            raise ValueError("no JSON payload")
        return self._payload


def _patch_get(monkeypatch, resp=None, exc=None):
    """Patch requests.get as seen by the module under test."""

    def fake_get(url, params=None, timeout=None):
        if exc is not None:
            raise exc
        return resp

    monkeypatch.setattr(trm.requests, "get", fake_get)


def test_success_returns_real_features(monkeypatch):
    # 14 synthetic antecedent daily values (T-14 .. T-1). FIXTURE, not real data.
    daily = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0,
             7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0]
    _patch_get(monkeypatch, resp=_FakeResp(200, {"daily": {"precipitation_sum": daily}}))

    out = trm.fetch_historical_rainfall_series(27.3, 88.6, "2015-06-15")

    assert set(out.keys()) == RAIN_KEYS
    assert out["rain_1d"] == 13.0
    assert out["rain_3d"] == 11.0 + 12.0 + 13.0
    assert out["rain_7d"] == sum(daily[-7:])
    assert out["antecedent_rain_14d"] == float(sum(daily))
    assert out["rain_intensity_max_3d"] == 13.0
    # A successful real response must not collapse to all-zeros.
    assert any(v != 0.0 for v in out.values())


def test_http_non_200_raises_and_does_not_zero_fill(monkeypatch):
    _patch_get(monkeypatch, resp=_FakeResp(status_code=503, payload=None,
                                           text="Service Unavailable"))
    with pytest.raises(trm.RainfallUnavailableError):
        trm.fetch_historical_rainfall_series(27.3, 88.6, "2015-06-15")


def test_network_exception_raises_and_does_not_zero_fill(monkeypatch):
    import requests
    _patch_get(monkeypatch, exc=requests.ConnectionError("DNS resolution failed"))
    with pytest.raises(trm.RainfallUnavailableError):
        trm.fetch_historical_rainfall_series(27.3, 88.6, "2015-06-15")


def test_timeout_exception_raises_and_does_not_zero_fill(monkeypatch):
    import requests
    _patch_get(monkeypatch, exc=requests.Timeout("read timed out"))
    with pytest.raises(trm.RainfallUnavailableError):
        trm.fetch_historical_rainfall_series(27.3, 88.6, "2015-06-15")


def test_malformed_json_raises_and_does_not_zero_fill(monkeypatch):
    # 200 OK but body is not the expected shape -> failure, not zeros.
    _patch_get(monkeypatch, resp=_FakeResp(200, {"unexpected": "shape"}))
    with pytest.raises(trm.RainfallUnavailableError):
        trm.fetch_historical_rainfall_series(27.3, 88.6, "2015-06-15")


def test_insufficient_observations_raises_and_does_not_pad_zeros(monkeypatch):
    # Only 5 days returned when 14 are required. Must be treated as unavailable,
    # never padded up to 14 with zeros.
    daily = [1.0, 2.0, 3.0, 4.0, 5.0]
    _patch_get(monkeypatch, resp=_FakeResp(200, {"daily": {"precipitation_sum": daily}}))
    with pytest.raises(trm.RainfallUnavailableError):
        trm.fetch_historical_rainfall_series(27.3, 88.6, "2015-06-15")


def test_null_day_raises_and_is_not_treated_as_zero(monkeypatch):
    # A present-but-null observation must not be silently read as 0 mm.
    daily = [1.0, 2.0, None, 4.0, 5.0, 6.0, 7.0,
             8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0]
    _patch_get(monkeypatch, resp=_FakeResp(200, {"daily": {"precipitation_sum": daily}}))
    with pytest.raises(trm.RainfallUnavailableError):
        trm.fetch_historical_rainfall_series(27.3, 88.6, "2015-06-15")
