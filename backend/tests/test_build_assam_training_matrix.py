"""
Offline tests for scripts/build_assam_training_matrix.py -- the Assam pilot
training-matrix builder that replaces the Sikkim elevation land-cover proxy with the
real categorical WorldCover feature while reusing the Sikkim terrain sampler and
Open-Meteo ERA5 rainfall fetch verbatim.

DEPENDENCY BUDGET: stdlib + numpy/pandas only. The builder imports rasterio (terrain),
requests/network (Open-Meteo) and sklearn (ml_pipeline) LAZILY inside main(), so the
module and every helper exercised here import offline. These tests never call main(),
never touch the network, and never train.

What they protect:
  * the 11-feature schema and its order (5 terrain + land_cover_class + 5 rainfall),
    with land_cover_class declared categorical -- the contract a trainer relies on;
  * nodata/non-finite terrain -> NaN and never filled, with correct missing counts;
  * the missing-value report counts the land-cover UNAVAILABLE sentinel and NaN feats;
  * missing terrain rasters fail fast (never silently build a partial matrix);
  * the real Assam positives snapshot loads to 59 dated, target=1 rows.
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'scripts')))

import build_assam_training_matrix as b
from app.services import worldcover as wc


# --------------------------------------------------------------------------- #
# Feature schema / categorical contract
# --------------------------------------------------------------------------- #
def test_feature_layout_matches_sikkim_trainer():
    assert b.TERRAIN_FEATURES == ["elevation", "slope", "aspect", "roughness", "tpi"]
    assert b.RAINFALL_FEATURES == [
        "rain_1d", "rain_3d", "rain_7d", "antecedent_rain_14d", "rain_intensity_max_3d"
    ]
    # 6 static (terrain + land cover) then 5 rainfall == the exact dynamic_features
    # order in scripts/train_real_models.py.
    assert b.STATIC_FEATURES == b.TERRAIN_FEATURES + ["land_cover_class"]
    assert b.ALL_FEATURES == b.STATIC_FEATURES + b.RAINFALL_FEATURES
    assert len(b.ALL_FEATURES) == 11


def test_land_cover_is_the_single_categorical_feature():
    assert wc.LANDCOVER_FEATURE_NAME == "land_cover_class"
    assert "land_cover_class" in b.STATIC_FEATURES
    assert wc.landcover_categorical_feature() == ["land_cover_class"]
    assert wc.LANDCOVER_IS_CATEGORICAL is True


# --------------------------------------------------------------------------- #
# Terrain nodata handling: -9999 / non-finite -> NaN, never filled
# --------------------------------------------------------------------------- #
def test_mask_terrain_missing_sets_nan_and_counts():
    df = pd.DataFrame({
        "elevation": [100.0, b.TERRAIN_NODATA, 300.0, np.inf],
        "slope":     [10.0, 20.0, np.nan, 5.0],
        "aspect":    [90.0, 180.0, 270.0, 360.0],
        "roughness": [1.0, 2.0, 3.0, 4.0],
        "tpi":       [0.1, -0.2, 0.3, -0.4],
    })
    counts = b.mask_terrain_missing(df)
    # -9999 and +inf in elevation -> 2 missing; one NaN in slope -> 1 missing.
    assert counts["elevation"] == 2
    assert counts["slope"] == 1
    assert counts["aspect"] == 0 and counts["roughness"] == 0 and counts["tpi"] == 0
    # The bad cells are NaN (UNAVAILABLE), NOT replaced by any real number.
    assert np.isnan(df["elevation"].iloc[1]) and np.isnan(df["elevation"].iloc[3])
    assert np.isnan(df["slope"].iloc[2])
    # Good values untouched.
    assert df["elevation"].iloc[0] == pytest.approx(100.0)
    assert df["elevation"].iloc[2] == pytest.approx(300.0)


def test_missing_report_counts_sentinel_and_nan():
    n = 4
    df = pd.DataFrame({f: np.ones(n, dtype="float32") for f in b.TERRAIN_FEATURES})
    for f in b.RAINFALL_FEATURES:
        df[f] = np.ones(n, dtype="float32")
    # One NaN terrain cell + one UNAVAILABLE land-cover cell.
    df.loc[0, "elevation"] = np.nan
    df["land_cover_class"] = np.array([wc.UNAVAILABLE_SENTINEL, 1, 3, 4], dtype="int32")

    rep = b._feature_missing_report(df, landcover_unavailable=1)
    assert rep["elevation"] == 1
    assert rep["land_cover_class"] == 1
    assert rep["rain_1d"] == 0
    # rows 0 has both a NaN terrain and the sentinel -> counted once.
    assert rep["_rows_with_any_missing_feature"] == 1
    assert rep["_landcover_unavailable_rows"] == 1


# --------------------------------------------------------------------------- #
# Fail-fast on missing terrain rasters (never build a partial matrix silently)
# --------------------------------------------------------------------------- #
def test_terrain_raster_map_missing_raises(tmp_path):
    with pytest.raises(SystemExit):
        b.terrain_raster_map(str(tmp_path), str(tmp_path))


# --------------------------------------------------------------------------- #
# Real positives snapshot loads to the expected shape
# --------------------------------------------------------------------------- #
def test_load_positive_events_real_snapshot():
    models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'models'))
    if not os.path.exists(os.path.join(models_dir, b.EVENTS_FILENAME)):
        pytest.skip("assam_events.json snapshot not present")
    pos = b.load_positive_events(models_dir)
    assert len(pos) == 59
    assert (pos["target"] == 1).all()
    assert str(pos["event_date"].dtype).startswith("datetime64")
    assert pos["event_date"].isna().sum() == 0
    assert pos["latitude"].notna().all() and pos["longitude"].notna().all()


def test_load_positive_events_missing_file_raises(tmp_path):
    with pytest.raises(SystemExit):
        b.load_positive_events(str(tmp_path))

# --------------------------------------------------------------------------- #
# Open-Meteo HTTP-429 backoff (the concurrency fix). Fully offline: the reused
# Sikkim fetcher is replaced by a fake, and sleep + jitter are injected, so nothing
# waits or touches the network. These lock in that the retry ONLY spaces out
# identical requests -- it never alters, substitutes, or fills a rainfall value, and
# non-429 failures still abort immediately.
# --------------------------------------------------------------------------- #
class _FakeRainErr(RuntimeError):
    """Stands in for train_real_models.RainfallUnavailableError so these tests never
    import the host-only Sikkim trainer (which needs rasterio/network)."""


_RAIN = {"rain_1d": 1.0, "rain_3d": 2.0, "rain_7d": 3.0,
         "antecedent_rain_14d": 4.0, "rain_intensity_max_3d": 5.0}


def _capture():
    """Return (recorded_sleeps, fake_sleep, fake_jitter). fake_jitter returns the
    upper bound so the backoff schedule is deterministic and assertable."""
    slept = []
    return slept, (lambda s: slept.append(s)), (lambda lo, hi: hi)


def test_reduced_default_concurrency():
    # Concurrency was lowered from 8; a low ceiling is what avoids concurrent-429s.
    assert 1 <= b.RAINFALL_DEFAULT_MAX_WORKERS <= 2


def test_is_http_429_detects_only_429():
    assert b._is_http_429(_FakeRainErr(
        "Historical rainfall request returned HTTP 429 (lat=26.1, lon=91.7): "
        "'Too many concurrent requests'")) is True
    assert b._is_http_429(_FakeRainErr("returned HTTP 500 server error")) is False
    assert b._is_http_429(_FakeRainErr("request failed: ConnectionError()")) is False


def test_backoff_retries_then_returns_unchanged_value():
    calls = {"n": 0}

    def fetch(lat, lon, date):
        calls["n"] += 1
        if calls["n"] <= 2:
            raise _FakeRainErr("returned HTTP 429 'Too many concurrent requests'")
        return dict(_RAIN)

    slept, sleep, jitter = _capture()
    out = b.fetch_rainfall_with_backoff(
        fetch, 26.1, 91.7, "2020-06-01", _FakeRainErr, sleep=sleep, jitter_uniform=jitter)
    # Value passes through unchanged; the retry never fabricates or fills anything.
    assert out == _RAIN
    assert calls["n"] == 3           # 2 x 429 + 1 success
    assert slept == [2.0, 4.0]       # exponential backoff between identical retries


def test_backoff_does_not_retry_non_429():
    calls = {"n": 0}

    def fetch(lat, lon, date):
        calls["n"] += 1
        raise _FakeRainErr("returned HTTP 500 server error")

    slept, sleep, jitter = _capture()
    with pytest.raises(_FakeRainErr):
        b.fetch_rainfall_with_backoff(
            fetch, 26.1, 91.7, "2020-06-01", _FakeRainErr, sleep=sleep, jitter_uniform=jitter)
    assert calls["n"] == 1           # raised immediately -- no retry
    assert slept == []               # and no waiting


def test_backoff_gives_up_after_max_retries_and_never_fills():
    calls = {"n": 0}

    def fetch(lat, lon, date):
        calls["n"] += 1
        raise _FakeRainErr("returned HTTP 429 too many concurrent requests")

    slept, sleep, jitter = _capture()
    with pytest.raises(_FakeRainErr):     # persistent 429 aborts; NO dict is returned
        b.fetch_rainfall_with_backoff(
            fetch, 26.1, 91.7, "2020-06-01", _FakeRainErr,
            max_retries=5, sleep=sleep, jitter_uniform=jitter)
    assert calls["n"] == 6                # 1 initial + 5 retries
    assert slept == [2.0, 4.0, 8.0, 16.0, 32.0]


def test_backoff_delay_is_capped():
    def fetch(lat, lon, date):
        raise _FakeRainErr("returned HTTP 429")

    slept, sleep, jitter = _capture()
    with pytest.raises(_FakeRainErr):
        b.fetch_rainfall_with_backoff(
            fetch, 26.1, 91.7, "2020-06-01", _FakeRainErr,
            max_retries=8, base_delay=2.0, cap_delay=60.0, sleep=sleep, jitter_uniform=jitter)
    assert slept == [2.0, 4.0, 8.0, 16.0, 32.0, 60.0, 60.0, 60.0]   # capped at 60s
