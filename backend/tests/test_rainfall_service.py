"""
Offline tests for app.services.rainfall_service.

Zero credentials, zero network, zero rasters. Every external collaborator is
injected: `imerg_fetcher`, `fallback_fetcher`, `session_factory`, `clock`, `cache`.

What these tests are here to defend:
  * IMERG is preferred; Open-Meteo is only ever a labelled fallback.
  * A fallback is stamped FALLBACK / is_fallback=True and is never described as an
    official live observation.
  * A failure yields UNAVAILABLE with NO numbers -- never a zero-filled series.
  * One fetch per state AOI per TTL; the four AOIs never share a cache entry.
  * Freshness metadata distinguishes a cached read from a fresh one.
  * The probe value is reused (one fewer round trip than the previous code).
  * The five feature names and their arithmetic match the fitted schema exactly.
"""

import ast
import os
from contextlib import contextmanager
from datetime import datetime, timedelta

import pytest

from app.services import rainfall_service as rs
from app.services import risk_inputs


TARGET = datetime(2026, 8, 20)
SIKKIM = "Sikkim"
STATES = ("Sikkim", "Assam", "Arunachal Pradesh", "Meghalaya")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
class FakeClock(object):
    """Deterministic, advanceable UTC clock."""

    def __init__(self, start=None):
        self.now = start or datetime(2026, 8, 20, 12, 0, 0)

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now = self.now + timedelta(seconds=seconds)


class FakeImerg(object):
    """
    Records every (day, bounds) it is asked for. `missing_before` makes granules
    404 so the backward probe has to walk; `fail_with` breaks it outright.
    """

    def __init__(self, missing_days=0, fail_with=None, value_for=None):
        self.missing_days = missing_days
        self.fail_with = fail_with
        self.value_for = value_for or (lambda day: 1.0 + day.day % 5)
        self.calls = []

    def __call__(self, session, day, bounds, run_type):
        self.calls.append((day.strftime("%Y-%m-%d"), tuple(sorted(bounds.items())), run_type))
        if self.fail_with is not None:
            raise self.fail_with
        if (TARGET.date() - day.date()).days <= self.missing_days:
            raise RuntimeError("EARTHDATA IMERG FETCH FAILED: 404 Not Found")
        return float(self.value_for(day))

    @property
    def days(self):
        return [c[0] for c in self.calls]


class FakeFallback(object):
    def __init__(self, fail_with=None, value_for=None):
        self.fail_with = fail_with
        self.value_for = value_for or (lambda k: 2.0 + k)
        self.calls = []

    def __call__(self, bounds, end_date, days):
        self.calls.append((end_date.strftime("%Y-%m-%d"), days))
        if self.fail_with is not None:
            raise self.fail_with
        return [float(self.value_for(k)) for k in range(days)]


def _session_factory():
    return object()


def _fresh_cache():
    return rs.RainfallCache(max_entries=32)


def _call(**kwargs):
    kwargs.setdefault("state_name", SIKKIM)
    kwargs.setdefault("target_date", TARGET)
    kwargs.setdefault("session_factory", _session_factory)
    kwargs.setdefault("clock", FakeClock())
    kwargs.setdefault("cache", _fresh_cache())
    return rs.get_state_rainfall(**kwargs)


def _read_source(*parts):
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(rs.__file__))), *parts)
    with open(path, "r", encoding="utf-8") as handle:
        return handle.read()


def _called_names(source):
    names = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name):
                names.add(func.id)
            elif isinstance(func, ast.Attribute):
                names.add(func.attr)
    return names


@contextmanager
def _env_override(**overrides):
    """
    Set env vars for the duration of a block and restore them afterwards.

    Done by hand rather than with monkeypatch because the offline shim does not
    always undo environment mutations; a leaked SIH_RAINFALL_* flag would
    silently change the behaviour of every later test in this module.
    A value of None deletes the variable.
    """
    previous = {name: os.environ.get(name) for name in overrides}
    try:
        for name, value in overrides.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = str(value)
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


# ---------------------------------------------------------------------------
# Contract: feature names / window semantics are the fitted ones
# ---------------------------------------------------------------------------
def test_feature_names_are_single_sourced_from_risk_inputs():
    assert rs.RAINFALL_FEATURE_NAMES is risk_inputs.RAINFALL_FEATURE_NAMES
    assert rs.RAINFALL_FEATURE_NAMES == (
        "rain_1d", "rain_3d", "rain_7d", "antecedent_rain_14d", "rain_intensity_max_3d",
    )


def test_window_length_matches_the_prediction_module():
    from app.services import sikkim_prediction as sp

    assert rs.RAINFALL_WINDOW_DAYS == sp.RAINFALL_WINDOW_DAYS == 14


def test_derivation_is_identical_to_the_prediction_module():
    from app.services import sikkim_prediction as sp

    series = [0.0, 12.5, 3.25, 0.0, 44.0, 1.0, 2.0, 0.5, 0.0, 7.75, 19.0, 0.0, 0.0, 31.5]
    assert rs.derive_rainfall_features(series) == sp._derive_rainfall_features(series)


def test_derivation_arithmetic():
    series = [1.0, 2.0, 3.0] + [1.0] * 11
    feats = rs.derive_rainfall_features(series)
    assert feats["rain_1d"] == 1.0
    assert feats["rain_3d"] == 6.0
    assert feats["rain_7d"] == 10.0
    assert feats["antecedent_rain_14d"] == 17.0
    assert feats["rain_intensity_max_3d"] == 3.0


def test_short_series_is_refused():
    with pytest.raises(ValueError):
        rs.derive_rainfall_features([1.0] * 13)


def test_run_type_is_normalised_and_validated():
    assert rs.normalise_run_type("early") == "Early"
    assert rs.normalise_run_type("FINAL") == "Final"
    with pytest.raises(ValueError):
        rs.normalise_run_type("Nowcast")


def test_bounds_come_from_the_canonical_aoi_accessor():
    from app.core.config_states import get_pilot_aoi_bounds

    for state in STATES:
        assert rs.resolve_bounds(state) == get_pilot_aoi_bounds(state)


def test_unknown_state_is_refused():
    with pytest.raises(KeyError):
        rs.resolve_bounds("Nagaland")


# ---------------------------------------------------------------------------
# Preferred path: real IMERG
# ---------------------------------------------------------------------------
def test_imerg_success_is_labelled_real():
    imerg = FakeImerg()
    record = _call(imerg_fetcher=imerg, fallback_fetcher=FakeFallback())

    assert record["data_quality_status"] == rs.QUALITY_REAL
    assert record["source_kind"] == rs.SOURCE_KIND_IMERG
    assert record["source"] == "IMERG_Early"
    assert record["is_fallback"] is False
    assert record["usable"] is True
    assert record["units"] == "mm"
    assert len(record["daily_series_mm"]) == 14
    assert set(record["features"]) == set(rs.RAINFALL_FEATURE_NAMES)


def test_required_contract_keys_are_present():
    record = _call(imerg_fetcher=FakeImerg())
    for key in ("source", "timestamp", "freshness", "coverage", "units",
                "data_quality_status"):
        assert key in record
    assert set(record["timestamp"]) == {
        "fetched_at_utc", "requested_date", "rainfall_observation_date",
    }
    assert record["coverage"]["aoi"] == rs.resolve_bounds(SIKKIM)
    assert record["coverage"]["window_days"] == 14
    assert record["coverage"]["aoi_uniform"] is True


def test_window_is_strictly_antecedent():
    imerg = FakeImerg()
    record = _call(imerg_fetcher=imerg)

    observed = record["timestamp"]["rainfall_observation_date"]
    assert observed == "2026-08-19"                       # T-1, never the target day
    assert record["timestamp"]["requested_date"] == "2026-08-20"
    assert "2026-08-20" not in imerg.days                 # event day never fetched
    assert imerg.days == [
        (TARGET - timedelta(days=1 + k)).strftime("%Y-%m-%d") for k in range(14)
    ]


def test_probe_value_is_reused_so_no_day_is_fetched_twice():
    imerg = FakeImerg()
    _call(imerg_fetcher=imerg)

    # 14 calls, not 15: the successful probe's value becomes day T-1.
    assert len(imerg.days) == 14
    assert len(set(imerg.days)) == 14


def test_probe_walks_back_over_missing_granules():
    imerg = FakeImerg(missing_days=3)
    record = _call(imerg_fetcher=imerg)

    assert record["data_quality_status"] == rs.QUALITY_REAL
    assert record["timestamp"]["rainfall_observation_date"] == "2026-08-16"
    assert record["freshness"]["probe_days_walked"] == 4
    assert record["freshness"]["observation_lag_days"] == 4


def test_a_non_404_imerg_error_does_not_keep_probing():
    imerg = FakeImerg(fail_with=PermissionError("EARTHDATA AUTHENTICATION REJECTED (HTTP 401)"))
    record = _call(imerg_fetcher=imerg, fallback_fetcher=FakeFallback(fail_with=RuntimeError("no net")))

    assert len(imerg.calls) == 1                          # refused immediately
    assert record["data_quality_status"] == rs.QUALITY_UNAVAILABLE
    assert "401" in record["unavailable_reason"]


def test_exhausted_probe_window_falls_through():
    imerg = FakeImerg(missing_days=999)
    record = _call(imerg_fetcher=imerg, fallback_fetcher=FakeFallback(fail_with=RuntimeError("no net")))

    assert record["data_quality_status"] == rs.QUALITY_UNAVAILABLE
    assert "no available IMERG Early observation" in record["unavailable_reason"]


# ---------------------------------------------------------------------------
# Fallback path
# ---------------------------------------------------------------------------
def test_fallback_is_used_only_after_imerg_fails():
    imerg = FakeImerg()
    fallback = FakeFallback()
    record = _call(imerg_fetcher=imerg, fallback_fetcher=fallback)

    assert record["source_kind"] == rs.SOURCE_KIND_IMERG
    assert fallback.calls == []                           # never touched on success


def test_fallback_record_is_explicitly_labelled():
    fallback = FakeFallback()
    record = _call(
        imerg_fetcher=FakeImerg(fail_with=PermissionError("BLOCKER: Missing NASA Earthdata credentials!")),
        fallback_fetcher=fallback,
    )

    assert record["data_quality_status"] == rs.QUALITY_FALLBACK
    assert record["source_kind"] == rs.SOURCE_KIND_FALLBACK
    assert record["is_fallback"] is True
    assert record["source"] == "Open-Meteo ERA5 archive (FALLBACK)"
    assert "FALLBACK" in record["note"]
    assert "not a calibrated probability" in record["note"].lower()
    assert record["caveats"] and any("not NASA GPM IMERG" in c for c in record["caveats"])
    assert len(record["daily_series_mm"]) == 14


def test_fallback_is_never_called_official_live_observation():
    record = _call(
        imerg_fetcher=FakeImerg(fail_with=RuntimeError("down")),
        fallback_fetcher=FakeFallback(),
    )
    # The disclaimer is explicit and the status never claims REAL.
    assert "NOT an official live satellite observation" in record["note"]
    assert record["data_quality_status"] != rs.QUALITY_REAL
    assert "IMERG" not in record["source"]


def test_fallback_window_is_also_strictly_antecedent():
    fallback = FakeFallback()
    record = _call(
        imerg_fetcher=FakeImerg(fail_with=RuntimeError("down")),
        fallback_fetcher=fallback,
    )
    assert fallback.calls == [("2026-08-19", 14)]
    assert record["timestamp"]["rainfall_observation_date"] == "2026-08-19"
    assert record["freshness"]["observation_lag_days"] == 1


def test_fallback_can_be_disabled_by_env():
    with _env_override(**{rs.ENV_FALLBACK_ENABLED: "0"}):
        fallback = FakeFallback()
        record = _call(
            imerg_fetcher=FakeImerg(fail_with=RuntimeError("down")),
            fallback_fetcher=fallback,
        )
    assert fallback.calls == []
    assert record["data_quality_status"] == rs.QUALITY_UNAVAILABLE
    assert any(a["status"] == "DISABLED" for a in record["attempts"])
    assert rs.fallback_enabled() is True          # the flag really was restored


# ---------------------------------------------------------------------------
# Refusal honesty
# ---------------------------------------------------------------------------
def test_both_sources_failing_yields_no_numbers():
    record = _call(
        imerg_fetcher=FakeImerg(fail_with=RuntimeError("imerg down")),
        fallback_fetcher=FakeFallback(fail_with=RuntimeError("open-meteo down")),
    )
    assert record["data_quality_status"] == rs.QUALITY_UNAVAILABLE
    assert record["usable"] is False
    assert record["daily_series_mm"] is None
    assert record["features"] is None
    assert record["source"] is None
    assert "imerg down" in record["unavailable_reason"]
    assert "open-meteo down" in record["unavailable_reason"]


def test_missing_credentials_surface_as_the_reason():
    def refuse():
        raise PermissionError("BLOCKER: Missing NASA Earthdata credentials!")

    record = _call(
        session_factory=refuse,
        imerg_fetcher=FakeImerg(),
        fallback_fetcher=FakeFallback(fail_with=RuntimeError("offline")),
    )
    assert record["data_quality_status"] == rs.QUALITY_UNAVAILABLE
    assert "Missing NASA Earthdata credentials" in record["unavailable_reason"]


@pytest.mark.parametrize("bad", [None, float("nan"), float("inf"), -3.0])
def test_bad_day_refuses(bad):
    def fetcher(session, day, bounds, run_type):
        return bad if day == TARGET - timedelta(days=5) else 1.0

    record = _call(
        imerg_fetcher=fetcher,
        fallback_fetcher=FakeFallback(fail_with=RuntimeError("offline")),
    )
    assert record["data_quality_status"] == rs.QUALITY_UNAVAILABLE
    assert record["daily_series_mm"] is None


def test_short_fallback_series_is_not_padded():
    def stingy(bounds, end_date, days):
        return [1.0] * (days - 2)

    record = _call(
        imerg_fetcher=FakeImerg(fail_with=RuntimeError("down")),
        fallback_fetcher=stingy,
    )
    assert record["data_quality_status"] == rs.QUALITY_UNAVAILABLE
    assert "refusing to pad" in record["unavailable_reason"]


def test_a_genuinely_dry_window_is_a_real_measurement():
    record = _call(imerg_fetcher=lambda s, d, b, r: 0.0)
    assert record["data_quality_status"] == rs.QUALITY_REAL
    assert record["features"]["antecedent_rain_14d"] == 0.0


# ---------------------------------------------------------------------------
# Cache + freshness
# ---------------------------------------------------------------------------
def test_second_read_is_served_from_cache():
    imerg = FakeImerg()
    cache = _fresh_cache()
    clock = FakeClock()

    first = _call(imerg_fetcher=imerg, cache=cache, clock=clock)
    calls_after_first = len(imerg.calls)
    clock.advance(60)
    second = _call(imerg_fetcher=imerg, cache=cache, clock=clock)

    assert len(imerg.calls) == calls_after_first          # zero extra network I/O
    assert first["freshness"]["cache_hit"] is False
    assert second["freshness"]["cache_hit"] is True
    assert second["freshness"]["age_seconds"] == 60.0
    assert second["features"] == first["features"]


def test_default_ttl_is_thirty_minutes():
    with _env_override(**{rs.ENV_CACHE_TTL: None}):
        assert rs.cache_ttl_seconds() == 1800.0
        record = _call(imerg_fetcher=FakeImerg())
    assert record["freshness"]["ttl_seconds"] == 1800.0


def test_entry_expires_after_the_ttl():
    imerg = FakeImerg()
    cache = _fresh_cache()
    clock = FakeClock()

    _call(imerg_fetcher=imerg, cache=cache, clock=clock)
    first = len(imerg.calls)
    clock.advance(1801)
    again = _call(imerg_fetcher=imerg, cache=cache, clock=clock)

    assert len(imerg.calls) > first                       # refetched
    assert again["freshness"]["cache_hit"] is False


def test_ttl_is_configurable():
    with _env_override(**{rs.ENV_CACHE_TTL: "45"}):
        imerg = FakeImerg()
        cache = _fresh_cache()
        clock = FakeClock()
        _call(imerg_fetcher=imerg, cache=cache, clock=clock)
        calls = len(imerg.calls)
        clock.advance(46)
        _call(imerg_fetcher=imerg, cache=cache, clock=clock)
    assert len(imerg.calls) > calls
    assert rs.cache_ttl_seconds() == 1800.0       # restored to the default


def test_expires_in_seconds_counts_down():
    cache = _fresh_cache()
    clock = FakeClock()
    _call(imerg_fetcher=FakeImerg(), cache=cache, clock=clock)
    clock.advance(300)
    hit = _call(imerg_fetcher=FakeImerg(), cache=cache, clock=clock)
    assert hit["freshness"]["expires_in_seconds"] == 1500.0


def test_each_state_aoi_gets_its_own_cache_entry():
    imerg = FakeImerg()
    cache = _fresh_cache()
    clock = FakeClock()
    seen = {}
    for state in STATES:
        record = rs.get_state_rainfall(
            state_name=state, target_date=TARGET, imerg_fetcher=imerg,
            session_factory=_session_factory, clock=clock, cache=cache,
        )
        assert record["freshness"]["cache_hit"] is False
        seen[state] = record["coverage"]["aoi"]

    assert cache.size() == 4
    # Distinct bboxes, so an AOI-blind key would have cross-served rainfall.
    assert len({tuple(sorted(b.items())) for b in seen.values()}) == 4


def test_cache_key_separates_run_type_and_date():
    keys = {
        rs.cache_key(SIKKIM, rs.resolve_bounds(SIKKIM), TARGET, "Early", 14),
        rs.cache_key(SIKKIM, rs.resolve_bounds(SIKKIM), TARGET, "Final", 14),
        rs.cache_key(SIKKIM, rs.resolve_bounds(SIKKIM), TARGET - timedelta(days=1), "Early", 14),
        rs.cache_key("Assam", rs.resolve_bounds("Assam"), TARGET, "Early", 14),
    }
    assert len(keys) == 4


def test_refusal_is_cached_briefly_but_stays_a_refusal():
    imerg = FakeImerg(fail_with=RuntimeError("imerg down"))
    fallback = FakeFallback(fail_with=RuntimeError("offline"))
    cache = _fresh_cache()
    clock = FakeClock()

    first = _call(imerg_fetcher=imerg, fallback_fetcher=fallback, cache=cache, clock=clock)
    calls = len(imerg.calls)
    clock.advance(10)
    second = _call(imerg_fetcher=imerg, fallback_fetcher=fallback, cache=cache, clock=clock)

    assert len(imerg.calls) == calls                      # no fan-out storm
    assert second["data_quality_status"] == rs.QUALITY_UNAVAILABLE
    assert second["daily_series_mm"] is None              # a refusal, never a zero
    assert second["freshness"]["cache_hit"] is True
    assert first["freshness"]["ttl_seconds"] == 120.0     # short negative TTL

    clock.advance(200)
    third = _call(imerg_fetcher=imerg, fallback_fetcher=fallback, cache=cache, clock=clock)
    assert len(imerg.calls) > calls                       # retried after expiry
    assert third["freshness"]["cache_hit"] is False


def test_cache_is_bounded():
    cache = rs.RainfallCache(max_entries=2)
    clock = FakeClock()
    for offset in range(5):
        rs.get_state_rainfall(
            state_name=SIKKIM, target_date=TARGET - timedelta(days=offset),
            imerg_fetcher=FakeImerg(missing_days=0), session_factory=_session_factory,
            clock=clock, cache=cache,
        )
    assert cache.size() == 2


def test_use_cache_false_bypasses_the_cache():
    cache = _fresh_cache()
    _call(imerg_fetcher=FakeImerg(), cache=cache, use_cache=False)
    assert cache.size() == 0


# ---------------------------------------------------------------------------
# Bounded fan-out
# ---------------------------------------------------------------------------
def test_probe_reach_is_configurable():
    with _env_override(**{rs.ENV_MAX_PROBE_DAYS: "3"}):
        imerg = FakeImerg(missing_days=999)
        record = _call(
            imerg_fetcher=imerg,
            fallback_fetcher=FakeFallback(fail_with=RuntimeError("offline")),
        )
    assert len(imerg.calls) == 3
    assert record["data_quality_status"] == rs.QUALITY_UNAVAILABLE


def test_deadline_stops_a_slow_walk():
    clock = FakeClock()

    def slow(session, day, bounds, run_type):
        clock.advance(20)
        raise RuntimeError("EARTHDATA IMERG FETCH FAILED: 404 Not Found")

    with _env_override(**{rs.ENV_DEADLINE: "30"}):
        record = _call(
            imerg_fetcher=slow, clock=clock,
            fallback_fetcher=FakeFallback(fail_with=RuntimeError("offline")),
        )
    assert record["data_quality_status"] == rs.QUALITY_UNAVAILABLE
    assert "budget of 30s exhausted" in record["unavailable_reason"]


def test_worst_case_call_count_is_bounded():
    with _env_override(**{rs.ENV_MAX_PROBE_DAYS: "30"}):
        imerg = FakeImerg(missing_days=5)
        _call(imerg_fetcher=imerg)
    # 6 probes (5 x 404 + 1 hit) + 13 remaining window days = 19, not 20.
    assert len(imerg.calls) == 19


# ---------------------------------------------------------------------------
# Provider adapter (consumed by the prediction path in a later step)
# ---------------------------------------------------------------------------
def test_provider_payload_keeps_every_legacy_key():
    record = _call(imerg_fetcher=FakeImerg())
    payload = rs.to_provider_payload(record)

    for key in ("source", "run_type", "aoi_uniform", "window_days", "requested_date",
                "rainfall_observation_date", "daily_series_mm", "features"):
        assert key in payload
    assert payload["source"] == "IMERG_Early"
    assert payload["run_type"] == "Early"
    assert payload["aoi_uniform"] is True
    assert payload["window_days"] == 14
    assert set(payload["features"]) == set(rs.RAINFALL_FEATURE_NAMES)


def test_provider_payload_matches_the_shipped_provider_shape():
    from app.services import sikkim_prediction as sp

    payload = rs.to_provider_payload(_call(imerg_fetcher=FakeImerg()))
    # Every key the shipped _rainfall_report reads must be present, so the
    # prediction path can delegate here without changing its response shape.
    for key in ("source", "run_type", "aoi_uniform", "window_days", "daily_series_mm",
                "features"):
        assert key in payload
    assert set(sp.RAINFALL_FEATURES).issubset(payload["features"])
    report = sp._rainfall_report(payload)
    assert report["source"] == "IMERG_Early"
    assert report["window_days"] == 14
    assert set(report["features"]) == set(sp.RAINFALL_FEATURES)


def test_provider_payload_adds_provenance():
    record = _call(imerg_fetcher=FakeImerg())
    payload = rs.to_provider_payload(record)
    for key in ("source_kind", "is_fallback", "data_quality_status", "units",
                "fetched_at_utc", "freshness", "coverage", "caveats", "note"):
        assert key in payload
    assert payload["data_quality_status"] == rs.QUALITY_REAL
    assert payload["is_fallback"] is False


def test_provider_payload_marks_fallback():
    record = _call(
        imerg_fetcher=FakeImerg(fail_with=RuntimeError("down")),
        fallback_fetcher=FakeFallback(),
    )
    payload = rs.to_provider_payload(record)
    assert payload["data_quality_status"] == rs.QUALITY_FALLBACK
    assert payload["is_fallback"] is True
    assert payload["run_type"] == "Early"


def test_provider_payload_refuses_an_unavailable_record():
    record = _call(
        imerg_fetcher=FakeImerg(fail_with=RuntimeError("imerg down")),
        fallback_fetcher=FakeFallback(fail_with=RuntimeError("offline")),
    )
    with pytest.raises(rs.RainfallUnavailable) as caught:
        rs.to_provider_payload(record)
    assert "imerg down" in str(caught.value)


# ---------------------------------------------------------------------------
# Source-level integrity: startup safety, memory safety, frozen collaborators
# ---------------------------------------------------------------------------
def test_startup_never_touches_the_rainfall_service():
    """
    Requirement 15: FastAPI startup must never block on a network rainfall
    request. The cheapest durable proof is that the app entrypoint does not
    reference this module at all.
    """
    source = _read_source("main.py")
    assert "rainfall_service" not in source
    assert "get_state_rainfall" not in source


def test_module_does_no_import_time_work():
    """
    Every network/env read must be inside a function body, so merely importing
    the service (which routes.py will do) cannot fetch or block.
    """
    tree = ast.parse(_read_source("services", "rainfall_service.py"))
    forbidden = {"get", "post", "urlopen", "request", "getenv", "environ"}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.Import, ast.ImportFrom)):
            continue
        assert not (_called_names(ast.unparse(node)) & forbidden), (
            "module-level statement performs deferred-only work: %s" % ast.unparse(node)
        )


def test_no_unbounded_response_read():
    """
    The 512 MB Render budget: the fallback body must be read with an explicit
    size cap, never via an unbounded .content / .read().
    """
    source = _read_source("services", "rainfall_service.py")
    assert "response.content" not in source
    assert "response.text" not in source
    assert ".read()" not in source
    assert "iter_content(chunk_size=" in source
    assert "MAX_FALLBACK_BODY_BYTES" in source
    # The cap must actually be applied to the body, not merely defined.
    assert source.count("MAX_FALLBACK_BODY_BYTES") >= 2


def test_no_raster_is_opened():
    """Rainfall is a point/AOI-mean series; this service must never open a raster."""
    source = _read_source("services", "rainfall_service.py")
    assert "rasterio" not in source
    assert ".tif" not in source


def test_thresholds_and_weather_ingestion_are_only_read_not_reimplemented():
    """
    Requirements 11 + 12: the Sikkim-derived thresholds and the IMERG auth
    guards stay where they are. This module may CALL weather_ingestion but must
    not restate a threshold constant or its own auth/redirect logic.
    """
    source = _read_source("services", "rainfall_service.py")
    assert "14.2" not in source          # the I = 14.2 * D^-0.62 coefficient
    assert "-0.62" not in source
    assert "should_strip_auth" not in source
    assert "Bearer" not in source
    assert "_fetch_imerg_day" in source  # delegated, not reimplemented
    assert "get_earthdata_session" in source


def test_thresholds_module_is_unmodified_by_this_work():
    """The frozen rainfall trigger must still be the Sikkim-derived one."""
    source = _read_source("models", "thresholds.py")
    assert "14.2" in source
    assert "-0.62" in source
    assert "rainfall_service" not in source
