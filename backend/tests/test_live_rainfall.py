"""
Offline tests for app.services.live_rainfall -- the near-real-time rainfall
monitor. No credentials, no network, no NetCDF: every collaborator (granule
fetcher, session factory, fallback fetcher, clock, cache) is injected.

What these tests actually pin down:
  * granule identity -- YYYY/DDD directory, minutes-of-day filename field, Early
    vs Late prefixes, and the confirmed [time][lon][lat] constraint order;
  * mm/hr -> mm conversion for a 30-minute interval (rate * 0.5);
  * the probe walk stops at the first published granule and REUSES its value;
  * 3 h / 6 h accumulations are null-with-reason when the run is short;
  * Early -> Late -> Open-Meteo -> UNAVAILABLE, with attempts recorded;
  * UNAVAILABLE carries no numbers;
  * the live cache is a separate object from the antecedent one;
  * structurally, the module cannot reach derive_rainfall_features() and never
    touches response.content / response.text.
"""

import ast
import os
import sys
from datetime import datetime, timedelta

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from app.services import live_rainfall as lr  # noqa: E402

MODULE_PATH = os.path.join(BACKEND_DIR, "app", "services", "live_rainfall.py")

SIKKIM_BOUNDS = {
    "min_lat": 27.0,
    "max_lat": 28.1,
    "min_lon": 88.0,
    "max_lon": 88.9,
}

# The probe anchor used throughout. now - 240 min (Early latency) = 09:10 ->
# floored to the 09:00 slot, whose minutes-of-day field is 0540.
NOW = datetime(2025, 9, 18, 13, 12, 45)
ANCHOR = datetime(2025, 9, 18, 9, 0, 0)


def _clear():
    lr.clear_cache()


class _FakeSession(object):
    """Stands in for the authenticated Earthdata session; never used to fetch."""

    def __init__(self):
        self.closed = False


def _session_factory():
    return _FakeSession()


def _fetcher_from(rates_by_slot, missing=()):
    """
    A granule fetcher backed by a dict of slot -> mm/hr, recording every call.
    Slots in `missing` raise GranuleUnavailable (the 404 case); unknown slots do
    too, so a walk cannot silently read a slot the test never defined.
    """
    calls = []

    def fetcher(session, slot_start, bounds, run_type):
        calls.append((slot_start, run_type))
        if slot_start in missing or slot_start not in rates_by_slot:
            raise lr.GranuleUnavailable("not published: %s" % slot_start)
        return rates_by_slot[slot_start]

    fetcher.calls = calls
    return fetcher


def _contiguous_rates(anchor, count, rate=2.0):
    return {
        anchor - timedelta(minutes=lr.GRANULE_MINUTES * step): rate
        for step in range(count)
    }


# ---------------------------------------------------------------------------
# Granule identity: directory, filename, constraint order
# ---------------------------------------------------------------------------
def test_floor_to_granule_snaps_to_half_hour_boundaries():
    assert lr.floor_to_granule(datetime(2025, 9, 18, 9, 29, 59)) == datetime(
        2025, 9, 18, 9, 0, 0
    )
    assert lr.floor_to_granule(datetime(2025, 9, 18, 9, 30, 0)) == datetime(
        2025, 9, 18, 9, 30, 0
    )
    assert lr.floor_to_granule(datetime(2025, 9, 18, 9, 59, 30)) == datetime(
        2025, 9, 18, 9, 30, 0
    )


def test_minutes_of_day_matches_the_filename_sequence_field():
    assert lr.minutes_of_day(datetime(2025, 9, 18, 0, 0)) == 0
    assert lr.minutes_of_day(datetime(2025, 9, 18, 9, 0)) == 540
    assert lr.minutes_of_day(datetime(2025, 9, 18, 9, 30)) == 570
    assert lr.minutes_of_day(datetime(2025, 9, 18, 23, 30)) == 1410


def test_granule_basename_is_the_confirmed_early_name():
    assert lr.granule_basename(lr.RUN_TYPE_EARLY, ANCHOR) == (
        "3B-HHR-E.MS.MRG.3IMERG.20250918-S090000-E092959.0540.V07B.HDF5"
    )


def test_granule_basename_uses_the_late_prefix_for_the_late_run():
    name = lr.granule_basename(lr.RUN_TYPE_LATE, ANCHOR)
    assert name.startswith("3B-HHR-L.")
    assert name.endswith(".0540.V07B.HDF5")


def test_granule_basename_pads_the_sequence_field_to_four_digits():
    assert ".0000." in lr.granule_basename(
        lr.RUN_TYPE_EARLY, datetime(2025, 9, 18, 0, 0)
    )
    assert ".0030." in lr.granule_basename(
        lr.RUN_TYPE_EARLY, datetime(2025, 9, 18, 0, 30)
    )


def test_granule_directory_url_uses_day_of_year_not_month():
    url = lr.granule_directory_url(lr.RUN_TYPE_EARLY, ANCHOR)
    assert url.endswith("/GPM_3IMERGHHE.07/2025/261")
    assert url.startswith(lr.OPENDAP_ROOT)


def test_late_directory_url_uses_the_late_collection():
    assert lr.granule_directory_url(lr.RUN_TYPE_LATE, ANCHOR).endswith(
        "/GPM_3IMERGHHL.07/2025/261"
    )


def test_subset_url_requests_precipitation_in_time_lon_lat_order():
    from app.services import weather_ingestion

    lat_min, lat_max, lon_min, lon_max = weather_ingestion.get_imerg_indices(
        SIKKIM_BOUNDS
    )
    url = lr.granule_subset_url(lr.RUN_TYPE_EARLY, ANCHOR, SIKKIM_BOUNDS)
    expected = "?precipitation[0:0][%d:%d][%d:%d]" % (
        lon_min,
        lon_max,
        lat_min,
        lat_max,
    )
    assert url.endswith(expected), url
    assert ".nc4?" in url
    assert lon_min < lon_max and lat_min < lat_max


def test_subset_url_is_an_aoi_subset_not_a_full_grid():
    url = lr.granule_subset_url(lr.RUN_TYPE_EARLY, ANCHOR, SIKKIM_BOUNDS)
    assert "[0:3599]" not in url
    assert "[0:1799]" not in url


# ---------------------------------------------------------------------------
# Units: the product publishes mm/hr, a 30-minute total is rate * 0.5
# ---------------------------------------------------------------------------
def test_granule_hours_is_a_half_hour():
    assert lr.VARIABLE_UNITS == "mm/hr"
    assert lr.GRANULE_MINUTES == 30
    assert lr.GRANULE_HOURS == 0.5


def test_interval_value_is_the_rate_halved():
    _clear()
    record = lr.get_latest_rainfall(
        "Sikkim",
        bounds=SIKKIM_BOUNDS,
        now=NOW,
        granule_fetcher=_fetcher_from(_contiguous_rates(ANCHOR, 12, rate=4.0)),
        session_factory=_session_factory,
        use_cache=False,
    )
    assert record["data_quality_status"] == lr.QUALITY_REAL
    assert record["latest_available_rainfall_mm"] == 2.0
    assert record["interval_minutes"] == 30
    assert record["units"] == "mm"


# ---------------------------------------------------------------------------
# The probe walk
# ---------------------------------------------------------------------------
def test_probe_starts_at_now_minus_publication_latency():
    _clear()
    fetcher = _fetcher_from(_contiguous_rates(ANCHOR, 12))
    record = lr.get_latest_rainfall(
        "Sikkim",
        bounds=SIKKIM_BOUNDS,
        now=NOW,
        granule_fetcher=fetcher,
        session_factory=_session_factory,
        use_cache=False,
    )
    assert fetcher.calls[0][0] == ANCHOR
    assert record["observed_at_utc"] == "2025-09-18T09:00:00Z"


def test_probe_walks_back_over_unpublished_slots_and_reuses_the_hit():
    _clear()
    hit = ANCHOR - timedelta(minutes=60)
    fetcher = _fetcher_from(
        _contiguous_rates(hit, 12), missing=(ANCHOR, ANCHOR - timedelta(minutes=30))
    )
    record = lr.get_latest_rainfall(
        "Sikkim",
        bounds=SIKKIM_BOUNDS,
        now=NOW,
        granule_fetcher=fetcher,
        session_factory=_session_factory,
        use_cache=False,
    )
    assert record["observed_at_utc"] == "2025-09-18T08:00:00Z"
    # The anchor is fetched exactly once: the window walk reuses the probe value.
    assert [slot for slot, _ in fetcher.calls].count(hit) == 1


def test_probe_walk_is_bounded_by_probe_granules():
    _clear()
    fetcher = _fetcher_from({})
    os.environ["SIH_LIVE_RAINFALL_PROBE_GRANULES"] = "4"
    try:
        record = lr.get_latest_rainfall(
            "Sikkim",
            bounds=SIKKIM_BOUNDS,
            now=NOW,
            granule_fetcher=fetcher,
            session_factory=_session_factory,
            include_late=False,
            include_fallback=False,
            use_cache=False,
        )
    finally:
        del os.environ["SIH_LIVE_RAINFALL_PROBE_GRANULES"]
    assert len(fetcher.calls) == 4
    assert record["data_quality_status"] == lr.QUALITY_UNAVAILABLE


# ---------------------------------------------------------------------------
# Accumulations: complete or null-with-reason, never a partial sum
# ---------------------------------------------------------------------------
def test_accumulate_window_requires_every_interval():
    total, reason = lr.accumulate_window([1.0] * 12, 6, 30)
    assert total == 12.0
    assert reason is None


def test_accumulate_window_refuses_a_short_run():
    total, reason = lr.accumulate_window([1.0] * 5, 3, 30)
    assert total is None
    assert "incomplete window" in reason
    assert "5 of 6" in reason


def test_full_window_reports_both_accumulations():
    _clear()
    record = lr.get_latest_rainfall(
        "Sikkim",
        bounds=SIKKIM_BOUNDS,
        now=NOW,
        granule_fetcher=_fetcher_from(_contiguous_rates(ANCHOR, 12, rate=2.0)),
        session_factory=_session_factory,
        use_cache=False,
    )
    assert record["granules_used"] == 12
    assert record["accum_3h_mm"] == 6.0
    assert record["accum_6h_mm"] == 12.0
    assert record["accum_3h_unavailable_reason"] is None
    assert record["accum_6h_unavailable_reason"] is None


def test_short_window_nulls_the_six_hour_accumulation_with_a_reason():
    _clear()
    record = lr.get_latest_rainfall(
        "Sikkim",
        bounds=SIKKIM_BOUNDS,
        now=NOW,
        granule_fetcher=_fetcher_from(_contiguous_rates(ANCHOR, 6, rate=2.0)),
        session_factory=_session_factory,
        use_cache=False,
    )
    assert record["granules_used"] == 6
    assert record["accum_3h_mm"] == 6.0
    assert record["accum_6h_mm"] is None
    assert "incomplete window" in record["accum_6h_unavailable_reason"]


def test_six_hour_accumulation_is_enabled_by_default():
    assert lr.ACCUMULATION_WINDOWS_HOURS == (3, 6)
    assert lr.window_granules() >= lr.granules_required(6)


# ---------------------------------------------------------------------------
# Source order and provenance
# ---------------------------------------------------------------------------
def test_early_is_preferred_when_it_is_published():
    _clear()
    record = lr.get_latest_rainfall(
        "Sikkim",
        bounds=SIKKIM_BOUNDS,
        now=NOW,
        granule_fetcher=_fetcher_from(_contiguous_rates(ANCHOR, 12)),
        session_factory=_session_factory,
        use_cache=False,
    )
    assert record["source_kind"] == lr.SOURCE_KIND_EARLY
    assert "GPM_3IMERGHHE.07" in record["source"]
    assert [a["source_kind"] for a in record["attempts"]] == [lr.SOURCE_KIND_EARLY]


def test_late_is_used_when_early_has_no_published_granule():
    _clear()
    late_anchor = lr.floor_to_granule(
        NOW - timedelta(minutes=lr.IMERG_PRODUCTS[lr.RUN_TYPE_LATE]["latency_minutes"])
    )
    rates = _contiguous_rates(late_anchor, 12, rate=1.0)

    def fetcher(session, slot_start, bounds, run_type):
        if run_type == lr.RUN_TYPE_EARLY or slot_start not in rates:
            raise lr.GranuleUnavailable("not published")
        return rates[slot_start]

    record = lr.get_latest_rainfall(
        "Sikkim",
        bounds=SIKKIM_BOUNDS,
        now=NOW,
        granule_fetcher=fetcher,
        session_factory=_session_factory,
        use_cache=False,
    )
    assert record["source_kind"] == lr.SOURCE_KIND_LATE
    assert record["data_quality_status"] == lr.QUALITY_REAL
    outcomes = {a["source_kind"]: a["outcome"] for a in record["attempts"]}
    assert outcomes[lr.SOURCE_KIND_EARLY] == "no_published_granule"
    assert outcomes[lr.SOURCE_KIND_LATE] == "ok"


def test_auth_rejection_stops_the_imerg_phase_and_is_recorded():
    _clear()

    def fetcher(session, slot_start, bounds, run_type):
        raise PermissionError("EARTHDATA AUTHENTICATION REJECTED (HTTP 401)")

    record = lr.get_latest_rainfall(
        "Sikkim",
        bounds=SIKKIM_BOUNDS,
        now=NOW,
        granule_fetcher=fetcher,
        session_factory=_session_factory,
        include_fallback=False,
        use_cache=False,
    )
    assert record["data_quality_status"] == lr.QUALITY_UNAVAILABLE
    assert record["attempts"][0]["outcome"] == "auth_rejected"
    assert [a["source_kind"] for a in record["attempts"]] == [lr.SOURCE_KIND_EARLY]


def test_missing_credentials_are_recorded_as_auth_unavailable():
    _clear()

    def broken_factory():
        raise PermissionError("BLOCKER: Missing NASA Earthdata credentials!")

    record = lr.get_latest_rainfall(
        "Sikkim",
        bounds=SIKKIM_BOUNDS,
        now=NOW,
        granule_fetcher=_fetcher_from(_contiguous_rates(ANCHOR, 12)),
        session_factory=broken_factory,
        include_fallback=False,
        use_cache=False,
    )
    assert record["data_quality_status"] == lr.QUALITY_UNAVAILABLE
    assert record["attempts"][0]["outcome"] == "auth_unavailable"


def _fallback_pairs(anchor, count, mm=1.5):
    return [(anchor - timedelta(minutes=60 * step), mm) for step in range(count)]


def test_open_meteo_fallback_is_labelled_and_used_last():
    _clear()
    hour_anchor = NOW.replace(minute=0, second=0, microsecond=0)

    def fetcher(session, slot_start, bounds, run_type):
        raise lr.GranuleUnavailable("not published")

    record = lr.get_latest_rainfall(
        "Sikkim",
        bounds=SIKKIM_BOUNDS,
        now=NOW,
        granule_fetcher=fetcher,
        session_factory=_session_factory,
        fallback_fetcher=lambda bounds, now: _fallback_pairs(hour_anchor, 6, 1.0),
        use_cache=False,
    )
    assert record["source_kind"] == lr.SOURCE_KIND_FALLBACK
    assert record["data_quality_status"] == lr.QUALITY_FALLBACK
    assert "FALLBACK" in record["source"]
    assert record["interval_minutes"] == 60
    assert record["latest_available_rainfall_mm"] == 1.0
    assert record["accum_3h_mm"] == 3.0
    assert record["accum_6h_mm"] == 6.0
    kinds = [a["source_kind"] for a in record["attempts"]]
    assert kinds == [
        lr.SOURCE_KIND_EARLY,
        lr.SOURCE_KIND_LATE,
        lr.SOURCE_KIND_FALLBACK,
    ]


def test_fallback_only_accumulates_a_contiguous_hour_run():
    _clear()
    hour_anchor = NOW.replace(minute=0, second=0, microsecond=0)
    pairs = [
        (hour_anchor, 2.0),
        (hour_anchor - timedelta(minutes=60), 2.0),
        # 2 hours missing here -- the run ends.
        (hour_anchor - timedelta(minutes=240), 9.0),
    ]

    def fetcher(session, slot_start, bounds, run_type):
        raise lr.GranuleUnavailable("not published")

    record = lr.get_latest_rainfall(
        "Sikkim",
        bounds=SIKKIM_BOUNDS,
        now=NOW,
        granule_fetcher=fetcher,
        session_factory=_session_factory,
        fallback_fetcher=lambda bounds, now: pairs,
        use_cache=False,
    )
    assert record["granules_used"] == 2
    assert record["accum_3h_mm"] is None
    assert record["accum_6h_mm"] is None
    assert "2 of 3" in record["accum_3h_unavailable_reason"]


def test_everything_failing_yields_unavailable_with_no_numbers():
    _clear()

    def fetcher(session, slot_start, bounds, run_type):
        raise lr.GranuleUnavailable("not published")

    def broken_fallback(bounds, now):
        raise RuntimeError("Open-Meteo unreachable")

    record = lr.get_latest_rainfall(
        "Sikkim",
        bounds=SIKKIM_BOUNDS,
        now=NOW,
        granule_fetcher=fetcher,
        session_factory=_session_factory,
        fallback_fetcher=broken_fallback,
        use_cache=False,
    )
    assert record["data_quality_status"] == lr.QUALITY_UNAVAILABLE
    for field in (
        "latest_available_rainfall_mm",
        "accum_3h_mm",
        "accum_6h_mm",
        "observed_at_utc",
        "age_minutes",
        "source",
        "source_kind",
    ):
        assert record[field] is None, field
    assert record["unavailable_reason"]
    assert record["fetched_at_utc"] == "2025-09-18T13:12:45Z"


# ---------------------------------------------------------------------------
# Freshness: the value is never called "current"
# ---------------------------------------------------------------------------
def test_freshness_labels_follow_the_configured_boundaries():
    near = lr.near_real_time_minutes()
    stale = lr.stale_after_minutes()
    assert lr.freshness_label_for(near) == lr.FRESHNESS_NEAR_REAL_TIME
    assert lr.freshness_label_for(near + 1) == lr.FRESHNESS_RECENT
    assert lr.freshness_label_for(stale) == lr.FRESHNESS_RECENT
    assert lr.freshness_label_for(stale + 1) == lr.FRESHNESS_STALE


def test_age_and_latency_are_reported_for_a_real_observation():
    _clear()
    record = lr.get_latest_rainfall(
        "Sikkim",
        bounds=SIKKIM_BOUNDS,
        now=NOW,
        granule_fetcher=_fetcher_from(_contiguous_rates(ANCHOR, 12)),
        session_factory=_session_factory,
        use_cache=False,
    )
    # 09:00:00 observed, 13:12:45 fetched -> 252.75 minutes.
    assert record["age_minutes"] == 252.8
    assert record["latency_minutes"] == record["age_minutes"]
    assert record["freshness_label"] == lr.FRESHNESS_RECENT
    assert record["is_stale"] is False
    assert record["expected_product_latency_minutes"] == 240
    assert record["staleness_threshold_minutes"] == lr.stale_after_minutes()


def test_a_very_old_observation_is_flagged_stale():
    """
    Early's own ~4 h publication latency sits below the default 6 h staleness
    threshold, so a fresh Early granule is RECENT rather than STALE. Tightening
    the threshold is what exercises the stale branch.
    """
    _clear()
    os.environ["SIH_LIVE_RAINFALL_STALE_MINUTES"] = "60"
    try:
        record = lr.get_latest_rainfall(
            "Sikkim",
            bounds=SIKKIM_BOUNDS,
            now=NOW,
            granule_fetcher=_fetcher_from(_contiguous_rates(ANCHOR, 12)),
            session_factory=_session_factory,
            use_cache=False,
        )
    finally:
        del os.environ["SIH_LIVE_RAINFALL_STALE_MINUTES"]
    assert record["age_minutes"] == 252.8
    assert record["is_stale"] is True
    assert record["freshness_label"] == lr.FRESHNESS_STALE
    assert record["staleness_threshold_minutes"] == 60.0


def test_the_value_field_is_named_latest_available_not_current():
    _clear()
    record = lr.get_latest_rainfall(
        "Sikkim",
        bounds=SIKKIM_BOUNDS,
        now=NOW,
        granule_fetcher=_fetcher_from(_contiguous_rates(ANCHOR, 12)),
        session_factory=_session_factory,
        use_cache=False,
    )
    assert "latest_available_rainfall_mm" in record
    assert not [key for key in record if key.startswith("current_")]


# ---------------------------------------------------------------------------
# AOI resolution
# ---------------------------------------------------------------------------
def test_bounds_default_to_the_canonical_pilot_aoi():
    from app.core import config_states

    for state in config_states.PILOT_AOIS:
        assert lr.resolve_bounds(state) == config_states.get_pilot_aoi_bounds(state)


def test_all_four_pilots_are_supported():
    assert lr.supported_states() == [
        "Arunachal Pradesh",
        "Assam",
        "Meghalaya",
        "Sikkim",
    ]


def test_unknown_state_raises_rather_than_guessing_an_aoi():
    try:
        lr.resolve_bounds("Atlantis")
    except KeyError:
        return
    raise AssertionError("an unknown state must not resolve to any AOI")


def test_the_record_echoes_the_aoi_it_subset():
    _clear()
    record = lr.get_latest_rainfall(
        "Meghalaya",
        now=NOW,
        granule_fetcher=_fetcher_from(_contiguous_rates(ANCHOR, 12)),
        session_factory=_session_factory,
        use_cache=False,
    )
    from app.core import config_states

    assert record["state"] == "Meghalaya"
    assert record["aoi_bounds"] == config_states.get_pilot_aoi_bounds("Meghalaya")


# ---------------------------------------------------------------------------
# Caching: separate storage, separate env namespace
# ---------------------------------------------------------------------------
def test_a_second_call_is_served_from_the_live_cache():
    _clear()
    fetcher = _fetcher_from(_contiguous_rates(ANCHOR, 12))
    first = lr.get_latest_rainfall(
        "Sikkim",
        bounds=SIKKIM_BOUNDS,
        now=NOW,
        granule_fetcher=fetcher,
        session_factory=_session_factory,
    )
    calls_after_first = len(fetcher.calls)
    second = lr.get_latest_rainfall(
        "Sikkim",
        bounds=SIKKIM_BOUNDS,
        now=NOW,
        granule_fetcher=fetcher,
        session_factory=_session_factory,
    )
    assert first["served_from_cache"] is False
    assert second["served_from_cache"] is True
    assert len(fetcher.calls) == calls_after_first
    _clear()


def test_the_live_cache_is_not_the_antecedent_cache():
    from app.services import rainfall_service

    assert lr.LiveRainfallCache is not rainfall_service.RainfallCache
    assert lr._CACHE is not getattr(rainfall_service, "_CACHE", None)


def test_the_env_namespace_is_disjoint_from_the_model_rainfall_one():
    names = [
        value
        for name, value in vars(lr).items()
        if name.startswith("ENV_") and isinstance(value, str)
    ]
    assert names
    for name in names:
        assert name.startswith("SIH_LIVE_RAINFALL_"), name


def test_a_refusal_gets_the_short_negative_ttl():
    assert lr.negative_cache_ttl_seconds() < lr.cache_ttl_seconds()


def test_different_bounds_do_not_share_a_cache_entry():
    other = dict(SIKKIM_BOUNDS)
    other["max_lat"] = 28.0
    assert lr.cache_key("Sikkim", SIKKIM_BOUNDS, "live") != lr.cache_key(
        "Sikkim", other, "live"
    )


def test_the_cache_is_bounded():
    cache = lr.LiveRainfallCache(max_entries=2)
    for index in range(5):
        cache.put(("state-%d" % index,), {"data_quality_status": lr.QUALITY_REAL}, 0.0)
    assert len(cache) == 2


# ---------------------------------------------------------------------------
# Structural guarantees (requirements 11, 12, 13 and the memory ceiling)
# ---------------------------------------------------------------------------
def _module_tree():
    with open(MODULE_PATH, "r") as handle:
        return ast.parse(handle.read())


def test_the_module_never_imports_rainfall_service():
    for node in ast.walk(_module_tree()):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert "rainfall_service" not in alias.name
        elif isinstance(node, ast.ImportFrom):
            names = [alias.name for alias in node.names]
            assert "rainfall_service" not in names
            assert "rainfall_service" not in (node.module or "")


def test_the_module_never_calls_derive_rainfall_features():
    for node in ast.walk(_module_tree()):
        if isinstance(node, ast.Name):
            assert node.id != "derive_rainfall_features"
        if isinstance(node, ast.Attribute):
            assert node.attr != "derive_rainfall_features"


def test_no_unbounded_body_read_in_executable_code():
    """
    response.content / response.text / read() would each materialise a whole body
    in RAM. The docstrings mention them by name, so this walks the AST rather than
    grepping the text.
    """
    banned_attributes = {"content", "text"}
    for node in ast.walk(_module_tree()):
        if isinstance(node, ast.Attribute) and node.attr in banned_attributes:
            raise AssertionError("unbounded body access: .%s" % node.attr)
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "read"
        ):
            raise AssertionError("unbounded read() call")


def test_streaming_ceilings_are_enforced_constants():
    assert lr.MAX_SUBSET_BODY_BYTES == 2 * 1024 * 1024
    assert lr.MAX_FALLBACK_BODY_BYTES == 512 * 1024
    assert 0 < lr.SUBSET_CHUNK_BYTES < lr.MAX_SUBSET_BODY_BYTES
    assert 0 < lr.FALLBACK_CHUNK_BYTES < lr.MAX_FALLBACK_BODY_BYTES


def test_one_subset_per_granule_not_one_per_grid_cell():
    """
    A 1.1 x 0.9 degree AOI spans ~99 IMERG cells. The whole call must still issue
    at most window_granules() requests -- one AOI subset per granule.
    """
    _clear()
    fetcher = _fetcher_from(_contiguous_rates(ANCHOR, 12))
    lr.get_latest_rainfall(
        "Sikkim",
        bounds=SIKKIM_BOUNDS,
        now=NOW,
        granule_fetcher=fetcher,
        session_factory=_session_factory,
        use_cache=False,
    )
    assert len(fetcher.calls) == 12
    assert len(fetcher.calls) <= lr.window_granules()


def test_open_meteo_parser_drops_nulls_and_future_hours():
    now = datetime(2025, 9, 18, 13, 0)
    payload = {
        "hourly": {
            "time": [
                "2025-09-18T11:00",
                "2025-09-18T12:00",
                "2025-09-18T13:00",
                "2025-09-18T14:00",
            ],
            "precipitation": [1.0, None, 3.0, 99.0],
        }
    }
    pairs = lr._parse_open_meteo_hours(payload, now)
    assert pairs == [
        (datetime(2025, 9, 18, 11, 0), 1.0),
        (datetime(2025, 9, 18, 13, 0), 3.0),
    ]


def test_open_meteo_parser_returns_nothing_for_an_all_null_window():
    payload = {
        "hourly": {
            "time": ["2025-09-18T11:00", "2025-09-18T12:00"],
            "precipitation": [None, None],
        }
    }
    assert lr._parse_open_meteo_hours(payload, datetime(2025, 9, 18, 13, 0)) == []


def test_module_import_does_no_network_or_credential_work():
    """
    Import-time work would run on every worker boot. The module body must contain
    no top-level calls beyond constant construction.
    """
    for node in _module_tree().body:
        assert not isinstance(node, ast.Expr) or isinstance(
            node.value, (ast.Constant, ast.Str)
        )


# ---------------------------------------------------------------------------
# Freshness parity: the FALLBACK path must be aged like the IMERG path
#
# Regression guard. A host run of the four pilots returned a populated value and
# accumulations from the Open-Meteo FALLBACK but no freshness structure, even
# though observed_at_utc and fetched_at_utc were both present. Freshness is a
# property of two timestamps, not of the source, so every record that has an
# observation must carry the block.
# ---------------------------------------------------------------------------
FRESHNESS_KEYS = (
    "observed_at_utc",
    "fetched_at_utc",
    "age_seconds",
    "age_minutes",
    "freshness_label",
    "is_stale",
    "staleness_threshold_minutes",
    "near_real_time_threshold_minutes",
    "measured_from",
    "cache_hit",
)


def _imerg_record(now=NOW):
    _clear()
    return lr.get_latest_rainfall(
        "Sikkim",
        bounds=SIKKIM_BOUNDS,
        now=now,
        granule_fetcher=_fetcher_from(_contiguous_rates(ANCHOR, 12, rate=4.0)),
        session_factory=_session_factory,
        include_fallback=False,
    )


def _fallback_record(now=NOW, hours=8, mm=1.5):
    _clear()
    hour_anchor = now.replace(minute=0, second=0, microsecond=0)

    def fetcher(session, slot_start, bounds, run_type):
        raise lr.GranuleUnavailable("not published")

    return lr.get_latest_rainfall(
        "Sikkim",
        bounds=SIKKIM_BOUNDS,
        now=now,
        granule_fetcher=fetcher,
        session_factory=_session_factory,
        fallback_fetcher=lambda bounds, now_: _fallback_pairs(
            hour_anchor, hours, mm=mm
        ),
    )


def test_measured_age_is_derived_from_the_two_timestamps_only():
    observed = datetime(2026, 8, 31, 10, 0, 0)
    fetched = datetime(2026, 8, 31, 10, 43, 34)
    assert lr.measured_age_seconds(observed, fetched) == 2614.0
    assert lr.measured_age_seconds(None, fetched) is None
    assert lr.measured_age_seconds(observed, None) is None


def test_the_fallback_record_carries_a_populated_freshness_block():
    record = _fallback_record()
    assert record["data_quality_status"] == lr.QUALITY_FALLBACK
    freshness = record["freshness"]
    assert freshness is not None
    for key in FRESHNESS_KEYS:
        assert key in freshness
    assert freshness["observed_at_utc"] == "2025-09-18T13:00:00Z"
    assert freshness["fetched_at_utc"] == "2025-09-18T13:12:45Z"
    assert freshness["age_seconds"] == 765.0
    assert freshness["age_minutes"] == 12.8
    assert freshness["freshness_label"] == lr.FRESHNESS_NEAR_REAL_TIME
    assert freshness["is_stale"] is False


def test_fallback_and_imerg_freshness_blocks_have_identical_shape():
    imerg = _imerg_record()
    fallback = _fallback_record()
    assert imerg["data_quality_status"] == lr.QUALITY_REAL
    assert fallback["data_quality_status"] == lr.QUALITY_FALLBACK
    assert sorted(imerg["freshness"]) == sorted(fallback["freshness"])
    assert sorted(imerg["freshness"]) == sorted(FRESHNESS_KEYS)
    for record in (imerg, fallback):
        assert record["freshness"]["age_seconds"] is not None
        assert record["freshness"]["freshness_label"] is not None
        assert record["freshness"]["is_stale"] is not None


def test_fallback_age_is_measured_not_taken_from_a_product_latency():
    """
    Open-Meteo has no published latency constant, so the age must come from the
    timestamps. The IMERG record keeps its expected_product_latency_minutes as
    documentation, but its measured age is what freshness uses.
    """
    fallback = _fallback_record()
    assert fallback["expected_product_latency_minutes"] is None
    assert fallback["age_minutes"] == 12.8

    imerg = _imerg_record()
    assert imerg["expected_product_latency_minutes"] == 240.0
    assert imerg["freshness"]["age_minutes"] == imerg["age_minutes"] == 252.8


def test_top_level_age_fields_mirror_the_freshness_block():
    for record in (_imerg_record(), _fallback_record()):
        freshness = record["freshness"]
        assert record["age_seconds"] == freshness["age_seconds"]
        assert record["age_minutes"] == freshness["age_minutes"]
        assert record["observed_at_utc"] == freshness["observed_at_utc"]
        assert record["fetched_at_utc"] == freshness["fetched_at_utc"]
        assert record["freshness_label"] == freshness["freshness_label"]
        assert record["is_stale"] == freshness["is_stale"]


def test_fallback_staleness_uses_the_same_threshold_rule_as_imerg():
    previous = os.environ.get(lr.ENV_STALE_MINUTES)
    os.environ[lr.ENV_STALE_MINUTES] = "10"
    try:
        fallback = _fallback_record()
        assert fallback["freshness"]["age_minutes"] == 12.8
        assert fallback["freshness"]["is_stale"] is True
        assert fallback["freshness"]["freshness_label"] == lr.FRESHNESS_STALE
        assert fallback["freshness"]["staleness_threshold_minutes"] == 10.0

        imerg = _imerg_record()
        assert imerg["freshness"]["is_stale"] is True
        assert imerg["freshness"]["freshness_label"] == lr.FRESHNESS_STALE
        assert imerg["freshness"]["staleness_threshold_minutes"] == 10.0
    finally:
        if previous is None:
            os.environ.pop(lr.ENV_STALE_MINUTES, None)
        else:
            os.environ[lr.ENV_STALE_MINUTES] = previous


def test_the_label_never_contradicts_is_stale_under_an_inverted_config():
    """
    Guards the label/flag agreement rule directly: with the stale threshold set
    below the near-real-time band, an aged observation must read STALE, not
    NEAR_REAL_TIME.
    """
    previous_stale = os.environ.get(lr.ENV_STALE_MINUTES)
    previous_nrt = os.environ.get(lr.ENV_NEAR_REAL_TIME_MINUTES)
    os.environ[lr.ENV_STALE_MINUTES] = "10"
    os.environ[lr.ENV_NEAR_REAL_TIME_MINUTES] = "90"
    try:
        assert lr.freshness_label_for(12.8) == lr.FRESHNESS_STALE
        assert lr.freshness_label_for(5.0) == lr.FRESHNESS_NEAR_REAL_TIME
    finally:
        for name, value in (
            (lr.ENV_STALE_MINUTES, previous_stale),
            (lr.ENV_NEAR_REAL_TIME_MINUTES, previous_nrt),
        ):
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def test_an_unavailable_record_still_carries_the_block_with_null_age():
    _clear()

    def fetcher(session, slot_start, bounds, run_type):
        raise lr.GranuleUnavailable("not published")

    record = lr.get_latest_rainfall(
        "Sikkim",
        bounds=SIKKIM_BOUNDS,
        now=NOW,
        granule_fetcher=fetcher,
        session_factory=_session_factory,
        include_fallback=False,
    )
    assert record["data_quality_status"] == lr.QUALITY_UNAVAILABLE
    freshness = record["freshness"]
    assert sorted(freshness) == sorted(FRESHNESS_KEYS)
    assert freshness["observed_at_utc"] is None
    assert freshness["fetched_at_utc"] == "2025-09-18T13:12:45Z"
    assert freshness["age_seconds"] is None
    assert freshness["age_minutes"] is None
    assert freshness["freshness_label"] is None
    assert freshness["is_stale"] is None


def test_cache_hit_flag_is_accurate_on_both_passes():
    _clear()
    hour_anchor = NOW.replace(minute=0, second=0, microsecond=0)

    def fetcher(session, slot_start, bounds, run_type):
        raise lr.GranuleUnavailable("not published")

    def call():
        return lr.get_latest_rainfall(
            "Sikkim",
            bounds=SIKKIM_BOUNDS,
            now=NOW,
            granule_fetcher=fetcher,
            session_factory=_session_factory,
            fallback_fetcher=lambda bounds, now_: _fallback_pairs(hour_anchor, 8),
        )

    first = call()
    assert first["cache_hit"] is False
    assert first["served_from_cache"] is False

    second = call()
    assert second["cache_hit"] is True
    assert second["served_from_cache"] is True
    # Everything except the delivery provenance is byte-identical.
    assert {k: v for k, v in second["freshness"].items() if k != "cache_hit"} == {
        k: v for k, v in first["freshness"].items() if k != "cache_hit"
    }
    assert first["freshness"]["cache_hit"] is False
    assert second["freshness"]["cache_hit"] is True

    _clear()
    third = call()
    assert third["cache_hit"] is False


def test_freshness_values_are_unchanged_rainfall_and_labels():
    """
    This fix is presentational: it must not move a millimetre of rainfall, nor
    relabel a source, nor reorder the chain.
    """
    fallback = _fallback_record(mm=1.5, hours=8)
    assert fallback["latest_available_rainfall_mm"] == 1.5
    assert fallback["accum_3h_mm"] == 4.5
    assert fallback["accum_6h_mm"] == 9.0
    assert fallback["source"] == lr.FALLBACK_SOURCE_LABEL
    assert fallback["source_kind"] == lr.SOURCE_KIND_FALLBACK
    assert [a["source_kind"] for a in fallback["attempts"]] == [
        lr.SOURCE_KIND_EARLY,
        lr.SOURCE_KIND_LATE,
        lr.SOURCE_KIND_FALLBACK,
    ]

    imerg = _imerg_record()
    assert imerg["latest_available_rainfall_mm"] == 2.0
    assert imerg["source_kind"] == lr.SOURCE_KIND_EARLY


def test_the_freshness_fix_did_not_reintroduce_a_current_named_field():
    for record in (_imerg_record(), _fallback_record()):
        for key in list(record) + list(record["freshness"]):
            assert not key.startswith("current_")
            assert key != "current_rainfall_mm"


# ---------------------------------------------------------------------------
# Cache-hit provenance inside the freshness block (requirements 1-8 of the
# cache fix). A monitoring consumer that reads only `freshness` must be able to
# tell a fresh acquisition from a replay, and a replay must cost nothing.
# ---------------------------------------------------------------------------
class _CountingProviders(object):
    """
    A pair of injected providers that count every call they receive, so a test
    can assert that a cache hit reached NEITHER IMERG nor Open-Meteo.
    """

    def __init__(self, now=NOW, hours=8, mm=1.5):
        self.imerg_calls = 0
        self.fallback_calls = 0
        self._anchor = now.replace(minute=0, second=0, microsecond=0)
        self._hours = hours
        self._mm = mm

    def granule_fetcher(self, session, slot_start, bounds, run_type):
        self.imerg_calls += 1
        raise lr.GranuleUnavailable("not published")

    def fallback_fetcher(self, bounds, now_):
        self.fallback_calls += 1
        return _fallback_pairs(self._anchor, self._hours, mm=self._mm)

    @property
    def total(self):
        return self.imerg_calls + self.fallback_calls


def _call_state(state, providers, now=NOW, bounds=None):
    return lr.get_latest_rainfall(
        state,
        bounds=bounds,
        now=now,
        granule_fetcher=providers.granule_fetcher,
        session_factory=_session_factory,
        fallback_fetcher=providers.fallback_fetcher,
    )


def test_two_identical_calls_hit_the_cache_with_zero_new_provider_calls():
    """
    The regression the host smoke test asked for, stated exactly: same state
    twice, provider calls counted, cache_hit flipping, timestamps frozen.
    """
    _clear()
    providers = _CountingProviders()

    first = _call_state("Assam", providers)
    calls_after_first = providers.total
    assert calls_after_first > 0

    second = _call_state("Assam", providers)

    assert first["freshness"]["cache_hit"] is False
    assert second["freshness"]["cache_hit"] is True
    assert providers.total == calls_after_first
    assert (
        second["freshness"]["fetched_at_utc"] == first["freshness"]["fetched_at_utc"]
    )
    assert second["fetched_at_utc"] == first["fetched_at_utc"]
    assert (
        second["freshness"]["observed_at_utc"] == first["freshness"]["observed_at_utc"]
    )
    assert second["latest_available_rainfall_mm"] == first[
        "latest_available_rainfall_mm"
    ]
    _clear()


def test_a_cache_hit_leaves_the_stored_record_untouched():
    """
    Requirement 4. The caller is handed a copy, so scribbling on the returned
    record -- as the annotation itself does -- cannot rewrite the cached
    observation. A third call must still read the original timestamps.
    """
    _clear()
    providers = _CountingProviders()
    first = _call_state("Assam", providers)
    second = _call_state("Assam", providers)
    calls_after_two = providers.total

    second["freshness"]["observed_at_utc"] = "1999-01-01T00:00:00Z"
    second["freshness"]["cache_hit"] = "tampered"
    second["attempts"][0]["source_kind"] = "tampered"

    third = _call_state("Assam", providers)
    assert third["freshness"]["observed_at_utc"] == first["freshness"][
        "observed_at_utc"
    ]
    assert third["freshness"]["cache_hit"] is True
    assert third["attempts"][0]["source_kind"] == lr.SOURCE_KIND_EARLY
    assert providers.total == calls_after_two
    _clear()


def test_the_cached_copy_is_not_the_object_the_caller_holds():
    cache = lr.LiveRainfallCache()
    record = {
        "data_quality_status": lr.QUALITY_REAL,
        "freshness": {"cache_hit": False, "observed_at_utc": "x"},
        "aoi_bounds": dict(SIKKIM_BOUNDS),
        "attempts": [{"source_kind": lr.SOURCE_KIND_EARLY}],
    }
    cache.put(("k",), record, 0.0)
    record["freshness"]["observed_at_utc"] = "mutated-after-put"

    got = cache.get(("k",), 0.0)
    assert got["freshness"]["observed_at_utc"] == "x"
    assert got["freshness"] is not record["freshness"]
    assert got["attempts"][0] is not record["attempts"][0]
    assert got["aoi_bounds"] is not record["aoi_bounds"]


def test_cache_expiry_still_causes_a_new_fetch():
    """Requirement 6: a zero TTL must not make the cache sticky."""
    _clear()
    previous = os.environ.get(lr.ENV_CACHE_TTL)
    os.environ[lr.ENV_CACHE_TTL] = "0"
    try:
        providers = _CountingProviders()
        first = _call_state("Assam", providers)
        calls_after_first = providers.total
        second = _call_state("Assam", providers)
        assert first["freshness"]["cache_hit"] is False
        assert second["freshness"]["cache_hit"] is False
        assert providers.total > calls_after_first
    finally:
        if previous is None:
            os.environ.pop(lr.ENV_CACHE_TTL, None)
        else:
            os.environ[lr.ENV_CACHE_TTL] = previous
    _clear()


def test_cache_keys_stay_state_aoi_and_run_type_specific():
    """Requirement 5, restated as three independent dimensions."""
    other_bounds = dict(SIKKIM_BOUNDS)
    other_bounds["max_lon"] = 89.4
    base = lr.cache_key("Sikkim", SIKKIM_BOUNDS, "live")
    assert base != lr.cache_key("Assam", SIKKIM_BOUNDS, "live")
    assert base != lr.cache_key("Sikkim", other_bounds, "live")
    assert base != lr.cache_key("Sikkim", SIKKIM_BOUNDS, "other")
    assert base == lr.cache_key("Sikkim", dict(SIKKIM_BOUNDS), "live")


def test_two_different_states_do_not_serve_each_others_cached_record():
    _clear()
    providers = _CountingProviders()
    assam = _call_state("Assam", providers)
    meghalaya = _call_state("Meghalaya", providers)
    assert assam["freshness"]["cache_hit"] is False
    assert meghalaya["freshness"]["cache_hit"] is False
    assert meghalaya["aoi_bounds"] != assam["aoi_bounds"]
    _clear()


def test_a_cache_hit_reports_the_same_values_and_source_as_the_acquisition():
    """Requirement 7: replay must not change a millimetre or a label."""
    _clear()
    providers = _CountingProviders()
    first = _call_state("Assam", providers)
    second = _call_state("Assam", providers)
    for key in (
        "latest_available_rainfall_mm",
        "accum_3h_mm",
        "accum_6h_mm",
        "source",
        "source_kind",
        "data_quality_status",
        "observed_at_utc",
    ):
        assert second[key] == first[key], key
    assert [a["source_kind"] for a in second["attempts"]] == [
        a["source_kind"] for a in first["attempts"]
    ]
    _clear()


def test_the_top_level_and_nested_cache_flags_agree():
    _clear()
    providers = _CountingProviders()
    for expected in (False, True):
        record = _call_state("Assam", providers)
        assert record["cache_hit"] is expected
        assert record["served_from_cache"] is expected
        assert record["freshness"]["cache_hit"] is expected
    _clear()


def test_an_unavailable_record_also_carries_the_cache_flag():
    _clear()

    def fetcher(session, slot_start, bounds, run_type):
        raise lr.GranuleUnavailable("not published")

    record = lr.get_latest_rainfall(
        "Assam",
        now=NOW,
        granule_fetcher=fetcher,
        session_factory=_session_factory,
        include_fallback=False,
    )
    assert record["data_quality_status"] == lr.QUALITY_UNAVAILABLE
    assert record["freshness"]["cache_hit"] is False
    _clear()

