"""
LIVE RAINFALL SERVICE -- one cached, provenance-stamped rainfall read per state AOI.

This module is the single place the serving path obtains near-real-time rainfall.
It exists because every /predict/<state>/grid request previously performed 15-44
sequential IMERG OPeNDAP round trips with no cache, and because an IMERG outage
took all four pilot consoles down with it.

WHAT IT GUARANTEES
  * NASA GPM IMERG is PREFERRED. The existing authentication guards in
    app.services.weather_ingestion (get_earthdata_session / _EarthdataAuthSession)
    are used unchanged; this module adds no credential handling of its own.
  * Open-Meteo ERA5 archive is a FALLBACK, used only after IMERG fails. It is
    ALWAYS stamped data_quality_status="FALLBACK", source_kind
    "OPEN_METEO_FALLBACK" and is_fallback=True. It is never described as an
    official live observation, and a probability computed from it is not a
    calibrated probability -- the record says so in `note` and `caveats`.
  * Nothing is ever fabricated. A window that cannot be obtained in full from
    ONE source yields data_quality_status="UNAVAILABLE" with the real reasons and
    NO numbers -- no zero fill, no imputation, no duplicated days, no partial
    series padded out.
  * The window is strictly antecedent (T-1..T-14 relative to the observation
    date), identical to the schema the models were fitted on. Feature names and
    their arithmetic are unchanged.
  * One fetch per state AOI per TTL, never per grid cell and never per frontend
    request. Cached entries carry explicit freshness metadata.
  * Latency-aware probing. A current-date request probes only the days IMERG could
    plausibly have published (see _probe_plan); it does not spend 30 sequential
    OPeNDAP round trips discovering that the product is not serving recent data,
    which previously consumed the whole wall-clock budget and starved the
    fallback. Historical requests keep the full 30-day reach.
  * Memory-flat: an entry is 14 floats plus small metadata, the cache is bounded,
    and no raster or NetCDF payload is retained (IMERG parsing still happens in
    weather_ingestion's tempfile-and-delete path).
  * Import-time work: none. No network call, no credential read, no cache warm.
    Nothing here may be reached from FastAPI startup.

Everything external is injectable (`imerg_fetcher`, `fallback_fetcher`,
`session_factory`, `clock`, `cache`) so the whole module is testable offline with
no credentials and no network.
"""

import json
import os
import threading
from collections import OrderedDict
from datetime import date as _date_cls, datetime, timedelta

from app.services import risk_inputs


# ---------------------------------------------------------------------------
# Contract vocabulary
# ---------------------------------------------------------------------------
# Single-sourced from risk_inputs so the live path and the training/serving
# resolver can never disagree about the five feature names.
RAINFALL_FEATURE_NAMES = risk_inputs.RAINFALL_FEATURE_NAMES

# Antecedent window semantics: T-1..T-14, event day excluded (no leakage).
RAINFALL_WINDOW_DAYS = 14

QUALITY_REAL = "REAL"
QUALITY_FALLBACK = "FALLBACK"
QUALITY_UNAVAILABLE = "UNAVAILABLE"

SOURCE_KIND_IMERG = "IMERG"
SOURCE_KIND_FALLBACK = "OPEN_METEO_FALLBACK"

FALLBACK_SOURCE_LABEL = "Open-Meteo ERA5 archive (FALLBACK)"

UNITS = "mm"

DEFAULT_RUN_TYPE = "Early"
RUN_TYPES = ("Early", "Late", "Final")

# The fallback window ends one day before the requested date, so it is strictly
# antecedent exactly like the IMERG window.
FALLBACK_LATENCY_DAYS = 1

OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
OPEN_METEO_TIMEOUT_SECONDS = 15

# Response bodies here are tiny JSON documents (a 14-element daily series), not
# rasters; this ceiling keeps that assumption enforced rather than assumed.
MAX_FALLBACK_BODY_BYTES = 256 * 1024
FALLBACK_CHUNK_BYTES = 16 * 1024

# ---------------------------------------------------------------------------
# Environment surface (read at CALL time, never at import time)
# ---------------------------------------------------------------------------
ENV_CACHE_TTL = "SIH_RAINFALL_CACHE_TTL_SECONDS"
ENV_NEGATIVE_TTL = "SIH_RAINFALL_NEGATIVE_TTL_SECONDS"
ENV_MAX_PROBE_DAYS = "SIH_RAINFALL_MAX_PROBE_DAYS"
ENV_DEADLINE = "SIH_RAINFALL_DEADLINE_SECONDS"
ENV_CACHE_MAX_ENTRIES = "SIH_RAINFALL_CACHE_MAX_ENTRIES"
ENV_FALLBACK_ENABLED = "SIH_RAINFALL_FALLBACK"
ENV_IMERG_RECENT_PROBE_DAYS = "SIH_RAINFALL_IMERG_RECENT_PROBE_DAYS"
ENV_IMERG_RECENT_GRACE_DAYS = "SIH_RAINFALL_IMERG_RECENT_GRACE_DAYS"

DEFAULT_CACHE_TTL_SECONDS = 1800.0        # 30 minutes, product-cadence independent
DEFAULT_NEGATIVE_TTL_SECONDS = 120.0      # short: a refusal must not stick around
DEFAULT_MAX_PROBE_DAYS = 30               # historical reach, unchanged
DEFAULT_DEADLINE_SECONDS = 120.0          # bounds the worst-case fan-out
DEFAULT_CACHE_MAX_ENTRIES = 16            # 4 pilots x a few dates/run types

# Near-real-time probing. IMERG publishes ONE granule per observation day,
# contiguously. If the newest plausibly-published day 404s, and so do the two
# before it, the product is not serving recent data for this account/AOI -- 27
# further round trips cannot change that outcome, they only burn the wall-clock
# budget the Open-Meteo fallback then needs. A HISTORICAL request keeps the full
# DEFAULT_MAX_PROBE_DAYS reach, because a historical hole genuinely can be an
# isolated missing granule.
DEFAULT_IMERG_RECENT_PROBE_DAYS = 3
DEFAULT_IMERG_RECENT_GRACE_DAYS = 2       # target this recent => NEAR_REAL_TIME

PROBE_MODE_NEAR_REAL_TIME = "NEAR_REAL_TIME"
PROBE_MODE_HISTORICAL = "HISTORICAL"

# Publication latency per product, in whole days. Used ONLY to skip candidate
# days that provably cannot be published yet -- never to skip a day that might
# exist. Early ~4-6 h and Late ~14 h both land inside one day, so for those two
# products this is a no-op and their behaviour is bit-identical to before.
IMERG_PRODUCT_LATENCY_DAYS = {"Early": 1, "Late": 1, "Final": 105}

# The IMERG phase may not consume the whole acquisition budget: without this a
# slow probe walk starves a fallback that would have answered in under a second.
# The TOTAL budget is unchanged -- this only splits it.
IMERG_PHASE_BUDGET_FRACTION = 0.75

def _env_number(name, default, minimum=0.0):
    raw = os.environ.get(name)
    if raw is None or str(raw).strip() == "":
        return float(default)
    try:
        value = float(str(raw).strip())
    except (TypeError, ValueError):
        return float(default)
    if value < minimum:
        return float(default)
    return value


def _env_flag(name, default=True):
    raw = os.environ.get(name)
    if raw is None or str(raw).strip() == "":
        return bool(default)
    return str(raw).strip().lower() not in ("0", "false", "no", "off")


def cache_ttl_seconds():
    return _env_number(ENV_CACHE_TTL, DEFAULT_CACHE_TTL_SECONDS)


def negative_cache_ttl_seconds():
    return _env_number(ENV_NEGATIVE_TTL, DEFAULT_NEGATIVE_TTL_SECONDS)


def max_probe_days():
    return int(_env_number(ENV_MAX_PROBE_DAYS, DEFAULT_MAX_PROBE_DAYS, minimum=1.0))


def deadline_seconds():
    return _env_number(ENV_DEADLINE, DEFAULT_DEADLINE_SECONDS, minimum=1.0)


def cache_max_entries():
    return int(_env_number(ENV_CACHE_MAX_ENTRIES, DEFAULT_CACHE_MAX_ENTRIES, minimum=1.0))


def fallback_enabled():
    """Open-Meteo fallback is ON by default and can be switched off per deployment."""
    return _env_flag(ENV_FALLBACK_ENABLED, True)


def imerg_recent_probe_days():
    """How many candidate days a NEAR_REAL_TIME request may probe."""
    return int(_env_number(
        ENV_IMERG_RECENT_PROBE_DAYS, DEFAULT_IMERG_RECENT_PROBE_DAYS, minimum=1.0
    ))


def imerg_recent_grace_days():
    """A target date this many days old (or newer) counts as NEAR_REAL_TIME."""
    return int(_env_number(
        ENV_IMERG_RECENT_GRACE_DAYS, DEFAULT_IMERG_RECENT_GRACE_DAYS, minimum=0.0
    ))


def imerg_product_latency_days(run_type):
    return int(IMERG_PRODUCT_LATENCY_DAYS.get(run_type, 1))


# ---------------------------------------------------------------------------
# Time helpers
# ---------------------------------------------------------------------------
def _utcnow():
    return datetime.utcnow()


def _as_datetime(value):
    """Coerce a datetime / date / 'YYYY-MM-DD' string into a naive datetime."""
    if isinstance(value, datetime):
        return value
    if isinstance(value, _date_cls):
        return datetime(value.year, value.month, value.day)
    if isinstance(value, str):
        return datetime.strptime(value.strip(), "%Y-%m-%d")
    raise ValueError(
        "target_date must be a datetime, date, or 'YYYY-MM-DD' string, got %r" % (value,)
    )


def _day(value):
    return value.strftime("%Y-%m-%d")


def _iso(value):
    return value.strftime("%Y-%m-%dT%H:%M:%SZ")


def _probe_plan(target_date, run_type, now):
    """
    Decide WHICH candidate days the latest-granule walk may try, from the
    product's real publication characteristics instead of a fixed 30.

    Returns (first_offset, last_offset, mode) where an offset of k means the day
    target_date - k.

      first_offset  the newest day that could plausibly be published already.
                    For Early/Late this is always 1 (their latency is hours), so
                    nothing changes for them; a current-date Final request stops
                    probing the ~104 days it cannot possibly have yet.
      last_offset   NEAR_REAL_TIME  -> first_offset + imerg_recent_probe_days() - 1
                    HISTORICAL      -> max_probe_days()  (the previous reach)
      mode          reported in the record so a short walk is never silent.

    last_offset is clamped by max_probe_days(), so that knob still bounds
    everything: setting it to 3 still yields at most 3 probes. If the newest
    publishable day lies beyond that reach the plan is EMPTY (last < first) and
    the caller refuses without issuing a single request.
    """
    reach_limit = max_probe_days()
    target_age_days = int((now.date() - target_date.date()).days)
    latency = imerg_product_latency_days(run_type)

    first_offset = max(1, latency - target_age_days)

    if target_age_days <= imerg_recent_grace_days():
        mode = PROBE_MODE_NEAR_REAL_TIME
        last_offset = first_offset + imerg_recent_probe_days() - 1
    else:
        mode = PROBE_MODE_HISTORICAL
        last_offset = reach_limit

    last_offset = min(last_offset, reach_limit)
    if first_offset > reach_limit:
        # Nothing within the configured reach can exist yet: an empty plan.
        return first_offset, first_offset - 1, mode
    return first_offset, last_offset, mode


def _latency_label(run_type):
    """Human-readable publication latency, for self-explaining error messages."""
    if run_type == "Early":
        return "4-6 h"
    if run_type == "Late":
        return "14 h"
    return "3.5 months"


class RainfallDeadlineExceeded(Exception):
    """The wall-clock budget for one rainfall acquisition ran out."""


class _Deadline(object):
    """
    Bounds one acquisition attempt. Without this, a granule-by-granule walk of 44
    days at a 30 s timeout each can hang a request for over 20 minutes before it
    finally refuses.
    """

    def __init__(self, clock, budget_seconds):
        self._clock = clock
        self._budget = float(budget_seconds)
        self._started = clock()

    def elapsed(self):
        return (self._clock() - self._started).total_seconds()

    def check(self, what, fraction=1.0):
        """
        `fraction` bounds ONE PHASE of the acquisition without changing the total
        budget: the IMERG walk gets IMERG_PHASE_BUDGET_FRACTION of it so that an
        overrunning probe cannot leave the Open-Meteo fallback with no time at
        all. The fallback itself checks with the full budget.
        """
        share = float(fraction)
        if share <= 0.0 or share > 1.0:
            share = 1.0
        budget = self._budget * share
        if self.elapsed() > budget:
            if share < 1.0:
                raise RainfallDeadlineExceeded(
                    "IMERG phase budget of %.0fs (%.0f%% of the %.0fs rainfall "
                    "acquisition budget, reserved so the fallback is not starved) "
                    "exhausted while %s" % (budget, share * 100.0, self._budget, what)
                )
            raise RainfallDeadlineExceeded(
                "rainfall acquisition budget of %.0fs exhausted while %s"
                % (self._budget, what)
            )


# ---------------------------------------------------------------------------
# Cache: bounded, TTL'd, thread-safe. Holds records only -- never raw payloads.
# ---------------------------------------------------------------------------
class RainfallCache(object):
    def __init__(self, max_entries=None):
        self._entries = OrderedDict()
        self._lock = threading.Lock()
        self._max_entries = max_entries

    def _limit(self):
        return int(self._max_entries or cache_max_entries())

    def get(self, key, now):
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                return None
            stored_at, ttl, record = entry
            age = (now - stored_at).total_seconds()
            if age > ttl:
                del self._entries[key]
                return None
            self._entries.move_to_end(key)
            return stored_at, ttl, age, record

    def put(self, key, now, ttl, record):
        with self._lock:
            self._entries[key] = (now, float(ttl), record)
            self._entries.move_to_end(key)
            while len(self._entries) > self._limit():
                self._entries.popitem(last=False)

    def clear(self):
        with self._lock:
            self._entries.clear()

    def size(self):
        with self._lock:
            return len(self._entries)


_CACHE = RainfallCache()


def clear_cache():
    """Drop every cached rainfall record (used by tests and by operators)."""
    _CACHE.clear()

# ---------------------------------------------------------------------------
# AOI + cache key
# ---------------------------------------------------------------------------
BBOX_KEYS = ("min_lat", "max_lat", "min_lon", "max_lon")


def resolve_bounds(state_name, bounds=None):
    """
    The AOI a rainfall read covers. Defaults to the canonical pilot AOI from
    app.core.config_states -- never a bbox restated here.
    """
    if bounds is not None:
        missing = [k for k in BBOX_KEYS if k not in bounds]
        if missing:
            raise ValueError("bounds is missing %s" % ", ".join(missing))
        return {k: float(bounds[k]) for k in BBOX_KEYS}
    from app.core.config_states import get_pilot_aoi_bounds

    return get_pilot_aoi_bounds(state_name)


def _centroid(bounds):
    return (
        (float(bounds["min_lat"]) + float(bounds["max_lat"])) / 2.0,
        (float(bounds["min_lon"]) + float(bounds["max_lon"])) / 2.0,
    )


def normalise_run_type(run_type):
    """
    Validate/normalise once, here, instead of leaving a free-form string to be
    lowercased deep inside weather_ingestion._fetch_imerg_day.
    """
    text = str(run_type or DEFAULT_RUN_TYPE).strip().lower()
    for known in RUN_TYPES:
        if text == known.lower():
            return known
    raise ValueError(
        "run_type must be one of %s, got %r" % (", ".join(RUN_TYPES), run_type)
    )


def cache_key(state_name, bounds, target_date, run_type, window_days):
    """
    The bbox is part of the key ON PURPOSE: the four pilot AOIs differ, so a
    date-only key would serve Assam's rainfall for Sikkim.
    """
    return (
        str(state_name),
        tuple(round(float(bounds[k]), 4) for k in BBOX_KEYS),
        _day(target_date),
        run_type,
        int(window_days),
    )


# ---------------------------------------------------------------------------
# Feature derivation (identical arithmetic to the fitted schema)
# ---------------------------------------------------------------------------
def derive_rainfall_features(daily, window_days=RAINFALL_WINDOW_DAYS):
    """
    daily[k] = precip mm for day T-(k+1); returns the five schema features.

        rain_1d               = day T-1
        rain_3d               = sum(T-1..T-3)
        rain_7d               = sum(T-1..T-7)
        antecedent_rain_14d   = sum(T-1..T-14)
        rain_intensity_max_3d = max(T-1..T-3)
    """
    if len(daily) < window_days:
        raise ValueError("need %d antecedent days, got %d" % (window_days, len(daily)))
    vals = [float(v) for v in daily]
    return {
        "rain_1d": vals[0],
        "rain_3d": float(sum(vals[0:3])),
        "rain_7d": float(sum(vals[0:7])),
        "antecedent_rain_14d": float(sum(vals[0:window_days])),
        "rain_intensity_max_3d": float(max(vals[0:3])),
    }


def _finite(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number or number in (float("inf"), float("-inf")):
        return None
    return number


def _validate_series(daily, window_days, source_kind):
    """A window must be complete and finite, or it is not a window."""
    if len(daily) != window_days:
        raise ValueError(
            "%s returned %d of %d antecedent days; refusing to pad the series"
            % (source_kind, len(daily), window_days)
        )
    for idx, value in enumerate(daily):
        if _finite(value) is None:
            raise ValueError(
                "%s day T-%d is missing or non-finite; refusing to substitute a value"
                % (source_kind, idx + 1)
            )
        if float(value) < 0.0:
            raise ValueError(
                "%s day T-%d is negative (%r), which is not physical precipitation"
                % (source_kind, idx + 1, value)
            )

# ---------------------------------------------------------------------------
# Preferred path: real IMERG
# ---------------------------------------------------------------------------
def _default_imerg_fetcher(session, day, bounds, run_type):
    """Thin passthrough to the UNCHANGED weather_ingestion granule reader."""
    from app.services import weather_ingestion

    return weather_ingestion._fetch_imerg_day(session, day, bounds, run_type)


def _default_session_factory():
    """
    The existing NASA Earthdata auth guard, used as-is. It raises PermissionError
    when no credentials are configured; that surfaces as an UNAVAILABLE reason and
    is never worked around.
    """
    from app.services import weather_ingestion

    return weather_ingestion.get_earthdata_session()


def _acquire_imerg_window(bounds, target_date, run_type, window_days,
                          imerg_fetcher, session_factory, deadline, now=None):
    """
    Walk backward for the latest available granule, then complete the antecedent
    window ending on it.

    Two bounded-fan-out properties matter here:
      * the value fetched by the successful probe is REUSED as day T-1 instead of
        being discarded and refetched -- one fewer OPeNDAP round trip per request;
      * WHICH days may be probed comes from _probe_plan, so a current-date request
        against a product that is not publishing recent data costs a handful of
        round trips instead of 30 (see _probe_plan for the reasoning). Historical
        requests keep the full reach.
    """
    first_offset, last_offset, probe_mode = _probe_plan(
        target_date, run_type, now or _utcnow()
    )
    probe_reach = max(0, last_offset - first_offset + 1)

    if probe_reach == 0:
        # The newest publishable day lies beyond the configured reach: every
        # request would provably 404, so none is issued.
        raise ValueError(
            "IMERG %s cannot have published any day within the configured %d-day "
            "probe reach for %s: with ~%s latency the newest publishable "
            "observation is T-%d. Refusing to issue a request that provably 404s."
            % (run_type, max_probe_days(), _day(target_date),
               _latency_label(run_type), first_offset)
        )

    session = session_factory()

    latest_date = None
    latest_value = None
    probe_days_walked = 0

    for offset in range(first_offset, last_offset + 1):
        deadline.check(
            "probing for the latest available IMERG %s granule" % run_type,
            fraction=IMERG_PHASE_BUDGET_FRACTION,
        )
        candidate = target_date - timedelta(days=offset)
        probe_days_walked = offset
        try:
            latest_value = float(imerg_fetcher(session, candidate, bounds, run_type))
            latest_date = candidate
            break
        except Exception as exc:
            # Only a missing granule justifies walking further back. Anything else
            # (auth rejection, no-data subset, transport failure) is a real error.
            if "404" not in str(exc):
                raise

    if latest_date is None:
        raise ValueError(
            "no available IMERG %s observation in the %d candidate day(s) T-%d..T-%d "
            "before %s (%s probe reach). IMERG %s publishes one granule per "
            "observation day with ~%s latency, so a contiguous run of 404s means the "
            "product is not serving these days for this account/AOI, not that a "
            "single granule is missing."
            % (run_type, probe_reach, first_offset, last_offset, _day(target_date),
               probe_mode, run_type, _latency_label(run_type))
        )

    daily = [latest_value]
    for k in range(1, window_days):
        deadline.check(
            "fetching the IMERG %s antecedent window" % run_type,
            fraction=IMERG_PHASE_BUDGET_FRACTION,
        )
        day = latest_date - timedelta(days=k)
        daily.append(float(imerg_fetcher(session, day, bounds, run_type)))

    _validate_series(daily, window_days, "IMERG")
    return {
        "observation_date": latest_date,
        "daily": daily,
        "probe_days_walked": probe_days_walked,
        "probe_first_offset": first_offset,
        "probe_reach": probe_reach,
        "probe_mode": probe_mode,
        "session_reused_probe_value": True,
    }

# ---------------------------------------------------------------------------
# Fallback path: Open-Meteo ERA5 archive, ALWAYS labelled as a fallback
# ---------------------------------------------------------------------------
def _default_fallback_fetcher(bounds, end_date, days):
    """
    Daily precipitation totals for the `days` calendar days ending on `end_date`,
    returned NEWEST FIRST so index k is day T-(k+1) -- the same convention as the
    IMERG series.

    Deliberately a separate call from weather_ingestion.fetch_open_meteo_forecast,
    which is forecast-only and cannot produce an antecedent series; that function
    is left untouched. Raises if the archive does not return every requested day
    with a finite value: a hole is a refusal, never a zero.
    """
    import requests

    lat, lon = _centroid(bounds)
    start_date = end_date - timedelta(days=days - 1)
    params = {
        "latitude": round(lat, 4),
        "longitude": round(lon, 4),
        "start_date": _day(start_date),
        "end_date": _day(end_date),
        "daily": "precipitation_sum",
        "timezone": "UTC",
    }
    response = requests.get(
        OPEN_METEO_ARCHIVE_URL,
        params=params,
        timeout=OPEN_METEO_TIMEOUT_SECONDS,
        stream=True,
    )
    try:
        response.raise_for_status()

        # Streamed with a hard ceiling, so a hostile/mis-routed response can never
        # be materialised in full: the same discipline the artifact store uses.
        chunks = []
        total = 0
        for chunk in response.iter_content(chunk_size=FALLBACK_CHUNK_BYTES):
            if not chunk:
                continue
            total += len(chunk)
            if total > MAX_FALLBACK_BODY_BYTES:
                raise ValueError(
                    "Open-Meteo archive body exceeds the %d byte ceiling; abandoning "
                    "the read" % MAX_FALLBACK_BODY_BYTES
                )
            chunks.append(chunk)
    finally:
        response.close()
    payload = json.loads(b"".join(chunks).decode("utf-8"))

    daily_block = payload.get("daily") or {}
    times = list(daily_block.get("time") or [])
    totals = list(daily_block.get("precipitation_sum") or [])
    if len(times) != len(totals):
        raise ValueError("Open-Meteo archive returned mismatched time/value arrays")

    by_day = {}
    for stamp, value in zip(times, totals):
        by_day[str(stamp)] = value

    series = []
    for k in range(days):
        wanted = _day(end_date - timedelta(days=k))
        if wanted not in by_day:
            raise ValueError(
                "Open-Meteo archive did not return %s; refusing to fill the gap" % wanted
            )
        number = _finite(by_day[wanted])
        if number is None:
            raise ValueError(
                "Open-Meteo archive value for %s is null/non-finite; refusing to "
                "substitute a value" % wanted
            )
        series.append(number)
    return series


def _acquire_fallback_window(bounds, target_date, window_days, fallback_fetcher, deadline):
    """
    The fallback window ends FALLBACK_LATENCY_DAYS before the requested date, so
    it is strictly antecedent exactly like the IMERG window (no event-day leakage).
    """
    deadline.check("fetching the Open-Meteo ERA5 fallback window")
    end_date = target_date - timedelta(days=FALLBACK_LATENCY_DAYS)
    daily = [float(v) for v in fallback_fetcher(bounds, end_date, window_days)]
    _validate_series(daily, window_days, "Open-Meteo ERA5 archive")
    return {"observation_date": end_date, "daily": daily}

# ---------------------------------------------------------------------------
# Record assembly
# ---------------------------------------------------------------------------
REAL_NOTE = (
    "Antecedent-only (T-1..T-14, event day excluded). One AOI-mean NASA GPM IMERG "
    "series applied uniformly to every cell."
)

FALLBACK_NOTE = (
    "FALLBACK SOURCE. NASA GPM IMERG was unavailable, so this antecedent series "
    "comes from the Open-Meteo ERA5 archive (T-1..T-14, event day excluded). It is "
    "NOT an official live satellite observation, and any probability derived from "
    "it is decision-support only, not a calibrated probability."
)

FALLBACK_CAVEATS = (
    "data_quality_status=FALLBACK: Open-Meteo ERA5 archive, not NASA GPM IMERG.",
    "ERA5 reanalysis is a different product from IMERG; the two are not interchangeable.",
    "Probabilities computed from fallback rainfall are not calibrated and must not "
    "be presented as official live observations.",
)


def _base_record(state_name, bounds, target_date, run_type, window_days, fetched_at, attempts):
    return {
        "state": state_name,
        "units": UNITS,
        "run_type": run_type,
        "coverage": {
            "state": state_name,
            "aoi": dict(bounds),
            "aoi_uniform": True,
            "window_days": int(window_days),
            "window_semantics": "T-1..T-%d (antecedent only)" % int(window_days),
        },
        "timestamp": {
            "fetched_at_utc": _iso(fetched_at),
            "requested_date": _day(target_date),
            "rainfall_observation_date": None,
        },
        "attempts": list(attempts),
    }


def _observed_record(state_name, bounds, target_date, run_type, window_days,
                     fetched_at, attempts, acquired, source, source_kind, quality, ttl):
    record = _base_record(
        state_name, bounds, target_date, run_type, window_days, fetched_at, attempts
    )
    observation_date = acquired["observation_date"]
    daily = acquired["daily"]
    lag_days = int((target_date.date() - observation_date.date()).days)

    record["timestamp"]["rainfall_observation_date"] = _day(observation_date)
    record["source"] = source
    record["source_kind"] = source_kind
    record["is_fallback"] = source_kind == SOURCE_KIND_FALLBACK
    record["data_quality_status"] = quality
    record["usable"] = True
    record["daily_series_mm"] = [round(float(v), 4) for v in daily]
    record["features"] = derive_rainfall_features(daily, window_days)
    record["freshness"] = {
        "cache_hit": False,
        "age_seconds": 0.0,
        "ttl_seconds": round(float(ttl), 3),
        "expires_in_seconds": round(float(ttl), 3),
        "observation_lag_days": lag_days,
        "probe_days_walked": acquired.get("probe_days_walked"),
        "probe_first_offset": acquired.get("probe_first_offset"),
        "probe_reach": acquired.get("probe_reach"),
        "probe_mode": acquired.get("probe_mode"),
    }
    if source_kind == SOURCE_KIND_FALLBACK:
        record["note"] = FALLBACK_NOTE
        record["caveats"] = list(FALLBACK_CAVEATS)
    else:
        record["note"] = REAL_NOTE
        record["caveats"] = []
    return record


def _unavailable_record(state_name, bounds, target_date, run_type, window_days,
                        fetched_at, attempts, ttl):
    record = _base_record(
        state_name, bounds, target_date, run_type, window_days, fetched_at, attempts
    )
    reasons = ["%s: %s" % (a["source_kind"], a["reason"]) for a in attempts if a.get("reason")]
    record["source"] = None
    record["source_kind"] = None
    record["is_fallback"] = False
    record["data_quality_status"] = QUALITY_UNAVAILABLE
    record["usable"] = False
    # No numbers whatsoever: an unavailable window carries no series and no
    # features, so nothing downstream can mistake it for a measurement.
    record["daily_series_mm"] = None
    record["features"] = None
    record["unavailable_reason"] = "; ".join(reasons) or "rainfall could not be obtained"
    record["note"] = (
        "No rainfall was obtained from IMERG or from the Open-Meteo ERA5 fallback. "
        "No values are reported; nothing was substituted, zero-filled or imputed."
    )
    record["caveats"] = []
    record["freshness"] = {
        "cache_hit": False,
        "age_seconds": 0.0,
        "ttl_seconds": round(float(ttl), 3),
        "expires_in_seconds": round(float(ttl), 3),
        "observation_lag_days": None,
        "probe_days_walked": None,
        "probe_first_offset": None,
        "probe_reach": None,
        "probe_mode": None,
    }
    return record


def _with_freshness(record, age_seconds, ttl_seconds, cache_hit):
    """A cached read must never look like a fresh one."""
    clone = dict(record)
    freshness = dict(record.get("freshness") or {})
    freshness["cache_hit"] = bool(cache_hit)
    freshness["age_seconds"] = round(float(age_seconds), 3)
    freshness["ttl_seconds"] = round(float(ttl_seconds), 3)
    freshness["expires_in_seconds"] = round(max(0.0, float(ttl_seconds) - float(age_seconds)), 3)
    clone["freshness"] = freshness
    return clone

# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def get_state_rainfall(state_name, target_date=None, run_type=DEFAULT_RUN_TYPE,
                       window_days=RAINFALL_WINDOW_DAYS, bounds=None,
                       imerg_fetcher=None, fallback_fetcher=None,
                       session_factory=None, clock=None, cache=None,
                       use_cache=True):
    """
    One cached, provenance-stamped antecedent rainfall read for a state AOI.

    IMERG is attempted first. Only if it fails is the Open-Meteo ERA5 archive
    tried, and that result is stamped FALLBACK. If both fail the record is
    UNAVAILABLE and carries no numbers.

    Never raises for a data problem -- the outcome is always a record whose
    data_quality_status says exactly what happened. ValueError is still raised for
    a caller error (unknown state, bad run_type, bad target_date).
    """
    clock = clock or _utcnow
    active_cache = cache if cache is not None else _CACHE

    run_type = normalise_run_type(run_type)
    window_days = int(window_days)
    if window_days < 3:
        raise ValueError("window_days must be at least 3, got %r" % (window_days,))
    resolved_bounds = resolve_bounds(state_name, bounds)
    target_date = _as_datetime(target_date) if target_date is not None else clock()

    key = cache_key(state_name, resolved_bounds, target_date, run_type, window_days)
    now = clock()

    if use_cache:
        hit = active_cache.get(key, now)
        if hit is not None:
            _stored_at, ttl, age, record = hit
            return _with_freshness(record, age, ttl, cache_hit=True)

    deadline = _Deadline(clock, deadline_seconds())
    attempts = []

    imerg_fetcher = imerg_fetcher or _default_imerg_fetcher
    session_factory = session_factory or _default_session_factory

    try:
        acquired = _acquire_imerg_window(
            resolved_bounds, target_date, run_type, window_days,
            imerg_fetcher, session_factory, deadline, now=now,
        )
        attempts.append({"source_kind": SOURCE_KIND_IMERG, "status": "OK", "reason": None})
        ttl = cache_ttl_seconds()
        record = _observed_record(
            state_name, resolved_bounds, target_date, run_type, window_days,
            now, attempts, acquired,
            source="IMERG_%s" % run_type, source_kind=SOURCE_KIND_IMERG,
            quality=QUALITY_REAL, ttl=ttl,
        )
        if use_cache:
            active_cache.put(key, now, ttl, record)
        return record
    except Exception as imerg_error:
        attempts.append({
            "source_kind": SOURCE_KIND_IMERG,
            "status": "FAILED",
            "reason": "%s: %s" % (type(imerg_error).__name__, imerg_error),
        })

    if not fallback_enabled():
        attempts.append({
            "source_kind": SOURCE_KIND_FALLBACK,
            "status": "DISABLED",
            "reason": "the Open-Meteo fallback is disabled via %s" % ENV_FALLBACK_ENABLED,
        })
    else:
        try:
            acquired = _acquire_fallback_window(
                resolved_bounds, target_date, window_days,
                fallback_fetcher or _default_fallback_fetcher, deadline,
            )
            attempts.append({
                "source_kind": SOURCE_KIND_FALLBACK, "status": "OK", "reason": None,
            })
            ttl = cache_ttl_seconds()
            record = _observed_record(
                state_name, resolved_bounds, target_date, run_type, window_days,
                now, attempts, acquired,
                source=FALLBACK_SOURCE_LABEL, source_kind=SOURCE_KIND_FALLBACK,
                quality=QUALITY_FALLBACK, ttl=ttl,
            )
            if use_cache:
                active_cache.put(key, now, ttl, record)
            return record
        except Exception as fallback_error:
            attempts.append({
                "source_kind": SOURCE_KIND_FALLBACK,
                "status": "FAILED",
                "reason": "%s: %s" % (type(fallback_error).__name__, fallback_error),
            })

    # Both sources refused. Cache the REFUSAL (briefly) so a credentials outage
    # cannot turn every request into another 15-44 request fan-out. What is cached
    # is an explicit unavailable record -- never a zero.
    ttl = negative_cache_ttl_seconds()
    record = _unavailable_record(
        state_name, resolved_bounds, target_date, run_type, window_days,
        now, attempts, ttl,
    )
    if use_cache:
        active_cache.put(key, now, ttl, record)
    return record

# ---------------------------------------------------------------------------
# Adapter for the prediction path
# ---------------------------------------------------------------------------
class RainfallUnavailable(Exception):
    """
    Raised by to_provider_payload when a record carries no usable rainfall. The
    prediction layer maps this onto its own PredictionUnavailable -> HTTP 503, so
    no caller ever receives a payload with invented numbers.
    """

    def __init__(self, reason, details=None):
        super().__init__(reason)
        self.reason = reason
        self.details = details or {}


def to_provider_payload(record):
    """
    Re-shape a record into the dict the prediction services already consume,
    ADDITIVELY: every key sikkim_prediction._default_rainfall_provider returned
    today is present and unchanged, plus the new provenance/freshness keys.
    """
    if not record.get("usable") or not record.get("features"):
        raise RainfallUnavailable(
            record.get("unavailable_reason") or "rainfall is unavailable",
            details={"attempts": record.get("attempts")},
        )
    timestamp = record.get("timestamp") or {}
    return {
        # --- keys the existing prediction path already relies on ---
        "source": record.get("source"),
        "run_type": record.get("run_type"),
        "aoi_uniform": (record.get("coverage") or {}).get("aoi_uniform", True),
        "window_days": (record.get("coverage") or {}).get("window_days"),
        "requested_date": timestamp.get("requested_date"),
        "rainfall_observation_date": timestamp.get("rainfall_observation_date"),
        "daily_series_mm": record.get("daily_series_mm"),
        "features": dict(record["features"]),
        # --- additive provenance the response was previously missing ---
        "source_kind": record.get("source_kind"),
        "is_fallback": record.get("is_fallback"),
        "data_quality_status": record.get("data_quality_status"),
        "units": record.get("units"),
        "fetched_at_utc": timestamp.get("fetched_at_utc"),
        "freshness": dict(record.get("freshness") or {}),
        "coverage": dict(record.get("coverage") or {}),
        "caveats": list(record.get("caveats") or []),
        "note": record.get("note"),
    }
