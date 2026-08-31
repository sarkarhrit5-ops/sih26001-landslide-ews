"""
LIVE RAINFALL MONITOR -- the LATEST AVAILABLE sub-daily rainfall per pilot AOI.

This module is deliberately SEPARATE from app.services.rainfall_service. That
module serves the model: a strictly antecedent T-1..T-14 DAILY window whose five
features the pilot models were fitted on. This module serves the operator: the
newest half-hourly observation NASA GPM IMERG has actually published, with its
age stated. The two never mix.

  * Nothing here is reachable from derive_rainfall_features(). This module does
    not import rainfall_service at all, so a live value cannot become a model
    feature by accident. The T-1..T-14 semantics are untouched.
  * The word "current" is not used for a value. IMERG Early publishes with ~4 h
    latency, so the newest observation is never now; the field is
    latest_available_rainfall_mm and every record carries age_minutes,
    freshness_label and is_stale.
  * FRESHNESS IS SOURCE-INDEPENDENT. Every record -- IMERG Early, IMERG Late,
    Open-Meteo FALLBACK -- carries the same `freshness` block, with age_seconds
    and age_minutes measured as fetched_at_utc - observed_at_utc and
    freshness_label / is_stale derived from that measured age by one shared
    rule. A FALLBACK observation is therefore exactly as auditable as an IMERG
    one; no source gets a null freshness.
  * SOURCE ORDER: IMERG Early half-hourly -> IMERG Late half-hourly ->
    Open-Meteo (labelled FALLBACK). If all three fail the record is UNAVAILABLE
    and carries NO numbers: no zero fill, no imputation, no partial window
    presented as a whole one.
  * A 3 h or 6 h accumulation is reported ONLY when every granule in that window
    was retrieved. A short run yields null plus the real reason.
  * ONE AOI subset per granule -- never one request per grid cell. Responses are
    streamed under a hard byte ceiling into a tempfile, parsed, and deleted; no
    NetCDF payload or raster is retained.

CONFIRMED PRODUCT FACTS (host .dds/.das/.dmr probe of GPM_3IMERGHHE.07):
  variable   precipitation
  dimensions time, lon, lat   (same order as the daily product)
  units      mm/hr  -> a 30-minute interval total is rate * 0.5
  grid       0.1 deg x 0.1 deg, lon 3600, lat 1800
  granules   48/day, directory YYYY/DDD, filename carries minutes-of-day

Authentication is the EXISTING guard: weather_ingestion.get_earthdata_session().
No second NASA client, no credential handling of its own. Everything external
(granule_fetcher, session_factory, fallback_fetcher, clock, cache) is injectable,
so the whole module is testable offline with no credentials and no network.

Import-time work: none. No network call, no credential read, no cache warm.
"""

import json
import os
import tempfile
import threading
from collections import OrderedDict
from datetime import datetime, timedelta

# ---------------------------------------------------------------------------
# Confirmed product surface
# ---------------------------------------------------------------------------
OPENDAP_ROOT = "https://gpm1.gesdisc.eosdis.nasa.gov/opendap/GPM_L3"

# The variable name and index order below are the ones the granule's own .dds
# declares. They are NOT guesses: requesting the wrong name returns HTTP 400 for
# an existing granule, which is exactly how the V06 'precipitationCal' name was
# caught on the daily product.
VARIABLE_NAME = "precipitation"
VARIABLE_UNITS = "mm/hr"
PRODUCT_VERSION = "V07B"

GRANULE_MINUTES = 30
# mm/hr over a 30-minute interval -> millimetres. If a future product version
# reports totals instead of a rate this factor is the single place to change, and
# the units assertion in the tests will fail first.
GRANULE_HOURS = GRANULE_MINUTES / 60.0

RUN_TYPE_EARLY = "Early"
RUN_TYPE_LATE = "Late"

SOURCE_KIND_EARLY = "IMERG_HHR_EARLY"
SOURCE_KIND_LATE = "IMERG_HHR_LATE"
SOURCE_KIND_FALLBACK = "OPEN_METEO_FALLBACK"

# Publication latency per run, in minutes. Used ONLY to pick the newest slot that
# could plausibly exist, never to declare a slot missing.
IMERG_PRODUCTS = {
    RUN_TYPE_EARLY: {
        "collection": "GPM_3IMERGHHE.07",
        "granule_prefix": "3B-HHR-E",
        "source_kind": SOURCE_KIND_EARLY,
        "label": "NASA GPM IMERG Early half-hourly (GPM_3IMERGHHE.07)",
        "latency_minutes": 240,
    },
    RUN_TYPE_LATE: {
        "collection": "GPM_3IMERGHHL.07",
        "granule_prefix": "3B-HHR-L",
        "source_kind": SOURCE_KIND_LATE,
        "label": "NASA GPM IMERG Late half-hourly (GPM_3IMERGHHL.07)",
        "latency_minutes": 840,
    },
}

# Attempt order. Early first because it is the freshest; Late second because a
# 14 h observation is still a real IMERG measurement.
IMERG_RUN_ORDER = (RUN_TYPE_EARLY, RUN_TYPE_LATE)

QUALITY_REAL = "REAL"
QUALITY_FALLBACK = "FALLBACK"
QUALITY_UNAVAILABLE = "UNAVAILABLE"

UNITS = "mm"

# Accumulation windows, in hours. 6 h is ON by default: at 30-minute granules it
# costs 12 subsets per state per TTL, which the cache amortises.
ACCUMULATION_WINDOWS_HOURS = (3, 6)

FRESHNESS_NEAR_REAL_TIME = "NEAR_REAL_TIME"
FRESHNESS_RECENT = "RECENT"
FRESHNESS_STALE = "STALE"

# Response bodies here are AOI subsets of a 0.1 deg grid -- a few hundred floats,
# a few kilobytes. The ceiling keeps that an enforced property rather than an
# assumption, so a mis-routed full-grid response is abandoned instead of read.
MAX_SUBSET_BODY_BYTES = 2 * 1024 * 1024
SUBSET_CHUNK_BYTES = 64 * 1024
SUBSET_TIMEOUT_SECONDS = 45

OPEN_METEO_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
OPEN_METEO_TIMEOUT_SECONDS = 15
MAX_FALLBACK_BODY_BYTES = 512 * 1024
FALLBACK_CHUNK_BYTES = 16 * 1024
FALLBACK_INTERVAL_MINUTES = 60
FALLBACK_SOURCE_LABEL = "Open-Meteo hourly precipitation (FALLBACK)"

# ---------------------------------------------------------------------------
# Environment surface (read at CALL time, never at import time). Every name is
# distinct from the antecedent service's SIH_RAINFALL_* knobs, so tuning the live
# monitor cannot change what the models are fed.
# ---------------------------------------------------------------------------
ENV_CACHE_TTL = "SIH_LIVE_RAINFALL_CACHE_TTL_SECONDS"
ENV_NEGATIVE_TTL = "SIH_LIVE_RAINFALL_NEGATIVE_TTL_SECONDS"
ENV_DEADLINE = "SIH_LIVE_RAINFALL_DEADLINE_SECONDS"
ENV_CACHE_MAX_ENTRIES = "SIH_LIVE_RAINFALL_CACHE_MAX_ENTRIES"
ENV_PROBE_GRANULES = "SIH_LIVE_RAINFALL_PROBE_GRANULES"
ENV_WINDOW_GRANULES = "SIH_LIVE_RAINFALL_WINDOW_GRANULES"
ENV_FALLBACK_ENABLED = "SIH_LIVE_RAINFALL_FALLBACK"
ENV_LATE_ENABLED = "SIH_LIVE_RAINFALL_LATE"
ENV_STALE_MINUTES = "SIH_LIVE_RAINFALL_STALE_MINUTES"
ENV_NEAR_REAL_TIME_MINUTES = "SIH_LIVE_RAINFALL_NEAR_REAL_TIME_MINUTES"

DEFAULT_CACHE_TTL_SECONDS = 900.0        # half the 30-minute product cadence
DEFAULT_NEGATIVE_TTL_SECONDS = 120.0     # a refusal must not stick around
DEFAULT_DEADLINE_SECONDS = 90.0
DEFAULT_CACHE_MAX_ENTRIES = 16
DEFAULT_PROBE_GRANULES = 8               # 4 h of 30-minute steps
DEFAULT_WINDOW_GRANULES = 12             # 6 h -> covers both accumulations
DEFAULT_STALE_MINUTES = 360
DEFAULT_NEAR_REAL_TIME_MINUTES = 90

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


def deadline_seconds():
    return _env_number(ENV_DEADLINE, DEFAULT_DEADLINE_SECONDS, minimum=1.0)


def cache_max_entries():
    return int(_env_number(ENV_CACHE_MAX_ENTRIES, DEFAULT_CACHE_MAX_ENTRIES, minimum=1.0))


def probe_granules():
    """How many 30-minute slots the latest-granule walk may try."""
    return int(_env_number(ENV_PROBE_GRANULES, DEFAULT_PROBE_GRANULES, minimum=1.0))


def window_granules():
    """
    How many contiguous granules to retrieve from the anchor backwards. The
    default covers the widest configured accumulation window; lowering it below
    12 makes accum_6h_mm report null-with-reason rather than a short sum.
    """
    return int(_env_number(ENV_WINDOW_GRANULES, DEFAULT_WINDOW_GRANULES, minimum=1.0))


def fallback_enabled():
    return _env_flag(ENV_FALLBACK_ENABLED, True)


def late_enabled():
    return _env_flag(ENV_LATE_ENABLED, True)


def stale_after_minutes():
    return _env_number(ENV_STALE_MINUTES, DEFAULT_STALE_MINUTES, minimum=1.0)


def near_real_time_minutes():
    return _env_number(
        ENV_NEAR_REAL_TIME_MINUTES, DEFAULT_NEAR_REAL_TIME_MINUTES, minimum=1.0
    )


def granules_required(hours):
    """Granules needed to cover `hours` completely at the product's cadence."""
    return int(round(float(hours) * 60.0 / GRANULE_MINUTES))

# ---------------------------------------------------------------------------
# Time + granule identity
# ---------------------------------------------------------------------------
def _utcnow():
    return datetime.utcnow()


def _iso(value):
    return value.strftime("%Y-%m-%dT%H:%M:%SZ")


def floor_to_granule(value):
    """The start of the 30-minute slot containing `value` (seconds dropped)."""
    minute = (value.minute // GRANULE_MINUTES) * GRANULE_MINUTES
    return value.replace(minute=minute, second=0, microsecond=0)


def minutes_of_day(slot_start):
    """The filename's 4-digit sequence field: minutes since 00:00 UTC."""
    return slot_start.hour * 60 + slot_start.minute


def granule_basename(run_type, slot_start):
    """
    The confirmed half-hourly basename, e.g.

        3B-HHR-E.MS.MRG.3IMERG.20250918-S090000-E092959.0540.V07B.HDF5

    The end stamp is the slot's LAST second (S090000-E092959), not the next
    slot's start, and the sequence field is minutes-of-day zero-padded to 4.
    """
    product = IMERG_PRODUCTS[run_type]
    slot_end = slot_start + timedelta(minutes=GRANULE_MINUTES) - timedelta(seconds=1)
    return "%s.MS.MRG.3IMERG.%s-S%s-E%s.%04d.%s.HDF5" % (
        product["granule_prefix"],
        slot_start.strftime("%Y%m%d"),
        slot_start.strftime("%H%M%S"),
        slot_end.strftime("%H%M%S"),
        minutes_of_day(slot_start),
        PRODUCT_VERSION,
    )


def granule_directory_url(run_type, slot_start):
    """Collection directory for the slot: <root>/<collection>/YYYY/DDD."""
    product = IMERG_PRODUCTS[run_type]
    return "%s/%s/%s/%s" % (
        OPENDAP_ROOT,
        product["collection"],
        slot_start.strftime("%Y"),
        slot_start.strftime("%j"),
    )

def granule_subset_url(run_type, slot_start, bounds):
    """
    The OPeNDAP request for ONE granule, subset to the AOI on the server.

    Index order is [time][lon][lat], as the granule's .dds declares -- the same
    order as the daily product. The grid arithmetic is reused verbatim from
    weather_ingestion.get_imerg_indices, which is product-independent: every
    IMERG L3 product shares the 0.1 deg lon 3600 / lat 1800 grid.
    """
    from app.services import weather_ingestion

    lat_min, lat_max, lon_min, lon_max = weather_ingestion.get_imerg_indices(bounds)
    query = "?%s[0:0][%d:%d][%d:%d]" % (
        VARIABLE_NAME, lon_min, lon_max, lat_min, lat_max
    )
    return "%s/%s.nc4%s" % (
        granule_directory_url(run_type, slot_start),
        granule_basename(run_type, slot_start),
        query,
    )


class GranuleUnavailable(Exception):
    """
    A granule that is not published (HTTP 404). This is the ONLY condition that
    justifies walking further back; anything else -- auth rejection, a no-data
    subset, a transport failure -- is a real error and propagates.
    """


def _default_granule_fetcher(session, slot_start, bounds, run_type):
    """
    Mean AOI precipitation RATE (mm/hr) for one half-hourly granule.

    Memory discipline, identical to the artifact store's: the body is streamed in
    SUBSET_CHUNK_BYTES pieces under a hard ceiling into a tempfile, parsed, and
    the tempfile deleted in a finally. There is no response.content, no
    response.text and no unbounded read(), and nothing survives the call except
    one float.

    An all-NaN / all-fill subset raises rather than returning 0.0: the check is
    weather_ingestion._mean_valid_precipitation, reused unchanged so the live
    monitor and the model path agree on what "no data" means.
    """
    import requests
    import xarray as xr

    from app.services import weather_ingestion

    url = granule_subset_url(run_type, slot_start, bounds)
    response = session.get(url, timeout=SUBSET_TIMEOUT_SECONDS, stream=True)
    try:
        if response.status_code == 404:
            raise GranuleUnavailable(
                "IMERG %s granule for %s is not published (HTTP 404)"
                % (run_type, _iso(slot_start))
            )
        if response.status_code in (401, 403):
            raise PermissionError(
                "EARTHDATA AUTHENTICATION REJECTED (HTTP %d) for the IMERG %s "
                "half-hourly granule at %s"
                % (response.status_code, run_type, _iso(slot_start))
            )
        response.raise_for_status()

        handle = tempfile.NamedTemporaryFile(suffix=".nc4", delete=False)
        try:
            total = 0
            for chunk in response.iter_content(chunk_size=SUBSET_CHUNK_BYTES):
                if not chunk:
                    continue
                total += len(chunk)
                if total > MAX_SUBSET_BODY_BYTES:
                    raise ValueError(
                        "IMERG subset body for %s exceeds the %d byte ceiling; "
                        "abandoning the read rather than materialising a full grid"
                        % (_iso(slot_start), MAX_SUBSET_BODY_BYTES)
                    )
                handle.write(chunk)
            handle.flush()
            temp_path = handle.name
        finally:
            handle.close()
    finally:
        response.close()

    try:
        with xr.open_dataset(temp_path, engine="h5netcdf") as dataset:
            values = dataset[VARIABLE_NAME].values
    finally:
        os.remove(temp_path)

    return float(weather_ingestion._mean_valid_precipitation(values))


def _default_session_factory():
    """The EXISTING Earthdata guard, used as-is. No credential handling here."""
    from app.services import weather_ingestion

    return weather_ingestion.get_earthdata_session()

def _default_fallback_fetcher(bounds, now):
    """
    Open-Meteo hourly precipitation at the AOI centroid -- the LABELLED fallback.

    This is ONE request for the AOI centre, never one per grid cell, and the body
    is streamed under MAX_FALLBACK_BODY_BYTES exactly like the IMERG path. It
    returns hourly totals in mm (Open-Meteo's native unit for `precipitation`),
    so no rate conversion applies; the caller marks the record FALLBACK and sets
    interval_minutes to 60.

    Returns a list of (slot_start_utc, mm) pairs ordered oldest -> newest, past
    hours only. An empty list means "no usable value", which the caller turns
    into UNAVAILABLE rather than a zero.
    """
    import requests

    lat = (float(bounds["min_lat"]) + float(bounds["max_lat"])) / 2.0
    lon = (float(bounds["min_lon"]) + float(bounds["max_lon"])) / 2.0
    params = {
        "latitude": round(lat, 4),
        "longitude": round(lon, 4),
        "hourly": "precipitation",
        "past_days": 1,
        "forecast_days": 1,
        "timezone": "UTC",
    }

    response = requests.get(
        OPEN_METEO_FORECAST_URL,
        params=params,
        timeout=OPEN_METEO_TIMEOUT_SECONDS,
        stream=True,
    )
    try:
        response.raise_for_status()
        chunks = []
        total = 0
        for chunk in response.iter_content(chunk_size=FALLBACK_CHUNK_BYTES):
            if not chunk:
                continue
            total += len(chunk)
            if total > MAX_FALLBACK_BODY_BYTES:
                raise ValueError(
                    "Open-Meteo fallback body exceeds the %d byte ceiling"
                    % MAX_FALLBACK_BODY_BYTES
                )
            chunks.append(chunk)
        payload = json.loads(b"".join(chunks).decode("utf-8"))
    finally:
        response.close()

    return _parse_open_meteo_hours(payload, now)

def _parse_open_meteo_hours(payload, now):
    """
    (slot_start, mm) pairs for PAST hours only, oldest -> newest.

    Non-finite / null hours are dropped rather than read as 0.0, and future hours
    are dropped entirely: this monitor reports observations, never forecasts. A
    payload whose usable past hours are all missing yields [] and therefore
    UNAVAILABLE, not a fabricated dry hour.
    """
    hourly = (payload or {}).get("hourly") or {}
    times = hourly.get("time") or []
    values = hourly.get("precipitation") or []

    horizon = now.replace(minute=0, second=0, microsecond=0)
    pairs = []
    for index, stamp in enumerate(times):
        if index >= len(values):
            break
        raw = values[index]
        if raw is None:
            continue
        try:
            slot = datetime.strptime(str(stamp)[:16], "%Y-%m-%dT%H:%M")
            amount = float(raw)
        except (TypeError, ValueError):
            continue
        if amount != amount:  # NaN
            continue
        if slot > horizon:
            continue
        pairs.append((slot, max(0.0, amount)))

    pairs.sort(key=lambda item: item[0])
    return pairs


def _copy_record(record):
    """
    An independent copy of a live-rainfall record.

    The nested members are copied explicitly rather than by dict(record): the
    cached record must never be reachable from what a caller holds, or annotating
    the returned copy would silently rewrite history in the cache (requirement 4).
    """
    copy = dict(record)
    for name in ("freshness", "aoi_bounds"):
        nested = copy.get(name)
        if isinstance(nested, dict):
            copy[name] = dict(nested)
    attempts = copy.get("attempts")
    if isinstance(attempts, list):
        copy["attempts"] = [
            dict(entry) if isinstance(entry, dict) else entry for entry in attempts
        ]
    return copy


def _mark_cache_hit(record, hit):
    """
    Stamp a record with its provenance-of-delivery, top level and inside the
    freshness block, so a consumer reading only `freshness` can still tell a
    fresh acquisition from a replay. Mutates the COPY it is handed, never the
    cached original.
    """
    record["cache_hit"] = bool(hit)
    record["served_from_cache"] = bool(hit)
    freshness = record.get("freshness")
    if isinstance(freshness, dict):
        freshness["cache_hit"] = bool(hit)
    return record


class LiveRainfallCache:
    """
    Bounded, TTL'd, thread-safe cache for live-rainfall records.

    Deliberately a SEPARATE object from rainfall_service.RainfallCache: the two
    hold different quantities with different meanings, and requirement 8 is that
    the live result never share storage with the antecedent one. Refusals get the
    short negative TTL so an outage does not persist past its cause.

    Every value crosses the boundary through _copy_record in BOTH directions, so
    a caller can annotate what it was handed (cache_hit) without reaching into
    the stored record. A shallow dict() copy would not be enough: `freshness`,
    `aoi_bounds` and `attempts` are nested, and a shallow copy would leave the
    caller holding the stored objects.
    """

    def __init__(self, max_entries=None):
        self._entries = OrderedDict()
        self._lock = threading.Lock()
        self._max_entries = max_entries

    def _limit(self):
        return int(self._max_entries or cache_max_entries())

    def get(self, key, monotonic):
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                return None
            expires_at, record = entry
            if monotonic >= expires_at:
                del self._entries[key]
                return None
            self._entries.move_to_end(key)
            return _copy_record(record)

    def put(self, key, record, monotonic):
        ttl = (
            negative_cache_ttl_seconds()
            if record.get("data_quality_status") == QUALITY_UNAVAILABLE
            else cache_ttl_seconds()
        )
        with self._lock:
            self._entries[key] = (monotonic + ttl, _copy_record(record))
            self._entries.move_to_end(key)
            while len(self._entries) > self._limit():
                self._entries.popitem(last=False)

    def clear(self):
        with self._lock:
            self._entries.clear()

    def __len__(self):
        with self._lock:
            return len(self._entries)


_CACHE = LiveRainfallCache()


def clear_cache():
    """Drop every cached live-rainfall record (tests, and operator reset)."""
    _CACHE.clear()


def cache_key(state_name, bounds, run_type):
    """
    Identity of a live-rainfall lookup. The bounds are part of the key so an
    explicit bbox override can never be served a state's cached record.
    """
    return (
        str(state_name or ""),
        str(run_type or ""),
        tuple(round(float(bounds[key]), 4) for key in ("min_lat", "max_lat", "min_lon", "max_lon")),
    )

def resolve_bounds(state_name, bounds=None):
    """
    The AOI to subset. An explicit bbox wins (tests, ad-hoc operator queries);
    otherwise the canonical pilot AOI is read from config_states -- the only
    sanctioned source of pilot geometry. Restating numbers inline is forbidden.
    """
    if bounds is not None:
        return {key: float(bounds[key]) for key in ("min_lat", "max_lat", "min_lon", "max_lon")}

    from app.core import config_states

    return config_states.get_pilot_aoi_bounds(state_name)


def supported_states():
    """The four pilot states, straight from the canonical registry."""
    from app.core import config_states

    return sorted(config_states.PILOT_AOIS)


class _Deadline:
    """
    Wall-clock budget for one live-rainfall call. Local to this module on purpose:
    importing rainfall_service._Deadline would couple the live monitor to the
    model path that requirement 13 keeps it away from.
    """

    def __init__(self, budget_seconds, monotonic):
        import time as _time

        self._time = _time
        self._budget = float(budget_seconds)
        self._started = monotonic if monotonic is not None else _time.monotonic()

    def elapsed(self):
        return self._time.monotonic() - self._started

    def remaining(self):
        return self._budget - self.elapsed()

    def expired(self):
        return self.remaining() <= 0.0


def freshness_label_for(age_minutes):
    """
    NEAR_REAL_TIME / RECENT / STALE from the observation's measured age.

    The stale threshold is tested FIRST so the label can never contradict
    is_stale. If an operator sets SIH_LIVE_RAINFALL_STALE_MINUTES below
    SIH_LIVE_RAINFALL_NEAR_REAL_TIME_MINUTES, a stale observation must still
    read STALE rather than being flattered by the near-real-time band.
    """
    if age_minutes is None:
        return None
    if age_minutes > stale_after_minutes():
        return FRESHNESS_STALE
    if age_minutes <= near_real_time_minutes():
        return FRESHNESS_NEAR_REAL_TIME
    return FRESHNESS_RECENT

def accumulate_window(interval_mm_newest_first, hours, interval_minutes):
    """
    Total rainfall over the most recent `hours`, or None WITH A REASON.

    `interval_mm_newest_first` is a contiguous run of interval totals ending at
    the anchor slot, newest first. A window is reported ONLY when every interval
    it needs is present; a short run returns (None, reason) rather than a sum of
    whatever happened to be retrieved. This is requirement 6: a partial window is
    never presented as a complete one.
    """
    needed = int(round(float(hours) * 60.0 / float(interval_minutes)))
    available = len(interval_mm_newest_first)
    if needed <= 0:
        return None, "window of %s h is not expressible at %s-minute intervals" % (
            hours, interval_minutes
        )
    if available < needed:
        return None, (
            "incomplete window: %d of %d contiguous %d-minute intervals were "
            "retrieved, so a %s h accumulation would be a partial sum"
            % (available, needed, interval_minutes, hours)
        )
    return round(float(sum(interval_mm_newest_first[:needed])), 4), None


def _accumulation_block(interval_mm_newest_first, interval_minutes):
    """The accum_<n>h_mm / accum_<n>h_unavailable_reason pairs for every window."""
    block = {}
    for hours in ACCUMULATION_WINDOWS_HOURS:
        total, reason = accumulate_window(
            interval_mm_newest_first, hours, interval_minutes
        )
        block["accum_%dh_mm" % hours] = total
        block["accum_%dh_unavailable_reason" % hours] = reason
    return block


def _empty_accumulation_block():
    """Windows for a record that carries no numbers at all."""
    block = {}
    for hours in ACCUMULATION_WINDOWS_HOURS:
        block["accum_%dh_mm" % hours] = None
        block["accum_%dh_unavailable_reason" % hours] = (
            "no observation was retrieved, so no accumulation exists"
        )
    return block

def measured_age_seconds(observed_at, fetched_at):
    """
    The MEASURED age of an observation: fetched_at - observed_at, in seconds.

    Nothing here is source-specific. IMERG half-hourly granules and the
    Open-Meteo FALLBACK hours are both timestamped observations, so both get
    their age from the same two timestamps rather than from a per-product
    latency constant. `None` only when there is no observation to age.
    """
    if observed_at is None or fetched_at is None:
        return None
    return round((fetched_at - observed_at).total_seconds(), 1)


def _freshness_block(observed_at, fetched_at):
    """
    The single freshness structure every record carries, whatever the source.

    It exists so a FALLBACK record is exactly as auditable as an IMERG one: the
    same keys, derived from the same measured age by the same rules. A record
    with no observation still gets the block, with the age-derived fields null
    -- an absent age is stated, never guessed at.
    """
    age_seconds = measured_age_seconds(observed_at, fetched_at)
    age_minutes = None if age_seconds is None else round(age_seconds / 60.0, 1)
    return {
        "observed_at_utc": _iso(observed_at) if observed_at is not None else None,
        "fetched_at_utc": _iso(fetched_at) if fetched_at is not None else None,
        "age_seconds": age_seconds,
        "age_minutes": age_minutes,
        "freshness_label": freshness_label_for(age_minutes),
        "is_stale": None if age_minutes is None else bool(
            age_minutes > stale_after_minutes()
        ),
        "staleness_threshold_minutes": stale_after_minutes(),
        "near_real_time_threshold_minutes": near_real_time_minutes(),
        "measured_from": "fetched_at_utc - observed_at_utc",
        "cache_hit": False,
    }


def _observed_record(
    state_name,
    bounds,
    observed_at,
    fetched_at,
    interval_mm,
    interval_minutes,
    interval_totals_newest_first,
    source,
    source_kind,
    data_quality_status,
    expected_latency_minutes,
    attempts,
    granules_used,
):
    """
    A record built from a REAL retrieval. The value field is deliberately named
    latest_available_rainfall_mm, never "current": IMERG Early publishes with
    hours of latency, so the newest observation is never now (requirement 5).

    Age and freshness come from _freshness_block, so an Open-Meteo FALLBACK
    record is aged by exactly the same measured rule as an IMERG one.
    """
    freshness = _freshness_block(observed_at, fetched_at)
    record = {
        "state": state_name,
        "aoi_bounds": dict(bounds),
        "latest_available_rainfall_mm": round(float(interval_mm), 4),
        "interval_minutes": int(interval_minutes),
        "units": UNITS,
        "observed_at_utc": _iso(observed_at),
        "fetched_at_utc": _iso(fetched_at),
        "age_seconds": freshness["age_seconds"],
        "age_minutes": freshness["age_minutes"],
        "latency_minutes": freshness["age_minutes"],
        "freshness_label": freshness["freshness_label"],
        "freshness": freshness,
        "is_stale": freshness["is_stale"],
        "staleness_threshold_minutes": stale_after_minutes(),
        "expected_product_latency_minutes": expected_latency_minutes,
        "source": source,
        "source_kind": source_kind,
        "data_quality_status": data_quality_status,
        "granules_used": int(granules_used),
        "attempts": list(attempts),
        "value_semantics": (
            "latest available observation for the AOI, NOT a nowcast and NOT the "
            "antecedent T-1..T-14 rainfall the prediction models consume"
        ),
    }
    record.update(_accumulation_block(interval_totals_newest_first, interval_minutes))
    return record

def _unavailable_record(state_name, bounds, fetched_at, attempts, reason):
    """
    A refusal. It carries NO numbers: no zero fill, no imputation, no last-known
    value dressed up as an observation. Every numeric field is null and the
    reason names what actually failed (requirement 12).
    """
    freshness = _freshness_block(None, fetched_at)
    record = {
        "state": state_name,
        "aoi_bounds": dict(bounds),
        "latest_available_rainfall_mm": None,
        "interval_minutes": None,
        "units": UNITS,
        "observed_at_utc": None,
        "fetched_at_utc": _iso(fetched_at),
        "age_seconds": None,
        "age_minutes": None,
        "latency_minutes": None,
        "freshness_label": None,
        "freshness": freshness,
        "is_stale": None,
        "staleness_threshold_minutes": stale_after_minutes(),
        "expected_product_latency_minutes": None,
        "source": None,
        "source_kind": None,
        "data_quality_status": QUALITY_UNAVAILABLE,
        "granules_used": 0,
        "attempts": list(attempts),
        "unavailable_reason": reason,
        "value_semantics": (
            "no observation could be retrieved; no value is reported rather than "
            "a fabricated zero"
        ),
    }
    record.update(_empty_accumulation_block())
    return record


def _record_attempt(attempts, source_kind, outcome, detail=None):
    """One line of the provenance trail: what was tried, and what came back."""
    entry = {"source_kind": source_kind, "outcome": outcome}
    if detail:
        entry["detail"] = str(detail)[:400]
    attempts.append(entry)
    return attempts

def _acquire_imerg_run(session, bounds, run_type, now, fetcher, deadline):
    """
    Retrieve the newest published granule for one IMERG run, then walk backwards
    contiguously to cover the widest accumulation window.

    Two distinct phases, and the distinction matters:

      PROBE   Start at floor_to_granule(now - publication latency) and step back
              at most probe_granules() slots until one granule exists. Only
              GranuleUnavailable (HTTP 404) advances the walk -- an auth failure
              or a no-data subset is a real error and propagates, because
              swallowing it would let the monitor report Late or Open-Meteo data
              while the real problem was a rejected token.

      WINDOW  From the anchor found above, continue backwards for
              window_granules() slots, STOPPING at the first gap. The anchor's
              own value is reused, never re-fetched.

    Returns (anchor_slot, [interval_mm newest-first]) or None if nothing existed.
    """
    product = IMERG_PRODUCTS[run_type]
    anchor_candidate = floor_to_granule(
        now - timedelta(minutes=product["latency_minutes"])
    )

    anchor = None
    anchor_rate = None
    for step in range(probe_granules()):
        if deadline.expired():
            break
        slot = anchor_candidate - timedelta(minutes=GRANULE_MINUTES * step)
        try:
            anchor_rate = fetcher(session, slot, bounds, run_type)
        except GranuleUnavailable:
            continue
        anchor = slot
        break

    if anchor is None:
        return None

    totals = [float(anchor_rate) * GRANULE_HOURS]
    for step in range(1, max(1, window_granules())):
        if deadline.expired():
            break
        slot = anchor - timedelta(minutes=GRANULE_MINUTES * step)
        try:
            rate = fetcher(session, slot, bounds, run_type)
        except GranuleUnavailable:
            break
        totals.append(float(rate) * GRANULE_HOURS)

    return anchor, totals

def get_latest_rainfall(
    state_name,
    bounds=None,
    now=None,
    granule_fetcher=None,
    session_factory=None,
    fallback_fetcher=None,
    clock=None,
    cache=None,
    use_cache=True,
    include_late=None,
    include_fallback=None,
):
    """
    The LATEST AVAILABLE sub-daily rainfall for a pilot AOI, with its age stated.

    Source order: IMERG Early HHR -> IMERG Late HHR -> Open-Meteo (FALLBACK) ->
    UNAVAILABLE. Every attempt is recorded in `attempts`, so a FALLBACK record
    always says what failed before it.

    This function is NOT a source of model features. It is unreachable from
    derive_rainfall_features(), it does not import rainfall_service, and its
    cache and environment namespace are separate (requirements 8 and 13).

    Every collaborator is injectable so the whole path is testable offline with
    no credentials and no network.
    """
    import time as _time

    resolved = resolve_bounds(state_name, bounds)
    fetch_clock = clock or _utcnow
    fetched_at = now or fetch_clock()
    monotonic = _time.monotonic()

    store = cache if cache is not None else _CACHE
    key = cache_key(state_name, resolved, "live")
    if use_cache and store is not None:
        cached = store.get(key, monotonic)
        if cached is not None:
            # store.get already handed back an independent deep-enough copy, so
            # stamping it cannot rewrite the cached observation timestamps.
            return _mark_cache_hit(cached, True)

    fetcher = granule_fetcher or _default_granule_fetcher
    make_session = session_factory or _default_session_factory
    fall_back = fallback_fetcher or _default_fallback_fetcher
    deadline = _Deadline(deadline_seconds(), monotonic)

    want_late = late_enabled() if include_late is None else bool(include_late)
    want_fallback = (
        fallback_enabled() if include_fallback is None else bool(include_fallback)
    )

    attempts = []
    record = _attempt_imerg_runs(
        state_name,
        resolved,
        fetched_at,
        fetcher,
        make_session,
        deadline,
        attempts,
        want_late,
    )
    if record is None and want_fallback:
        record = _attempt_fallback(
            state_name, resolved, fetched_at, fall_back, attempts
        )
    if record is None:
        record = _unavailable_record(
            state_name,
            resolved,
            fetched_at,
            attempts,
            "no IMERG half-hourly granule and no fallback observation could be "
            "retrieved; no rainfall value is reported",
        )

    _mark_cache_hit(record, False)
    if use_cache and store is not None:
        store.put(key, record, monotonic)
    return record

def _attempt_imerg_runs(
    state_name, bounds, fetched_at, fetcher, make_session, deadline, attempts, want_late
):
    """
    Try Early, then Late. The session is built ONCE and shared by both runs --
    they live on the same host behind the same Earthdata guard.

    A credential failure is recorded and ends the IMERG phase immediately rather
    than being retried against Late: the token is the same, so a second rejection
    is certain and would only burn the budget.
    """
    runs = [RUN_TYPE_EARLY] + ([RUN_TYPE_LATE] if want_late else [])

    try:
        session = make_session()
    except Exception as auth_error:
        _record_attempt(
            attempts, SOURCE_KIND_EARLY, "auth_unavailable", auth_error
        )
        return None

    for run_type in runs:
        product = IMERG_PRODUCTS[run_type]
        if deadline.expired():
            _record_attempt(
                attempts, product["source_kind"], "skipped_deadline_exhausted"
            )
            continue
        try:
            outcome = _acquire_imerg_run(
                session, bounds, run_type, fetched_at, fetcher, deadline
            )
        except PermissionError as auth_error:
            _record_attempt(
                attempts, product["source_kind"], "auth_rejected", auth_error
            )
            return None
        except Exception as error:
            _record_attempt(attempts, product["source_kind"], "error", error)
            continue

        if outcome is None:
            _record_attempt(
                attempts,
                product["source_kind"],
                "no_published_granule",
                "no granule existed within the %d-slot probe window"
                % probe_granules(),
            )
            continue

        anchor, totals = outcome
        _record_attempt(
            attempts,
            product["source_kind"],
            "ok",
            "anchor %s, %d contiguous granules" % (_iso(anchor), len(totals)),
        )
        return _observed_record(
            state_name=state_name,
            bounds=bounds,
            observed_at=anchor,
            fetched_at=fetched_at,
            interval_mm=totals[0],
            interval_minutes=GRANULE_MINUTES,
            interval_totals_newest_first=totals,
            source=product["label"],
            source_kind=product["source_kind"],
            data_quality_status=QUALITY_REAL,
            expected_latency_minutes=product["latency_minutes"],
            attempts=attempts,
            granules_used=len(totals),
        )

    return None

def _attempt_fallback(state_name, bounds, fetched_at, fall_back, attempts):
    """
    The LABELLED Open-Meteo fallback. data_quality_status is FALLBACK, never REAL,
    and the source string says FALLBACK too, so a downstream reader cannot mistake
    a reanalysis/forecast-model hour for an IMERG satellite observation.

    Its hours are 60 minutes wide, so a 3 h window needs 3 of them and a 6 h
    window needs 6 -- the same completeness rule, applied at the coarser cadence.
    """
    try:
        pairs = fall_back(bounds, fetched_at)
    except Exception as error:
        _record_attempt(attempts, SOURCE_KIND_FALLBACK, "error", error)
        return None

    if not pairs:
        _record_attempt(
            attempts,
            SOURCE_KIND_FALLBACK,
            "no_usable_hours",
            "the fallback returned no finite past-hour precipitation value",
        )
        return None

    ordered = sorted(pairs, key=lambda item: item[0], reverse=True)
    anchor = ordered[0][0]

    # Only a contiguous run ending at the anchor may feed an accumulation.
    totals = [float(ordered[0][1])]
    expected = anchor
    for slot, amount in ordered[1:]:
        expected = expected - timedelta(minutes=FALLBACK_INTERVAL_MINUTES)
        if slot != expected:
            break
        totals.append(float(amount))

    _record_attempt(
        attempts,
        SOURCE_KIND_FALLBACK,
        "ok",
        "anchor %s, %d contiguous hours" % (_iso(anchor), len(totals)),
    )
    return _observed_record(
        state_name=state_name,
        bounds=bounds,
        observed_at=anchor,
        fetched_at=fetched_at,
        interval_mm=totals[0],
        interval_minutes=FALLBACK_INTERVAL_MINUTES,
        interval_totals_newest_first=totals,
        source=FALLBACK_SOURCE_LABEL,
        source_kind=SOURCE_KIND_FALLBACK,
        data_quality_status=QUALITY_FALLBACK,
        expected_latency_minutes=None,
        attempts=attempts,
        granules_used=len(totals),
    )

