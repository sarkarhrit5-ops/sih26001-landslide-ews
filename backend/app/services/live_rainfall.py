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
  * SOURCE ORDER: PPS IMERG Early half-hourly -> PPS IMERG Late half-hourly ->
    GES DISC IMERG Early half-hourly -> GES DISC IMERG Late half-hourly ->
    Open-Meteo (labelled FALLBACK). If everything fails the record is UNAVAILABLE
    and carries NO numbers: no zero fill, no imputation, no partial window
    presented as a whole one.
  * PPS NEAR-REAL-TIME (jsimpsonhttps) is tried FIRST because it publishes NRT
    granules the GES DISC OPeNDAP collection does not carry yet. It is a separate
    provider in every respect: its own credentials (PPS_USERNAME/PPS_PASSWORD, or
    PPS_EMAIL for both -- EARTHDATA_* does NOT authenticate jsimpsonhttps), its
    own YYYYMM directory layout, its own V07C/.RT-H5 naming, and its own 429
    cooldown. Granule existence is read from the monthly directory LISTING, so the
    anchor is the newest slot the server actually names rather than a
    latency guess. A PPS success is IMERG_HHR_EARLY_PPS / IMERG_HHR_LATE_PPS at
    data_quality_status REAL -- never FALLBACK. Any PPS failure records its reason
    and hands over to the GES DISC path with that path's behaviour unchanged.
  * PPS AOI READS ARE RANGE HYPERSLABS UNDER A HARD CEILING. jsimpsonhttps is a
    file server, not OPeNDAP: there is no server-side subsetting. The AOI window
    is read out of the remote HDF5 through _HttpRangeReader, which counts every
    byte and raises PPSAOIReadTooLarge at MAX_PPS_RANGE_BYTES. Nothing is written
    to disk, no response body is buffered whole, and a granule whose chunk layout
    would push the AOI read past the ceiling is ABANDONED in favour of GES DISC.
    The ceiling is never widened and the full granule is never downloaded.
  * A 3 h or 6 h accumulation is reported ONLY when every granule in that window
    was retrieved. A short run yields null plus the real reason.
  * ONE AOI subset per granule -- never one request per grid cell. Responses are
    streamed under a hard byte ceiling into a tempfile, parsed, and deleted; no
    NetCDF payload or raster is retained.
  * FALLBACK RATE-LIMIT DISCIPLINE. Open-Meteo limits by client IP, so an HTTP 429
    is a fact about this deployment rather than about one AOI. A 429 is raised as
    OpenMeteoRateLimited, recorded explicitly in `attempts`, and arms a PROVIDER-
    scoped cooldown (Retry-After when the provider supplied a usable one, else
    fallback_cooldown_seconds()) that suppresses the fallback for all four pilot
    AOIs. While it is armed no Open-Meteo request is made at all. This only ever
    lowers the request rate; no retry is added and no interval is shortened, and a
    suppressed provider still ends at UNAVAILABLE with no value.
  * IDENTICAL CONCURRENT LOOKUPS ARE SINGLE-FLIGHTED. Two simultaneous requests
    for the same state+AOI cost one acquisition; the second is served the first's
    cache entry. Different AOIs are never serialised against each other.

CONFIRMED PRODUCT FACTS (host .dds/.das/.dmr probe of GPM_3IMERGHHE.07):
  variable   precipitation
  dimensions time, lon, lat   (same order as the daily product)
  units      mm/hr  -> a 30-minute interval total is rate * 0.5
  grid       0.1 deg x 0.1 deg, lon 3600, lat 1800
  granules   48/day, directory YYYY/DDD, filename carries minutes-of-day

GES DISC authentication is the EXISTING guard:
weather_ingestion.get_earthdata_session(). PPS is a DIFFERENT host that guard does
not cover, so it has its own minimal session factory reading PPS_USERNAME /
PPS_PASSWORD (or PPS_EMAIL for both). Neither factory's credentials are logged or
returned; only environment-variable NAMES ever appear in an error.

The PPS attempt defaults ON exactly when those PPS variables are configured, so a
deployment without them behaves precisely as it did before instead of recording a
guaranteed auth failure on every call; SIH_LIVE_RAINFALL_PPS forces either state
explicitly. Everything external (granule_fetcher, session_factory,
fallback_fetcher, pps_granule_fetcher, pps_listing_fetcher, pps_session_factory,
include_pps, clock, cache) is injectable, so the whole module is testable offline
with no credentials and no network.

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

# ---------------------------------------------------------------------------
# PPS near-real-time surface (jsimpsonhttps). A SECOND NASA provider, tried
# BEFORE GES DISC, with its own credentials, its own directory layout and its own
# retrieval mechanism.
#
# jsimpsonhttps is a plain authenticated HTTPS file server with directory
# listings. It is NOT an OPeNDAP server: there is no ?precipitation[...]
# constraint and no .nc4 translation, so the GES DISC subsetting mechanism does
# not apply here. The AOI subset is instead taken by an HTTP Range hyperslab read
# of the remote HDF5 (see _HttpRangeReader), under a hard byte ceiling. Nothing is
# written to disk and no full granule is ever fetched: if the granule's chunk
# layout would make the AOI read exceed the ceiling, the read is ABANDONED and the
# ladder falls through to the existing GES DISC path.
# ---------------------------------------------------------------------------
PPS_ROOT = "https://jsimpsonhttps.pps.eosdis.nasa.gov/imerg"
PPS_RUN_DIRS = {RUN_TYPE_EARLY: "early", RUN_TYPE_LATE: "late"}

# Host-confirmed NRT naming, e.g.
#   3B-HHR-E.MS.MRG.3IMERG.20260831-S000000-E002959.0000.V07C.RT-H5
# Version and suffix BOTH differ from GES DISC (V07B / .HDF5); the GES DISC
# helpers above are therefore left exactly as they are.
PPS_PRODUCT_VERSION = "V07C"
PPS_GRANULE_SUFFIX = ".RT-H5"
PPS_DATASET_PATH = "/Grid/precipitation"

SOURCE_KIND_EARLY_PPS = "IMERG_HHR_EARLY_PPS"
SOURCE_KIND_LATE_PPS = "IMERG_HHR_LATE_PPS"

PPS_PRODUCTS = {
    RUN_TYPE_EARLY: {
        "granule_prefix": "3B-HHR-E",
        "source_kind": SOURCE_KIND_EARLY_PPS,
        "label": "NASA GPM IMERG Early half-hourly, PPS near-real-time (V07C)",
        # Informational only. Discovery is listing-based, so this number never
        # decides whether a slot exists.
        "latency_minutes": 240,
    },
    RUN_TYPE_LATE: {
        "granule_prefix": "3B-HHR-L",
        "source_kind": SOURCE_KIND_LATE_PPS,
        "label": "NASA GPM IMERG Late half-hourly, PPS near-real-time (V07C)",
        "latency_minutes": 840,
    },
}
PPS_RUN_ORDER = (RUN_TYPE_EARLY, RUN_TYPE_LATE)

# The AOI hyperslab is a few hundred float32 cells. The ceiling is the SAME 2 MiB
# the GES DISC subset uses, and it is a hard stop, never a soft target.
MAX_PPS_RANGE_BYTES = 2 * 1024 * 1024
PPS_RANGE_BLOCK_BYTES = 64 * 1024
PPS_TIMEOUT_SECONDS = 30
# A month of 30-minute listings is ~1440 filenames of ~70 bytes plus markup.
MAX_PPS_LISTING_BYTES = 4 * 1024 * 1024
PPS_LISTING_CHUNK_BYTES = 64 * 1024

# Credentials come from the environment ONLY. PPS registration issues the same
# e-mail address as both username and password, so PPS_EMAIL populates both when
# the explicit pair is absent. Never logged, never echoed into a record.
ENV_PPS_USERNAME = "PPS_USERNAME"
ENV_PPS_PASSWORD = "PPS_PASSWORD"
ENV_PPS_EMAIL = "PPS_EMAIL"

# Recorded when the bounded AOI read had to be abandoned. Distinct from "error"
# so the operator can tell a layout limitation from a broken provider.
OUTCOME_CEILING_ABORT = "aoi_read_exceeds_ceiling"

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

# Open-Meteo rate-limits by CLIENT IP, not by coordinate. A 429 for one AOI is
# therefore a statement about this deployment, not about Sikkim -- which is why
# the cooldown below is provider-scoped and shared by all four pilot AOIs.
HTTP_TOO_MANY_REQUESTS = 429

# Attempt outcomes recorded for the fallback provider. Named constants because the
# operator UI renders them verbatim and the tests assert on them.
OUTCOME_RATE_LIMITED = "rate_limited"
OUTCOME_RATE_LIMITED_COOLDOWN = "rate_limited_cooldown"

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
ENV_FALLBACK_COOLDOWN = "SIH_LIVE_RAINFALL_FALLBACK_COOLDOWN_SECONDS"
ENV_PPS_ENABLED = "SIH_LIVE_RAINFALL_PPS"
ENV_PPS_COOLDOWN = "SIH_LIVE_RAINFALL_PPS_COOLDOWN_SECONDS"

DEFAULT_CACHE_TTL_SECONDS = 900.0        # half the 30-minute product cadence
DEFAULT_NEGATIVE_TTL_SECONDS = 120.0     # a refusal must not stick around
DEFAULT_DEADLINE_SECONDS = 90.0
DEFAULT_CACHE_MAX_ENTRIES = 16
DEFAULT_PROBE_GRANULES = 8               # 4 h of 30-minute steps
DEFAULT_WINDOW_GRANULES = 12             # 6 h -> covers both accumulations
DEFAULT_STALE_MINUTES = 360
DEFAULT_NEAR_REAL_TIME_MINUTES = 90

# How long the Open-Meteo provider is left alone after it answers 429 without a
# usable Retry-After. One positive cache TTL: long enough that the four AOIs stop
# generating traffic, short enough that a transient limit clears on its own. This
# LOWERS the request rate; it is not a retry budget and never shortens one.
DEFAULT_FALLBACK_COOLDOWN_SECONDS = 900.0
# A provider-supplied Retry-After is honoured inside these bounds. The floor stops
# a "Retry-After: 0" from defeating the cooldown; the ceiling stops one bad header
# from disabling the fallback for a day.
MIN_FALLBACK_COOLDOWN_SECONDS = 60.0
MAX_FALLBACK_COOLDOWN_SECONDS = 3600.0

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


def fallback_cooldown_seconds():
    """
    How long to leave Open-Meteo alone after an unqualified 429. Clamped to the
    same bounds a provider-supplied Retry-After is clamped to, so an operator
    cannot configure the cooldown away entirely.
    """
    configured = _env_number(
        ENV_FALLBACK_COOLDOWN,
        DEFAULT_FALLBACK_COOLDOWN_SECONDS,
        minimum=MIN_FALLBACK_COOLDOWN_SECONDS,
    )
    return min(configured, MAX_FALLBACK_COOLDOWN_SECONDS)


def late_enabled():
    return _env_flag(ENV_LATE_ENABLED, True)


def pps_credentials_configured():
    """
    Whether PPS credentials are present in the environment. Reads only presence,
    never a value, and never logs one.
    """
    username = os.environ.get(ENV_PPS_USERNAME)
    password = os.environ.get(ENV_PPS_PASSWORD)
    email = os.environ.get(ENV_PPS_EMAIL)
    return bool((username and password) or email)


def pps_enabled():
    """
    Whether the PPS near-real-time path is tried before GES DISC.

    Default: ON exactly when PPS credentials are configured. PPS cannot be a
    source of this deployment without them, so an unconfigured deployment behaves
    precisely as it did before rather than logging a guaranteed auth failure on
    every call. Setting SIH_LIVE_RAINFALL_PPS=1 forces the attempt anyway (which
    surfaces the missing credentials as an explicit auth_unavailable attempt), and
    SIH_LIVE_RAINFALL_PPS=0 disables it even when credentials exist.
    """
    return _env_flag(ENV_PPS_ENABLED, pps_credentials_configured())


def pps_cooldown_seconds():
    """
    How long to leave PPS alone after an unqualified 429. Same bounds and same
    only-ever-lowers-the-rate semantics as the Open-Meteo cooldown; a separate
    gate because the two are different providers with different limits.
    """
    configured = _env_number(
        ENV_PPS_COOLDOWN,
        DEFAULT_FALLBACK_COOLDOWN_SECONDS,
        minimum=MIN_FALLBACK_COOLDOWN_SECONDS,
    )
    return min(configured, MAX_FALLBACK_COOLDOWN_SECONDS)


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


# ---------------------------------------------------------------------------
# PPS granule identity and discovery
# ---------------------------------------------------------------------------
def pps_directory_url(run_type, slot_start):
    """
    The PPS monthly directory for a slot: .../imerg/early/YYYYMM/

    Deliberately NOT granule_directory_url's YYYY/DDD layout -- PPS groups NRT
    granules by calendar month.
    """
    return "%s/%s/%s/" % (
        PPS_ROOT,
        PPS_RUN_DIRS[run_type],
        slot_start.strftime("%Y%m"),
    )


def pps_granule_basename(run_type, slot_start):
    """
    The PPS NRT filename for a slot, e.g.
      3B-HHR-E.MS.MRG.3IMERG.20260831-S000000-E002959.0000.V07C.RT-H5

    Same S/E/minutes-of-day rule as the GES DISC name, different version and
    suffix. Used for URL construction only -- existence is decided by the
    listing, never by this string.
    """
    product = PPS_PRODUCTS[run_type]
    slot_end = slot_start + timedelta(minutes=GRANULE_MINUTES) - timedelta(seconds=1)
    return "%s.MS.MRG.3IMERG.%s-S%s-E%s.%04d.%s%s" % (
        product["granule_prefix"],
        slot_start.strftime("%Y%m%d"),
        slot_start.strftime("%H%M%S"),
        slot_end.strftime("%H%M%S"),
        minutes_of_day(slot_start),
        PPS_PRODUCT_VERSION,
        PPS_GRANULE_SUFFIX,
    )


def pps_granule_url(run_type, slot_start, filename=None):
    """Absolute URL of one PPS granule. `filename` wins so a listing-supplied
    name (which may carry a version PPS has moved on to) is used verbatim."""
    return "%s%s" % (
        pps_directory_url(run_type, slot_start),
        filename or pps_granule_basename(run_type, slot_start),
    )


def pps_parse_granule_name(name):
    """
    (slot_start, run_type) for a PPS HHR filename, or None if it is not one.

    The slot start is read from the file's own -SHHMMSS field rather than from the
    minutes-of-day sequence number, so a malformed sequence cannot shift an
    observation timestamp.
    """
    import re

    match = re.match(
        r"^(3B-HHR-[EL])\.MS\.MRG\.3IMERG\."
        r"(\d{8})-S(\d{6})-E(\d{6})\.(\d{4})\.(V\d+[A-Z]?)(\.RT-H5)$",
        str(name or "").strip(),
    )
    if match is None:
        return None
    prefix, day, start, _end, _seq, _version, _suffix = match.groups()
    try:
        slot_start = datetime.strptime(day + start, "%Y%m%d%H%M%S")
    except ValueError:
        return None
    if slot_start != floor_to_granule(slot_start):
        return None
    for run_type, product in PPS_PRODUCTS.items():
        if product["granule_prefix"] == prefix:
            return slot_start, run_type
    return None


def pps_index_from_listing(body, run_type):
    """
    {slot_start: filename} for every HHR granule of `run_type` named in a PPS
    directory listing body. Parsing the listing is what makes requirement 9
    true: nothing here assumes a file exists.
    """
    index = {}
    for token in _pps_listing_tokens(body):
        parsed = pps_parse_granule_name(token)
        if parsed is None:
            continue
        slot_start, parsed_run = parsed
        if parsed_run != run_type:
            continue
        index[slot_start] = token
    return index


def _pps_listing_tokens(body):
    """Candidate filenames in an Apache-style HTML or plain-text listing."""
    import re

    return re.findall(r"3B-HHR-[EL][^\s\"'<>]*?\.RT-H5", str(body or ""))


def pps_latest_slot_at_or_before(index, now):
    """
    The newest published slot at or before the current observation slot
    (requirement 10). Future-dated granules are ignored rather than trusted.
    """
    ceiling = floor_to_granule(now)
    candidates = [slot for slot in index if slot <= ceiling]
    if not candidates:
        return None
    return max(candidates)


class PPSRateLimited(Exception):
    """PPS answered HTTP 429. Carries the provider's Retry-After when usable."""

    def __init__(self, retry_after_seconds=None):
        self.retry_after_seconds = retry_after_seconds
        if retry_after_seconds is None:
            detail = "PPS returned HTTP 429 (no usable Retry-After header)"
        else:
            detail = "PPS returned HTTP 429, Retry-After %.0fs" % float(
                retry_after_seconds
            )
        super().__init__(detail)


class PPSAOIReadTooLarge(Exception):
    """
    The bounded AOI hyperslab read would have exceeded MAX_PPS_RANGE_BYTES.

    Raised INSTEAD of continuing, so a granule whose chunk layout makes an AOI
    subset impossible costs a bounded number of range requests and then hands the
    ladder to GES DISC. The ceiling is never widened and the full granule is never
    fetched.
    """


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


class _HttpRangeReader:
    """
    A minimal seekable, read-only file-like view of a remote HTTP object, served
    by Range requests under a HARD cumulative byte ceiling.

    This is what lets h5py read the AOI hyperslab out of a remote HDF5 without
    downloading it: h5py seeks to the superblock, then to the dataset's b-tree,
    then to just the bytes covering the requested lon/lat window. Every byte read
    is counted; crossing `max_bytes` raises PPSAOIReadTooLarge instead of
    continuing, which is the ceiling-abort the ladder falls through on.

    Nothing is buffered whole: only the current block is held, and no file is ever
    written to disk. `size` comes from Content-Length on the initial HEAD.
    """

    def __init__(self, session, url, size, max_bytes, block_bytes=None, timeout=None):
        self._session = session
        self._url = url
        self._size = int(size)
        self._max_bytes = int(max_bytes)
        self._block = int(block_bytes or PPS_RANGE_BLOCK_BYTES)
        self._timeout = PPS_TIMEOUT_SECONDS if timeout is None else timeout
        self._pos = 0
        self._bytes_read = 0
        self._requests = 0
        self._cache_start = None
        self._cache = b""

    # -- introspection used by the attempt trail and the tests ---------------
    @property
    def bytes_read(self):
        return self._bytes_read

    @property
    def range_requests(self):
        return self._requests

    # -- file-like surface h5py needs ---------------------------------------
    def seekable(self):
        return True

    def readable(self):
        return True

    def writable(self):
        return False

    def tell(self):
        return self._pos

    def seek(self, offset, whence=0):
        if whence == 0:
            target = int(offset)
        elif whence == 1:
            target = self._pos + int(offset)
        elif whence == 2:
            target = self._size + int(offset)
        else:
            raise ValueError("unsupported whence %r" % (whence,))
        self._pos = max(0, target)
        return self._pos

    def read(self, length=-1):
        if length is None or length < 0:
            # An unbounded read is a full download by another name. Refused.
            remaining = self._size - self._pos
            if remaining > self._max_bytes:
                raise PPSAOIReadTooLarge(
                    "an unbounded read of %d bytes was requested from %s; refused"
                    % (remaining, self._url)
                )
            length = remaining
        length = min(int(length), max(0, self._size - self._pos))
        if length <= 0:
            return b""

        out = bytearray()
        while len(out) < length:
            chunk = self._block_at(self._pos + len(out))
            if not chunk:
                break
            want = length - len(out)
            out.extend(chunk[:want])
        self._pos += len(out)
        return bytes(out)

    def _block_at(self, position):
        """Bytes at `position`, from the single cached block or a new Range GET."""
        if (
            self._cache_start is not None
            and self._cache_start <= position < self._cache_start + len(self._cache)
        ):
            return self._cache[position - self._cache_start :]

        start = position
        end = min(self._size, start + self._block) - 1
        if end < start:
            return b""

        span = end - start + 1
        if self._bytes_read + span > self._max_bytes:
            raise PPSAOIReadTooLarge(
                "the AOI read of %s would exceed the %d byte ceiling after %d bytes "
                "in %d range requests; abandoning rather than downloading the "
                "granule" % (self._url, self._max_bytes, self._bytes_read, self._requests)
            )

        response = self._session.get(
            self._url,
            headers={"Range": "bytes=%d-%d" % (start, end)},
            timeout=self._timeout,
            stream=True,
        )
        try:
            _raise_for_pps_status(response, self._url)
            if response.status_code != 206:
                # A 200 here means the server ignored the Range header and is
                # about to hand us the whole file. Refuse it.
                raise PPSAOIReadTooLarge(
                    "PPS ignored the Range header for %s (HTTP %d); refusing a "
                    "full-granule download" % (self._url, response.status_code)
                )
            payload = bytearray()
            for piece in response.iter_content(PPS_RANGE_BLOCK_BYTES):
                if not piece:
                    continue
                payload.extend(piece)
                if len(payload) > span:
                    raise PPSAOIReadTooLarge(
                        "PPS returned more than the requested range for %s"
                        % (self._url,)
                    )
        finally:
            response.close()

        self._requests += 1
        self._bytes_read += len(payload)
        self._cache_start = start
        self._cache = bytes(payload)
        return self._cache

    def close(self):
        self._cache = b""
        self._cache_start = None


def _raise_for_pps_status(response, url):
    """Map PPS HTTP status onto the ladder's own vocabulary."""
    status = int(getattr(response, "status_code", 0))
    if status == 404:
        raise GranuleUnavailable("PPS object is not published (HTTP 404): %s" % url)
    if status in (401, 403):
        raise PermissionError(
            "PPS AUTHENTICATION REJECTED (HTTP %d) for %s. Check %s / %s (or %s); "
            "EARTHDATA credentials do not authenticate jsimpsonhttps."
            % (status, url, ENV_PPS_USERNAME, ENV_PPS_PASSWORD, ENV_PPS_EMAIL)
        )
    if status == HTTP_TOO_MANY_REQUESTS:
        raise PPSRateLimited(
            _parse_retry_after(getattr(response, "headers", {}).get("Retry-After"))
        )
    if status >= 400:
        raise RuntimeError("PPS request failed (HTTP %d) for %s" % (status, url))


def _default_pps_session_factory():
    """
    An authenticated PPS session built from the ENVIRONMENT ONLY.

    PPS issues the registered e-mail address as both username and password, so
    PPS_EMAIL populates both when the explicit pair is absent. Credentials are
    never logged, never returned and never placed in a record; only the variable
    NAMES appear in the error text.
    """
    import requests

    username = os.environ.get(ENV_PPS_USERNAME)
    password = os.environ.get(ENV_PPS_PASSWORD)
    email = os.environ.get(ENV_PPS_EMAIL)
    if not (username and password):
        if email:
            username = username or email
            password = password or email
    if not (username and password):
        raise PermissionError(
            "BLOCKER: Missing PPS near-real-time credentials. Set %s and %s (or %s). "
            "EARTHDATA_* credentials do not authenticate jsimpsonhttps."
            % (ENV_PPS_USERNAME, ENV_PPS_PASSWORD, ENV_PPS_EMAIL)
        )

    session = requests.Session()
    session.auth = (username, password)
    return session


def _default_pps_listing_fetcher(session, run_type, slot_start):
    """
    The monthly PPS directory listing, streamed under MAX_PPS_LISTING_BYTES and
    parsed into {slot_start: filename}. One request per run per call.
    """
    url = pps_directory_url(run_type, slot_start)
    response = session.get(url, timeout=PPS_TIMEOUT_SECONDS, stream=True)
    try:
        _raise_for_pps_status(response, url)
        body = bytearray()
        for chunk in response.iter_content(PPS_LISTING_CHUNK_BYTES):
            if not chunk:
                continue
            body.extend(chunk)
            if len(body) > MAX_PPS_LISTING_BYTES:
                raise RuntimeError(
                    "PPS listing for %s exceeds the %d byte ceiling"
                    % (url, MAX_PPS_LISTING_BYTES)
                )
    finally:
        response.close()

    return pps_index_from_listing(body.decode("utf-8", "replace"), run_type)


def _default_pps_granule_fetcher(session, slot_start, bounds, run_type, filename=None):
    """
    The AOI mean precipitation RATE (mm/hr) for one PPS granule, taken by an HTTP
    Range hyperslab read of the remote HDF5.

    No temporary file, no response.content, no whole-granule download: h5py reads
    through _HttpRangeReader, which counts every byte and raises
    PPSAOIReadTooLarge at MAX_PPS_RANGE_BYTES. The AOI indices and the no-data
    rule are the EXISTING ones, so a PPS value and a GES DISC value are computed
    identically.
    """
    import h5py

    from app.services import weather_ingestion

    url = pps_granule_url(run_type, slot_start, filename)
    head = session.head(url, timeout=PPS_TIMEOUT_SECONDS)
    try:
        _raise_for_pps_status(head, url)
        size = head.headers.get("Content-Length")
    finally:
        close = getattr(head, "close", None)
        if close is not None:
            close()
    if not size:
        raise RuntimeError(
            "PPS did not report Content-Length for %s; a bounded range read is "
            "not possible" % url
        )

    lat_min, lat_max, lon_min, lon_max = weather_ingestion.get_imerg_indices(bounds)
    reader = _HttpRangeReader(session, url, int(size), MAX_PPS_RANGE_BYTES)
    try:
        with h5py.File(reader, "r") as handle:
            dataset = handle[PPS_DATASET_PATH]
            values = dataset[0, lon_min : lon_max + 1, lat_min : lat_max + 1]
    finally:
        reader.close()

    return float(weather_ingestion._mean_valid_precipitation(values))


def _default_session_factory():
    """The EXISTING Earthdata guard, used as-is. No credential handling here."""
    from app.services import weather_ingestion

    return weather_ingestion.get_earthdata_session()


class OpenMeteoRateLimited(Exception):
    """
    The Open-Meteo fallback answered HTTP 429.

    Carried as its own type so _attempt_fallback can arm the shared cooldown for
    this case ONLY. A timeout, a 500 or a malformed body must not silence the
    provider for every state -- those are per-call failures, a 429 is not.

    retry_after_seconds is the provider's own request when it supplied a usable
    Retry-After, else None.
    """

    def __init__(self, retry_after_seconds=None):
        self.retry_after_seconds = retry_after_seconds
        if retry_after_seconds is None:
            detail = "Open-Meteo returned HTTP 429 (no usable Retry-After header)"
        else:
            detail = "Open-Meteo returned HTTP 429, Retry-After %.0fs" % float(
                retry_after_seconds
            )
        super().__init__(detail)


def _parse_retry_after(raw):
    """
    Retry-After as a float number of seconds, or None when unusable.

    RFC 9110 allows either delta-seconds or an HTTP-date. Both are accepted; an
    absent, malformed, negative or already-past value returns None so the caller
    falls back to the configured cooldown rather than to zero.
    """
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None
    try:
        seconds = float(text)
    except (TypeError, ValueError):
        seconds = None
    if seconds is not None:
        return seconds if seconds > 0.0 else None

    try:
        from email.utils import parsedate_to_datetime

        when = parsedate_to_datetime(text)
    except Exception:
        return None
    if when is None:
        return None
    try:
        if when.tzinfo is not None:
            when = when.astimezone(tz=None).replace(tzinfo=None)
            reference = datetime.now()
        else:
            reference = _utcnow()
        delta = (when - reference).total_seconds()
    except Exception:
        return None
    return delta if delta > 0.0 else None


class _FallbackCooldown:
    """
    A PROVIDER-scoped, monotonic cooldown gate (one instance per provider: see
    _FALLBACK_COOLDOWN for Open-Meteo and _PPS_COOLDOWN for PPS).

    Deliberately not keyed by state: Open-Meteo rate-limits by client IP, so a 429
    raised while serving Sikkim is a fact about this deployment. Keying it per AOI
    would let the other three pilots each re-earn the same 429 within seconds --
    the exact behaviour that kept the Render instance rate-limited.

    While armed, _attempt_fallback makes ZERO network calls and records
    OUTCOME_RATE_LIMITED_COOLDOWN. This can only reduce the request rate; nothing
    here shortens an interval or adds a retry.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._until = None
        self._reason = None

    def arm(self, seconds, monotonic, reason=None):
        """Suppress the provider for `seconds`, clamped to the sanctioned bounds."""
        span = max(
            MIN_FALLBACK_COOLDOWN_SECONDS,
            min(float(seconds), MAX_FALLBACK_COOLDOWN_SECONDS),
        )
        with self._lock:
            candidate = monotonic + span
            # Never shorten an existing cooldown: a second 429 must not become a
            # way to get back to the provider sooner.
            if self._until is None or candidate > self._until:
                self._until = candidate
                self._reason = reason
            return self._until

    def remaining(self, monotonic):
        """Seconds left, or None when the provider may be called."""
        with self._lock:
            if self._until is None:
                return None
            left = self._until - monotonic
            if left <= 0.0:
                self._until = None
                self._reason = None
                return None
            return left

    def reason(self):
        with self._lock:
            return self._reason

    def clear(self):
        with self._lock:
            self._until = None
            self._reason = None


_FALLBACK_COOLDOWN = _FallbackCooldown()

# A SECOND, independent instance for PPS. Same never-shortens, provider-scoped
# semantics; separate state because a NASA rate limit and an Open-Meteo rate limit
# are unrelated facts and must not silence each other.
_PPS_COOLDOWN = _FallbackCooldown()


def clear_pps_cooldown():
    """Re-permit PPS immediately (tests, and operator reset)."""
    _PPS_COOLDOWN.clear()


def pps_cooldown_remaining(monotonic=None):
    """Seconds until PPS may be called again, or None if it may be now."""
    import time as _time

    return _PPS_COOLDOWN.remaining(
        _time.monotonic() if monotonic is None else monotonic
    )


def clear_fallback_cooldown():
    """Re-permit Open-Meteo immediately (tests, and operator reset)."""
    _FALLBACK_COOLDOWN.clear()


def fallback_cooldown_remaining(monotonic=None):
    """Seconds until Open-Meteo may be called again, or None if it may be now."""
    import time as _time

    return _FALLBACK_COOLDOWN.remaining(
        _time.monotonic() if monotonic is None else monotonic
    )

def _default_fallback_fetcher(bounds, now):
    """
    Open-Meteo hourly precipitation at the AOI centroid -- the LABELLED fallback.

    This is ONE request for the AOI centre, never one per grid cell, and the body
    is streamed under MAX_FALLBACK_BODY_BYTES exactly like the IMERG path. It
    returns hourly totals in mm (Open-Meteo's native unit for `precipitation`),
    so no rate conversion applies; the caller marks the record FALLBACK and sets
    interval_minutes to 60.

    HTTP 429 is raised as OpenMeteoRateLimited, NOT as a generic HTTPError: the
    caller has to be able to tell "the provider asked us to stop" apart from "the
    provider is broken", because only the former arms the shared cooldown.

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
        # Checked BEFORE raise_for_status so the rate-limit signal, and any
        # Retry-After the provider volunteered, survive as structured data.
        if getattr(response, "status_code", None) == HTTP_TOO_MANY_REQUESTS:
            headers = getattr(response, "headers", None) or {}
            raise OpenMeteoRateLimited(
                retry_after_seconds=_parse_retry_after(headers.get("Retry-After"))
            )
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


class _KeyedLocks:
    """
    One lock per cache key, so identical concurrent lookups collapse into a single
    upstream acquisition (a "single flight").

    Scope is deliberately the CACHE KEY -- state + run type + rounded bounds --
    and nothing coarser: four dashboards asking about four different AOIs must
    still proceed in parallel. Only a second caller asking the SAME question waits,
    and what it waits for is the first caller's cache entry, not a second fetch.

    The map is reference-counted so it cannot grow without bound: the last waiter
    to leave a key removes it.
    """

    def __init__(self):
        self._guard = threading.Lock()
        self._locks = {}

    def acquire(self, key):
        with self._guard:
            entry = self._locks.get(key)
            if entry is None:
                entry = [threading.Lock(), 0]
                self._locks[key] = entry
            entry[1] += 1
            lock = entry[0]
        lock.acquire()
        return lock

    def release(self, key, lock):
        lock.release()
        with self._guard:
            entry = self._locks.get(key)
            if entry is None:
                return
            entry[1] -= 1
            if entry[1] <= 0:
                del self._locks[key]

    def __len__(self):
        with self._guard:
            return len(self._locks)


_INFLIGHT = _KeyedLocks()


def clear_cache():
    """
    Drop every cached live-rainfall record (tests, and operator reset).

    The provider cooldown is cleared with it: both exist to suppress redundant
    upstream work, and an operator reset that left Open-Meteo silenced would be
    surprising. Nothing here fabricates or revives a value -- the next call
    re-acquires from the sources or reports UNAVAILABLE.
    """
    _CACHE.clear()
    _FALLBACK_COOLDOWN.clear()
    _PPS_COOLDOWN.clear()


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

def _acquire_pps_run(
    session, bounds, run_type, now, fetcher, listing_fetcher, deadline
):
    """
    Retrieve the newest PUBLISHED PPS granule for one run, then walk backwards
    contiguously to cover the widest accumulation window.

    The difference from _acquire_imerg_run is discovery: the anchor comes from the
    monthly directory LISTING, not from a publication-latency guess, so "no
    granule" means the server does not list one rather than "we looked in the
    wrong place". A slot the listing does not name is a gap, exactly as an HTTP
    404 is on the GES DISC path.

    The window may cross a month boundary, so a second listing is fetched (once)
    when the walk steps into the previous month.

    Returns (anchor_slot, [interval_mm newest-first]) or None if nothing existed.
    """
    indexes = {}

    def index_for(slot):
        month = slot.strftime("%Y%m")
        if month not in indexes:
            indexes[month] = listing_fetcher(session, run_type, slot)
        return indexes[month] or {}

    anchor_index = index_for(now)
    anchor = pps_latest_slot_at_or_before(anchor_index, now)
    if anchor is None:
        return None

    anchor_rate = fetcher(session, anchor, bounds, run_type, anchor_index.get(anchor))
    totals = [float(anchor_rate) * GRANULE_HOURS]

    for step in range(1, max(1, window_granules())):
        if deadline.expired():
            break
        slot = anchor - timedelta(minutes=GRANULE_MINUTES * step)
        try:
            listing = index_for(slot)
        except Exception:
            # A missing neighbouring month is a gap in the window, not a reason to
            # discard the anchor observation we already hold.
            break
        filename = listing.get(slot)
        if filename is None:
            break
        try:
            rate = fetcher(session, slot, bounds, run_type, filename)
        except GranuleUnavailable:
            break
        totals.append(float(rate) * GRANULE_HOURS)

    return anchor, totals


def _attempt_pps_runs(
    state_name,
    bounds,
    fetched_at,
    fetcher,
    listing_fetcher,
    make_session,
    deadline,
    attempts,
    want_late,
    monotonic=None,
):
    """
    Try PPS Early, then PPS Late, BEFORE the GES DISC path.

    Failure here is never terminal: every outcome is recorded and None is
    returned, so the caller proceeds to GES DISC with its existing behaviour
    intact (requirement 18). A PPS success is labelled IMERG_HHR_*_PPS at
    QUALITY_REAL -- never FALLBACK.
    """
    import time as _time

    runs = [RUN_TYPE_EARLY] + ([RUN_TYPE_LATE] if want_late else [])
    clock = _time.monotonic() if monotonic is None else monotonic

    cooling = _PPS_COOLDOWN.remaining(clock)
    if cooling is not None:
        for run_type in runs:
            _record_attempt(
                attempts,
                PPS_PRODUCTS[run_type]["source_kind"],
                OUTCOME_RATE_LIMITED_COOLDOWN,
                "PPS is rate-limited; suppressed for a further %.0fs without a "
                "request (%s)" % (cooling, _PPS_COOLDOWN.reason() or "no detail"),
            )
        return None

    try:
        session = make_session()
    except Exception as auth_error:
        _record_attempt(
            attempts, SOURCE_KIND_EARLY_PPS, "auth_unavailable", auth_error
        )
        return None

    for run_type in runs:
        product = PPS_PRODUCTS[run_type]
        if deadline.expired():
            _record_attempt(
                attempts, product["source_kind"], "skipped_deadline_exhausted"
            )
            continue
        try:
            outcome = _acquire_pps_run(
                session, bounds, run_type, fetched_at, fetcher, listing_fetcher, deadline
            )
        except PermissionError as auth_error:
            # One credential, one verdict: retrying Late would be certain to fail.
            _record_attempt(
                attempts, product["source_kind"], "auth_rejected", auth_error
            )
            return None
        except PPSRateLimited as limited:
            span = limited.retry_after_seconds
            if span is None:
                span = pps_cooldown_seconds()
            _PPS_COOLDOWN.arm(span, clock, str(limited))
            _record_attempt(
                attempts, product["source_kind"], OUTCOME_RATE_LIMITED, limited
            )
            return None
        except PPSAOIReadTooLarge as too_large:
            # The ceiling held. Hand over to GES DISC rather than widening it.
            _record_attempt(
                attempts, product["source_kind"], OUTCOME_CEILING_ABORT, too_large
            )
            continue
        except GranuleUnavailable as missing:
            _record_attempt(
                attempts, product["source_kind"], "no_published_granule", missing
            )
            continue
        except Exception as error:
            _record_attempt(attempts, product["source_kind"], "error", error)
            continue

        if outcome is None:
            _record_attempt(
                attempts,
                product["source_kind"],
                "no_published_granule",
                "the PPS monthly listing names no granule at or before the "
                "current 30-minute slot",
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
    pps_granule_fetcher=None,
    pps_listing_fetcher=None,
    pps_session_factory=None,
    include_pps=None,
):
    """
    The LATEST AVAILABLE sub-daily rainfall for a pilot AOI, with its age stated.

    Source order: PPS IMERG Early HHR -> PPS IMERG Late HHR -> GES DISC IMERG
    Early HHR -> GES DISC IMERG Late HHR -> Open-Meteo (FALLBACK) -> UNAVAILABLE.
    Every attempt is recorded in `attempts`, so a FALLBACK record always says what
    failed before it. The PPS phase is skipped (recording nothing) when no PPS
    credentials are configured and include_pps / SIH_LIVE_RAINFALL_PPS do not force
    it, so an unconfigured deployment behaves exactly as it did before.

    This function is NOT a source of model features. It is unreachable from
    derive_rainfall_features(), it does not import rainfall_service, and its
    cache and environment namespace are separate (requirements 8 and 13).

    Every collaborator is injectable so the whole path is testable offline with
    no credentials and no network.

    Concurrency: identical lookups are single-flighted (see _KeyedLocks), so four
    dashboards refreshing the same state at the same moment cost ONE acquisition.
    Distinct AOIs are never serialised against each other.
    """
    import time as _time

    resolved = resolve_bounds(state_name, bounds)
    fetch_clock = clock or _utcnow
    monotonic = _time.monotonic()

    store = cache if cache is not None else _CACHE
    key = cache_key(state_name, resolved, "live")
    if use_cache and store is not None:
        cached = store.get(key, monotonic)
        if cached is not None:
            # store.get already handed back an independent deep-enough copy, so
            # stamping it cannot rewrite the cached observation timestamps.
            return _mark_cache_hit(cached, True)

    acquire_kwargs = dict(
        state_name=state_name,
        resolved=resolved,
        now=now,
        fetch_clock=fetch_clock,
        granule_fetcher=granule_fetcher,
        session_factory=session_factory,
        fallback_fetcher=fallback_fetcher,
        include_late=include_late,
        include_fallback=include_fallback,
        pps_granule_fetcher=pps_granule_fetcher,
        pps_listing_fetcher=pps_listing_fetcher,
        pps_session_factory=pps_session_factory,
        include_pps=include_pps,
    )

    if not (use_cache and store is not None):
        # No cache to coalesce through: run directly, exactly as before.
        return _acquire_record(monotonic=monotonic, **acquire_kwargs)

    lock = _INFLIGHT.acquire(key)
    try:
        # Re-check under the key's lock: whoever we queued behind has just stored
        # its result, and replaying that is what makes the second caller free.
        monotonic = _time.monotonic()
        cached = store.get(key, monotonic)
        if cached is not None:
            return _mark_cache_hit(cached, True)

        record = _acquire_record(monotonic=monotonic, **acquire_kwargs)
        store.put(key, record, monotonic)
        return record
    finally:
        _INFLIGHT.release(key, lock)


def _acquire_record(
    state_name,
    resolved,
    now,
    fetch_clock,
    monotonic,
    granule_fetcher,
    session_factory,
    fallback_fetcher,
    include_late,
    include_fallback,
    pps_granule_fetcher=None,
    pps_listing_fetcher=None,
    pps_session_factory=None,
    include_pps=None,
):
    """
    One acquisition through the full source ladder: PPS IMERG Early -> PPS IMERG
    Late -> GES DISC IMERG Early -> GES DISC IMERG Late -> Open-Meteo FALLBACK ->
    UNAVAILABLE.

    PPS is tried first because it publishes NRT granules the GES DISC OPeNDAP
    collection does not carry yet. It runs inside the SAME _Deadline, so the total
    budget is unchanged; the GES DISC and Open-Meteo phases below are untouched.

    Storing is the caller's job, because only the caller knows whether a cache is
    in play.
    """
    fetched_at = now or fetch_clock()

    fetcher = granule_fetcher or _default_granule_fetcher
    make_session = session_factory or _default_session_factory
    fall_back = fallback_fetcher or _default_fallback_fetcher
    pps_fetcher = pps_granule_fetcher or _default_pps_granule_fetcher
    pps_listing = pps_listing_fetcher or _default_pps_listing_fetcher
    make_pps_session = pps_session_factory or _default_pps_session_factory
    deadline = _Deadline(deadline_seconds(), monotonic)

    want_late = late_enabled() if include_late is None else bool(include_late)
    want_fallback = (
        fallback_enabled() if include_fallback is None else bool(include_fallback)
    )
    want_pps = pps_enabled() if include_pps is None else bool(include_pps)

    attempts = []
    record = None
    if want_pps:
        record = _attempt_pps_runs(
            state_name,
            resolved,
            fetched_at,
            pps_fetcher,
            pps_listing,
            make_pps_session,
            deadline,
            attempts,
            want_late,
            monotonic,
        )
    if record is None:
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
            state_name, resolved, fetched_at, fall_back, attempts, monotonic
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

def _attempt_fallback(
    state_name, bounds, fetched_at, fall_back, attempts, monotonic=None
):
    """
    The LABELLED Open-Meteo fallback. data_quality_status is FALLBACK, never REAL,
    and the source string says FALLBACK too, so a downstream reader cannot mistake
    a reanalysis/forecast-model hour for an IMERG satellite observation.

    Its hours are 60 minutes wide, so a 3 h window needs 3 of them and a 6 h
    window needs 6 -- the same completeness rule, applied at the coarser cadence.

    RATE-LIMIT DISCIPLINE. Open-Meteo limits by client IP, so the cooldown gate
    consulted here is shared by all four pilot AOIs:
      * armed  -> return immediately, ZERO network calls, outcome
                  rate_limited_cooldown with the seconds remaining;
      * a 429  -> arm the cooldown to Retry-After when the provider supplied a
                  usable one, else to fallback_cooldown_seconds(), and record
                  outcome rate_limited;
      * any other failure -> unchanged generic `error`, cooldown NOT armed.
    Either way the caller ends at UNAVAILABLE with no value: a suppressed provider
    is a reason, never a substitute for an observation.
    """
    import time as _time

    if monotonic is None:
        monotonic = _time.monotonic()

    remaining = _FALLBACK_COOLDOWN.remaining(monotonic)
    if remaining is not None:
        reason = _FALLBACK_COOLDOWN.reason()
        _record_attempt(
            attempts,
            SOURCE_KIND_FALLBACK,
            OUTCOME_RATE_LIMITED_COOLDOWN,
            "provider suppressed for a further %.0fs after %s; no request was made"
            % (remaining, reason or "an earlier HTTP 429"),
        )
        return None

    try:
        pairs = fall_back(bounds, fetched_at)
    except OpenMeteoRateLimited as limited:
        requested = limited.retry_after_seconds
        span = fallback_cooldown_seconds() if requested is None else float(requested)
        _FALLBACK_COOLDOWN.arm(
            span,
            monotonic,
            reason="HTTP 429 while serving %s" % (state_name or "an AOI"),
        )
        _record_attempt(
            attempts,
            SOURCE_KIND_FALLBACK,
            OUTCOME_RATE_LIMITED,
            "%s; provider suppressed for %.0fs (%s)"
            % (
                limited,
                max(
                    MIN_FALLBACK_COOLDOWN_SECONDS,
                    min(span, MAX_FALLBACK_COOLDOWN_SECONDS),
                ),
                "Retry-After honoured" if requested is not None else "configured cooldown",
            ),
        )
        return None
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

