"""
Read-only resolver for the East Sikkim pilot landslide-event geometry.

This is the single source of truth for the 82 real NASA Global Landslide Catalog
(GLC) landslide positives inside the canonical pilot AOI -- the exact inventory the
pilot model was trained on. It is intentionally dependency-light (standard library
+ app.core.config_states only; NO fastapi / pandas / ml imports) so it can be
imported by the API layer, by the snapshot generator, and by the offline test
harness alike. Because it imports no web/ML stack, the event-serving path is now
unit-testable offline for the first time.

DATA-INTEGRITY CONTRACT (mirrors app/api/routes.py):
  * Every field is copied from a real source record; nothing is defaulted,
    synthesised, or hard-coded.
  * Two real sources are supported, in this order:
      1. A committed, validated snapshot artifact
         (backend/data/models/sikkim_events.json) -- part of the shipped Sikkim
         evidence bundle, so the endpoint no longer depends on the 8.5 MB raw
         catalog being present on the server.
      2. The raw NASA GLC catalog (backend/data/raw/glc_legacy.csv), filtered
         live with the SAME rule used by scripts/train_real_models.py.
    Both, by construction, yield the identical 82 events; the snapshot is just a
    durable materialisation of source (2) -- see
    scripts/build_sikkim_events_snapshot.py.
  * If neither source is present the resolver returns events=None and the caller
    refuses with HTTP 503 DATA_UNAVAILABLE -- it never returns a partial or
    invented list.
"""
import csv
import json
import os

from app.core.config_states import get_pilot_aoi_bounds

# location_accuracy values the GLC pipeline treats as spatially precise; anything
# else is counted as >= ~5 km uncertainty (matches the provenance heuristic that
# reports 78.0% of the pilot events as spatially uncertain).
PRECISE_ACCURACY = frozenset({"1km", "exact", "100m"})

# Public field order of a single event record (matches the frontend LandslideEvent
# interface in frontend/src/services/api.ts).
EVENT_FIELDS = (
    "latitude", "longitude", "event_date", "event_title", "landslide_category",
    "landslide_trigger", "landslide_size", "location_accuracy", "fatality_count",
    "spatial_uncertainty", "source_name",
)

# backend/app/services/pilot_events.py -> two levels up is backend/.
_BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEFAULT_CSV_PATH = os.path.join(_BACKEND_DIR, "data", "raw", "glc_legacy.csv")
DEFAULT_SNAPSHOT_PATH = os.path.join(_BACKEND_DIR, "data", "models", "sikkim_events.json")

SNAPSHOT_SCHEMA_VERSION = "1.0.0"


def _classify(accuracy):
    """(is_precise, spatial_uncertainty_label) for a raw location_accuracy string."""
    is_precise = (accuracy or "").strip().lower() in PRECISE_ACCURACY
    return is_precise, "precise_lt_5km" if is_precise else "approximate_ge_5km"


def _within_aoi(aoi, lat, lon):
    return (aoi["min_lat"] <= lat <= aoi["max_lat"]
            and aoi["min_lon"] <= lon <= aoi["max_lon"])


def load_events_from_csv(csv_path=None):
    """
    Reproduce the canonical pilot positives directly from the raw GLC catalog.

    Filter is identical to scripts/train_real_models.py: AOI bbox from
    config_states.get_pilot_aoi_bounds -> drop rows with empty/nan event_date ->
    de-duplicate on (latitude, longitude, event_date). Standard library only.

    Returns (aoi, events | None, precise_count). events is None only when the CSV
    file itself is absent; an empty CSV yields ([], 0), not None.
    """
    aoi = get_pilot_aoi_bounds("Sikkim")
    path = csv_path or DEFAULT_CSV_PATH
    if not os.path.exists(path):
        return aoi, None, 0

    seen = set()
    events = []
    precise = 0
    with open(path, "r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            try:
                lat = float(row.get("latitude"))
                lon = float(row.get("longitude"))
            except (TypeError, ValueError):
                continue
            if not _within_aoi(aoi, lat, lon):
                continue
            event_date = (row.get("event_date") or "").strip()
            if event_date == "" or event_date.lower() == "nan":
                continue
            key = (lat, lon, event_date)
            if key in seen:
                continue
            seen.add(key)

            accuracy = (row.get("location_accuracy") or "").strip()
            is_precise, uncertainty = _classify(accuracy)
            if is_precise:
                precise += 1

            fatalities_raw = (row.get("fatality_count") or "").strip()
            try:
                fatalities = int(float(fatalities_raw)) if fatalities_raw != "" else None
            except (TypeError, ValueError):
                fatalities = None

            events.append({
                "latitude": lat,
                "longitude": lon,
                "event_date": event_date,
                "event_title": (row.get("event_title") or "").strip() or None,
                "landslide_category": (row.get("landslide_category") or "").strip() or None,
                "landslide_trigger": (row.get("landslide_trigger") or "").strip() or None,
                "landslide_size": (row.get("landslide_size") or "").strip() or None,
                "location_accuracy": accuracy or None,
                "fatality_count": fatalities,
                "spatial_uncertainty": uncertainty,
                "source_name": (row.get("source_name") or "").strip() or None,
            })
    return aoi, events, precise


def _coerce_event(rec):
    """
    Whitelist a snapshot record down to EVENT_FIELDS.

    Returns None when latitude/longitude cannot be parsed. The spatial_uncertainty
    label is ALWAYS recomputed from location_accuracy rather than trusted from the
    file, so a hand-edited or stale label can never contradict the accuracy value.
    """
    if not isinstance(rec, dict):
        return None
    try:
        lat = float(rec["latitude"])
        lon = float(rec["longitude"])
    except (KeyError, TypeError, ValueError):
        return None
    out = {k: rec.get(k) for k in EVENT_FIELDS}
    out["latitude"] = lat
    out["longitude"] = lon
    _, out["spatial_uncertainty"] = _classify(rec.get("location_accuracy"))
    return out


def load_events_from_snapshot(json_path=None):
    """
    Load the committed, validated events snapshot if present and well-formed.

    Returns (aoi, events | None, precise_count). events is None when the snapshot
    is absent or structurally invalid (so the caller can fall back to the CSV). The
    AOI returned is always the live canonical AOI, never a value read from the file,
    and any stored event outside that AOI is dropped rather than served.
    """
    aoi = get_pilot_aoi_bounds("Sikkim")
    path = json_path or DEFAULT_SNAPSHOT_PATH
    if not os.path.exists(path):
        return aoi, None, 0
    try:
        with open(path, "r", encoding="utf-8") as fh:
            doc = json.load(fh)
        raw_events = doc.get("events")
    except (ValueError, OSError):
        return aoi, None, 0
    if not isinstance(raw_events, list) or not raw_events:
        return aoi, None, 0

    events = []
    precise = 0
    for rec in raw_events:
        ev = _coerce_event(rec)
        if ev is None:
            continue
        if not _within_aoi(aoi, ev["latitude"], ev["longitude"]):
            continue
        if ev["spatial_uncertainty"] == "precise_lt_5km":
            precise += 1
        events.append(ev)
    if not events:
        return aoi, None, 0
    return aoi, events, precise


def resolve_pilot_events():
    """
    Resolve the pilot event geometry from the best available REAL source.

    Order: (1) committed validated snapshot, (2) raw GLC catalog. Returns
    (aoi, events | None, precise_count, source) where source is
    "validated_snapshot" | "raw_glc_catalog" | None. events is None only when BOTH
    sources are absent, in which case the caller must refuse with HTTP 503.
    """
    aoi, events, precise = load_events_from_snapshot()
    if events is not None:
        return aoi, events, precise, "validated_snapshot"
    aoi, events, precise = load_events_from_csv()
    if events is not None:
        return aoi, events, precise, "raw_glc_catalog"
    return aoi, None, 0, None


def spatial_uncertainty_summary(events, precise):
    """Summary block shared by the API response and the snapshot document."""
    approximate = len(events) - precise
    return {
        "precise_lt_5km": precise,
        "approximate_ge_5km": approximate,
        "pct_approximate_ge_5km": (
            round(100.0 * approximate / len(events), 1) if events else 0.0
        ),
    }


def build_snapshot_document(events, precise, aoi, generated_at=None):
    """
    Assemble the JSON document written to data/models/sikkim_events.json.

    A pure projection of `events`: it adds only provenance describing HOW the list
    was derived (source file + filter), never a coordinate or count that is not
    already implied by `events` itself.
    """
    return {
        "snapshot_schema_version": SNAPSHOT_SCHEMA_VERSION,
        "state": "Sikkim",
        "pilot_area": "East Sikkim",
        "aoi": aoi,
        "count": len(events),
        "source": (
            "NASA Global Landslide Catalog (glc_legacy.csv), AOI-filtered; "
            "de-duplicated on (latitude, longitude, event_date)."
        ),
        "derivation": {
            "raw_source_file": "backend/data/raw/glc_legacy.csv",
            "filter": (
                "canonical pilot AOI (config_states.get_pilot_aoi_bounds('Sikkim')) "
                "-> drop rows with empty/nan event_date -> de-duplicate on "
                "(latitude, longitude, event_date)"
            ),
            "generator": "backend/scripts/build_sikkim_events_snapshot.py",
            "precise_accuracy_values": sorted(PRECISE_ACCURACY),
            "generated_at": generated_at,
        },
        "spatial_uncertainty_summary": spatial_uncertainty_summary(events, precise),
        "events": events,
    }
