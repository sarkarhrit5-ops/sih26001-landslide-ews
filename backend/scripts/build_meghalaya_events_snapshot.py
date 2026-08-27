"""
Generate backend/data/models/meghalaya_events.json -- the committed snapshot of the
real NASA GLC landslide positives inside the canonical Meghalaya pilot AOI.

This mirrors scripts/build_arunachal_events_snapshot.py (which in turn mirrors the
Assam and Sikkim builders), and deliberately reuses the state-agnostic helpers in
app.services.pilot_events (the precise-accuracy set, the AOI containment test, the
spatial-uncertainty summary and the public event-field order) so the classification
rules cannot drift from the other pilots. It does NOT modify pilot_events or any
Sikkim / Assam / Arunachal artifact.

POSITIVES RULE (chosen for Meghalaya): identical to the other pilots -- every real
GLC landslide whose (lat, lon) falls inside the canonical Meghalaya pilot AOI is a
positive (pure bbox), after dropping rows with no event_date and de-duplicating on
(latitude, longitude, event_date). Because Meghalaya's pilot box (the East Khasi +
Jaintia Hills belt) clips small parts of neighbouring jurisdictions (Assam to the
north, and Bangladesh's Sylhet division to the south), this snapshot ALSO records,
per event, the catalog's admin_division_name and country_name, and an aggregate
jurisdiction_provenance block, so the reader can see how many positives are
administratively "Meghalaya" versus neighbouring states/countries. No record is
fabricated and none is dropped on jurisdiction grounds.

FILENAME NOTE (data-prep vs serving): the canonical snapshot filename is
'meghalaya_events.json', matching the state id ("meghalaya") and the sibling
artifacts. Unlike the Arunachal pilot -- whose two-word state name "Arunachal
Pradesh" forces a space-vs-underscore reconciliation in the serving resolver --
"Meghalaya" is a single word, so the serving resolver
app.services.pilot_events._snapshot_path_for_state("Meghalaya") derives exactly
"meghalaya_events.json", the same path this generator writes and the Meghalaya
data-prep consumers read. No path workaround is needed. Nothing here modifies that
shared resolver.

This script only READS the raw catalog and WRITES the snapshot JSON (CRLF, 2-space
indent, matching the sibling artifacts in backend/data/models/). It is pure
standard library (csv + json) and needs no network, so it is fully reproducible
offline.

Usage (from the backend/ directory):
    python scripts/build_meghalaya_events_snapshot.py
"""
import csv
import datetime as _dt
import json
import os
import sys
import unicodedata

# Make 'app' importable when run as a bare script.
_BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from app.core.config_states import get_pilot_aoi_bounds  # noqa: E402
from app.services.pilot_events import (  # noqa: E402  (reuse; do not re-implement)
    EVENT_FIELDS,
    PRECISE_ACCURACY,
    _classify,
    _within_aoi,
    spatial_uncertainty_summary,
)

STATE_NAME = "Meghalaya"
PILOT_AREA = "East Khasi + Jaintia Hills belt"
SNAPSHOT_SCHEMA_VERSION = "1.0.0"

DEFAULT_CSV_PATH = os.path.join(_BACKEND_DIR, "data", "raw", "glc_legacy.csv")
# Canonical filename. "Meghalaya" is a single word, so this equals the serving
# resolver's derived path (see module docstring FILENAME NOTE); written explicitly
# for symmetry with the sibling builders.
DEFAULT_SNAPSHOT_PATH = os.path.join(
    _BACKEND_DIR, "data", "models", "meghalaya_events.json"
)

# Extra provenance fields carried per event beyond the shared EVENT_FIELDS.
JURISDICTION_FIELDS = ("admin_division_name", "country_name")


def _norm(value):
    """Trim a raw string cell to a clean value or None."""
    return (value or "").strip() or None


def _is_meghalaya_admin(admin):
    """
    True if an admin_division_name refers to Meghalaya.

    The GLC catalog spells the division inconsistently (e.g. "Meghalaya" and the
    diacritic form "Meghalaya" with a macron on the second 'a'), so the match strips
    diacritics, lower-cases, and tests for the "meghalaya" substring. This affects
    ONLY the aggregate `labelled_meghalaya` tally; the full, verbatim distribution is
    always preserved in by_admin_division_name, so no spelling variant is hidden by
    this normalisation.
    """
    if not admin:
        return False
    folded = unicodedata.normalize("NFKD", admin)
    folded = "".join(ch for ch in folded if not unicodedata.combining(ch))
    return "meghalaya" in folded.strip().lower()


def load_events_from_csv(csv_path=None):
    """
    Reproduce the Meghalaya positives from the raw GLC catalog with the SAME filter
    as the other pilots (AOI bbox -> drop empty/nan event_date -> de-dup on
    lat/lon/date), additionally capturing admin_division_name and country_name.

    Returns (aoi, events | None, precise_count). events is None only when the CSV
    file itself is absent.
    """
    aoi = get_pilot_aoi_bounds(STATE_NAME)
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
                "event_title": _norm(row.get("event_title")),
                "landslide_category": _norm(row.get("landslide_category")),
                "landslide_trigger": _norm(row.get("landslide_trigger")),
                "landslide_size": _norm(row.get("landslide_size")),
                "location_accuracy": accuracy or None,
                "fatality_count": fatalities,
                "spatial_uncertainty": uncertainty,
                "source_name": _norm(row.get("source_name")),
                # Jurisdiction provenance (Meghalaya box clips neighbours; see docstring).
                "admin_division_name": _norm(row.get("admin_division_name")),
                "country_name": _norm(row.get("country_name")),
            })
    return aoi, events, precise


def independent_event_dates(events):
    """
    Count distinct event_date values among the positives. This is the figure the
    Option-A/Option-C gate cares about (independent event-dates), NOT the raw
    positive count; it is recorded here for provenance, computed straight from the
    events. Invents nothing.
    """
    return len({ev["event_date"] for ev in events if ev.get("event_date")})


def jurisdiction_provenance(events):
    """
    Aggregate the admin_division_name / country_name of the positives so the reader
    can see how many are administratively 'Meghalaya' vs. neighbouring
    jurisdictions. Pure projection of `events`; invents nothing.
    """
    by_admin = {}
    by_country = {}
    labelled_meghalaya = 0
    for ev in events:
        admin = ev.get("admin_division_name")
        country = ev.get("country_name")
        admin_key = admin if admin is not None else "<blank>"
        country_key = country if country is not None else "<blank>"
        by_admin[admin_key] = by_admin.get(admin_key, 0) + 1
        by_country[country_key] = by_country.get(country_key, 0) + 1
        if _is_meghalaya_admin(admin):
            labelled_meghalaya += 1

    def _sorted_desc(d):
        return dict(sorted(d.items(), key=lambda kv: (-kv[1], kv[0])))

    return {
        "labelled_meghalaya": labelled_meghalaya,
        "other_or_blank": len(events) - labelled_meghalaya,
        "by_admin_division_name": _sorted_desc(by_admin),
        "by_country_name": _sorted_desc(by_country),
    }


def build_snapshot_document(events, precise, aoi, generated_at=None):
    """Assemble the JSON document written to data/models/meghalaya_events.json."""
    return {
        "snapshot_schema_version": SNAPSHOT_SCHEMA_VERSION,
        "state": STATE_NAME,
        "pilot_area": PILOT_AREA,
        "aoi": aoi,
        "count": len(events),
        "independent_event_dates": independent_event_dates(events),
        "positives_rule": (
            "in-box (pure bounding box), identical to the Sikkim, Assam and "
            "Arunachal pilots; positives are NOT filtered by administrative "
            "jurisdiction -- the breakdown is recorded in jurisdiction_provenance "
            "instead."
        ),
        "source": (
            "NASA Global Landslide Catalog (glc_legacy.csv), AOI-filtered; "
            "de-duplicated on (latitude, longitude, event_date)."
        ),
        "derivation": {
            "raw_source_file": "backend/data/raw/glc_legacy.csv",
            "filter": (
                "canonical pilot AOI "
                "(config_states.get_pilot_aoi_bounds('Meghalaya')) -> drop "
                "rows with empty/nan event_date -> de-duplicate on "
                "(latitude, longitude, event_date)"
            ),
            "generator": "backend/scripts/build_meghalaya_events_snapshot.py",
            "event_fields": list(EVENT_FIELDS) + list(JURISDICTION_FIELDS),
            "precise_accuracy_values": sorted(PRECISE_ACCURACY),
            "generated_at": generated_at,
        },
        "jurisdiction_provenance": jurisdiction_provenance(events),
        "spatial_uncertainty_summary": spatial_uncertainty_summary(events, precise),
        "events": events,
    }


def main():
    aoi, events, precise = load_events_from_csv()
    if events is None:
        raise SystemExit(
            "Raw GLC catalog not found at %s -- cannot build snapshot." % DEFAULT_CSV_PATH
        )
    generated_at = _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    doc = build_snapshot_document(events, precise, aoi, generated_at=generated_at)

    out_path = DEFAULT_SNAPSHOT_PATH
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    # newline="\r\n" makes every "\n" json.dump writes a CRLF, matching the sibling
    # JSON artifacts in data/models/ regardless of the host platform.
    with open(out_path, "w", encoding="utf-8", newline="\r\n") as fh:
        json.dump(doc, fh, indent=2, ensure_ascii=False)
        fh.write("\n")

    prov = doc["jurisdiction_provenance"]
    summary = doc["spatial_uncertainty_summary"]
    print("Wrote %d Meghalaya-AOI events (%d independent event-dates; %d admin-labelled "
          "'Meghalaya' / %d other-or-blank; %d precise / %d approximate, %.1f%% "
          "approximate) to %s"
          % (doc["count"], doc["independent_event_dates"], prov["labelled_meghalaya"],
             prov["other_or_blank"], summary["precise_lt_5km"],
             summary["approximate_ge_5km"], summary["pct_approximate_ge_5km"], out_path))


if __name__ == "__main__":
    main()
