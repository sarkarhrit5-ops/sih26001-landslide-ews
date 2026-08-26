"""
Offline tests for app.services.pilot_events -- the read-only resolver behind
GET /api/v1/validation/sikkim/events.

These run under the out-of-repo shim (outputs/run_all.py) with pytest ABSENT: the
module imports only the standard library + app.core.config_states, so no fastapi /
pandas / ml stack is required. This is the first automated coverage of the
event-serving path (routes.py could never be imported offline).

What is asserted:
  * the raw GLC filter yields exactly the 82 canonical positives (18 precise / 64
    approximate / 78.0% approximate) recorded in sikkim_provenance.json;
  * the committed snapshot serves those same 82 events WITHOUT the raw CSV;
  * the snapshot is byte-faithful to the CSV (no fabricated or dropped record);
  * resolve order is snapshot -> raw CSV -> refuse; and the label / AOI invariants
    hold on stored data.
Tests that depend on a real data file skip honestly when it is absent rather than
passing vacuously.
"""
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.services import pilot_events  # noqa: E402

EXPECTED_TOTAL = 82
EXPECTED_PRECISE = 18
EXPECTED_APPROX = 64
EXPECTED_PCT_APPROX = 78.0

_REQUIRED_KEYS = set(pilot_events.EVENT_FIELDS)


def _assert_canonical(aoi, events, precise, source_label):
    assert events is not None, "%s returned None" % source_label
    assert len(events) == EXPECTED_TOTAL, (
        "%s served %d events, expected %d" % (source_label, len(events), EXPECTED_TOTAL)
    )
    assert precise == EXPECTED_PRECISE
    assert len(events) - precise == EXPECTED_APPROX
    summary = pilot_events.spatial_uncertainty_summary(events, precise)
    assert summary["pct_approximate_ge_5km"] == EXPECTED_PCT_APPROX
    for ev in events:
        assert _REQUIRED_KEYS.issubset(ev.keys())
        assert pilot_events._within_aoi(aoi, ev["latitude"], ev["longitude"])
        # the served label must always agree with the raw accuracy value
        _, expected_label = pilot_events._classify(ev["location_accuracy"])
        assert ev["spatial_uncertainty"] == expected_label


def test_csv_yields_82_canonical_events():
    """The raw-catalog filter reproduces the exact provenance-recorded inventory."""
    if not os.path.exists(pilot_events.DEFAULT_CSV_PATH):
        pytest.skip("raw GLC catalog (glc_legacy.csv) not present in this checkout")
    aoi, events, precise = pilot_events.load_events_from_csv()
    _assert_canonical(aoi, events, precise, "load_events_from_csv")


def test_snapshot_serves_82_offline():
    """The committed snapshot serves the same 82 events without the raw CSV."""
    if not os.path.exists(pilot_events.DEFAULT_SNAPSHOT_PATH):
        pytest.skip("events snapshot not generated (run scripts/build_sikkim_events_snapshot.py)")
    aoi, events, precise = pilot_events.load_events_from_snapshot()
    _assert_canonical(aoi, events, precise, "load_events_from_snapshot")


def test_snapshot_matches_csv_exactly():
    """The snapshot is a faithful materialisation of the CSV -- nothing invented."""
    if not (os.path.exists(pilot_events.DEFAULT_CSV_PATH)
            and os.path.exists(pilot_events.DEFAULT_SNAPSHOT_PATH)):
        pytest.skip("need both raw CSV and snapshot present to cross-check")
    _, csv_events, csv_precise = pilot_events.load_events_from_csv()
    _, snap_events, snap_precise = pilot_events.load_events_from_snapshot()
    assert csv_precise == snap_precise

    def _key(ev):
        return (ev["latitude"], ev["longitude"], ev["event_date"])

    assert sorted(map(_key, csv_events)) == sorted(map(_key, snap_events))
    # full-record equivalence on the whitelisted fields (order-independent)
    csv_by_key = {_key(e): e for e in csv_events}
    for ev in snap_events:
        ref = csv_by_key[_key(ev)]
        for field in pilot_events.EVENT_FIELDS:
            assert ev[field] == ref[field], "mismatch on %r for %r" % (field, _key(ev))


def test_coerce_event_recomputes_label_and_rejects_bad_coords():
    # stored label deliberately wrong -> must be recomputed from location_accuracy
    ev = pilot_events._coerce_event({
        "latitude": "27.5", "longitude": "88.5", "event_date": "01/01/2020",
        "location_accuracy": "1km", "spatial_uncertainty": "approximate_ge_5km",
    })
    assert ev is not None
    assert ev["latitude"] == 27.5 and ev["longitude"] == 88.5
    assert ev["spatial_uncertainty"] == "precise_lt_5km"
    # unparseable coordinates -> dropped
    assert pilot_events._coerce_event({"latitude": "n/a", "longitude": "88.5"}) is None
    assert pilot_events._coerce_event("not a dict") is None


def _write(path, text):
    with open(path, "w", encoding="utf-8", newline="") as fh:
        fh.write(text)


def test_snapshot_aoi_invariant_drops_out_of_area_rows(tmp_path):
    """A stored event outside the canonical AOI is dropped, not served."""
    snap = tmp_path / "snap.json"
    doc = {
        "events": [
            {"latitude": 27.5, "longitude": 88.5, "event_date": "01/01/2020",
             "location_accuracy": "1km"},                      # inside AOI -> kept
            {"latitude": 10.0, "longitude": 10.0, "event_date": "01/01/2020",
             "location_accuracy": "exact"},                    # outside AOI -> dropped
        ]
    }
    _write(str(snap), json.dumps(doc))
    _, events, precise = pilot_events.load_events_from_snapshot(json_path=str(snap))
    assert events is not None and len(events) == 1
    assert events[0]["latitude"] == 27.5
    assert precise == 1


def test_resolver_prefers_snapshot_then_csv_then_refuses(tmp_path, monkeypatch):
    """resolve_pilot_events(): snapshot wins; else CSV; else (None, source=None)."""
    snap = tmp_path / "events.json"
    csv_file = tmp_path / "glc.csv"

    _write(str(snap), json.dumps({
        "events": [{"latitude": 27.6, "longitude": 88.6, "event_date": "02/02/2021",
                    "location_accuracy": "exact"}]
    }))
    _write(str(csv_file),
           "latitude,longitude,event_date,location_accuracy\r\n"
           "27.5,88.5,01/01/2020 12:00:00 AM,1km\r\n")

    monkeypatch.setattr(pilot_events, "DEFAULT_SNAPSHOT_PATH", str(snap))
    monkeypatch.setattr(pilot_events, "DEFAULT_CSV_PATH", str(csv_file))

    # 1) snapshot present -> served from snapshot
    aoi, events, precise, source = pilot_events.resolve_pilot_events()
    assert source == "validated_snapshot"
    assert len(events) == 1 and events[0]["latitude"] == 27.6

    # 2) snapshot absent -> fall back to raw CSV
    monkeypatch.setattr(pilot_events, "DEFAULT_SNAPSHOT_PATH", str(tmp_path / "missing.json"))
    aoi, events, precise, source = pilot_events.resolve_pilot_events()
    assert source == "raw_glc_catalog"
    assert len(events) == 1 and events[0]["latitude"] == 27.5

    # 3) both absent -> refuse (caller raises 503)
    monkeypatch.setattr(pilot_events, "DEFAULT_CSV_PATH", str(tmp_path / "missing.csv"))
    aoi, events, precise, source = pilot_events.resolve_pilot_events()
    assert events is None
    assert source is None
    assert aoi == pilot_events.get_pilot_aoi_bounds("Sikkim")


def test_build_snapshot_document_is_pure_projection():
    """The snapshot doc reports only counts implied by the events passed in."""
    events = [
        {"latitude": 27.5, "longitude": 88.5, "event_date": "01/01/2020",
         "event_title": None, "landslide_category": None, "landslide_trigger": None,
         "landslide_size": None, "location_accuracy": "1km", "fatality_count": None,
         "spatial_uncertainty": "precise_lt_5km", "source_name": None},
        {"latitude": 27.6, "longitude": 88.6, "event_date": "02/02/2021",
         "event_title": None, "landslide_category": None, "landslide_trigger": None,
         "landslide_size": None, "location_accuracy": "25km", "fatality_count": None,
         "spatial_uncertainty": "approximate_ge_5km", "source_name": None},
    ]
    aoi = pilot_events.get_pilot_aoi_bounds("Sikkim")
    doc = pilot_events.build_snapshot_document(events, precise=1, aoi=aoi)
    assert doc["count"] == 2
    assert doc["spatial_uncertainty_summary"]["precise_lt_5km"] == 1
    assert doc["spatial_uncertainty_summary"]["approximate_ge_5km"] == 1
    assert doc["spatial_uncertainty_summary"]["pct_approximate_ge_5km"] == 50.0
    assert doc["events"] is events


# ---------------------------------------------------------------------------
# State parameterisation (2nd pilot = Assam). The resolver is now state-aware so a
# second pilot is served by the SAME code path with its own AOI + committed events
# snapshot; every assertion below also pins that the Sikkim DEFAULT stays unchanged.
# ---------------------------------------------------------------------------
ASSAM_INSIDE = (26.0, 92.0)     # inside the Assam AOI, OUTSIDE the Sikkim AOI
SIKKIM_INSIDE = (27.5, 88.5)    # inside the Sikkim AOI, OUTSIDE the Assam AOI


def test_default_state_is_sikkim():
    assert pilot_events.DEFAULT_STATE_NAME == "Sikkim"


def test_snapshot_path_is_sikkim_default_but_derived_for_other_states(monkeypatch):
    # Sikkim reads the module-level DEFAULT_SNAPSHOT_PATH *at call time* -- this is
    # exactly what keeps monkeypatch-based tests and existing callers working, so
    # overriding the constant must still be honoured for Sikkim.
    monkeypatch.setattr(pilot_events, "DEFAULT_SNAPSHOT_PATH", "/sentinel/sikkim.json")
    assert pilot_events._snapshot_path_for_state("Sikkim") == "/sentinel/sikkim.json"
    assert pilot_events._snapshot_path_for_state("sikkim") == "/sentinel/sikkim.json"
    assert pilot_events._snapshot_path_for_state("  SIKKIM  ") == "/sentinel/sikkim.json"

    # Any OTHER pilot derives <data/models>/<state>_events.json and is UNAFFECTED by
    # the Sikkim constant (case- and whitespace-insensitive).
    assert pilot_events._snapshot_path_for_state("Assam").endswith(
        os.path.join("data", "models", "assam_events.json")
    )
    assert pilot_events._snapshot_path_for_state("  Assam ").endswith(
        os.path.join("data", "models", "assam_events.json")
    )


def test_snapshot_loader_uses_state_selected_aoi(tmp_path):
    """The same snapshot file is filtered to whichever pilot AOI the state selects."""
    snap = tmp_path / "snap.json"
    _write(str(snap), json.dumps({"events": [
        {"latitude": ASSAM_INSIDE[0], "longitude": ASSAM_INSIDE[1],
         "event_date": "01/01/2020", "location_accuracy": "1km"},
        {"latitude": SIKKIM_INSIDE[0], "longitude": SIKKIM_INSIDE[1],
         "event_date": "01/01/2020", "location_accuracy": "1km"},
    ]}))

    aoi_a, events_a, _ = pilot_events.load_events_from_snapshot(
        json_path=str(snap), state_name="Assam")
    assert aoi_a == pilot_events.get_pilot_aoi_bounds("Assam")
    assert [(e["latitude"], e["longitude"]) for e in events_a] == [ASSAM_INSIDE]

    aoi_s, events_s, _ = pilot_events.load_events_from_snapshot(
        json_path=str(snap), state_name="Sikkim")
    assert aoi_s == pilot_events.get_pilot_aoi_bounds("Sikkim")
    assert [(e["latitude"], e["longitude"]) for e in events_s] == [SIKKIM_INSIDE]

    # Default (no state_name) MUST remain Sikkim.
    _, events_default, _ = pilot_events.load_events_from_snapshot(json_path=str(snap))
    assert [(e["latitude"], e["longitude"]) for e in events_default] == [SIKKIM_INSIDE]


def test_csv_loader_shares_file_but_filters_by_state_aoi(tmp_path):
    """The raw catalog is one shared file; only the AOI (hence kept rows) is per-state."""
    csv_file = tmp_path / "glc.csv"
    _write(str(csv_file),
           "latitude,longitude,event_date,location_accuracy\r\n"
           "%s,%s,01/01/2020 12:00:00 AM,1km\r\n"
           "%s,%s,01/01/2020 12:00:00 AM,1km\r\n"
           % (ASSAM_INSIDE[0], ASSAM_INSIDE[1], SIKKIM_INSIDE[0], SIKKIM_INSIDE[1]))

    aoi_a, events_a, _ = pilot_events.load_events_from_csv(
        csv_path=str(csv_file), state_name="Assam")
    assert aoi_a == pilot_events.get_pilot_aoi_bounds("Assam")
    assert [(e["latitude"], e["longitude"]) for e in events_a] == [ASSAM_INSIDE]

    _, events_s, _ = pilot_events.load_events_from_csv(
        csv_path=str(csv_file), state_name="Sikkim")
    assert [(e["latitude"], e["longitude"]) for e in events_s] == [SIKKIM_INSIDE]


def test_resolve_assam_from_committed_snapshot_offline():
    """resolve_pilot_events('Assam') serves the committed Assam snapshot, AOI-filtered."""
    path = pilot_events._snapshot_path_for_state("Assam")
    if not os.path.exists(path):
        pytest.skip("committed assam_events.json not present in this checkout")
    aoi, events, precise, source = pilot_events.resolve_pilot_events(state_name="Assam")
    assert source == "validated_snapshot"
    assert events and len(events) >= 1
    assert aoi == pilot_events.get_pilot_aoi_bounds("Assam")
    for ev in events:
        assert pilot_events._within_aoi(aoi, ev["latitude"], ev["longitude"])
        assert _REQUIRED_KEYS.issubset(ev.keys())
        _, expected_label = pilot_events._classify(ev["location_accuracy"])
        assert ev["spatial_uncertainty"] == expected_label
