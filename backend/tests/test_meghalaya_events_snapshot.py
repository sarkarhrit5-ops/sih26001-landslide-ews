"""
Focused offline regression for the Meghalaya events snapshot + serving path.

Contrast with test_arunachal_events_snapshot.py
-----------------------------------------------
The Arunachal tests pin a WORKAROUND: because "Arunachal Pradesh" is two words,
pilot_events._snapshot_path_for_state() lower-cases it to the SPACE form
"arunachal pradesh_events.json", which does not match the committed underscore
artifact, so its route must pass an explicit json_path.

"Meghalaya" is a SINGLE word, so no such workaround is needed and these tests pin
that simpler, healthier property directly: the shared per-state derivation yields
exactly the committed filename, the DEFAULT (json_path=None) load finds the
snapshot, and resolve_pilot_events() reports source="validated_snapshot" at full
parity with Sikkim/Assam -- so the Meghalaya events route can call the shared
resolver verbatim without any path hack.

These tests are stdlib-only (pilot_events imports only stdlib + config_states, and
risk_inputs is likewise import-safe offline), so they run in the offline shim
harness. They require the committed snapshot
backend/data/models/meghalaya_events.json (produced by
scripts/build_meghalaya_events_snapshot.py) to be present.
"""

import json
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.services import pilot_events
from app.services import risk_inputs


_STATE = "Meghalaya"


def _snapshot_path():
    """The committed artifact path, in the same data/models dir as the Sikkim one."""
    return os.path.join(
        os.path.dirname(pilot_events.DEFAULT_SNAPSHOT_PATH),
        "meghalaya_events.json",
    )


# ---------------------------------------------------------------------------
# (1) NO space bug: single-word state derives exactly the committed filename, and
#     that file exists on disk. (This is the inverse of the Arunachal space-bug test.)
# ---------------------------------------------------------------------------
def test_default_per_state_snapshot_path_is_the_committed_underscore_artifact():
    derived = pilot_events._snapshot_path_for_state(_STATE)
    assert os.path.basename(derived) == "meghalaya_events.json"  # no space, single word
    assert os.path.exists(derived), "committed validated snapshot must be present"


# ---------------------------------------------------------------------------
# (2) Because the derived path is correct, the DEFAULT snapshot load finds the real
#     inventory and the naive resolver reports it as the validated snapshot -- no
#     explicit json_path needed (unlike Arunachal).
# ---------------------------------------------------------------------------
def test_default_load_finds_snapshot_and_resolver_reports_validated_snapshot():
    _aoi, events, _precise = pilot_events.load_events_from_snapshot(state_name=_STATE)
    assert events is not None and len(events) > 0

    _aoi, _events, _precise, source = pilot_events.resolve_pilot_events(state_name=_STATE)
    assert source == "validated_snapshot"


# ---------------------------------------------------------------------------
# (3) The recovered inventory is AOI-consistent and not fabricated.
# ---------------------------------------------------------------------------
def test_default_load_recovers_the_real_validated_snapshot():
    aoi, events, precise = pilot_events.load_events_from_snapshot(state_name=_STATE)

    assert events is not None and len(events) > 0
    assert 0 <= precise <= len(events)

    # The AOI is the LIVE canonical one for the state (never read from the file).
    assert aoi == pilot_events.get_pilot_aoi_bounds(_STATE)

    # Every served event lies inside the canonical Meghalaya pilot AOI.
    for ev in events:
        assert risk_inputs.point_within_pilot_aoi(
            ev["latitude"], ev["longitude"], state_name=_STATE
        ), ev

    # No fabrication: the loader cannot return more events than the file holds.
    with open(_snapshot_path(), "r", encoding="utf-8") as fh:
        raw = json.load(fh)
    assert isinstance(raw.get("events"), list)
    assert len(events) <= len(raw["events"])


# ---------------------------------------------------------------------------
# (4) The committed snapshot's provenance is internally consistent: the jurisdiction
#     tallies and the spatial-uncertainty summary both add up to the event count,
#     and the independent-event-date count is <= the raw positive count (the reason
#     the pilot is Option-C by construction).
# ---------------------------------------------------------------------------
def test_snapshot_document_provenance_is_internally_consistent():
    with open(_snapshot_path(), "r", encoding="utf-8") as fh:
        doc = json.load(fh)

    count = doc["count"]
    assert count == len(doc["events"])

    prov = doc["jurisdiction_provenance"]
    assert prov["labelled_meghalaya"] + prov["other_or_blank"] == count
    assert sum(prov["by_admin_division_name"].values()) == count
    assert sum(prov["by_country_name"].values()) == count

    summary = doc["spatial_uncertainty_summary"]
    assert summary["precise_lt_5km"] + summary["approximate_ge_5km"] == count

    # Independent event-dates drive the Option-A/Option-C gate; must be <= positives
    # and > 0 for a non-empty snapshot.
    assert 0 < doc["independent_event_dates"] <= count


# ---------------------------------------------------------------------------
# (5) The summary the events route returns is computed from the real events, and is
#     internally consistent (precise + approximate == count).
# ---------------------------------------------------------------------------
def test_spatial_uncertainty_summary_is_consistent_with_the_served_events():
    _aoi, events, precise = pilot_events.load_events_from_snapshot(state_name=_STATE)
    summary = pilot_events.spatial_uncertainty_summary(events, precise)
    assert summary["precise_lt_5km"] == precise
    assert summary["approximate_ge_5km"] == len(events) - precise
    assert summary["precise_lt_5km"] + summary["approximate_ge_5km"] == len(events)
