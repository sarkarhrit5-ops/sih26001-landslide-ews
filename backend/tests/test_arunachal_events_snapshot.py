"""
Focused offline regression for the ONE piece of novel logic in the Arunachal
Pradesh endpoints: the /validation/arunachal/events route resolves its committed
validated snapshot by passing an EXPLICIT json_path, working around a latent
filename bug in the shared pilot_events module WITHOUT modifying that module.

Why this test exists
--------------------
pilot_events._snapshot_path_for_state() derives the per-state snapshot filename by
lower-casing the state name:

    "<state>".strip().lower() + "_events.json"

For single-word states this is fine ("Assam" -> "assam_events.json", matching the
committed artifact, so the Assam route can just call resolve_pilot_events()). But
for the TWO-word "Arunachal Pradesh" it yields the SPACE form
"arunachal pradesh_events.json", whereas the committed artifact uses the canonical
UNDERSCORE form "arunachal_pradesh_events.json". So the default derived path does
NOT find the snapshot, and resolve_pilot_events() would silently skip the validated
snapshot and fall through to the raw CSV (or 503 if the CSV is absent too).

The route fixes this by composing the public pilot_events loaders directly with an
explicit json_path pointing at the underscore artifact -- preserving
resolve_pilot_events()'s snapshot-first / raw-CSV-fallback order and reporting
source_artifact="validated_snapshot" exactly like Sikkim/Assam.

These tests are stdlib-only (pilot_events imports only stdlib + config_states, and
risk_inputs is likewise import-safe offline), so they run in the offline shim
harness -- unlike the route itself, which needs fastapi. They pin BOTH halves: the
bug is real (the default path misses the snapshot) AND the route's explicit-path
recipe recovers the real 50-event validated inventory. If anyone later "fixes"
_snapshot_path_for_state to underscore-ify two-word states, tests (1)/(2) here
break and prompt revisiting the route so the workaround can be retired.
"""

import json
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.services import pilot_events
from app.services import risk_inputs


_STATE = "Arunachal Pradesh"


def _underscore_snapshot_path():
    """Derive the committed artifact path EXACTLY as get_arunachal_events() does."""
    return os.path.join(
        os.path.dirname(pilot_events.DEFAULT_SNAPSHOT_PATH),
        "arunachal_pradesh_events.json",
    )


# ---------------------------------------------------------------------------
# (1) The bug is real: the shared per-state derivation yields the SPACE filename,
#     which is NOT the committed underscore artifact and does not exist on disk.
# ---------------------------------------------------------------------------
def test_default_per_state_snapshot_path_has_the_space_bug():
    derived = pilot_events._snapshot_path_for_state(_STATE)
    assert os.path.basename(derived) == "arunachal pradesh_events.json"  # note the SPACE
    assert not os.path.exists(derived), (
        "the space-form path must NOT exist; if it does, the workaround is obsolete"
    )
    # ...whereas the canonical underscore artifact the route targets DOES exist.
    underscore = _underscore_snapshot_path()
    assert os.path.basename(underscore) == "arunachal_pradesh_events.json"
    assert os.path.exists(underscore), "committed validated snapshot must be present"


# ---------------------------------------------------------------------------
# (2) Consequence of the bug: the DEFAULT (json_path=None) snapshot load misses the
#     artifact, and the naive resolve_pilot_events() therefore does NOT report the
#     validated snapshot -- which is precisely why the route cannot use it verbatim.
# ---------------------------------------------------------------------------
def test_default_load_misses_snapshot_and_naive_resolver_does_not_report_it():
    _aoi, events, _precise = pilot_events.load_events_from_snapshot(state_name=_STATE)
    assert events is None, (
        "default derived path must miss the snapshot (space vs underscore)"
    )

    # The naive resolver falls through PAST the validated snapshot. It never reports
    # "validated_snapshot" for Arunachal -- it is either the raw CSV (if present in
    # this checkout) or None -- so the route must not rely on it.
    _aoi, _events, _precise, source = pilot_events.resolve_pilot_events(state_name=_STATE)
    assert source != "validated_snapshot"
    assert source in (None, "raw_glc_catalog")


# ---------------------------------------------------------------------------
# (3) The route's fix works: loading with the EXPLICIT underscore json_path recovers
#     the real validated inventory, AOI-filtered to the canonical Arunachal AOI.
# ---------------------------------------------------------------------------
def test_explicit_json_path_recovers_the_real_validated_snapshot():
    snapshot_path = _underscore_snapshot_path()
    aoi, events, precise = pilot_events.load_events_from_snapshot(
        json_path=snapshot_path, state_name=_STATE
    )

    # Real events came back (this is what lets the route set
    # source_artifact="validated_snapshot", at parity with Sikkim/Assam).
    assert events is not None and len(events) > 0
    assert 0 <= precise <= len(events)

    # The AOI is the LIVE canonical one for the state (never read from the file).
    assert aoi == pilot_events.get_pilot_aoi_bounds(_STATE)

    # Every served event lies inside the canonical Arunachal pilot AOI -- out-of-AOI
    # records are dropped, never served.
    for ev in events:
        assert risk_inputs.point_within_pilot_aoi(
            ev["latitude"], ev["longitude"], state_name=_STATE
        ), ev

    # No fabrication: the loader cannot return more events than the file holds.
    with open(snapshot_path, "r", encoding="utf-8") as fh:
        raw = json.load(fh)
    assert isinstance(raw.get("events"), list)
    assert len(events) <= len(raw["events"])


# ---------------------------------------------------------------------------
# (4) The summary block the route returns is computed from the real events, and is
#     internally consistent (precise + approximate == count).
# ---------------------------------------------------------------------------
def test_spatial_uncertainty_summary_is_consistent_with_the_served_events():
    _aoi, events, precise = pilot_events.load_events_from_snapshot(
        json_path=_underscore_snapshot_path(), state_name=_STATE
    )
    summary = pilot_events.spatial_uncertainty_summary(events, precise)
    assert summary["precise_lt_5km"] == precise
    assert summary["approximate_ge_5km"] == len(events) - precise
    assert summary["precise_lt_5km"] + summary["approximate_ge_5km"] == len(events)
