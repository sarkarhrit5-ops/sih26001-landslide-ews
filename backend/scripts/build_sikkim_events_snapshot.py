"""
Generate backend/data/models/sikkim_events.json -- the committed, validated
snapshot of the 82 real NASA GLC landslide positives inside the canonical East
Sikkim pilot AOI.

The snapshot is a durable, dependency-light materialisation of the SAME filter the
serving endpoint (and scripts/train_real_models.py) apply to the raw GLC catalog,
so the read-only /api/v1/validation/sikkim/events endpoint can serve the real event
geometry without the 8.5 MB raw catalog having to be present on the server.

This script only READS the raw catalog and WRITES the snapshot JSON. It does not
touch the model, the training matrix, or any other artifact, and it never
fabricates a record. Re-run it whenever the raw catalog changes; the accompanying
test (tests/test_pilot_events.py) asserts the snapshot still matches the CSV.

The output is written with CRLF line endings and 2-space indentation to match the
sibling evidence artifacts in backend/data/models/.

Usage (from the backend/ directory):
    python scripts/build_sikkim_events_snapshot.py
"""
import datetime as _dt
import json
import os
import sys

# Make 'app' importable when run as a bare script.
_BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from app.services import pilot_events  # noqa: E402


def main():
    aoi, events, precise = pilot_events.load_events_from_csv()
    if events is None:
        raise SystemExit(
            "Raw GLC catalog not found at %s -- cannot build snapshot."
            % pilot_events.DEFAULT_CSV_PATH
        )
    generated_at = _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    doc = pilot_events.build_snapshot_document(events, precise, aoi, generated_at=generated_at)

    out_path = pilot_events.DEFAULT_SNAPSHOT_PATH
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    # newline="\r\n" makes every "\n" json.dump writes a CRLF, matching the sibling
    # JSON artifacts in data/models/ regardless of the host platform.
    with open(out_path, "w", encoding="utf-8", newline="\r\n") as fh:
        json.dump(doc, fh, indent=2, ensure_ascii=False)
        fh.write("\n")

    summary = doc["spatial_uncertainty_summary"]
    print("Wrote %d events (%d precise / %d approximate, %.1f%% approximate) to %s"
          % (doc["count"], summary["precise_lt_5km"], summary["approximate_ge_5km"],
             summary["pct_approximate_ge_5km"], out_path))


if __name__ == "__main__":
    main()
