"""
PUBLISH PILOT TERRAIN ARTIFACTS (host-only, one-off)

Companion to app/services/pilot_artifact_store.py. That module DOWNLOADS the five
terrain rasters per state on a fresh deployment; this script prepares what it downloads
from the copies that already exist on a machine that has run the prepare_* drivers.

Covers all FOUR states -- Assam, Arunachal Pradesh, Meghalaya and Sikkim -- i.e. 20
objects. Sikkim contributes its published sweep-family names (sikkim_dem.tif,
sikkim_<name>.tif); its byte-identical serving twins (east_sikkim_dem.tif,
real_<name>.tif) are NOT separate objects and get no manifest entry, because the runtime
places them as links to the same verified bytes. --verify prints that mapping.

It does two things and nothing else:

  1. Hashes each local raster and writes a manifest -- {filename: {bytes, sha256}} --
     which is what lets the runtime reject a truncated or corrupted download instead of
     leaving a partial file where the dashboard reads availability.
  2. Prints the upload commands for the storage backend you name. It does NOT upload,
     because that needs credentials this repo must never contain.

Nothing is written inside the repository tree except the manifest, and only when
--manifest points there. No .tif is copied, moved or staged for Git.

USAGE
    cd backend
    python scripts/publish_pilot_artifacts.py --manifest pilot_manifest.json
    python scripts/publish_pilot_artifacts.py --manifest pilot_manifest.json --verify
    python scripts/publish_pilot_artifacts.py --manifest /tmp/manifest.json --uploader aws \
        --destination s3://my-bucket/pilots

--verify re-reads an existing manifest and re-checks every entry against the local file
(size AND SHA-256) without rewriting anything; exit 0 only when all entries match.

Then upload the 20 rasters plus the manifest to the same prefix, make them public-read,
and set on the deployment:

    SIH_PILOT_ARTIFACT_BASE_URL=https://<your-public-prefix>
    SIH_PILOT_ARTIFACT_CACHE_DIR=/var/data/pilot-terrain   (if a persistent disk is mounted)

The base URL is never hard-coded anywhere in the codebase.
"""

import argparse
import json
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.services.pilot_artifact_store import (  # noqa: E402
    PILOT_ARTIFACT_STATES,
    SIKKIM_STATE_NAME,
    artifact_wiring,
    sha256_of_file,
)

UPLOADERS = {
    "aws": 'aws s3 cp "{path}" "{destination}/{filename}" --acl public-read',
    "rclone": 'rclone copyto "{path}" "{destination}/{filename}"',
    "gcloud": 'gcloud storage cp "{path}" "{destination}/{filename}"',
    "curl": 'curl -T "{path}" "{destination}/{filename}"',
}


def collect(states, data_dir=None):
    """[(state, feature, path)] for all five rasters of each state, in a stable order.

    Sikkim contributes its PUBLISHED (sweep-family) names only -- sikkim_dem.tif and
    sikkim_<name>.tif. Its serving twins (east_sikkim_dem.tif, real_<name>.tif) are
    byte-identical locally, so uploading them again would double 245 MB for no gain;
    the runtime places them as links to the same verified bytes instead. Listing them
    here would claim objects exist in storage that do not.
    """
    items = []
    for state_name in states:
        paths = artifact_wiring(state_name)["paths"](data_dir)
        for feature, path in sorted(paths.items()):
            items.append((state_name, feature, path))
    return items


def alias_report(states, data_dir=None):
    """[(alias_path, published_path)] for states that place extra links (Sikkim only)."""
    rows = []
    for state_name in states:
        wiring = artifact_wiring(state_name)
        aliases = wiring["aliases"](data_dir)
        if not aliases:
            continue
        published = wiring["paths"](data_dir)
        for feature, alias_path in sorted(aliases.items()):
            rows.append((alias_path, published[feature]))
    return rows


def build_manifest(items):
    """Hash what is present; report what is not. Returns (entries, missing)."""
    entries = {}
    missing = []
    for state_name, feature, path in items:
        filename = os.path.basename(path)
        if not (os.path.exists(path) and os.path.getsize(path) > 0):
            missing.append((state_name, feature, path))
            continue
        size = os.path.getsize(path)
        print("  hashing %-34s %9.1f MB" % (filename, size / (1024.0 * 1024.0)))
        entries[filename] = {"bytes": size, "sha256": sha256_of_file(path)}
    return entries, missing


def verify_manifest(entries, items):
    """
    Re-check every manifest entry against its local file. Returns
    (ok, [(filename, reason)]) -- reason is None for a match.

    This is the same size-then-SHA-256 comparison the runtime performs after a download,
    run against the source copies, so a manifest can never claim a digest the local
    raster does not actually have.
    """
    from app.services.pilot_artifact_store import verify_artifact
    by_filename = {os.path.basename(path): path for _s, _f, path in items}
    rows = []
    for filename in sorted(entries):
        path = by_filename.get(filename)
        if path is None:
            rows.append((filename, "no local artifact corresponds to this entry"))
            continue
        rows.append((filename, verify_artifact(path, entries[filename])))
    for filename in sorted(by_filename):
        if filename not in entries:
            rows.append((filename, "local artifact has no manifest entry"))
    return all(reason is None for _f, reason in rows), rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--manifest", required=True,
                        help="path to write the manifest JSON to (or read, with --verify)")
    parser.add_argument("--data-dir", default=None,
                        help="data root override (default: the backend's own)")
    parser.add_argument("--states", default=",".join(PILOT_ARTIFACT_STATES),
                        help="comma-separated subset of states")
    parser.add_argument("--verify", action="store_true",
                        help="re-check an existing manifest against the local files; "
                             "writes nothing")
    parser.add_argument("--uploader", choices=sorted(UPLOADERS), default=None,
                        help="print upload commands in this flavour")
    parser.add_argument("--destination", default="<destination-prefix>",
                        help="bucket/prefix used in the printed upload commands")
    args = parser.parse_args(argv)

    states = [name.strip() for name in args.states.split(",") if name.strip()]
    unknown = [name for name in states if name not in PILOT_ARTIFACT_STATES]
    if unknown:
        parser.error("unknown state(s): %s (known: %s)"
                     % (", ".join(unknown), ", ".join(PILOT_ARTIFACT_STATES)))

    items = collect(states, args.data_dir)
    manifest_path = os.path.abspath(args.manifest)

    if args.verify:
        from app.services.pilot_artifact_store import normalize_manifest
        with open(manifest_path, "r", encoding="utf-8") as handle:
            entries = normalize_manifest(json.load(handle))
        print("Verifying %d manifest entry/entries in %s\n" % (len(entries), manifest_path))
        ok, rows = verify_manifest(entries, items)
        for filename, reason in rows:
            print("  %-34s %s" % (filename, "OK" if reason is None else "FAIL: " + reason))
        aliases = alias_report(states, args.data_dir)
        if aliases:
            print("\nServing-name links placed at runtime (NOT uploaded, no manifest "
                  "entry):")
            for alias_path, published_path in aliases:
                print("  %-24s -> %s" % (os.path.basename(alias_path),
                                         os.path.basename(published_path)))
        print("\n%s" % ("ALL ENTRIES MATCH" if ok else "MANIFEST DOES NOT MATCH LOCAL FILES"))
        return 0 if ok else 1

    print("Hashing %d artifact(s) for: %s" % (len(items), ", ".join(states)))
    entries, missing = build_manifest(items)

    if missing:
        # Honest partial output: publishing a manifest for a raster you do not have would
        # make the runtime fail verification for no reason.
        print("\nNOT PRESENT LOCALLY (excluded from the manifest):")
        for state_name, feature, path in missing:
            print("  %-20s %-10s %s" % (state_name, feature, path))
        print("  Run scripts/prepare_<state>_terrain.py for these before publishing.")

    total = sum(entry["bytes"] for entry in entries.values())
    payload = {"artifacts": entries}
    os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")

    print("\nManifest: %s" % manifest_path)
    print("Artifacts: %d, total %.1f MB" % (len(entries), total / (1024.0 * 1024.0)))
    print("Suggested SIH_PILOT_ARTIFACT_MAX_TOTAL_MB >= %d"
          % (int(total / (1024.0 * 1024.0)) + 64))

    aliases = alias_report(states, args.data_dir)
    if aliases:
        print("\nServing-name links placed at runtime (NOT uploaded, no manifest entry):")
        for alias_path, published_path in aliases:
            print("  %-24s -> %s" % (os.path.basename(alias_path),
                                     os.path.basename(published_path)))

    if args.uploader:
        template = UPLOADERS[args.uploader]
        print("\nUpload commands (%s):" % args.uploader)
        by_filename = {os.path.basename(path): path for _s, _f, path in items}
        for filename in sorted(entries):
            print("  " + template.format(path=by_filename[filename],
                                         destination=args.destination.rstrip("/"),
                                         filename=filename))
        print("  " + template.format(path=manifest_path,
                                     destination=args.destination.rstrip("/"),
                                     filename=os.path.basename(manifest_path)))

    return 0 if entries and not missing else 1


if __name__ == "__main__":
    raise SystemExit(main())
