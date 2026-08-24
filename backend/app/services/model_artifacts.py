"""
Canonical model-artifact persistence for the East Sikkim (Sikkim) pilot.

WHY THIS MODULE EXISTS
----------------------
`app.services.state_validation` only reports a pilot as VALIDATED_PILOT when
real, persisted validation evidence exists on disk. That gate
(`state_validation._evidence_paths` / `load_validation_evidence`) reads three
non-empty files from ``backend/data/models/``:

    sikkim_model.pkl
    sikkim_metrics.json           (must contain validation_metrics."PR-AUC" / "ROC-AUC")
    sikkim_feature_schema.json

Nothing in this repository previously WROTE those files, so the gate was
permanently starved of evidence and Sikkim stayed VALIDATION_REQUIRED. This
module is the writer -- and a strict reader -- for exactly that contract. It adds
one further, gate-independent reproducibility record, ``sikkim_provenance.json``.

HARD RULES ENFORCED HERE
------------------------
* Artifacts are only ever written from values produced by a real run. This module
  never invents, defaults, zero-fills or back-fills a metric, a model, or a
  provenance claim.
* Documentary / historical reference numbers (see
  `app.models.ml_pipeline.DOCUMENTARY_REFERENCE_METRICS`) can never be laundered
  into computed evidence: the caller must explicitly declare the metric source,
  and a metric block that is value-identical to a known documentary block is
  refused.
* The loader has NO fallback. Absent evidence -> MISSING. Unreadable or
  structurally invalid evidence -> INVALID. It never substitutes a hardcoded
  model or hardcoded metrics.
* The directory and the three gate filenames are FIXED by the existing gate and
  must not be changed here.
* Writes are staged then atomically moved, with the gate-critical metrics file
  moved LAST, so the gate can never observe a half-written evidence set as
  complete.
* Deliberately dependency-light (stdlib + optional joblib/pickle) so it can be
  imported and unit-tested without lightgbm / sklearn / rasterio / pandas.
"""

import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone

# ---------------------------------------------------------------------------
# Canonical contract
# ---------------------------------------------------------------------------
DEFAULT_STATE_NAME = "Sikkim"

# Mirrors app.services.state_validation._evidence_paths EXACTLY. If that gate is
# ever changed, this must be changed with it (and vice versa).
ARTIFACT_FILENAME_TEMPLATES = {
    "model": "{state}_model.pkl",
    "metrics": "{state}_metrics.json",
    "schema": "{state}_feature_schema.json",
    "provenance": "{state}_provenance.json",
}

# The three artifacts the existing validation gate actually requires.
GATE_REQUIRED_ARTIFACTS = ("model", "metrics", "schema")
# Everything this module writes (provenance is additive; the gate ignores it).
ALL_ARTIFACTS = ("model", "metrics", "schema", "provenance")

# Metric keys. The first two are what the gate requires; all six are what the
# existing compute_metrics() in ml_pipeline actually produces.
GATE_REQUIRED_METRIC_KEYS = ("PR-AUC", "ROC-AUC")
FULL_METRIC_KEYS = (
    "PR-AUC", "ROC-AUC", "Precision", "Recall", "F1", "False Alarm Rate",
)

METRICS_SCHEMA_VERSION = "1.0.0"
FEATURE_SCHEMA_VERSION = "1.0.0"
PROVENANCE_SCHEMA_VERSION = "1.0.0"

# The ONLY accepted declaration that a metric block came from a live validation
# run. Anything else (including omission) is refused by the writer.
COMPUTED_METRICS_SOURCE = "computed_from_validation_run"

# Allowed vocabulary for provenance input_status entries.
INPUT_STATUS_VALUES = ("REAL", "UNAVAILABLE", "NOT_USED", "DERIVED_PROXY")

ARTIFACT_STATUS_VALID = "VALID"
ARTIFACT_STATUS_MISSING = "MISSING"
ARTIFACT_STATUS_INVALID = "INVALID"

# Human-readable meanings for the features the training pipeline currently uses.
# These are DESCRIPTIONS ONLY -- no performance numbers, no validation claims.
FEATURE_MEANINGS = {
    "elevation": "Terrain elevation in metres, sampled from the Copernicus GLO-30 DEM.",
    "slope": "Terrain slope in degrees, derived from the DEM.",
    "aspect": "Terrain aspect in degrees clockwise from north, derived from the DEM.",
    "roughness": "Local terrain roughness, derived from the DEM.",
    "tpi": "Topographic Position Index, derived from the DEM.",
    "land_cover_class": (
        "DERIVED PROXY (not an observed land-cover product): elevation-binned class "
        "1 = tree cover / dense forest (<3000 m), 2 = shrubland / alpine scrub "
        "(3000-4200 m), 3 = bare rock / sparse vegetation / snow (>4200 m)."
    ),
    "rain_1d": "Precipitation total on day T-1 in mm (antecedent, no future leakage).",
    "rain_3d": "Cumulative precipitation over T-3..T-1 in mm.",
    "rain_7d": "Cumulative precipitation over T-7..T-1 in mm.",
    "antecedent_rain_14d": "Cumulative precipitation over T-14..T-1 in mm.",
    "rain_intensity_max_3d": "Maximum single-day precipitation over T-3..T-1 in mm/day.",
}

_UNDOCUMENTED_FEATURE_MEANING = (
    "UNDOCUMENTED: no description is recorded for this feature in "
    "model_artifacts.FEATURE_MEANINGS."
)


class ArtifactPersistenceError(RuntimeError):
    """Raised when validation artifacts cannot be persisted honestly."""


class ArtifactValidationError(ArtifactPersistenceError):
    """Raised when a proposed artifact bundle fails its integrity checks."""


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
def default_artifact_dir():
    """backend/data/models -- the directory the existing validation gate reads."""
    return os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "data", "models")
    )


def canonical_artifact_paths(state_name=DEFAULT_STATE_NAME, base_dir=None):
    """
    Returns {artifact_kind: absolute_path} for a state.

    The 'model', 'metrics' and 'schema' entries are byte-for-byte the same paths
    that state_validation._evidence_paths() checks.
    """
    clean_state_name = state_name.lower().replace(" ", "_")
    if base_dir is None:
        base_dir = default_artifact_dir()
    base_dir = os.path.abspath(base_dir)
    return {
        kind: os.path.join(base_dir, template.format(state=clean_state_name))
        for kind, template in ARTIFACT_FILENAME_TEMPLATES.items()
    }


# ---------------------------------------------------------------------------
# Small honest helpers
# ---------------------------------------------------------------------------
def _utc_timestamp():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def get_git_sha(repo_dir=None):
    """
    Returns the current commit SHA, or the literal string "UNKNOWN" when it cannot
    be determined. "UNKNOWN" is an honest label, not a fabricated value.
    """
    if repo_dir is None:
        repo_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_dir, capture_output=True, text=True, timeout=15, check=False,
        )
        sha = (out.stdout or "").strip()
        return sha if out.returncode == 0 and sha else "UNKNOWN"
    except Exception:
        return "UNKNOWN"


def collect_software_versions(module_names=None):
    """
    Reports the installed version of each module, or "NOT_INSTALLED" when absent.
    Never guesses a version.
    """
    if module_names is None:
        module_names = ("numpy", "pandas", "sklearn", "lightgbm", "joblib",
                        "rasterio", "pyarrow", "requests")
    versions = {"python": sys.version.split()[0]}
    for name in module_names:
        try:
            mod = __import__(name)
            versions[name] = str(getattr(mod, "__version__", "UNKNOWN_VERSION"))
        except Exception:
            versions[name] = "NOT_INSTALLED"
    return versions


def _is_real_number(value):
    if isinstance(value, bool) or value is None:
        return False
    if not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


# ---------------------------------------------------------------------------
# Documentary-metric quarantine
# ---------------------------------------------------------------------------
def documentary_metric_blocks():
    """
    Returns the known documentary/historical metric blocks so the writer can
    refuse to persist them as computed evidence.

    Deliberately loaded LAZILY from ml_pipeline instead of being duplicated here,
    so no historical performance number is hardcoded into the artifact layer. If
    ml_pipeline cannot be imported (its scientific dependencies are absent), this
    returns an empty list and the explicit metrics_source declaration remains the
    active safeguard.
    """
    try:
        from app.models.ml_pipeline import DOCUMENTARY_REFERENCE_METRICS
    except Exception:
        return []
    blocks = []
    holdout = DOCUMENTARY_REFERENCE_METRICS.get("temporal_holdout_metrics", {})
    if isinstance(holdout, dict):
        for block in holdout.values():
            if isinstance(block, dict):
                blocks.append(block)
    return blocks


def _matches_documentary_block(metrics, documentary_blocks):
    """
    True only when every shared metric key matches a documentary block on at
    least the two gate-required keys. A single coincidental value is NOT enough --
    we refuse whole-block reuse, not legitimate numbers that happen to collide.
    """
    for block in documentary_blocks or []:
        if not isinstance(block, dict):
            continue
        shared = [k for k in block if k in metrics]
        if len(shared) < len(GATE_REQUIRED_METRIC_KEYS):
            continue
        if not all(k in shared for k in GATE_REQUIRED_METRIC_KEYS):
            continue
        if all(float(block[k]) == float(metrics[k]) for k in shared):
            return True
    return False


# ---------------------------------------------------------------------------
# Document builders (pure functions -- unit-testable with no scientific deps)
# ---------------------------------------------------------------------------
def build_metrics_document(
    validation_metrics,
    primary_model_name,
    primary_evaluation,
    model_comparison=None,
    sample_counts=None,
    decision=None,
    feature_set=None,
    holdout_details=None,
    dataset_provenance_reference=None,
    metrics_source=COMPUTED_METRICS_SOURCE,
    generated_at=None,
):
    """
    Assembles sikkim_metrics.json.

    `validation_metrics` MUST be the dict returned by
    ml_pipeline.compute_metrics() for the primary evaluation of the run. It is
    copied verbatim -- no rounding, rescaling, or substitution happens here.
    """
    doc = {
        "metrics_schema_version": METRICS_SCHEMA_VERSION,
        "state": DEFAULT_STATE_NAME,
        "pilot_area": "East Sikkim",
        # Gate-critical block. state_validation reads doc["validation_metrics"].
        "validation_metrics": dict(validation_metrics or {}),
        "metrics_source": metrics_source,
        "status": "REAL/COMPUTED",
        "primary_model": primary_model_name,
        "primary_evaluation": primary_evaluation,
        "feature_set": feature_set,
        "holdout_details": holdout_details or {},
        "sample_counts": sample_counts or {},
        "model_comparison": model_comparison or {},
        "model_decision": decision or {},
        "dataset_provenance_reference": dataset_provenance_reference,
        "generated_at": generated_at or _utc_timestamp(),
        "note": (
            "All numbers in validation_metrics and model_comparison were computed by "
            "ml_pipeline.compute_metrics() during the run identified by "
            "dataset_provenance_reference and the companion provenance artifact. They "
            "are not copied from any documentary/historical reference figures."
        ),
    }
    return doc


def build_feature_schema_document(
    feature_names,
    dtypes=None,
    meanings=None,
    feature_set_name=None,
    target_column="target",
    generated_at=None,
):
    """
    Assembles sikkim_feature_schema.json from the feature list ACTUALLY handed to
    the persisted model. `feature_names` order is authoritative and preserved.
    """
    names = list(feature_names or [])
    resolved_meanings = {}
    resolved_dtypes = {}
    for name in names:
        if meanings and name in meanings:
            resolved_meanings[name] = meanings[name]
        else:
            resolved_meanings[name] = FEATURE_MEANINGS.get(
                name, _UNDOCUMENTED_FEATURE_MEANING
            )
        resolved_dtypes[name] = str((dtypes or {}).get(name, "UNKNOWN"))
    return {
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "state": DEFAULT_STATE_NAME,
        "feature_set_name": feature_set_name,
        "feature_names": names,
        # Explicit positional contract for inference-time column ordering.
        "feature_order": list(range(len(names))),
        "n_features": len(names),
        "dtype": resolved_dtypes,
        "meaning": resolved_meanings,
        "target_column": target_column,
        "generated_at": generated_at or _utc_timestamp(),
        "note": (
            "Captured from the exact feature matrix used to fit the persisted model. "
            "Inference must supply these features in this order."
        ),
    }


def build_provenance_document(
    aoi,
    model_type,
    model_hyperparameters,
    feature_list,
    random_seed,
    input_status,
    sample_counts=None,
    glc_source=None,
    glc_event_count=None,
    rainfall_source=None,
    dem_source=None,
    terrain_derivative_method=None,
    exposure_source=None,
    spatial_split=None,
    temporal_split=None,
    negative_sampling=None,
    leakage_controls=None,
    dataset_artifact=None,
    model_serialization=None,
    state=DEFAULT_STATE_NAME,
    pilot_area="East Sikkim",
    code_version=None,
    software_versions=None,
    generated_at=None,
    extra=None,
):
    """
    Assembles sikkim_provenance.json -- everything needed to reproduce the run and
    to state the reality status of every input. All values must be supplied by the
    caller from the live run; this function does not infer scientific facts.
    """
    doc = {
        "provenance_schema_version": PROVENANCE_SCHEMA_VERSION,
        "state": state,
        "pilot_area": pilot_area,
        "aoi": aoi,
        "glc_source": glc_source,
        "glc_event_count": glc_event_count,
        "sample_counts": sample_counts or {},
        "rainfall_source": rainfall_source,
        "dem_source": dem_source,
        "terrain_derivative_method": terrain_derivative_method,
        "exposure_source": exposure_source,
        "model_type": model_type,
        "model_hyperparameters": dict(model_hyperparameters or {}),
        "model_serialization": model_serialization,
        "feature_list": list(feature_list or []),
        "spatial_split": spatial_split,
        "temporal_split": temporal_split,
        "negative_sampling": negative_sampling,
        "leakage_controls": leakage_controls or {},
        "random_seed": random_seed,
        "dataset_artifact": dataset_artifact,
        "code_version": code_version if code_version is not None else get_git_sha(),
        "software_versions": software_versions or collect_software_versions(),
        "input_status": dict(input_status or {}),
        "generation_timestamp": generated_at or _utc_timestamp(),
    }
    if extra:
        doc["additional_context"] = extra
    return doc


# ---------------------------------------------------------------------------
# Validation (the success gate)
# ---------------------------------------------------------------------------
def validate_metrics_document(doc, documentary_blocks=None, require_full_metrics=True):
    """Returns a list of problem strings; empty list means acceptable."""
    problems = []
    if not isinstance(doc, dict):
        return ["metrics: document is not a JSON object"]

    metrics = doc.get("validation_metrics")
    if not isinstance(metrics, dict) or not metrics:
        return ["metrics: 'validation_metrics' block is missing or empty"]

    required = FULL_METRIC_KEYS if require_full_metrics else GATE_REQUIRED_METRIC_KEYS
    for key in required:
        if key not in metrics:
            problems.append("metrics: missing required metric '%s'" % key)
        elif not _is_real_number(metrics[key]):
            problems.append(
                "metrics: metric '%s' is not a finite number (%r)" % (key, metrics[key])
            )

    if doc.get("metrics_source") != COMPUTED_METRICS_SOURCE:
        problems.append(
            "metrics: 'metrics_source' must be %r to certify the numbers came from a "
            "live validation run (got %r)" % (COMPUTED_METRICS_SOURCE, doc.get("metrics_source"))
        )

    if documentary_blocks is None:
        documentary_blocks = documentary_metric_blocks()
    numeric_metrics = {k: v for k, v in metrics.items() if _is_real_number(v)}
    if numeric_metrics and _matches_documentary_block(numeric_metrics, documentary_blocks):
        problems.append(
            "metrics: validation_metrics is value-identical to a known "
            "documentary/historical reference block; documentary figures must never "
            "be persisted as computed validation evidence"
        )

    if not doc.get("primary_model"):
        problems.append("metrics: 'primary_model' is not recorded")
    if not doc.get("primary_evaluation"):
        problems.append("metrics: 'primary_evaluation' is not recorded")
    return problems


def validate_feature_schema_document(doc):
    problems = []
    if not isinstance(doc, dict):
        return ["schema: document is not a JSON object"]

    names = doc.get("feature_names")
    if not isinstance(names, list) or not names:
        return ["schema: 'feature_names' is missing or empty"]
    if not all(isinstance(n, str) and n for n in names):
        problems.append("schema: 'feature_names' must be a list of non-empty strings")
    if len(set(names)) != len(names):
        problems.append("schema: 'feature_names' contains duplicates")
    if doc.get("n_features") != len(names):
        problems.append(
            "schema: 'n_features' (%r) does not match len(feature_names) (%d)"
            % (doc.get("n_features"), len(names))
        )
    order = doc.get("feature_order")
    if not isinstance(order, list) or len(order) != len(names):
        problems.append("schema: 'feature_order' must be a list parallel to feature_names")
    for block_name in ("dtype", "meaning"):
        block = doc.get(block_name)
        if not isinstance(block, dict):
            problems.append("schema: '%s' block is missing" % block_name)
            continue
        missing = [n for n in names if n not in block]
        if missing:
            problems.append(
                "schema: '%s' block does not cover feature(s): %s"
                % (block_name, ", ".join(missing))
            )
    if not doc.get("feature_schema_version"):
        problems.append("schema: 'feature_schema_version' is not recorded")
    return problems


def validate_provenance_document(doc):
    problems = []
    if not isinstance(doc, dict):
        return ["provenance: document is not a JSON object"]

    required = ("state", "aoi", "model_type", "feature_list", "random_seed",
                "generation_timestamp", "input_status", "code_version")
    for key in required:
        if doc.get(key) in (None, "", [], {}):
            problems.append("provenance: required field '%s' is missing or empty" % key)

    aoi = doc.get("aoi")
    if isinstance(aoi, dict):
        for key in ("min_lat", "max_lat", "min_lon", "max_lon"):
            if not _is_real_number(aoi.get(key)):
                problems.append("provenance: aoi.%s is missing or not numeric" % key)
    elif aoi is not None:
        problems.append("provenance: 'aoi' must be an object with lat/lon bounds")

    status = doc.get("input_status")
    if isinstance(status, dict):
        for name, value in status.items():
            if value not in INPUT_STATUS_VALUES:
                problems.append(
                    "provenance: input_status[%r] = %r is not one of %s"
                    % (name, value, ", ".join(INPUT_STATUS_VALUES))
                )
    return problems


def validate_model_object(model):
    problems = []
    if model is None:
        return ["model: no fitted estimator was supplied"]
    if not callable(getattr(model, "predict_proba", None)):
        problems.append(
            "model: object does not expose a callable predict_proba(); it does not "
            "look like a fitted classifier"
        )
    return problems


def validate_evidence_bundle(model, metrics_doc, schema_doc, provenance_doc,
                             documentary_blocks=None, require_full_metrics=True):
    """
    The complete pre-write success gate. Returns a list of problems; an empty list
    means the bundle may be persisted.

    Also cross-checks that the feature list recorded in the schema and the
    provenance agree, so the persisted model, its reported metrics and its feature
    contract cannot describe different things.
    """
    problems = []
    problems.extend(validate_model_object(model))
    problems.extend(validate_metrics_document(
        metrics_doc, documentary_blocks=documentary_blocks,
        require_full_metrics=require_full_metrics,
    ))
    problems.extend(validate_feature_schema_document(schema_doc))
    problems.extend(validate_provenance_document(provenance_doc))

    schema_names = (schema_doc or {}).get("feature_names") if isinstance(schema_doc, dict) else None
    prov_names = (provenance_doc or {}).get("feature_list") if isinstance(provenance_doc, dict) else None
    if isinstance(schema_names, list) and isinstance(prov_names, list) and schema_names != prov_names:
        problems.append(
            "bundle: feature list disagreement between feature schema and provenance "
            "(%r vs %r)" % (schema_names, prov_names)
        )
    return problems


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------
def _dump_model(model, path):
    """
    Serializes the estimator, preferring joblib (as declared in requirements.txt)
    and falling back to stdlib pickle. Returns the serializer name so it can be
    recorded in provenance -- the mechanism is never left implicit.
    NOTE: the '.pkl' extension is mandated by the existing validation gate.
    """
    try:
        import joblib
    except Exception:
        import pickle
        with open(path, "wb") as fh:
            pickle.dump(model, fh, protocol=4)
        return "pickle"
    joblib.dump(model, path)
    return "joblib"


def _load_model(path):
    try:
        import joblib
    except Exception:
        import pickle
        with open(path, "rb") as fh:
            return pickle.load(fh)
    return joblib.load(path)


def _write_json(doc, path):
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=2, sort_keys=False)
        fh.write("\n")


# ---------------------------------------------------------------------------
# Save (atomic, success-gated)
# ---------------------------------------------------------------------------
def save_model_evidence(
    model,
    metrics_doc,
    schema_doc,
    provenance_doc,
    state_name=DEFAULT_STATE_NAME,
    base_dir=None,
    documentary_blocks=None,
    require_full_metrics=True,
):
    """
    Validates and then atomically persists the four artifacts.

    Raises ArtifactValidationError -- writing NOTHING -- if any part of the bundle
    fails the success gate. A failed or partial run therefore cannot leave behind
    a misleading validation artifact.

    Atomicity: all four files are written into a staging directory inside the
    target directory, re-read and re-validated from disk, then moved into place
    with os.replace(). The gate-critical metrics file is moved LAST, so any
    interrupted write leaves the gate reporting an honest "incomplete" state
    rather than a spuriously complete one.

    Returns {artifact_kind: path} on success.
    """
    problems = validate_evidence_bundle(
        model, metrics_doc, schema_doc, provenance_doc,
        documentary_blocks=documentary_blocks,
        require_full_metrics=require_full_metrics,
    )
    if problems:
        raise ArtifactValidationError(
            "Refusing to persist validation artifacts; the evidence bundle is not "
            "trustworthy:\n  - " + "\n  - ".join(problems)
        )

    paths = canonical_artifact_paths(state_name, base_dir)
    target_dir = os.path.dirname(paths["metrics"])
    os.makedirs(target_dir, exist_ok=True)

    staging = tempfile.mkdtemp(prefix=".staging_artifacts_", dir=target_dir)
    try:
        staged = {kind: os.path.join(staging, os.path.basename(paths[kind]))
                  for kind in ALL_ARTIFACTS}

        serializer = _dump_model(model, staged["model"])
        provenance_doc = dict(provenance_doc)
        provenance_doc.setdefault("model_serialization", None)
        if not provenance_doc.get("model_serialization"):
            provenance_doc["model_serialization"] = serializer

        _write_json(metrics_doc, staged["metrics"])
        _write_json(schema_doc, staged["schema"])
        _write_json(provenance_doc, staged["provenance"])

        # Re-read from disk and re-validate before anything becomes visible.
        for kind in ALL_ARTIFACTS:
            if not (os.path.exists(staged[kind]) and os.path.getsize(staged[kind]) > 0):
                raise ArtifactPersistenceError(
                    "Staged artifact '%s' is missing or empty; aborting without "
                    "publishing any file." % kind
                )
        with open(staged["metrics"], "r", encoding="utf-8") as fh:
            reread_metrics = json.load(fh)
        with open(staged["schema"], "r", encoding="utf-8") as fh:
            reread_schema = json.load(fh)
        with open(staged["provenance"], "r", encoding="utf-8") as fh:
            reread_provenance = json.load(fh)
        reread_problems = (
            validate_metrics_document(
                reread_metrics, documentary_blocks=documentary_blocks,
                require_full_metrics=require_full_metrics)
            + validate_feature_schema_document(reread_schema)
            + validate_provenance_document(reread_provenance)
        )
        if reread_problems:
            raise ArtifactPersistenceError(
                "Staged artifacts failed re-validation after write; publishing "
                "nothing:\n  - " + "\n  - ".join(reread_problems)
            )

        # Publish. 'metrics' is deliberately last (gate-critical).
        for kind in ("provenance", "model", "schema", "metrics"):
            os.replace(staged[kind], paths[kind])
    finally:
        shutil.rmtree(staging, ignore_errors=True)

    return paths


# ---------------------------------------------------------------------------
# Load / verify (no fallbacks, ever)
# ---------------------------------------------------------------------------
def verify_artifact_set(state_name=DEFAULT_STATE_NAME, base_dir=None,
                        require_full_metrics=False):
    """
    Classifies the on-disk artifact set WITHOUT deserializing the model.

    status is one of:
      VALID   -- all gate-required artifacts present, non-empty, structurally sound
      MISSING -- one or more gate-required artifacts absent or zero-length
      INVALID -- all present but at least one is unreadable or structurally wrong

    `provenance` is reported separately: it is additive and its absence never
    downgrades the gate-required verdict.

    Note: `require_full_metrics` defaults to False here so that verification
    mirrors the existing gate (which requires only PR-AUC and ROC-AUC) rather
    than being stricter than the thing it reports on.
    """
    paths = canonical_artifact_paths(state_name, base_dir)
    result = {
        "status": ARTIFACT_STATUS_MISSING,
        "paths": paths,
        "missing": [],
        "problems": [],
        "metrics": None,
        "feature_schema": None,
        "provenance": None,
        "provenance_present": False,
        "gate_compatible": False,
    }

    missing = [
        kind for kind in GATE_REQUIRED_ARTIFACTS
        if not (os.path.exists(paths[kind]) and os.path.getsize(paths[kind]) > 0)
    ]
    result["missing"] = missing
    if missing:
        result["status"] = ARTIFACT_STATUS_MISSING
        result["problems"].append(
            "Missing or empty gate-required artifact(s): " + ", ".join(missing)
        )
        return result

    docs = {}
    for kind in ("metrics", "schema"):
        try:
            with open(paths[kind], "r", encoding="utf-8") as fh:
                docs[kind] = json.load(fh)
        except Exception as exc:
            result["status"] = ARTIFACT_STATUS_INVALID
            result["problems"].append("%s: unreadable (%s)" % (kind, exc))
            return result

    problems = validate_metrics_document(
        docs["metrics"], require_full_metrics=require_full_metrics
    )
    problems.extend(validate_feature_schema_document(docs["schema"]))

    if os.path.exists(paths["provenance"]) and os.path.getsize(paths["provenance"]) > 0:
        result["provenance_present"] = True
        try:
            with open(paths["provenance"], "r", encoding="utf-8") as fh:
                result["provenance"] = json.load(fh)
        except Exception as exc:
            result["problems"].append("provenance: unreadable (%s)" % exc)

    result["metrics"] = docs["metrics"]
    result["feature_schema"] = docs["schema"]
    result["problems"].extend(problems)
    result["status"] = ARTIFACT_STATUS_INVALID if problems else ARTIFACT_STATUS_VALID

    # Independent check that the artifact would satisfy the existing gate.
    gate_metrics = docs["metrics"].get("validation_metrics") if isinstance(docs["metrics"], dict) else None
    result["gate_compatible"] = (
        isinstance(gate_metrics, dict)
        and all(k in gate_metrics for k in GATE_REQUIRED_METRIC_KEYS)
    )
    return result


def load_model_evidence(state_name=DEFAULT_STATE_NAME, base_dir=None,
                        load_model=False, require_full_metrics=False):
    """
    Loads persisted evidence, or reports MISSING / INVALID.

    There is NO fallback: this never returns a hardcoded model, hardcoded metrics,
    documentary reference figures, or synthesised values. When `load_model` is
    True and deserialization fails (including because joblib/pickle cannot
    reconstruct the estimator), the status becomes INVALID and 'model' stays None.
    """
    result = verify_artifact_set(
        state_name=state_name, base_dir=base_dir,
        require_full_metrics=require_full_metrics,
    )
    result["model"] = None
    if not load_model or result["status"] != ARTIFACT_STATUS_VALID:
        return result
    try:
        result["model"] = _load_model(result["paths"]["model"])
    except Exception as exc:
        result["status"] = ARTIFACT_STATUS_INVALID
        result["problems"].append("model: could not be deserialized (%s)" % exc)
        result["model"] = None
    return result
