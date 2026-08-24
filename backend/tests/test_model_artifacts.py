"""
UNIT TESTS for backend/app/services/model_artifacts.py.

SCOPE: these tests verify ARTIFACT BEHAVIOUR ONLY -- canonical paths,
serialization, validation, atomicity and honest failure reporting. They make NO
assertion whatsoever about scientific model performance, and they must never be
read as evidence that a model has been trained or validated.

They deliberately require NO DEM, NO rainfall, NO network, NO LightGBM/sklearn
training, NO geopandas and NO rasterio. A tiny in-file dummy estimator stands in
for a fitted classifier, and every filesystem interaction happens under pytest's
tmp_path.
"""

import json
import os
import sys

import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.services import model_artifacts as ma


# ---------------------------------------------------------------------------
# Test doubles and fixtures
# ---------------------------------------------------------------------------
class DummyEstimator:
    """
    Minimal stand-in for a fitted classifier. It exposes predict_proba so it
    satisfies the writer's duck-typed model check, and it is picklable because it
    is defined at module level. It is NOT a real model and produces no science.
    """

    def __init__(self, n_features=3, tag="unit-test-dummy"):
        self.n_features = n_features
        self.tag = tag

    def predict_proba(self, X):
        return [[0.5, 0.5] for _ in range(len(X))]


class NotAModel:
    """Object with no predict_proba -- must be rejected by the writer."""


FEATURES = ["elevation", "slope", "rain_3d"]

# Metric values below are ARBITRARY UNIT-TEST FIXTURES chosen to be structurally
# valid. They are not measurements of anything and carry no scientific meaning.
FIXTURE_METRICS = {
    "PR-AUC": 0.5,
    "ROC-AUC": 0.5,
    "Precision": 0.5,
    "Recall": 0.5,
    "F1": 0.5,
    "False Alarm Rate": 0.5,
}


def make_metrics_doc(**overrides):
    doc = ma.build_metrics_document(
        validation_metrics=dict(FIXTURE_METRICS),
        primary_model_name="LightGBM",
        primary_evaluation="temporal_holdout / static_plus_rainfall",
        feature_set="static_plus_rainfall",
        model_comparison={
            "spatial_holdout": {"static_only": {}, "static_plus_rainfall": {}},
            "temporal_holdout": {"static_only": {}, "static_plus_rainfall": {}},
        },
        sample_counts={"total_samples": 10},
        decision={"final_recommendation": "Option C: unit-test fixture"},
        dataset_provenance_reference="/unit/test/training_matrix.parquet",
    )
    doc.update(overrides)
    return doc


def make_schema_doc(features=None):
    return ma.build_feature_schema_document(
        feature_names=features if features is not None else list(FEATURES),
        dtypes={"elevation": "float32", "slope": "float32", "rain_3d": "float32"},
        feature_set_name="static_plus_rainfall",
    )


def make_provenance_doc(features=None):
    return ma.build_provenance_document(
        aoi={"min_lat": 27.0, "max_lat": 28.1, "min_lon": 88.0, "max_lon": 88.9},
        model_type="LightGBM (unit-test dummy)",
        model_hyperparameters={"n_estimators": 100, "random_state": 42},
        feature_list=features if features is not None else list(FEATURES),
        random_seed=42,
        input_status={
            "dem_copernicus_glo30": "REAL",
            "land_cover_class": "DERIVED_PROXY",
            "imerg_satellite_rainfall": "NOT_USED",
        },
        code_version="unit-test-sha",
        software_versions={"python": "unit-test"},
    )


def save_bundle(tmp_dir, model=None, metrics=None, schema=None, provenance=None):
    return ma.save_model_evidence(
        model=model if model is not None else DummyEstimator(),
        metrics_doc=metrics if metrics is not None else make_metrics_doc(),
        schema_doc=schema if schema is not None else make_schema_doc(),
        provenance_doc=provenance if provenance is not None else make_provenance_doc(),
        state_name="Sikkim",
        base_dir=str(tmp_dir),
        documentary_blocks=[],
    )


# ---------------------------------------------------------------------------
# 1. Canonical artifact paths
# ---------------------------------------------------------------------------
def test_canonical_paths_match_the_existing_validation_gate_filenames():
    paths = ma.canonical_artifact_paths("Sikkim", base_dir="/tmp/models")
    assert os.path.basename(paths["model"]) == "sikkim_model.pkl"
    assert os.path.basename(paths["metrics"]) == "sikkim_metrics.json"
    assert os.path.basename(paths["schema"]) == "sikkim_feature_schema.json"
    assert os.path.basename(paths["provenance"]) == "sikkim_provenance.json"
    for p in paths.values():
        assert os.path.dirname(p) == os.path.abspath("/tmp/models")


def test_canonical_paths_normalise_state_name_like_the_gate_does():
    paths = ma.canonical_artifact_paths("Arunachal Pradesh", base_dir="/tmp/models")
    assert os.path.basename(paths["model"]) == "arunachal_pradesh_model.pkl"


def test_default_artifact_dir_is_backend_data_models():
    d = ma.default_artifact_dir()
    assert os.path.basename(d) == "models"
    assert os.path.basename(os.path.dirname(d)) == "data"


def test_canonical_paths_agree_with_state_validation_gate_if_importable():
    """
    The writer and the gate must never drift apart. Skipped when
    state_validation's own heavy imports are unavailable in this environment.
    """
    sv = pytest.importorskip(
        "app.services.state_validation",
        reason="state_validation pulls in scientific dependencies",
    )
    gate = sv._evidence_paths("Sikkim", base_dir="/tmp/models")
    mine = ma.canonical_artifact_paths("Sikkim", base_dir="/tmp/models")
    for kind in ("model", "metrics", "schema"):
        assert gate[kind] == mine[kind]
    assert set(sv.REQUIRED_METRIC_KEYS) <= set(ma.GATE_REQUIRED_METRIC_KEYS)


# ---------------------------------------------------------------------------
# 2-5. Successful serialization of each artifact
# ---------------------------------------------------------------------------
def test_model_is_serialized_and_reloadable(tmp_path):
    paths = save_bundle(tmp_path, model=DummyEstimator(tag="round-trip"))
    assert os.path.getsize(paths["model"]) > 0
    loaded = ma.load_model_evidence(base_dir=str(tmp_path), load_model=True)
    assert loaded["status"] == ma.ARTIFACT_STATUS_VALID
    assert loaded["model"] is not None
    assert loaded["model"].tag == "round-trip"


def test_metrics_artifact_is_written_in_the_shape_the_gate_reads(tmp_path):
    paths = save_bundle(tmp_path)
    with open(paths["metrics"], "r", encoding="utf-8") as fh:
        doc = json.load(fh)
    # The gate does: metrics = doc.get("validation_metrics", doc)
    assert isinstance(doc["validation_metrics"], dict)
    for key in ma.GATE_REQUIRED_METRIC_KEYS:
        assert key in doc["validation_metrics"]
    assert doc["validation_metrics"] == FIXTURE_METRICS
    assert doc["primary_model"] == "LightGBM"
    assert doc["primary_evaluation"] == "temporal_holdout / static_plus_rainfall"
    # The full model comparison and the Option A/C recommendation are preserved.
    assert set(doc["model_comparison"]) == {"spatial_holdout", "temporal_holdout"}
    assert set(doc["model_comparison"]["temporal_holdout"]) == {
        "static_only", "static_plus_rainfall"
    }
    assert doc["model_decision"]["final_recommendation"].startswith("Option C")
    assert doc["metrics_source"] == ma.COMPUTED_METRICS_SOURCE


def test_feature_schema_artifact_records_names_order_dtype_meaning_and_version(tmp_path):
    paths = save_bundle(tmp_path)
    with open(paths["schema"], "r", encoding="utf-8") as fh:
        doc = json.load(fh)
    assert doc["feature_names"] == FEATURES          # order preserved verbatim
    assert doc["feature_order"] == [0, 1, 2]
    assert doc["n_features"] == 3
    assert doc["dtype"]["elevation"] == "float32"
    assert set(doc["meaning"]) == set(FEATURES)
    assert doc["feature_schema_version"] == ma.FEATURE_SCHEMA_VERSION
    # The land-cover proxy must be described as a proxy wherever it appears.
    assert "DERIVED PROXY" in ma.FEATURE_MEANINGS["land_cover_class"]


def test_feature_schema_marks_unknown_features_as_undocumented_rather_than_inventing():
    doc = ma.build_feature_schema_document(feature_names=["some_new_feature"])
    assert doc["meaning"]["some_new_feature"].startswith("UNDOCUMENTED")
    assert doc["dtype"]["some_new_feature"] == "UNKNOWN"


def test_provenance_artifact_records_reproduction_facts_and_input_status(tmp_path):
    paths = save_bundle(tmp_path)
    with open(paths["provenance"], "r", encoding="utf-8") as fh:
        doc = json.load(fh)
    assert doc["aoi"]["min_lat"] == 27.0 and doc["aoi"]["max_lon"] == 88.9
    assert doc["random_seed"] == 42
    assert doc["feature_list"] == FEATURES
    assert doc["input_status"]["imerg_satellite_rainfall"] == "NOT_USED"
    assert doc["input_status"]["land_cover_class"] == "DERIVED_PROXY"
    assert doc["generation_timestamp"]
    assert doc["code_version"] == "unit-test-sha"
    # The serializer actually used is recorded, never left implicit.
    assert doc["model_serialization"] in ("joblib", "pickle")


def test_provenance_rejects_input_status_values_outside_the_vocabulary():
    doc = make_provenance_doc()
    doc["input_status"]["dem_copernicus_glo30"] = "PROBABLY_FINE"
    problems = ma.validate_provenance_document(doc)
    assert any("input_status" in p for p in problems)


def test_unknown_git_sha_is_labelled_not_invented():
    sha = ma.get_git_sha(repo_dir="/nonexistent-directory-for-unit-test")
    assert sha == "UNKNOWN"


def test_absent_package_versions_are_labelled_not_invented():
    versions = ma.collect_software_versions(["definitely_not_a_real_module_xyz"])
    assert versions["definitely_not_a_real_module_xyz"] == "NOT_INSTALLED"
    assert versions["python"]


# ---------------------------------------------------------------------------
# 6. Missing artifact detection
# ---------------------------------------------------------------------------
def test_missing_artifacts_report_missing_and_never_fabricate(tmp_path):
    result = ma.load_model_evidence(base_dir=str(tmp_path), load_model=True)
    assert result["status"] == ma.ARTIFACT_STATUS_MISSING
    assert sorted(result["missing"]) == ["metrics", "model", "schema"]
    assert result["metrics"] is None
    assert result["feature_schema"] is None
    assert result["model"] is None
    assert result["gate_compatible"] is False


def test_zero_length_artifact_counts_as_missing(tmp_path):
    paths = save_bundle(tmp_path)
    open(paths["metrics"], "w").close()  # truncate to 0 bytes
    result = ma.verify_artifact_set(base_dir=str(tmp_path))
    assert result["status"] == ma.ARTIFACT_STATUS_MISSING
    assert "metrics" in result["missing"]


def test_absent_provenance_does_not_invalidate_the_gate_required_set(tmp_path):
    paths = save_bundle(tmp_path)
    os.remove(paths["provenance"])
    result = ma.verify_artifact_set(base_dir=str(tmp_path))
    assert result["status"] == ma.ARTIFACT_STATUS_VALID
    assert result["provenance_present"] is False
    assert result["provenance"] is None


# ---------------------------------------------------------------------------
# 7. Invalid / incomplete artifact detection
# ---------------------------------------------------------------------------
def test_unreadable_metrics_json_reports_invalid_not_valid(tmp_path):
    paths = save_bundle(tmp_path)
    with open(paths["metrics"], "w", encoding="utf-8") as fh:
        fh.write("{ this is not valid json")
    result = ma.load_model_evidence(base_dir=str(tmp_path))
    assert result["status"] == ma.ARTIFACT_STATUS_INVALID
    assert any("unreadable" in p for p in result["problems"])
    assert result["metrics"] is None


def test_metrics_json_without_required_keys_reports_invalid(tmp_path):
    paths = save_bundle(tmp_path)
    with open(paths["metrics"], "w", encoding="utf-8") as fh:
        json.dump({"validation_metrics": {"Precision": 0.5}}, fh)
    result = ma.verify_artifact_set(base_dir=str(tmp_path))
    assert result["status"] == ma.ARTIFACT_STATUS_INVALID
    assert result["gate_compatible"] is False


def test_undeserializable_model_reports_invalid_and_returns_no_model(tmp_path):
    paths = save_bundle(tmp_path)
    with open(paths["model"], "wb") as fh:
        fh.write(b"not a serialized estimator")
    result = ma.load_model_evidence(base_dir=str(tmp_path), load_model=True)
    assert result["status"] == ma.ARTIFACT_STATUS_INVALID
    assert result["model"] is None
    assert any("model" in p for p in result["problems"])


def test_schema_with_inconsistent_counts_is_rejected():
    doc = make_schema_doc()
    doc["n_features"] = 99
    problems = ma.validate_feature_schema_document(doc)
    assert any("n_features" in p for p in problems)


def test_writer_rejects_an_object_that_is_not_a_fitted_classifier(tmp_path):
    with pytest.raises(ma.ArtifactValidationError):
        save_bundle(tmp_path, model=NotAModel())
    assert os.listdir(str(tmp_path)) == []


def test_writer_rejects_a_missing_model(tmp_path):
    with pytest.raises(ma.ArtifactValidationError):
        ma.save_model_evidence(
            model=None,
            metrics_doc=make_metrics_doc(),
            schema_doc=make_schema_doc(),
            provenance_doc=make_provenance_doc(),
            base_dir=str(tmp_path),
            documentary_blocks=[],
        )
    assert os.listdir(str(tmp_path)) == []


def test_writer_rejects_a_feature_list_disagreement_between_schema_and_provenance(tmp_path):
    with pytest.raises(ma.ArtifactValidationError):
        save_bundle(
            tmp_path,
            schema=make_schema_doc(features=["elevation", "slope", "rain_3d"]),
            provenance=make_provenance_doc(features=["elevation", "slope"]),
        )
    assert os.listdir(str(tmp_path)) == []


# ---------------------------------------------------------------------------
# 8. Computed metrics are required
# ---------------------------------------------------------------------------
def test_writer_refuses_metrics_missing_pr_auc_or_roc_auc(tmp_path):
    for dropped in ma.GATE_REQUIRED_METRIC_KEYS:
        metrics = dict(FIXTURE_METRICS)
        del metrics[dropped]
        doc = make_metrics_doc(validation_metrics=metrics)
        with pytest.raises(ma.ArtifactValidationError):
            save_bundle(tmp_path, metrics=doc)
    assert os.listdir(str(tmp_path)) == []


def test_writer_refuses_non_numeric_or_non_finite_metrics(tmp_path):
    for bad in ("n/a", None, float("nan"), float("inf"), True):
        metrics = dict(FIXTURE_METRICS)
        metrics["PR-AUC"] = bad
        with pytest.raises(ma.ArtifactValidationError):
            save_bundle(tmp_path, metrics=make_metrics_doc(validation_metrics=metrics))
    assert os.listdir(str(tmp_path)) == []


def test_writer_refuses_an_empty_metrics_block(tmp_path):
    with pytest.raises(ma.ArtifactValidationError):
        save_bundle(tmp_path, metrics=make_metrics_doc(validation_metrics={}))
    assert os.listdir(str(tmp_path)) == []


def test_writer_requires_an_explicit_computed_source_declaration(tmp_path):
    with pytest.raises(ma.ArtifactValidationError):
        save_bundle(tmp_path, metrics=make_metrics_doc(metrics_source="looks_about_right"))
    with pytest.raises(ma.ArtifactValidationError):
        save_bundle(tmp_path, metrics=make_metrics_doc(metrics_source=None))
    assert os.listdir(str(tmp_path)) == []


# ---------------------------------------------------------------------------
# 9. Documentary / hardcoded metrics cannot become computed evidence
# ---------------------------------------------------------------------------
# The literals below are quoted ONLY to prove they are refused. They are the
# historical documentary figures from ml_pipeline.DOCUMENTARY_REFERENCE_METRICS
# and must never be persisted as the result of a validation run.
DOCUMENTARY_LIGHTGBM = {
    "PR-AUC": 0.7762, "ROC-AUC": 0.9190, "False Alarm Rate": 0.0317,
    "Precision": 0.7778, "Recall": 0.3684, "F1": 0.5000,
}
DOCUMENTARY_RANDOM_FOREST = {
    "PR-AUC": 0.7792, "ROC-AUC": 0.9319, "False Alarm Rate": 0.0476,
    "Precision": 0.7500, "Recall": 0.4737, "F1": 0.5806,
}


@pytest.mark.parametrize("block", [DOCUMENTARY_LIGHTGBM, DOCUMENTARY_RANDOM_FOREST])
def test_documentary_metric_blocks_cannot_be_persisted_as_computed_evidence(tmp_path, block):
    doc = make_metrics_doc(validation_metrics=dict(block))
    with pytest.raises(ma.ArtifactValidationError) as excinfo:
        ma.save_model_evidence(
            model=DummyEstimator(),
            metrics_doc=doc,
            schema_doc=make_schema_doc(),
            provenance_doc=make_provenance_doc(),
            base_dir=str(tmp_path),
            documentary_blocks=[DOCUMENTARY_LIGHTGBM, DOCUMENTARY_RANDOM_FOREST],
        )
    assert "documentary" in str(excinfo.value).lower()
    assert os.listdir(str(tmp_path)) == []


def test_documentary_block_detection_is_used_by_default(tmp_path):
    """
    With no explicit documentary_blocks argument the module looks them up from
    ml_pipeline. If ml_pipeline is not importable here the lookup returns an empty
    list, so this asserts the default path is at least wired up and that the
    explicit source declaration still guards the write.
    """
    blocks = ma.documentary_metric_blocks()
    assert isinstance(blocks, list)
    if not blocks:
        pytest.skip("ml_pipeline not importable (scientific dependencies absent)")
    doc = make_metrics_doc(validation_metrics=dict(DOCUMENTARY_LIGHTGBM))
    with pytest.raises(ma.ArtifactValidationError):
        ma.save_model_evidence(
            model=DummyEstimator(), metrics_doc=doc,
            schema_doc=make_schema_doc(), provenance_doc=make_provenance_doc(),
            base_dir=str(tmp_path),
        )


def test_a_single_coincidental_value_is_not_treated_as_documentary_reuse(tmp_path):
    """
    The quarantine must refuse whole-block reuse without punishing a legitimate
    computed number that happens to collide on one metric.
    """
    metrics = dict(FIXTURE_METRICS)
    metrics["PR-AUC"] = 0.7762  # collides on one key only
    paths = ma.save_model_evidence(
        model=DummyEstimator(),
        metrics_doc=make_metrics_doc(validation_metrics=metrics),
        schema_doc=make_schema_doc(),
        provenance_doc=make_provenance_doc(),
        base_dir=str(tmp_path),
        documentary_blocks=[DOCUMENTARY_LIGHTGBM, DOCUMENTARY_RANDOM_FOREST],
    )
    assert os.path.exists(paths["metrics"])


# ---------------------------------------------------------------------------
# 10. Partial writes never look valid
# ---------------------------------------------------------------------------
def test_a_rejected_write_leaves_no_partial_artifact_set(tmp_path):
    with pytest.raises(ma.ArtifactValidationError):
        save_bundle(tmp_path, metrics=make_metrics_doc(validation_metrics={}))
    # Nothing published, and nothing left behind for the gate to trip over.
    assert os.listdir(str(tmp_path)) == []
    assert ma.verify_artifact_set(base_dir=str(tmp_path))["status"] == ma.ARTIFACT_STATUS_MISSING


def test_each_individually_missing_gate_artifact_makes_the_set_incomplete(tmp_path):
    for kind in ma.GATE_REQUIRED_ARTIFACTS:
        paths = save_bundle(tmp_path)
        assert ma.verify_artifact_set(base_dir=str(tmp_path))["status"] == ma.ARTIFACT_STATUS_VALID
        os.remove(paths[kind])
        result = ma.verify_artifact_set(base_dir=str(tmp_path))
        assert result["status"] == ma.ARTIFACT_STATUS_MISSING
        assert kind in result["missing"]
        for remaining in paths.values():
            if os.path.exists(remaining):
                os.remove(remaining)


def test_writing_leaves_no_staging_directory_behind(tmp_path):
    save_bundle(tmp_path)
    leftovers = [n for n in os.listdir(str(tmp_path)) if n.startswith(".staging_artifacts_")]
    assert leftovers == []


def test_republishing_overwrites_cleanly_and_stays_valid(tmp_path):
    save_bundle(tmp_path)
    save_bundle(tmp_path, model=DummyEstimator(tag="second-run"))
    result = ma.load_model_evidence(base_dir=str(tmp_path), load_model=True)
    assert result["status"] == ma.ARTIFACT_STATUS_VALID
    assert result["model"].tag == "second-run"
    assert sorted(os.listdir(str(tmp_path))) == [
        "sikkim_feature_schema.json",
        "sikkim_metrics.json",
        "sikkim_model.pkl",
        "sikkim_provenance.json",
    ]


def test_validate_evidence_bundle_reports_every_problem_at_once():
    problems = ma.validate_evidence_bundle(
        model=None,
        metrics_doc={"validation_metrics": {}},
        schema_doc={"feature_names": []},
        provenance_doc={},
        documentary_blocks=[],
    )
    assert len(problems) >= 4
    joined = " | ".join(problems)
    assert "model" in joined and "metrics" in joined
    assert "schema" in joined and "provenance" in joined
