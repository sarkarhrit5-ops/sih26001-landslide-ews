"""
Focused tests for the explanation path in app.models.ml_pipeline.

The behaviour under test is a removal: explain_risk used to return invented
feature importances (slope 0.42 / rain_3d 0.28 / roughness 0.18) whenever a model
or feature vector was missing, and the only caller passed (None, None). These
tests pin down that the honest UNAVAILABLE result is what comes back instead, and
that a real ranking is computed only from SHAP output.

DEPENDENCY BUDGET: no real SHAP run. shap.TreeExplainer is replaced by a local
double so the ranking and failure paths can be exercised without shap, LightGBM
or a trained model.
"""

import os
import sys

import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# ml_pipeline imports lightgbm/shap/sklearn at module scope; in an environment
# without them this whole file is skipped rather than reported as a failure.
pytest.importorskip("lightgbm")
pytest.importorskip("shap")
pytest.importorskip("sklearn")

import numpy as np

from app.models import ml_pipeline
from app.models.ml_pipeline import (
    EXPLANATION_STATUS_REAL,
    EXPLANATION_STATUS_UNAVAILABLE,
    explain_risk,
    explanation_unavailable,
    rank_shap_contributions,
)
from app.services.model_artifacts import FEATURE_MEANINGS

FEATURES = ["slope", "elevation", "roughness"]


class _FakeExplainer:
    """Stands in for shap.TreeExplainer so no trained model is needed."""

    def __init__(self, model):
        self.model = model
        self.values = getattr(model, "shap_values", [[0.10, -0.40, 0.05]])

    def shap_values(self, features):
        return self.values


class _Model:
    def __init__(self, shap_values=None):
        if shap_values is not None:
            self.shap_values = shap_values


def _use_fake_explainer(monkeypatch, factory=_FakeExplainer):
    monkeypatch.setattr(ml_pipeline.shap, "TreeExplainer", factory, raising=False)


# ---------------------------------------------------------------------------
# The removed fallback
# ---------------------------------------------------------------------------
def test_no_model_and_no_features_yields_an_honest_refusal():
    result = explain_risk(None, None)
    assert result["status"] == EXPLANATION_STATUS_UNAVAILABLE
    assert result["top_features"] == [], "importances must not be invented"
    assert len(result["reasons"]) == 2, "both missing inputs must be named"
    joined = " ".join(result["reasons"])
    assert "model" in joined and "feature vector" in joined


def test_refusal_never_carries_the_old_hardcoded_importances():
    import json

    encoded = json.dumps(explain_risk(None, None))
    for invented in ("0.42", "0.28", "0.18"):
        assert invented not in encoded


def test_refusal_shape_matches_the_success_shape(monkeypatch):
    _use_fake_explainer(monkeypatch)
    real = explain_risk(_Model(), np.zeros((1, 3)), feature_names=FEATURES)
    refused = explain_risk(None, None)
    # A UI reading `explanation.top_features` must not have to special-case the
    # refusal: the key is always present.
    assert "top_features" in real and "top_features" in refused
    assert real["status"] == EXPLANATION_STATUS_REAL


def test_missing_features_alone_is_reported(monkeypatch):
    _use_fake_explainer(monkeypatch)
    result = explain_risk(_Model(), None, feature_names=FEATURES)
    assert result["status"] == EXPLANATION_STATUS_UNAVAILABLE
    assert len(result["reasons"]) == 1
    assert "nothing to explain" in result["reasons"][0]


def test_explanation_unavailable_states_that_nothing_was_substituted():
    result = explanation_unavailable(["because"])
    assert result["status"] == EXPLANATION_STATUS_UNAVAILABLE
    assert "NOT substituted" in result["message"]
    assert result["reasons"] == ["because"]
    assert result["top_features"] == []


# ---------------------------------------------------------------------------
# Failures are reported, not swallowed
# ---------------------------------------------------------------------------
def test_shap_failure_is_reported_instead_of_falling_back(monkeypatch):
    def exploding_explainer(model):
        raise RuntimeError("explainer does not support this model")

    _use_fake_explainer(monkeypatch, exploding_explainer)
    result = explain_risk(_Model(), np.zeros((1, 3)), feature_names=FEATURES)
    assert result["status"] == EXPLANATION_STATUS_UNAVAILABLE
    assert result["top_features"] == []
    assert "explainer does not support this model" in " ".join(result["reasons"])


# ---------------------------------------------------------------------------
# Rankings come from the SHAP output
# ---------------------------------------------------------------------------
def test_ranking_is_ordered_by_mean_absolute_shap_value():
    ranked, note = rank_shap_contributions([[0.10, -0.40, 0.05]], FEATURES)
    assert note is None
    assert [item["feature"] for item in ranked] == ["elevation", "slope", "roughness"]
    assert ranked[0]["importance"] == 0.4, "sign must not change magnitude"
    assert ranked[0]["description"] == FEATURE_MEANINGS["elevation"]


def test_ranking_averages_across_multiple_rows():
    ranked, note = rank_shap_contributions([[0.2, 0.0, 0.0], [0.0, 0.4, 0.0]],
                                           FEATURES)
    assert note is None
    importances = {item["feature"]: item["importance"] for item in ranked}
    assert importances == {"slope": 0.1, "elevation": 0.2, "roughness": 0.0}


def test_misaligned_shap_output_is_refused_rather_than_guessed():
    ranked, note = rank_shap_contributions([[0.1, 0.2]], FEATURES)
    assert ranked == []
    assert "does not align" in note


def test_unnamed_features_are_refused_rather_than_labelled():
    ranked, note = rank_shap_contributions([[0.1, 0.2, 0.3]], None)
    assert ranked == []
    assert "Feature names were not supplied" in note


def test_real_explanation_reports_its_source_and_raw_values(monkeypatch):
    _use_fake_explainer(monkeypatch)
    result = explain_risk(_Model([[0.10, -0.40, 0.05]]), np.zeros((1, 3)),
                          feature_names=FEATURES)
    assert result["status"] == EXPLANATION_STATUS_REAL
    assert "shap.TreeExplainer" in result["source"]
    assert result["shap_values"] == [[0.10, -0.40, 0.05]]
    assert len(result["top_features"]) == len(FEATURES)


def test_real_explanation_without_names_keeps_values_but_adds_a_note(monkeypatch):
    _use_fake_explainer(monkeypatch)
    result = explain_risk(_Model(), np.zeros((1, 3)))
    assert result["status"] == EXPLANATION_STATUS_REAL
    assert result["top_features"] == []
    assert "note" in result, "an empty ranking must explain itself"
