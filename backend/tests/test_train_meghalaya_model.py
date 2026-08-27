"""
Offline tests for scripts/train_meghalaya_model.py -- the Meghalaya pilot trainer
that loads the completed 136x11 matrix and reproduces the Sikkim/Assam/Arunachal
methodology, changing exactly one thing relative to Sikkim: land_cover_class is
fed to the LightGBM primary model as a NOMINAL categorical (real ESA WorldCover),
not a numeric ordinal proxy.

DEPENDENCY BUDGET: stdlib + numpy/pandas only. The trainer imports pyarrow (parquet
read), scikit-learn + lightgbm (via app.models.ml_pipeline) and joblib LAZILY inside
main(), so the module and every pure helper exercised here import offline. These tests
never call main(), never read the parquet, never touch the network, and never train a
real model. The scikit-learn / LightGBM code paths are exercised through injected fakes
that record how they were called, so we can prove the wiring without the libraries.

What they protect:
  * the 11-feature schema and order (5 terrain + land_cover_class + 5 rainfall) match
    the Sikkim/Assam/Arunachal trainer, with land_cover_class at index 5 and the
    temporal cutoff 2014;
  * rows with any non-finite feature OR the land-cover UNAVAILABLE sentinel are DROPPED,
    never filled/imputed, with correct counts (the trainer half of "abort, never fill");
  * the categorical view converts ONLY land_cover_class to a pandas category over the
    fixed WorldCover vocabulary, leaving values and other columns untouched;
  * the persisted LightGBM primary model is fit with categorical_feature declared, while
    the Logistic Regression / Random Forest comparison rows stay numeric (Sikkim-identical);
  * the reproducibility-consistency invariant (baseline LightGBM == refit primary) holds;
  * the Meghalaya label overrides are truthful WorldCover wording and relabelling never
    mutates metrics/features.
"""
import os
import sys
import types
from contextlib import contextmanager

import numpy as np
import pandas as pd
import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'scripts')))

import train_meghalaya_model as t
from app.services import worldcover as wc


SIKKIM_DYNAMIC_ORDER = [
    "elevation", "slope", "aspect", "roughness", "tpi",
    "land_cover_class",
    "rain_1d", "rain_3d", "rain_7d", "antecedent_rain_14d", "rain_intensity_max_3d",
]


# --------------------------------------------------------------------------- #
# Feature schema / categorical contract
# --------------------------------------------------------------------------- #
def test_feature_order_matches_sikkim_dynamic_features():
    assert t.TERRAIN_FEATURES == ["elevation", "slope", "aspect", "roughness", "tpi"]
    assert t.RAINFALL_FEATURES == [
        "rain_1d", "rain_3d", "rain_7d", "antecedent_rain_14d", "rain_intensity_max_3d"
    ]
    assert t.STATIC_FEATURES == t.TERRAIN_FEATURES + ["land_cover_class"]
    assert t.DYNAMIC_FEATURES == SIKKIM_DYNAMIC_ORDER
    assert len(t.STATIC_FEATURES) == 6
    assert len(t.DYNAMIC_FEATURES) == 11
    assert t.DYNAMIC_FEATURES.index("land_cover_class") == 5


def test_temporal_cutoff_and_state_labels():
    # Reuse Sikkim's cutoff verbatim (verified non-degenerate for Meghalaya:
    # 12 positives train <=2014 / 22 test >=2015; distinct dates 10 vs 19).
    assert t.TEMPORAL_CUTOFF_YEAR == 2014
    assert t.STATE_NAME == "Meghalaya"
    assert t.PILOT_AREA and "Sikkim" not in t.PILOT_AREA


def test_land_cover_is_the_single_categorical_feature():
    assert wc.LANDCOVER_FEATURE_NAME == "land_cover_class"
    assert wc.landcover_categorical_feature() == ["land_cover_class"]
    assert wc.LANDCOVER_IS_CATEGORICAL is True
    assert "land_cover_class" in t.STATIC_FEATURES
    assert "land_cover_class" in t.DYNAMIC_FEATURES


# --------------------------------------------------------------------------- #
# Missing-feature row drop (never fill)
# --------------------------------------------------------------------------- #
def _matrix_with_missing():
    # 5 rows: r0 clean, r1 NaN slope, r2 landcover UNAVAILABLE(-1), r3 inf rain_3d, r4 clean
    base = {f: np.arange(5, dtype="float32") + 1.0 for f in t.TERRAIN_FEATURES}
    for f in t.RAINFALL_FEATURES:
        base[f] = np.arange(5, dtype="float32") + 10.0
    df = pd.DataFrame(base)
    df["land_cover_class"] = np.array([1, 2, wc.UNAVAILABLE_SENTINEL, 4, 6], dtype="int32")
    df["target"] = np.array([1, 0, 1, 0, 1], dtype="int64")
    # Coordinates inside the canonical Meghalaya pilot AOI (25.0-25.99N, 91.0-92.8E).
    df["latitude"] = np.linspace(25.0, 25.99, 5).astype("float64")
    df["longitude"] = np.linspace(91.0, 92.8, 5).astype("float64")
    df["event_date"] = pd.to_datetime(
        ["2010-06-01", "2012-07-02", "2014-08-03", "2016-09-04", "2017-10-05"]
    )
    df.loc[1, "slope"] = np.nan
    df.loc[3, "rain_3d"] = np.inf
    return df


def test_drop_missing_feature_rows_counts_and_never_fills():
    df = _matrix_with_missing()
    original = df.copy(deep=True)
    clean, report = t.drop_missing_feature_rows(df)

    # r0 and r4 survive; r1/r2/r3 dropped.
    assert report["rows_in"] == 5
    assert report["rows_out"] == 2
    assert report["rows_dropped"] == 3
    assert report["dropped_nonfinite_numeric"] == 2      # NaN slope + inf rain_3d
    assert report["dropped_landcover_unavailable"] == 1  # sentinel row
    assert report["fill_or_impute_performed"] is False

    # Surviving rows keep their EXACT original values (no imputation/fill).
    assert list(clean["target"]) == [1, 1]
    assert clean["land_cover_class"].tolist() == [1, 6]
    assert clean["slope"].tolist() == [1.0, 5.0]
    assert clean["rain_3d"].tolist() == [10.0, 14.0]

    # The input frame is untouched by the drop helper.
    pd.testing.assert_frame_equal(df, original)


def test_drop_missing_feature_rows_all_clean_keeps_everything():
    df = _matrix_with_missing()
    df.loc[1, "slope"] = 2.0
    df.loc[3, "rain_3d"] = 13.0
    df.loc[2, "land_cover_class"] = 3
    clean, report = t.drop_missing_feature_rows(df)
    assert report["rows_dropped"] == 0
    assert report["rows_out"] == 5


# --------------------------------------------------------------------------- #
# Categorical view
# --------------------------------------------------------------------------- #
def test_lgbm_categorical_view_only_land_cover_and_values_preserved():
    df = pd.DataFrame(
        {
            "elevation": np.array([100.0, 200.0, 300.0], dtype="float32"),
            "land_cover_class": np.array([1, 4, 6], dtype="int32"),
        }
    )
    original = df.copy(deep=True)
    cats = list(wc.ASSAM_LANDCOVER_GROUP_CODES)
    view = t.lgbm_categorical_view(df, cats)

    # land_cover_class becomes categorical over the FULL WorldCover vocabulary.
    assert isinstance(view["land_cover_class"].dtype, pd.CategoricalDtype)
    assert list(view["land_cover_class"].cat.categories) == cats
    # underlying values unchanged
    assert view["land_cover_class"].astype("int64").tolist() == [1, 4, 6]
    # other columns untouched (dtype + values)
    assert view["elevation"].dtype == np.float32
    assert view["elevation"].tolist() == [100.0, 200.0, 300.0]
    # original frame not mutated (still integer land cover)
    pd.testing.assert_frame_equal(df, original)
    assert not isinstance(df["land_cover_class"].dtype, pd.CategoricalDtype)


def test_lgbm_categorical_view_no_column_is_noop():
    df = pd.DataFrame({"elevation": [1.0, 2.0]})
    view = t.lgbm_categorical_view(df, [1, 2, 3])
    pd.testing.assert_frame_equal(df, view)


# --------------------------------------------------------------------------- #
# GLC quality inputs to the Option A/C decision
# --------------------------------------------------------------------------- #
def test_compute_glc_quality_matches_sikkim_rule():
    pos = pd.DataFrame(
        {
            "location_accuracy": ["1km", "exact", "5km", "10km", "5km"],
            "event_date": pd.to_datetime(
                ["2011-01-01", "2011-01-01", "2013-06-06", "2014-07-07", "2016-08-08"]
            ),
        }
    )
    info = t.compute_glc_quality(pos)
    # high-accuracy = {1km, exact} -> 2; low = 3 -> 60%
    assert info["pct_low_accuracy"] == 60.0
    assert info["total_usable_events"] == 5
    # distinct event DATES: 2011-01-01 shared -> 4 unique
    assert info["independent_events"] == 4


def test_compute_glc_quality_empty():
    info = t.compute_glc_quality(pd.DataFrame({"location_accuracy": [], "event_date": []}))
    assert info == {"pct_low_accuracy": 0.0, "independent_events": 0, "total_usable_events": 0}


# --------------------------------------------------------------------------- #
# Meghalaya label overrides (truthful WorldCover wording; relabel is value-safe)
# --------------------------------------------------------------------------- #
def test_land_cover_meaning_is_worldcover_not_proxy():
    m = t.meghalaya_land_cover_meaning()
    low = m.lower()
    assert "worldcover" in low
    assert "categorical" in low
    assert "not an elevation proxy" in low
    # every WorldCover group label is named
    for label in wc.ASSAM_LANDCOVER_GROUP_LABELS.values():
        assert label in m


def test_meghalaya_feature_meanings_overrides_only_land_cover():
    meanings = t.meghalaya_feature_meanings()
    assert list(meanings.keys()) == ["land_cover_class"]


def test_relabel_for_meghalaya_changes_only_labels():
    metrics_like = {
        "state": "Sikkim",
        "pilot_area": "East Sikkim",
        "validation_metrics": {"PR-AUC": 0.42, "ROC-AUC": 0.71},
        "primary_model": "LightGBM",
    }
    out = t.relabel_for_meghalaya(dict(metrics_like))
    assert out["state"] == "Meghalaya"
    assert out["pilot_area"] == t.PILOT_AREA
    # nothing else altered
    assert out["validation_metrics"] == {"PR-AUC": 0.42, "ROC-AUC": 0.71}
    assert out["primary_model"] == "LightGBM"

    # schema-like doc has no pilot_area -> the key is NOT invented
    schema_like = {"state": "Sikkim", "feature_names": ["a", "b"]}
    out2 = t.relabel_for_meghalaya(dict(schema_like))
    assert out2["state"] == "Meghalaya"
    assert "pilot_area" not in out2
    assert out2["feature_names"] == ["a", "b"]


# --------------------------------------------------------------------------- #
# Training-wiring proof via fakes (no sklearn / lightgbm installed here)
# --------------------------------------------------------------------------- #
class _FakeLGBM:
    """Records how fit() was called and what dtype land_cover_class had."""

    def __init__(self):
        self.categorical_feature = "UNSET"
        self.fit_lc_dtype = None
        self.predict_lc_dtype = None

    def fit(self, X, y, categorical_feature=None):
        self.categorical_feature = categorical_feature
        self.fit_lc_dtype = X["land_cover_class"].dtype
        self.n_train = len(X)
        return self

    def predict_proba(self, X):
        self.predict_lc_dtype = X["land_cover_class"].dtype
        n = len(X)
        p = np.full(n, 0.5)
        return np.column_stack([1.0 - p, p])


class _FakeLinear:
    """Fake LR/RF that record the land_cover_class dtype they were fit on."""

    last_fit_lc_dtype = {}

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def fit(self, X, y):
        _FakeLinear.last_fit_lc_dtype[type(self).__name__] = X["land_cover_class"].dtype
        self._n = len(X)
        return self

    def predict_proba(self, X):
        n = len(X)
        p = np.full(n, 0.3)
        return np.column_stack([1.0 - p, p])


class _FakeLR(_FakeLinear):
    pass


class _FakeRF(_FakeLinear):
    pass


def _fake_ml():
    return types.SimpleNamespace(
        PRIMARY_MODEL_NAME="LightGBM",
        build_primary_model=lambda: _FakeLGBM(),
        # deterministic function of the probabilities so identical inputs -> identical dict
        compute_metrics=lambda y_true, proba: {
            "mean_proba": round(float(np.mean(proba)), 6),
            "n": int(len(proba)),
        },
    )


@contextmanager
def _inject_fake_sklearn():
    _FakeLinear.last_fit_lc_dtype = {}
    saved = {k: sys.modules.get(k) for k in ("sklearn", "sklearn.linear_model", "sklearn.ensemble")}
    sk = types.ModuleType("sklearn")
    lin = types.ModuleType("sklearn.linear_model")
    ens = types.ModuleType("sklearn.ensemble")
    lin.LogisticRegression = _FakeLR
    ens.RandomForestClassifier = _FakeRF
    sk.linear_model = lin
    sk.ensemble = ens
    sys.modules["sklearn"] = sk
    sys.modules["sklearn.linear_model"] = lin
    sys.modules["sklearn.ensemble"] = ens
    try:
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v


def _xy():
    X_train = pd.DataFrame(
        {
            "elevation": np.array([100.0, 200.0, 300.0, 400.0], dtype="float32"),
            "land_cover_class": np.array([1, 4, 6, 2], dtype="int32"),
        }
    )
    X_test = pd.DataFrame(
        {
            "elevation": np.array([150.0, 250.0], dtype="float32"),
            "land_cover_class": np.array([1, 6], dtype="int32"),
        }
    )
    y_train = pd.Series([1, 0, 1, 0], dtype="int64")
    y_test = pd.Series([1, 0], dtype="int64")
    return X_train, X_test, y_train, y_test


def test_train_primary_categorical_declares_categorical_feature():
    ml = _fake_ml()
    X_train, X_test, y_train, y_test = _xy()
    cats = list(wc.ASSAM_LANDCOVER_GROUP_CODES)
    model, metrics = t.train_primary_categorical(ml, X_train, X_test, y_train, y_test, cats)

    # categorical_feature was declared to LightGBM
    assert model.categorical_feature == ["land_cover_class"]
    # the model saw a CATEGORICAL land_cover_class at fit AND predict time
    assert isinstance(model.fit_lc_dtype, pd.CategoricalDtype)
    assert isinstance(model.predict_lc_dtype, pd.CategoricalDtype)
    # metrics came from ml.compute_metrics on the held-out proba (0.5 constant)
    assert metrics == {"mean_proba": 0.5, "n": 2}
    # inputs were not mutated to categorical
    assert not isinstance(X_train["land_cover_class"].dtype, pd.CategoricalDtype)


def test_evaluate_baselines_only_lgbm_is_categorical_and_consistent():
    ml = _fake_ml()
    X_train, X_test, y_train, y_test = _xy()
    cats = list(wc.ASSAM_LANDCOVER_GROUP_CODES)

    with _inject_fake_sklearn():
        res = t.evaluate_baselines_meghalaya(ml, X_train, X_test, y_train, y_test, cats)

    # same shape as the Sikkim helper: three named comparison rows
    assert set(res.keys()) == {"Logistic Regression", "Random Forest", "LightGBM"}

    # LR and RF were fit on the NUMERIC land_cover_class (integer), exactly like Sikkim
    lr_dtype = _FakeLinear.last_fit_lc_dtype["_FakeLR"]
    rf_dtype = _FakeLinear.last_fit_lc_dtype["_FakeRF"]
    assert np.issubdtype(lr_dtype, np.integer)
    assert np.issubdtype(rf_dtype, np.integer)
    assert not isinstance(lr_dtype, pd.CategoricalDtype)
    assert not isinstance(rf_dtype, pd.CategoricalDtype)

    # The LightGBM comparison entry reproduces the standalone refit metrics on the
    # same split -- this is what lets the runner's reproducibility gate
    # (primary_metrics == tm_dy_res["LightGBM"]) pass deterministically.
    _m, refit_metrics = t.train_primary_categorical(
        ml, X_train, X_test, y_train, y_test, cats
    )
    assert res["LightGBM"] == refit_metrics
