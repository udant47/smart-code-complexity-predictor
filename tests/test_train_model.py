"""Tests for src.train_model (Phase 10).

The experiment runs in quick mode on the small deterministic test table
from test_preprocessing (structural features only). These tests check the
protocol (no leakage, correct selection, saved artifacts), not model
quality on real data.
"""

import json

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin

from src import config
from src import preprocessing as pp
from src import train_model as tm
from src.feature_extractor import FILE_FEATURE_NAMES
from tests.test_preprocessing import fixture_table


# --------------------------------------------------------------------------- #
# Small pieces
# --------------------------------------------------------------------------- #
def test_nesting_depth_rule():
    X = pd.DataFrame({"maximum_loop_nesting_depth": [0, 0, 1, 1, 2, 2, 5]})
    y = ["constant", "constant", "linear", "linear", "quadratic", "cubic", "cubic"]
    rule = tm.NestingDepthRule().fit(X, y)
    pred = rule.predict(pd.DataFrame({"maximum_loop_nesting_depth": [0, 1, 6, 3]}))
    assert list(pred[:2]) == ["constant", "linear"]
    assert pred[2] == "cubic"  # depths >= 4 share one bucket
    assert pred[3] in y  # unseen depth -> overall majority


def test_metrics_perfect_and_known_values():
    y = ["constant", "linear", "linear", "cubic"]
    assert tm.metrics(y, y)["macro_f1"] == 1.0
    assert tm.metrics(y, ["constant", "linear", "cubic", "cubic"])["accuracy"] == pytest.approx(0.75)


def test_bootstrap_ci_bounds():
    y = pd.Series(["constant", "linear", "cubic", "linear"] * 10)
    groups = pd.Series(np.repeat(np.arange(10), 4))
    assert tm.group_bootstrap_ci(y, y.values, groups, n=200) == (1.0, 1.0)
    noisy = y.values.copy()
    noisy[::3] = "np"
    lo, hi = tm.group_bootstrap_ci(y, noisy, groups, n=200)
    assert 0 <= lo <= hi <= 1


def test_selection_prefers_simpler_model_on_near_ties():
    results = pd.DataFrame({
        "model": ["Hist Gradient Boosting", "Logistic Regression", "Dummy (most frequent)"],
        "rcv_macro_f1": [0.430, 0.427, 0.05],
        "simplicity": [5, 2, 0],
    })
    assert results.loc[tm.select(results, tolerance=0.005), "model"] == "Logistic Regression"
    assert results.loc[tm.select(results, tolerance=0.001), "model"] == "Hist Gradient Boosting"


def test_cv_folds_are_grouped():
    df = fixture_table()
    split = pp.make_split(df, "problem")
    dev = split.indices("train").append(split.indices("validation"))
    g = split.groups.loc[dev].values
    folds = tm.repeated_cv(df.loc[dev, "label"], split.groups.loc[dev], seed=0, repeats=2)
    assert len(folds) == 10
    for a, b in folds:
        assert not set(g[a]) & set(g[b])


def test_candidates_include_required_models():
    names = [c.name for c in tm.candidates(quick=True)]
    for required in ("Dummy (most frequent)", "Logistic Regression", "Decision Tree", "Random Forest",
                     "Gradient Boosting"):
        assert required in names


# --------------------------------------------------------------------------- #
# Leakage: test rows never reach tuning, comparison or selection
# --------------------------------------------------------------------------- #
class SpyClassifier(BaseEstimator, ClassifierMixin):
    seen: list = []

    def __init__(self, depth: int = 1):
        self.depth = depth

    def fit(self, X, y):
        SpyClassifier.seen.append(set(X.index))
        self.classes_ = np.unique(y)
        self.majority_ = pd.Series(y).value_counts().index[0]
        return self

    def predict(self, X):
        return np.full(len(X), self.majority_)


def test_test_rows_are_used_only_for_the_final_evaluation(monkeypatch, tmp_path):
    df = fixture_table()
    split = pp.make_split(df, "problem")
    spy = tm.Candidate("Spy", "none", SpyClassifier(), {"model__depth": [1, 2]}, 0, ("all",))
    monkeypatch.setattr(tm, "candidates", lambda quick=False, seed=0: [spy])
    monkeypatch.setattr(tm, "_extra_checks", lambda *a, **k: {"random_split_test_macro_f1": 0.0})
    SpyClassifier.seen = []
    tm.run(quick=True, out_dir=tmp_path, models_dir=tmp_path, df=df, split=split,
           save_model=False, verbose=False, n_jobs=1)

    train, val, test = (set(split.indices(p)) for p in ("train", "validation", "test"))
    dev = train | val
    *before_final, final = SpyClassifier.seen
    assert len(before_final) == 2 * 5 + tm.RCV_REPEATS * 5  # grid search + repeated CV
    for rows in before_final[:10]:
        assert rows <= train  # tuning: training rows only
    for rows in before_final[10:]:
        assert rows <= dev  # comparison: train + validation only
    assert final == dev
    assert not any(rows & test for rows in SpyClassifier.seen)


# --------------------------------------------------------------------------- #
# Full quick run on the test table
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def quick_run(tmp_path_factory):
    out = tmp_path_factory.mktemp("models")
    df = fixture_table()
    split = pp.make_split(df, "problem")
    result = tm.run(quick=True, out_dir=out, models_dir=out, df=df, split=split, verbose=False, n_jobs=1)
    return result, out, df, split


def test_comparison_table_has_all_models_and_valid_metrics(quick_run):
    result, out, _, _ = quick_run
    table = pd.read_csv(out / "model_comparison.csv")
    assert {c.name for c in tm.candidates(quick=True)} <= set(table["model"])
    assert set(table["features"]) == {"all"}  # the test table has no pattern columns
    for col in ("rcv_macro_f1", "test_macro_f1", "test_accuracy", "test_weighted_f1",
                "test_macro_precision", "test_macro_recall", "test_balanced_accuracy"):
        assert table[col].between(0, 1).all(), col
    assert (table["test_macro_f1_ci_low"] <= table["test_macro_f1_ci_high"]).all()


def test_selected_model_follows_the_rule(quick_run):
    result, _, _, _ = quick_run
    i = tm.select(result["results"])
    assert result["selected_key"] == result["results"].loc[i, "key"]


def test_artifacts_saved_and_model_predicts(quick_run):
    result, out, df, split = quick_run
    model = joblib.load(out / config.MODEL_PATH.name)
    meta = json.loads((out / config.MODEL_METADATA_PATH.name).read_text(encoding="utf-8"))
    assert meta["selected_model"] == result["selected"]
    assert meta["features"] == list(FILE_FEATURE_NAMES)
    assert meta["classes"] == pp.CLASS_ORDER
    pred = model.predict(df.loc[split.indices("test"), meta["features"]])
    assert set(pred) <= set(pp.CLASS_ORDER)
    for name in ("confusion_matrices.png", "confusion_matrix_selected.png", "confusion_matrix_selected.csv",
                 "model_report.md", "model_comparison.csv"):
        assert (out / name).exists(), name


def test_report_states_what_is_and_is_not_established(quick_run):
    _, out, _, _ = quick_run
    report = (out / "model_report.md").read_text(encoding="utf-8")
    for text in ("What this experiment establishes", "does **not** show accuracy on everyday code",
                 "Why this model was selected", "not used for selection"):
        assert text in report


def test_extra_checks_reported(quick_run):
    checks = quick_run[0]["metadata"]["extra_checks"]
    assert 0 <= checks["random_split_test_macro_f1"] <= 1


# --------------------------------------------------------------------------- #
# Real artifacts (only when trained)
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not config.MODEL_METADATA_PATH.exists(), reason="model not trained")
def test_real_model_metadata():
    meta = json.loads(config.MODEL_METADATA_PATH.read_text(encoding="utf-8"))
    assert meta["features"] == pp.feature_columns(meta["feature_set"])
    assert 0 < meta["test_metrics"]["macro_f1"] <= 1
    assert "Not validated on everyday code" in meta["scope"]
