"""Tests for src.explainability (Phase 14).

A small Random Forest is trained on a synthetic table whose label depends on
two known features, so the tests can check that the methods find them. The
values are meaningless otherwise; the tests check the output format and the
mathematical properties (additivity, ordering, staleness), not accuracy.
The last section runs against the real model when it exists.
"""

import json
import textwrap

import joblib
import numpy as np
import pandas as pd
import pytest
import sklearn
from sklearn.ensemble import RandomForestClassifier

from src import model_integrity
from src import config
from src import explainability as ex
from src import predictor as pr
from src import preprocessing as pp
from src.pattern_features import PATTERN_FEATURE_NAMES
from tests.test_preprocessing import fixture_table

FEATURES = pp.feature_columns("patterns")
LOOPY = textwrap.dedent('''
    def count_pairs(a):
        total = 0
        for i in range(len(a)):
            for j in range(i + 1, len(a)):
                if a[i] + a[j] == 0:
                    total += 1
        return total
''')


def synthetic_table() -> pd.DataFrame:
    """Label = f(max_input_loop_depth, sort_call_count); everything else is noise."""
    df = fixture_table(n_problems=84, per_problem=5)
    rng = np.random.default_rng(1)
    for name in PATTERN_FEATURE_NAMES:
        df[name] = rng.integers(0, 3, size=len(df))
    depth = rng.integers(0, 4, size=len(df))
    sort = rng.integers(0, 2, size=len(df))
    df["max_input_loop_depth"] = depth
    df["sort_call_count"] = sort
    df["loop_count"] = depth  # a perfectly correlated partner of max_input_loop_depth
    df["label"] = np.select([depth == 0, (depth == 1) & (sort == 1), depth == 1, depth == 2],
                            ["constant", "nlogn", "linear", "quadratic"], "cubic")
    return df


@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    out = tmp_path_factory.mktemp("model")
    df = synthetic_table()
    split = pp.make_split(df, "problem")
    dev = split.indices("train").append(split.indices("validation"))
    est = RandomForestClassifier(n_estimators=40, min_samples_leaf=2, class_weight="balanced_subsample", random_state=0)
    pipe = pp.make_pipeline("tree", est, FEATURES).fit(df.loc[dev, FEATURES], df.loc[dev, "label"])
    # the real classes include logn and np; add one row each so the model knows all seven
    if set(pipe.classes_) != set(pp.CLASS_ORDER):
        extra = df.loc[dev].head(2).copy()
        extra["label"] = ["logn", "np"]
        X = pd.concat([df.loc[dev, FEATURES], extra[FEATURES]])
        y = pd.concat([df.loc[dev, "label"], extra["label"]])
        pipe = pp.make_pipeline("tree", est, FEATURES).fit(X, y)
    joblib.dump(pipe, out / "complexity_model.joblib")
    joblib.dump(pipe.named_steps["preprocess"], out / "preprocessing_pipeline.joblib")
    meta = {"selected_model": "Random Forest", "feature_set": "patterns", "features": FEATURES,
            "classes": pp.CLASS_ORDER, "test_metrics": {"macro_f1": None},
            "versions": {"scikit-learn": sklearn.__version__}}
    meta = model_integrity.attach(meta, out)
    (out / "model_metadata.json").write_text(json.dumps(meta), encoding="utf-8")
    model = pr.load_model(out / "complexity_model.joblib", out / "model_metadata.json",
                          out / "preprocessing_pipeline.joblib")
    g = ex.compute_global_explanation(df, split, model, n_repeats=3, verbose=False)
    return {"dir": out, "df": df, "split": split, "model": model, "global": g}


def row_values(df, i=0):
    return {f: float(df.iloc[i][f]) for f in FEATURES}


# --------------------------------------------------------------------------- #
# Feature catalogue
# --------------------------------------------------------------------------- #
def test_catalogue_covers_every_model_feature_once():
    cat = ex.feature_catalogue()
    assert list(cat["feature"]) == FEATURES
    assert set(cat.columns) == {"feature", "label", "group", "kind", "description"}
    assert cat["label"].str.len().min() > 3 and cat["description"].str.len().min() > 10
    assert (cat["group"] != "Other").all()
    grouped = [f for fs in ex.FEATURE_GROUPS.values() for f in fs]
    assert sorted(grouped) == sorted(FEATURES)  # groups partition the features


def test_describe_feature():
    d = ex.describe_feature("max_input_loop_depth")
    assert d["group"] == "Loops" and d["kind"] == "algorithm pattern" and "loop" in d["description"]
    with pytest.raises(KeyError):
        ex.describe_feature("not_a_feature")


# --------------------------------------------------------------------------- #
# Global importance
# --------------------------------------------------------------------------- #
def test_native_importance_format(setup):
    nat = ex.native_importance(setup["model"].pipeline, FEATURES)
    assert list(nat.columns) == ["feature", "importance", "std_across_trees"]
    assert nat["importance"].sum() == pytest.approx(1.0)
    assert nat["importance"].is_monotonic_decreasing
    assert set(nat["feature"].head(3)) & {"max_input_loop_depth", "loop_count", "sort_call_count"}


def test_global_file_format(setup):
    g = setup["global"]
    for key in ("model_sha256", "settings", "baseline", "native", "permutation", "groups", "tree_path_mean_abs",
                "rank_agreement", "correlated", "training_quantiles", "caveats", "versions"):
        assert key in g
    assert len(g["native"]) == len(g["permutation"]) == len(g["tree_path_mean_abs"]) == 57
    assert {r["group"] for r in g["groups"]} == set(ex.FEATURE_GROUPS)
    assert set(g["permutation"][0]["per_class"]) == {pr.BIGO_OF[c] for c in setup["model"].classes}
    assert len(g["training_quantiles"]["code_lines"]) == 101
    assert any("not causation" in c for c in g["caveats"])
    json.dumps(g)  # serialisable


def test_permutation_finds_the_informative_features(setup):
    perm = ex.global_importance_table(setup["global"], "permutation")
    top = set(perm["feature"].head(3))
    assert "sort_call_count" in top  # the only carrier of the O(n) / O(n log n) distinction
    noise = perm.set_index("feature").loc["comment_ratio", "importance"]
    assert noise < 0.02
    assert list(perm["rank"]) == list(range(1, 58))


def test_correlated_partners_split_importance_but_the_group_does_not(setup):
    g = setup["global"]
    perm = {r["feature"]: r["importance"] for r in g["permutation"]}
    loops = {r["group"]: r["importance"] for r in g["groups"]}["Loops"]
    # loop_count duplicates max_input_loop_depth, so shuffling either alone hurts little; shuffling the group hurts a lot
    assert loops > max(perm["loop_count"], perm["max_input_loop_depth"]) + 0.1
    assert {"feature": "loop_count", "spearman": 1.0} in g["correlated"]["max_input_loop_depth"]


def test_permutation_is_reproducible(setup):
    m, df, split = setup["model"], setup["df"], setup["split"]
    te = split.indices("test")
    cols = {"sort_call_count": ["sort_call_count"]}
    a = ex.permutation_importance(m.pipeline, df.loc[te, FEATURES], df.loc[te, "label"], cols, 3, seed=5)
    b = ex.permutation_importance(m.pipeline, df.loc[te, FEATURES], df.loc[te, "label"], cols, 3, seed=5)
    pd.testing.assert_frame_equal(a, b)


def test_tables_for_the_frontend(setup):
    g = setup["global"]
    for kind in ("permutation", "native", "tree_path"):
        t = ex.global_importance_table(g, kind, top=5)
        assert len(t) == 5 and {"rank", "feature", "importance", "label", "group", "description",
                                "strongly_correlated_with"} <= set(t.columns)
        assert t["importance"].is_monotonic_decreasing
    with pytest.raises(ValueError):
        ex.global_importance_table(g, "magic")
    grp = ex.group_importance_table(g)
    assert list(grp.columns) == ["group", "importance", "std", "positive_repeats", "features", "n_features"]
    assert grp["n_features"].sum() == 57
    ct = ex.class_importance_table(g, "O(n log n)", top=4)
    assert len(ct) == 4 and "sort_call_count" in set(ct["feature"])
    with pytest.raises(KeyError):
        ex.class_importance_table(g, "O(n^7)")


def test_share_below():
    q = list(np.linspace(0, 100, 101))
    assert ex.share_below(-1, q) == 0.0
    assert ex.share_below(50, q) == pytest.approx(50.0)
    assert ex.share_below(1e9, q) == 100.0
    assert ex.share_below(5, None) is None and ex.share_below(float("nan"), q) is None


def test_outputs_report_and_staleness(setup, tmp_path):
    g = dict(setup["global"])
    model_file = tmp_path / "model.joblib"
    model_file.write_bytes(b"version 1")
    g["model_sha256"] = ex._sha256(model_file)
    ex.write_outputs(g, tmp_path / "report", tmp_path / "explainability.json")
    report = (tmp_path / "report" / "explainability_report.md").read_text(encoding="utf-8")
    assert "association, not causation" in report and "not the reason for any single prediction" in report
    assert (tmp_path / "report" / "permutation_importance.png").exists()
    assert (tmp_path / "report" / "group_importance.png").exists()

    loaded = ex.load_global_explanation(tmp_path / "explainability.json", model_file)
    assert loaded.available and not loaded.stale and loaded.data["settings"]["n_repeats"] == 3
    model_file.write_bytes(b"version 2")  # model retrained
    stale = ex.load_global_explanation(tmp_path / "explainability.json", model_file)
    assert stale.available and stale.stale and "rerun" in stale.message
    missing = ex.load_global_explanation(tmp_path / "nope.json", model_file)
    assert not missing.available and "python -m src.explainability" in missing.message


# --------------------------------------------------------------------------- #
# Per-prediction explanations
# --------------------------------------------------------------------------- #
def test_local_explanation_format_and_additivity(setup):
    m, df = setup["model"], setup["df"]
    for i in (0, 7, 33):
        e = ex.explain_features(row_values(df, i), model=m, global_data=setup["global"])
        assert e.available and e.method == "tree_path" and e.error is None
        assert e.additivity_error < 1e-9
        total = e.base_value * 100 + sum(c.contribution for c in e.contributions)
        assert total == pytest.approx(e.vote_share * 100, abs=0.01)  # displayed values are rounded
        assert sum(e.groups.values()) == pytest.approx(total - e.base_value * 100, abs=1e-2)
        assert len(e.contributions) == 57
        mags = [abs(c.contribution) for c in e.contributions]
        assert mags == sorted(mags, reverse=True)
        assert all(c.contribution > 0 and c.direction == "toward" for c in e.top_toward)
        assert all(c.contribution < 0 and c.direction == "away from" for c in e.top_away)
        assert e.explained_class == e.predicted_class


def test_local_explanation_dict_for_the_frontend(setup):
    e = ex.explain_features(row_values(setup["df"]), model=setup["model"], top=3,
                            global_data=setup["global"])
    d = e.to_dict()
    assert set(d) == {"available", "method", "method_description", "explained_class", "predicted_class", "vote_share",
                      "base_value", "contributions", "top_toward", "top_away", "groups", "additivity_error",
                      "summary", "notes", "caveats", "error"}
    assert set(d["contributions"][0]) == {"feature", "label", "group", "value", "contribution", "direction",
                                          "share_of_training_below", "description", "test_permutation_importance"}
    assert d["contributions"][0]["test_permutation_importance"] is not None
    assert len(d["top_toward"]) <= 3 and len(d["top_away"]) <= 3
    assert d["contributions"][0]["share_of_training_below"] is not None
    assert any("not causes" in s for s in d["summary"])
    assert any("not why the code has this complexity" in c for c in d["caveats"])
    t = e.table(top=5)
    assert len(t) == 5 and "Contribution (points)" in t.columns
    json.dumps(d)


def test_the_explanation_points_at_the_deciding_feature(setup):
    df = setup["df"]
    row = df[(df["max_input_loop_depth"] == 1) & (df["sort_call_count"] == 1)].index[0]
    e = ex.explain_features(row_values(df, df.index.get_loc(row)), model=setup["model"], target="O(n log n)")
    assert "sort_call_count" in [c.feature for c in e.top_toward]


def test_explaining_another_class(setup):
    e = ex.explain_features(row_values(setup["df"]), model=setup["model"], target="O(2^n)")
    assert e.explained_class == "O(2^n)" and e.additivity_error < 1e-9
    assert ex.explain_features(row_values(setup["df"]), model=setup["model"], target="np").explained_class == "O(2^n)"
    with pytest.raises(ValueError):
        ex.explain_features(row_values(setup["df"]), model=setup["model"], target="O(n^9)")
    with pytest.raises(ValueError):
        ex.explain_features(row_values(setup["df"]), model=setup["model"], method="lime")


def test_bad_input_and_missing_model_are_reported_not_invented(setup, monkeypatch):
    vals = row_values(setup["df"])
    del vals["code_lines"]
    e = ex.explain_features(vals, model=setup["model"])
    assert not e.available and e.error["code"] == "feature_schema_error" and e.contributions == []
    vals = row_values(setup["df"])
    vals["code_lines"] = float("inf")
    assert ex.explain_features(vals, model=setup["model"]).error["code"] == "feature_schema_error"

    def missing(*a, **k):
        raise pr.ModelUnavailable("model_missing", "Trained model files not found.")

    monkeypatch.setattr(pr, "get_model", missing)
    e = ex.explain_features(row_values(setup["df"]))
    assert not e.available and e.error["code"] == "model_missing"


def test_low_confidence_note(setup, monkeypatch):
    monkeypatch.setattr(ex, "LOW_CONFIDENCE", 1.01)  # every vote counts as unsure
    e = ex.explain_features(row_values(setup["df"]), model=setup["model"], global_data={})
    assert any("uncertain vote" in n for n in e.notes)
    assert any("Global context" in n for n in e.notes)  # no global data given


def test_weak_features_are_flagged(setup):
    g = json.loads(json.dumps(setup["global"]))
    for r in g["permutation"]:
        r["importance"] = -0.01  # pretend nothing helped on the test set
    e = ex.explain_features(row_values(setup["df"]), model=setup["model"], global_data=g)
    assert any("did not lower its test-set score" in n for n in e.notes)


def test_explain_source(setup):
    e = ex.explain_source(LOOPY, model=setup["model"])
    assert e.available and len(e.contributions) == 57
    by = {c.feature: c for c in e.contributions}
    assert by["max_input_loop_depth"].value == 2
    bad = ex.explain_source("def broken(:\n    pass\n", model=setup["model"])
    assert not bad.available and bad.error["code"] == "syntax"
    assert ex.explain_source(42, model=setup["model"]).error["code"] == "invalid_input_type"


def test_shap_missing_is_a_clear_error(setup, monkeypatch):
    monkeypatch.setattr(ex, "shap_available", lambda: False)
    e = ex.explain_features(row_values(setup["df"]), model=setup["model"], method="shap")
    assert not e.available and e.error["code"] == "shap_not_installed" and "pip install shap" in e.error["hint"]


@pytest.mark.skipif(not ex.shap_available(), reason="optional shap package not installed")
def test_shap_explanation_is_additive_and_agrees(setup):
    vals = row_values(setup["df"], 5)
    s = ex.explain_features(vals, model=setup["model"], method="shap")
    t = ex.explain_features(vals, model=setup["model"])
    assert s.available and s.method == "shap" and s.additivity_error < 1e-6
    assert s.vote_share == t.vote_share
    assert s.contributions[0].feature in {c.feature for c in t.contributions[:5]}


# --------------------------------------------------------------------------- #
# Predictor integration
# --------------------------------------------------------------------------- #
def test_predict_with_explanation(setup):
    d = setup["dir"]
    paths = {"model_path": d / "complexity_model.joblib", "metadata_path": d / "model_metadata.json",
             "preprocessing_path": d / "preprocessing_pipeline.joblib"}
    r = pr.predict(LOOPY, explain=True, **paths)
    assert r.explanation["available"] and r.explanation["explained_class"] == r.prediction.big_o
    assert r.explanation["vote_share"] == pytest.approx(r.prediction.probability, abs=1e-4)
    assert pr.predict(LOOPY, **paths).explanation is None  # optional, off by default


# --------------------------------------------------------------------------- #
# Real model (only when trained locally)
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not config.MODEL_PATH.exists(), reason="real model not trained")
def test_real_model_explanation():
    e = ex.explain_source(LOOPY)
    assert e.available and len(e.contributions) == 57 and e.additivity_error < 1e-9
    if ex.GLOBAL_PATH.exists():
        assert e.contributions[0].share_of_training_below is not None


@pytest.mark.skipif(not ex.GLOBAL_PATH.exists(), reason="global explanation not computed")
def test_real_global_file_is_consistent():
    g = ex.load_global_explanation()
    assert g.available
    assert g.data["baseline"]["matches_metadata"]
    assert len(g.data["permutation"]) == 57
    assert not g.stale
