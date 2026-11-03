"""Tests for src.tune_model and src.evaluate_model (Phase 11).

Small runs on the deterministic test table from test_preprocessing (with
pattern columns added), plus a tiny BigO(Bench)-format file written from the
documented schema. These check the protocol and outputs, not real-data quality.
"""

import json

import joblib
import numpy as np
import pandas as pd
import pytest

from src import config
from src import evaluate_model as ev
from src import preprocessing as pp
from src import tune_model
from src.pattern_features import PATTERN_FEATURE_NAMES
from tests.test_preprocessing import fixture_table


@pytest.fixture(scope="module")
def table():
    df = fixture_table()
    rng = np.random.default_rng(1)
    for name in PATTERN_FEATURE_NAMES:
        df[name] = rng.integers(0, 3, size=len(df))
    tags = ["dp", "math", "dp,math", "greedy", "sortings", "binary search", "bitmasks"]
    df["tags"] = [tags[int(p.split("_")[0]) % len(tags)] for p in df["problem"]]
    return df


@pytest.fixture(scope="module")
def split(table):
    return pp.make_split(table, "problem")


@pytest.fixture(scope="module")
def tuning(table, split, tmp_path_factory):
    path = tmp_path_factory.mktemp("tune") / "tuning.json"
    return tune_model.run(quick=True, df=table, split=split, out_path=path, n_jobs=1, verbose=False)


# --------------------------------------------------------------------------- #
# Tuning
# --------------------------------------------------------------------------- #
def test_tuning_result_structure(tuning):
    assert tuning["adopted"] in ("tuned", "phase10")
    assert set(tuning["searches"]) == {"patterns", "patterns_no_size"}
    source = tuning["tuned"] if tuning["adopted"] == "tuned" else tuning["before"]
    assert tuning["chosen"] == {k: source[k] for k in ("model", "feature_set", "params")}
    assert tuning["chosen"]["model"] in tune_model.FOREST_MODELS
    for part in ("before", "tuned"):
        assert 0 <= tuning[part]["rcv_macro_f1"] <= 1
    assert tuning["rcv_gain"] == pytest.approx(tuning["tuned"]["rcv_macro_f1"] - tuning["before"]["rcv_macro_f1"])


def test_tuning_adopts_only_above_margin(tuning):
    assert (tuning["adopted"] == "tuned") == (tuning["rcv_gain"] > tune_model.ADOPT_MARGIN)


def test_tuning_never_sees_test_rows(table, split, monkeypatch, tmp_path):
    seen = []
    original = tune_model.RandomizedSearchCV.fit

    def spy(self, X, y=None, **kw):
        seen.append(set(X.index))
        return original(self, X, y, **kw)

    monkeypatch.setattr(tune_model.RandomizedSearchCV, "fit", spy)
    tune_model.run(quick=True, df=table, split=split, out_path=tmp_path / "t.json", n_jobs=1, verbose=False)
    test_rows = set(split.indices("test"))
    assert seen and all(not (rows & test_rows) for rows in seen)


def test_tuning_file_saved(tuning, tmp_path):
    path = tmp_path / "t.json"
    path.write_text(json.dumps(tuning, default=str))
    assert tune_model.load_chosen(path)["chosen"]["feature_set"] in pp.FEATURE_SETS


# --------------------------------------------------------------------------- #
# BigO(Bench) helpers
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("raw, expected", [
    ("o(1)", "constant"), ("O(log(n))", "logn"), ("o(n)", "linear"), ("o(n*logn)", "nlogn"),
    ("O(n log n)", "nlogn"), ("o(n^2)", "quadratic"), ("o(n^3)", "cubic"), ("O(2^n)", "np"),
    ("o(n*m)", None), ("o(n+m)", None), ("O((n+m)log(n+m))", None), ("", None), (None, None),
])
def test_bigobench_label_mapping(raw, expected):
    assert ev.normalise_bigobench_label(raw) == expected


def test_bigobench_problem_id():
    assert ev.bigobench_problem_id({"problem_name": "992_C. Nastya and a Wardrobe"}) == "0992_C"
    assert ev.bigobench_problem_id({"problem_name": "x", "problem_id": "1548_E1"}) == "1548_E1"
    assert ev.bigobench_problem_id({"problem_name": "no id here"}) is None


def bigobench_file(path, overlap_problem):
    recs = [
        {"problem_name": "1001_A. Sum", "problem_id": "1", "solution_id": "a", "time_complexity_inferred": "o(1)",
         "solution_code": "a, b = map(int, input().split())\nprint(a + b)\n"},
        {"problem_name": "1001_B. Loop", "problem_id": "2", "solution_id": "b", "time_complexity_inferred": "o(n)",
         "solution_code": "n = int(input())\nfor i in range(n):\n    print(i)\n"},
        {"problem_name": "1001_C. Grid", "problem_id": "3", "solution_id": "c", "time_complexity_inferred": "o(n*m)",
         "solution_code": "x = 1\n"},
        {"problem_name": "1001_D. Py2", "problem_id": "4", "solution_id": "d", "time_complexity_inferred": "o(n)",
         "solution_code": "print 'hi'\n"},
        {"problem_name": f"{int(overlap_problem.split('_')[0])}_{overlap_problem.split('_')[1]}. Seen",
         "problem_id": "5", "solution_id": "e", "time_complexity_inferred": "o(n^2)",
         "solution_code": "n = int(input())\nfor i in range(n):\n    for j in range(n):\n        pass\n"},
    ]
    path.write_text("\n".join(json.dumps(r) for r in recs) + "\n", encoding="utf-8")
    return path


def test_load_bigobench_counts_exclusions(table, tmp_path):
    seen = table["problem"].iloc[0]
    path = bigobench_file(tmp_path / "time_complexity_test_set.jsonl", seen)
    ext, stats = ev.load_bigobench(path, set(table["problem"]))
    assert stats["records"] == 5
    assert stats["unmapped_label"] == 1 and stats["labels_unmapped"] == {"o(n*m)": 1}
    assert stats["unparsable"] == 1
    assert stats["overlapping_problem"] == 1
    assert len(ext) == 3 and ext["overlap"].sum() == 1
    assert set(PATTERN_FEATURE_NAMES) <= set(ext.columns)


# --------------------------------------------------------------------------- #
# Full evaluation run
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def evaluation(table, split, tuning, tmp_path_factory):
    out = tmp_path_factory.mktemp("eval")
    ext_path = bigobench_file(out / "bigo.jsonl", table["problem"].iloc[0])
    result = ev.run(df=table, split=split, tuning=tuning, out_dir=out, models_dir=out,
                    external_path=ext_path, verbose=False)
    return result, out


def test_test_metrics_and_per_class(evaluation, split):
    r, _ = evaluation
    m = r["test"]["metrics"]
    for k in ("accuracy", "macro_f1", "weighted_f1", "macro_precision", "macro_recall", "balanced_accuracy"):
        assert 0 <= m[k] <= 1
    assert m["macro_f1_ci"][0] <= m["macro_f1_ci"][1]
    assert list(r["test"]["per_class"].columns) == ["precision", "recall", "f1", "support"]
    assert r["test"]["cm"].sum() == len(split.indices("test"))


def test_generalization_sections_present(evaluation):
    r, _ = evaluation
    h = r["holdouts"]
    assert 0 <= h["problem_grouped_oof"]["macro_f1"] <= 1 and 0 <= h["contest_grouped_oof"]["macro_f1"] <= 1
    assert h["families"], "family hold-outs expected"
    for fam in h["families"].values():
        assert fam["macro_f1_change"] == pytest.approx(
            fam["family_held_out"]["macro_f1"] - fam["family_seen_problem_held_out"]["macro_f1"], abs=1e-4)
    assert r["implementations"]["test_problems"] > 0
    assert "all test programs" in r["slices"].index
    assert r["external"]["unseen_problems"]["n"] == 2


def test_family_mask_separates_training_rows(table):
    for tag in ("dp", "math", "sortings"):
        has = ev.family_mask(table, tag)
        assert has.any() and (~has).any()
        assert table.loc[has, "tags"].str.contains(tag).all()
        assert not table.loc[~has, "tags"].str.split(",").apply(lambda ts, t=tag: t in ts).any()


def test_failure_analysis(evaluation):
    r, _ = evaluation
    f = r["failures"]
    assert sum(f["distance_%"].values()) == pytest.approx(100, abs=0.2)
    assert set(f["confident_errors"].columns) >= {"sample_id", "true", "predicted", "confidence"}
    assert "src" not in f["confident_errors"].columns  # no source code copied


def test_outputs_and_saved_models(evaluation, table, split):
    r, out = evaluation
    for name in ("evaluation_report.md", "evaluation_results.json", "confusion_matrix_test.png", "per_class_test.png",
                 "family_holdout.png", "structural_slices.png", "failure_cases.csv"):
        assert (out / name).exists(), name
    model = joblib.load(out / config.MODEL_PATH.name)
    pre = joblib.load(out / ev.PREPROCESSING_PATH.name)
    meta = json.loads((out / config.MODEL_METADATA_PATH.name).read_text(encoding="utf-8"))
    X = table.loc[split.indices("test")]
    assert set(model.predict(X)) <= set(pp.CLASS_ORDER)
    assert pre.transform(X).shape[1] == len(meta["features"])
    assert meta["phase11"] is True and meta["features"] == pp.feature_columns(meta["feature_set"])
    report = (out / "evaluation_report.md").read_text(encoding="utf-8")
    for text in ("## 1. Tuning", "### c. Algorithm families excluded from training", "## 4. Failure cases",
                 "should not be", "How the project could be improved"):
        assert text in report


def test_reproducible(table, split, tuning, tmp_path):
    a = ev.run(df=table, split=split, tuning=tuning, out_dir=tmp_path / "a", save=False, verbose=False)
    b = ev.run(df=table, split=split, tuning=tuning, out_dir=tmp_path / "b", save=False, verbose=False)
    assert a["test"]["metrics"] == b["test"]["metrics"]
    assert (a["test"]["pred"] == b["test"]["pred"]).all()
