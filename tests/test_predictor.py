"""Tests for src.predictor (Phase 12).

A small model is trained once on the deterministic test table and saved to a
temporary folder in the same format as ``python -m src.evaluate_model``, so
the pipeline is tested end to end without depending on the real model.
Its predictions are meaningless; these tests check behaviour, not accuracy.
The last section runs the same checks against the real model when it exists.
"""

import json
import textwrap

import joblib
import numpy as np
import pandas as pd
import pytest
import sklearn
from sklearn.ensemble import RandomForestClassifier

from src import config
from src import predictor as pr
from src import preprocessing as pp
from src.pattern_features import PATTERN_FEATURE_NAMES
from tests.test_preprocessing import fixture_table

MERGE_SORT = textwrap.dedent('''
    def merge(left, right):
        result = []
        i = j = 0
        while i < len(left) and j < len(right):
            if left[i] <= right[j]:
                result.append(left[i])
                i += 1
            else:
                result.append(right[j])
                j += 1
        return result + left[i:] + right[j:]


    def merge_sort(items):
        if len(items) <= 1:
            return items
        mid = len(items) // 2
        return merge(merge_sort(items[:mid]), merge_sort(items[mid:]))
''')

SCRIPT = textwrap.dedent('''
    n = int(input())
    a = list(map(int, input().split()))
    a.sort()
    best = 0
    for i in range(1, n):
        best = max(best, a[i] - a[i - 1])
    print(best)
''')


@pytest.fixture(scope="module")
def model_dir(tmp_path_factory):
    out = tmp_path_factory.mktemp("model")
    df = fixture_table()
    rng = np.random.default_rng(0)
    for name in PATTERN_FEATURE_NAMES:
        df[name] = rng.integers(0, 3, size=len(df))
    features = pp.feature_columns("patterns")
    pipe = pp.make_pipeline("tree", RandomForestClassifier(n_estimators=20, random_state=0), features)
    pipe.fit(df[features], df["label"])
    joblib.dump(pipe, out / "complexity_model.joblib")
    joblib.dump(pipe.named_steps["preprocess"], out / "preprocessing_pipeline.joblib")
    meta = {
        "selected_model": "Random Forest", "feature_set": "patterns", "features": features,
        "classes": pp.CLASS_ORDER, "trained_on": "test fixture",
        "test_metrics": {"macro_f1": 0.5, "macro_f1_ci": [0.4, 0.6], "accuracy": 0.5},
        "confidence_reliability_test": [
            {"from": 0.0, "to": 0.4, "n": 10, "accuracy": 0.2}, {"from": 0.4, "to": 0.6, "n": 10, "accuracy": 0.4},
            {"from": 0.6, "to": 0.8, "n": 10, "accuracy": 0.6}, {"from": 0.8, "to": 1.0, "n": 10, "accuracy": 0.8}],
        "versions": {"scikit-learn": sklearn.__version__},
        "scope": "test fixture",
    }
    (out / "model_metadata.json").write_text(json.dumps(meta), encoding="utf-8")
    return out


def paths(d):
    return {"model_path": d / "complexity_model.joblib", "metadata_path": d / "model_metadata.json",
            "preprocessing_path": d / "preprocessing_pipeline.joblib"}


# --------------------------------------------------------------------------- #
# Real example inputs
# --------------------------------------------------------------------------- #
def test_merge_sort_full_result(model_dir):
    r = pr.predict(MERGE_SORT, **paths(model_dir))
    assert r.success and r.errors == []
    assert set(r.file_metrics) == set(pp.feature_columns("patterns"))
    assert [f.function_id for f in r.functions] == ["merge", "merge_sort"]
    ms = r.functions[1]
    assert ms.static_estimate["time_complexity"] == "O(n log n)"  # rule-based analyser
    assert ms.structural_category == "Low"
    assert r.structural_category["kind"].startswith("rule-based")
    assert r.prediction is not None and r.prediction.big_o in pr.BIGO_OF.values()
    assert sum(r.prediction.probabilities.values()) == pytest.approx(1, abs=1e-3)
    assert list(r.prediction.probabilities) == [pr.BIGO_OF[c] for c in pp.CLASS_ORDER]
    assert r.model.available and r.model.feature_set == "patterns" and r.model.n_features == 57


def test_script_without_functions(model_dir):
    r = pr.predict(SCRIPT, **paths(model_dir))
    assert r.success and r.functions == []
    assert any(w.code == "no_functions" for w in r.warnings)
    assert r.prediction is not None  # whole-file prediction still made
    assert r.file_metrics["sort_call_count"] == 1


def test_per_function_predictions_are_flagged_experimental(model_dir):
    r = pr.predict(MERGE_SORT, **paths(model_dir))
    preds = [f.ml_prediction for f in r.functions if f.ml_prediction]
    assert preds and all(p.experimental for p in preds)
    assert any(w.code == "experimental_function_predictions" for w in r.warnings)
    r2 = pr.predict(MERGE_SORT, per_function_ml=False, **paths(model_dir))
    assert all(f.ml_prediction is None for f in r2.functions)


def test_bytes_input(model_dir):
    r = pr.predict(MERGE_SORT.encode("utf-8"), **paths(model_dir))
    assert r.success and len(r.functions) == 2


def test_measured_accuracy_attached_and_probability_not_invented(model_dir):
    p = pr.predict(MERGE_SORT, **paths(model_dir)).prediction
    m = p.measured_accuracy_in_range
    assert m is not None and m["probability_range"][0] <= p.probability <= m["probability_range"][1]
    assert p.margin == pytest.approx(p.probability - p.probabilities[p.runner_up], abs=1e-3)


def test_low_confidence_warning(model_dir, monkeypatch):
    monkeypatch.setattr(pr, "LOW_CONFIDENCE", 1.01)
    r = pr.predict(MERGE_SORT, **paths(model_dir))
    assert any(w.code == "low_confidence" for w in r.warnings)


# --------------------------------------------------------------------------- #
# Errors for unsupported input
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("src, code", [
    ("", "empty"), ("   \n", "empty"), ("def f(:\n    pass\n", "syntax"),
    ("def f():\nreturn 1\n", "indentation"), ("x = 1\n" * 6000, "too_large"),
], ids=["empty", "blank", "syntax", "indentation", "too-large"])  # short ids: Windows limits env var length
def test_invalid_code_returns_clear_errors(model_dir, src, code):
    r = pr.predict(src, **paths(model_dir))
    assert not r.success and r.errors[0].code == code and r.errors[0].message
    assert r.prediction is None and r.functions == [] and r.file_metrics is None


@pytest.mark.parametrize("bad", [None, 42, ["code"]])
def test_wrong_input_type(model_dir, bad):
    r = pr.predict(bad, **paths(model_dir))
    assert not r.success and r.errors[0].code == "invalid_input_type"


# --------------------------------------------------------------------------- #
# Model problems: clear errors, never fake predictions
# --------------------------------------------------------------------------- #
def test_missing_model_gives_error_and_no_prediction(tmp_path):
    r = pr.predict(MERGE_SORT, **paths(tmp_path))
    assert r.success  # static analysis still works
    assert r.prediction is None and all(f.ml_prediction is None for f in r.functions)
    assert r.errors[0].code == "model_missing" and "evaluate_model" in r.errors[0].hint
    assert r.model.available is False
    assert r.functions[1].static_estimate["time_complexity"] == "O(n log n)"


def _copy(src_dir, dst_dir):
    for name in ("complexity_model.joblib", "model_metadata.json", "preprocessing_pipeline.joblib"):
        (dst_dir / name).write_bytes((src_dir / name).read_bytes())
    return dst_dir


def test_reordered_feature_schema_is_rejected(model_dir, tmp_path):
    d = _copy(model_dir, tmp_path)
    meta = json.loads((d / "model_metadata.json").read_text())
    meta["features"] = list(reversed(meta["features"]))
    (d / "model_metadata.json").write_text(json.dumps(meta))
    r = pr.predict(MERGE_SORT, **paths(d))
    assert r.prediction is None and r.errors[0].code == "model_incompatible"


def test_unknown_feature_set_is_rejected(model_dir, tmp_path):
    d = _copy(model_dir, tmp_path)
    meta = json.loads((d / "model_metadata.json").read_text())
    meta["feature_set"] = "all"  # 39 features, but the model expects 57
    (d / "model_metadata.json").write_text(json.dumps(meta))
    assert pr.predict(MERGE_SORT, **paths(d)).errors[0].code == "model_incompatible"


def test_corrupt_model_file(model_dir, tmp_path):
    d = _copy(model_dir, tmp_path)
    (d / "complexity_model.joblib").write_bytes(b"not a model")
    r = pr.predict(MERGE_SORT, **paths(d))
    assert r.prediction is None and r.errors[0].code == "model_load_failed"


def test_version_mismatch_is_a_warning(model_dir, tmp_path):
    d = _copy(model_dir, tmp_path)
    meta = json.loads((d / "model_metadata.json").read_text())
    meta["versions"]["scikit-learn"] = "0.0.1"
    (d / "model_metadata.json").write_text(json.dumps(meta))
    r = pr.predict(MERGE_SORT, **paths(d))
    assert r.prediction is not None and any(w.code == "version_mismatch" for w in r.warnings)


# --------------------------------------------------------------------------- #
# Feature order, caching, no retraining, no execution
# --------------------------------------------------------------------------- #
def test_feature_vector_preserves_training_order():
    feats = pp.feature_columns("patterns")
    values = {f: i for i, f in enumerate(reversed(feats))}  # dict in a different order
    X = pr.feature_vector(feats, values)
    assert list(X.columns) == feats
    assert X.iloc[0].tolist() == [values[f] for f in feats]
    with pytest.raises(ValueError, match="missing"):
        pr.feature_vector(feats, {k: v for k, v in values.items() if k != feats[0]})
    bad = dict(values)
    bad[feats[3]] = float("inf")
    with pytest.raises(ValueError, match="non-finite"):
        pr.feature_vector(feats, bad)


def test_model_is_loaded_once_and_never_refitted(model_dir, monkeypatch):
    pr._cached_model.cache_clear()
    loads = []
    real = pr.load_model
    monkeypatch.setattr(pr, "load_model", lambda *a, **k: loads.append(1) or real(*a, **k))

    def no_fit(*a, **k):
        raise AssertionError("the model must not be retrained during prediction")

    monkeypatch.setattr(RandomForestClassifier, "fit", no_fit)
    for _ in range(3):
        assert pr.predict(MERGE_SORT, **paths(model_dir)).prediction is not None
    assert len(loads) == 1


def test_model_reloads_when_files_change(model_dir, tmp_path):
    d = _copy(model_dir, tmp_path)
    a = pr.get_model(**paths(d))
    assert pr.get_model(**paths(d)) is a
    import os
    st = (d / "model_metadata.json").stat()
    os.utime(d / "model_metadata.json", ns=(st.st_atime_ns, st.st_mtime_ns + 10_000_000))
    assert pr.get_model(**paths(d)) is not a


def test_submitted_code_is_never_executed(model_dir, tmp_path):
    marker = tmp_path / "ran.txt"
    src = f"open({str(marker)!r}, 'w').write('x')\ndef f(n):\n    return f(n - 1) if n else open({str(marker)!r}, 'w')\n"
    pr.predict(src, **paths(model_dir))
    assert not marker.exists()


def test_result_is_json_serialisable_and_table(model_dir):
    r = pr.predict(MERGE_SORT, **paths(model_dir))
    d = json.loads(json.dumps(r.to_dict()))
    assert d["success"] is True and d["prediction"]["big_o"]
    assert d["functions"][1]["static_estimate"]["time_complexity"] == "O(n log n)"
    table = r.functions_table()
    assert list(table["Function"]) == ["merge", "merge_sort"]
    assert "Structural category" in table.columns
    assert json.loads(pr.predict_json(SCRIPT, **paths(model_dir)))["success"] is True


# --------------------------------------------------------------------------- #
# The real model (only when trained)
# --------------------------------------------------------------------------- #
REAL = config.MODEL_PATH.exists() and config.MODEL_METADATA_PATH.exists() and pr.PREPROCESSING_PATH.exists()


@pytest.mark.skipif(not REAL, reason="real model not trained")
@pytest.mark.parametrize("src", [MERGE_SORT, SCRIPT])
def test_real_model_predictions(src):
    pr._cached_model.cache_clear()
    r = pr.predict(src)
    assert r.success and not [e for e in r.errors], r.errors
    assert r.model.available and r.model.n_features == len(r.model and pp.feature_columns(r.model.feature_set))
    assert sum(r.prediction.probabilities.values()) == pytest.approx(1, abs=1e-3)
    assert r.prediction.measured_accuracy_in_range is not None
