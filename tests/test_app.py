"""Tests for the Streamlit frontend (Phase 16).

* ``src.ui.logic`` is tested directly (validation, comparisons, artifacts).
* The app itself is run headlessly with Streamlit's AppTest: pages render,
  the analyzer handles empty, invalid, unsupported and oversized input, and
  model problems are shown as friendly messages, never as tracebacks.
"""

import json
import textwrap

import pandas as pd
import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

from src import config  # noqa: E402
from src.predictor import predict  # noqa: E402
from src.ui import logic  # noqa: E402
from src.ui.samples import COMPARE_AFTER, COMPARE_BEFORE, SAMPLES  # noqa: E402

APP = str(config.PROJECT_ROOT / "app.py")
TIMEOUT = 180


def page_text(at) -> str:
    return "\n".join(m.value for m in at.markdown)


def no_crash(at):
    assert not at.exception, [e.value for e in at.exception]
    assert "Traceback" not in page_text(at)


def analyze(code: str, language: str = "Python"):
    at = AppTest.from_file(APP, default_timeout=TIMEOUT)
    at.run()
    at.selectbox(key="analyzer_language").set_value(language)
    at.text_area(key="analyzer_code").set_value(code)
    at.run()
    btn = next(b for b in at.button if b.label == "Analyze")
    if not btn.disabled:
        btn.click().run()
    return at


# --------------------------------------------------------------------------- #
# logic (no Streamlit)
# --------------------------------------------------------------------------- #
def test_check_input():
    ok = logic.check_input("x = 1\n")
    assert ok.ok and ok.lines == 1 and ok.chars == 6
    assert logic.check_input("a\nb").lines == 2 and logic.check_input("").lines == 0
    assert [e["code"] for e in logic.check_input("   \n").errors] == ["empty"]
    assert [e["code"] for e in logic.check_input("x = 1", "Java").errors] == ["unsupported_language"]
    big = logic.check_input("x = 1\n" * (config.MAX_SOURCE_LINES + 1))
    assert not big.ok and big.errors[0]["code"] == "too_large"
    wide = logic.check_input("#" * (config.MAX_SOURCE_BYTES + 10))
    assert not wide.ok and wide.errors[0]["code"] == "too_large"
    mid = logic.check_input("x = 1\n" * (logic.WARN_LINES + 1))
    assert mid.ok and mid.warnings


def test_only_python_is_supported():
    assert [k for k, v in logic.LANGUAGES.items() if v["supported"]] == ["Python"]


def test_overview_and_function_frame_come_from_results():
    r = predict(SAMPLES["Merge sort (divide and conquer)"]).to_dict()
    o = logic.overview(r)
    assert o["function_count"] == 2 and o["structural_category"] == "Low"
    assert o["derived_score"] == r["derived_score"]["highest"]
    df = logic.functions_frame(r)
    assert list(df["Function"]) == ["merge", "merge_sort"]
    assert df.loc[1, "Time (static)"] == "O(n log n)"


def test_compare_uses_real_differences():
    a, b = predict(COMPARE_BEFORE).to_dict(), predict(COMPARE_AFTER).to_dict()
    num = logic.compare_numeric(a, b).set_index("Metric")
    row = num.loc["Maximum nesting depth"]
    assert row["Change"] == row["After"] - row["Before"]
    assert row["Before"] == a["file_summary"]["maximum_nesting_depth"]
    shown = logic.compare_overview(a, b)
    assert shown.map(lambda v: isinstance(v, str)).all().all()  # Arrow-safe display table
    assert "Structural category" in set(shown["Metric"])
    fn = logic.compare_functions(a, b)
    assert list(fn["Function"]) == ["count_zero_pairs"] and fn.loc[0, "Status"] == "unchanged id"
    assert fn.loc[0, "Time before"] == "O(n^2)" and fn.loc[0, "Time after"] == "O(n)"


def test_compare_added_and_removed_functions():
    a = predict("def f(x):\n    return x\n").to_dict()
    b = predict("def g(x):\n    return x\n").to_dict()
    fn = logic.compare_functions(a, b).set_index("Function")
    assert fn.loc["f", "Status"] == "removed" and fn.loc["g", "Status"] == "added"


def test_confusion_scores():
    cm = pd.DataFrame([[3, 1], [0, 4]], index=["a", "b"], columns=["a", "b"])
    s = logic.confusion_scores(cm)
    assert s["n"] == 8 and s["accuracy"] == pytest.approx(7 / 8)
    assert s["macro_f1"] == pytest.approx((2 * 0.75 / 1.75 + 2 * 0.8 * 1 / 1.8) / 2)


def test_artifacts_missing_and_corrupt(tmp_path):
    good = tmp_path / "good.json"
    good.write_text(json.dumps({"a": 1}))
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    art = logic.load_artifacts({"model_metadata": good, "evaluation_results": bad,
                                "model_comparison": tmp_path / "none.csv"})
    assert art.data["model_metadata"] == {"a": 1}
    assert "evaluation_results" in art.errors and "model_comparison" in art.missing
    assert art.missing["model_comparison"].startswith("python -m")


@pytest.mark.skipif(not logic.ARTIFACTS["model_comparison"].exists(), reason="model comparison not run")
def test_real_comparison_table_and_confusion_matrix():
    art = logic.load_artifacts()
    t = logic.comparison_table(art.data["model_comparison"])
    assert t["Test macro F1"].is_monotonic_decreasing
    if "confusion_matrix" in art.data and "model_metadata" in art.data:
        s = logic.confusion_scores(art.data["confusion_matrix"])
        assert s["macro_f1"] == pytest.approx(art.data["model_metadata"]["test_metrics"]["macro_f1"], abs=0.002)


# --------------------------------------------------------------------------- #
# App (headless)
# --------------------------------------------------------------------------- #
def test_app_starts_and_every_page_renders():
    at = AppTest.from_file(APP, default_timeout=TIMEOUT)
    at.run()
    no_crash(at)
    text = page_text(at)
    assert "SmartCode" in text and "Intelligent Static Code Analysis" in text
    assert "Analyze Code. Understand Complexity." in text
    for page in ["Compare versions", "Model evaluation", "Documentation", "Analyzer"]:
        at.sidebar.radio(key="nav").set_value(page).run()
        no_crash(at)


def test_analyze_sample_shows_dashboard():
    at = analyze(SAMPLES["Merge sort (divide and conquer)"])
    no_crash(at)
    text = page_text(at)
    for label in ("Structural complexity", "Derived Complexity Score", "Code lines", "Functions", "Classes", "Loops",
                  "Cyclomatic complexity", "Max nesting depth"):
        assert label.upper() in text.upper()
    assert len(at.tabs) >= 5
    assert any("merge_sort" in str(df.value.values) for df in at.dataframe)


def test_empty_code_is_rejected():
    at = analyze("")
    no_crash(at)
    assert "There is no code to analyse" in page_text(at)


def test_invalid_syntax_is_reported_safely():
    at = analyze("def broken(:\n    return <script>alert(1)</script>\n")
    no_crash(at)
    text = page_text(at)
    assert "Invalid Python syntax" in text
    assert "<script>" not in text  # user text is escaped


def test_unsupported_language_is_not_analysed():
    at = analyze("function f() { return 1; }", language="JavaScript")
    no_crash(at)
    assert next(b for b in at.button if b.label == "Analyze").disabled
    assert "JavaScript is not supported" in page_text(at)


def test_large_input_is_rejected():
    at = analyze("x = 1\n" * (config.MAX_SOURCE_LINES + 5))
    no_crash(at)
    assert "the limit is" in page_text(at)


def test_internal_failure_shows_reference_not_traceback(monkeypatch):
    import src.predictor as pr

    def boom(*a, **k):
        raise RuntimeError("secret internal detail")

    monkeypatch.setattr(pr, "predict", boom)
    at = analyze("def unique_failure_case(a):\n    return a * 2\n")
    no_crash(at)
    text = page_text(at)
    assert "internal error" in text and "reference" in text
    assert "secret internal detail" not in text


def test_compare_page():
    at = AppTest.from_file(APP, default_timeout=TIMEOUT)
    at.run()
    at.sidebar.radio(key="nav").set_value("Compare versions").run()
    next(b for b in at.button if b.label == "Compare").click().run()
    no_crash(at)
    assert "Summary of changes" in page_text(at)
    assert len(at.dataframe) >= 1


# --------------------------------------------------------------------------- #
# Model problems (rendered from real predictor results)
# --------------------------------------------------------------------------- #
RENDER = '''
import streamlit as st
from src.ui import components as ui
from src.ui.views import results
ui.inject_theme()
results.render(st.session_state["fake_result"], key="t")
'''


def _render_result(result: dict):
    at = AppTest.from_string(RENDER, default_timeout=TIMEOUT)
    at.session_state["fake_result"] = result
    at.run()
    return at


def _model_files(tmp_path, **override):
    """A copy of the paths predict() needs, optionally broken."""
    paths = {"model_path": tmp_path / "m.joblib", "metadata_path": tmp_path / "meta.json",
             "preprocessing_path": tmp_path / "pre.joblib"}
    for k, v in override.items():
        if v is not None:
            paths[k].write_bytes(v) if isinstance(v, bytes) else paths[k].write_text(v)
    return paths


CODE = textwrap.dedent('''
    def f(xs):
        total = 0
        for x in xs:
            total += x
        return total
''')


def test_missing_model_keeps_static_results(tmp_path):
    r = predict(CODE, **_model_files(tmp_path)).to_dict()
    assert r["errors"][0]["code"] == "model_missing"
    at = _render_result(r)
    no_crash(at)
    text = page_text(at)
    assert "ML prediction unavailable" in text and "has not been trained" in text
    assert "static analysis below" in text
    assert "DERIVED COMPLEXITY SCORE" in text.upper()


def test_corrupt_model_is_reported(tmp_path):
    meta = {"feature_set": "patterns", "features": __import__("src.preprocessing", fromlist=["x"]).feature_columns("patterns")}
    paths = _model_files(tmp_path, model_path=b"not a pickle", preprocessing_path=b"junk", metadata_path=json.dumps(meta))
    r = predict(CODE, **paths).to_dict()
    assert r["errors"][0]["code"] == "model_load_failed"
    at = _render_result(r)
    no_crash(at)
    assert "could not be loaded" in page_text(at)


def test_feature_mismatch_is_reported(tmp_path):
    paths = _model_files(tmp_path, model_path=b"x", preprocessing_path=b"x",
                         metadata_path=json.dumps({"feature_set": "patterns", "features": ["only_one"]}))
    r = predict(CODE, **paths).to_dict()
    assert r["errors"][0]["code"] == "model_incompatible"
    at = _render_result(r)
    no_crash(at)
    assert "does not match this version" in page_text(at)


def test_corrupt_metadata_is_reported(tmp_path):
    paths = _model_files(tmp_path, model_path=b"x", preprocessing_path=b"x", metadata_path="{broken")
    r = predict(CODE, **paths).to_dict()
    at = _render_result(r)
    no_crash(at)
    assert "could not be read" in page_text(at)
