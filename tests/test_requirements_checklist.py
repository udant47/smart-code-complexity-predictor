"""Phase 18 requirement checklist.

One readable test (or parametrised group) per item of the Phase 18 testing
checklist, exercised through the public API. Many items are also covered in
more depth by the module-specific test files; this file makes the mapping
from requirement to test explicit.
"""

import json
import textwrap

import numpy as np
import pandas as pd
import pytest

from src import config
from src import preprocessing as pp
from src.algorithm_analyzer import analyze_algorithms
from src.code_parser import parse_source
from src.examples import EXAMPLES, get_example
from src.feature_extractor import FILE_FEATURE_NAMES, extract_features
from src.function_analyzer import analyze_functions
from src.pattern_features import PATTERN_FEATURE_NAMES
from src import predictor as pr
from tests.test_predictor import model_dir, paths  # noqa: F401  (module-scoped fixture: small trained model)


def dd(code: str) -> str:
    return textwrap.dedent(code).lstrip("\n")


# =========================================================================== #
# PARSER
# =========================================================================== #
def test_parser_valid_code():
    r = parse_source("x = 1\ny = x + 2\n")
    assert r.success and r.error is None and r.tree is not None
    assert r.line_count == 3 and r.char_count == len("x = 1\ny = x + 2\n")


@pytest.mark.parametrize("code, kind", [
    ("def f(:\n    pass\n", "syntax"),
    ("x = (1, 2\n", "syntax"),
    ("def f():\nreturn 1\n", "indentation"),
    ("if True:\n\tx = 1\n        y = 2\n", "indentation"),
])
def test_parser_invalid_syntax(code, kind):
    r = parse_source(code)
    assert not r.success and r.tree is None
    assert r.error.kind == kind and r.error.lineno is not None and r.error.message


@pytest.mark.parametrize("code", ["", "   ", "\n\n\t\n"])
def test_parser_empty_input(code):
    r = parse_source(code)
    assert not r.success and r.error.kind == "empty" and r.error.hint


def test_parser_functions():
    r = parse_source(dd('''
        def a(x, y=1, *args, z, **kw):
            return x

        async def b():
            pass

        @decorator
        def c():
            pass
    '''))
    assert [(f.name, f.kind, f.is_async) for f in r.functions] == [
        ("a", "function", False), ("b", "function", True), ("c", "function", False)]
    assert r.functions[0].parameters == ("x", "y", "*args", "z", "**kw")
    assert r.functions[2].decorators == ("decorator",)
    assert r.functions[0].lineno == 1 and r.functions[0].end_lineno == 2


def test_parser_classes():
    r = parse_source(dd('''
        class Shape:
            def area(self):
                return 0

            class Meta:
                def info(self):
                    return "m"
    '''))
    assert [c.qualname for c in r.classes] == ["Shape", "Shape.Meta"]
    assert [(f.qualname, f.kind, f.class_name) for f in r.functions] == [
        ("Shape.area", "method", "Shape"), ("Shape.Meta.info", "method", "Shape.Meta")]


def test_parser_nested_functions():
    r = parse_source(dd('''
        def outer(xs):
            def middle(y):
                def inner(z):
                    return z
                return inner(y)
            return [middle(x) for x in xs]
    '''))
    assert [(f.qualname, f.kind, f.depth, f.parent) for f in r.functions] == [
        ("outer", "function", 0, None),
        ("outer.middle", "nested_function", 1, "outer"),
        ("outer.middle.inner", "nested_function", 2, "outer.middle"),
    ]
    assert r.functions[2].source.startswith("def inner(z):")


# =========================================================================== #
# FEATURE EXTRACTION
# =========================================================================== #
def features(code: str) -> dict:
    rep = extract_features(parse_source(dd(code)))
    assert rep.success
    return rep.file_features


def test_features_loops():
    f = features('''
        def f(xs):
            for x in xs:
                for y in xs:
                    pass
            while xs:
                xs.pop()
            return [x for x in xs]
    ''')
    assert (f["for_loop_count"], f["while_loop_count"], f["loop_count"]) == (2, 1, 3)
    assert f["nested_loop_count"] == 1 and f["maximum_loop_nesting_depth"] == 2
    assert f["comprehension_count"] == 1  # comprehensions are counted separately from loops


def test_features_conditionals():
    f = features('''
        def f(x):
            if x > 0:
                return 1
            elif x < 0:
                return -1
            else:
                return 0 if x is None else 0
    ''')
    assert (f["if_count"], f["elif_count"], f["else_count"], f["conditional_expression_count"]) == (1, 1, 1, 1)


def test_features_recursion():
    f = features('''
        def fact(n):
            return 1 if n <= 1 else n * fact(n - 1)

        def walk(node):
            for child in node.children:
                walk(child)
            helper(node)
    ''')
    assert f["recursive_call_count"] == 2
    fa = analyze_functions(parse_source(get_example("recursive_fibonacci").code))
    assert fa.functions[0].recursive_call_count == 2


def test_features_nesting_depth():
    f = features(get_example("deep_nesting").code)
    assert f["maximum_nesting_depth"] == 6
    assert features("x = 1\n")["maximum_nesting_depth"] == 0
    f = features('''
        def f(xs):
            for x in xs:
                try:
                    with open(x) as fh:
                        if fh:
                            pass
                except OSError:
                    pass
    ''')
    assert f["maximum_nesting_depth"] == 4  # for > try > with > if


@pytest.mark.parametrize("code, expected", [
    ("def f():\n    return 1\n", 1),
    ("def f(x):\n    if x:\n        return 1\n    return 2\n", 2),
    ("def f(x):\n    if x and y or z:\n        return 1\n", 4),  # if + two boolean operators
    ("def f(xs):\n    for x in xs:\n        if x:\n            pass\n    while xs:\n        pass\n", 4),
    ("def f(x):\n    try:\n        pass\n    except A:\n        pass\n    except B:\n        pass\n", 3),
    ("def f(xs):\n    return [x for x in xs if x]\n", 3),  # comprehension loop + its if
])
def test_features_cyclomatic_complexity(code, expected):
    fa = analyze_functions(parse_source(code))
    assert fa.functions[0].cyclomatic_complexity == expected
    assert extract_features(parse_source(code)).function_features[0]["cyclomatic_complexity"] == expected


METRIC_CORPUS = [e.code for e in EXAMPLES] + [
    dd('''
        """Module docstring."""
        import os  # comment after code
        # standalone comment

        class Repo:
            """Class docstring
            on two lines."""
            def load(self, paths):
                out = []
                for p in paths:
                    if os.path.exists(p):
                        try:
                            out.append(p)
                        except Exception:
                            continue
                return out

            def nested(self):
                def helper(x):
                    while x:
                        x -= 1
                    return x
                return helper(3)
        '''),
]


@pytest.mark.parametrize("code", METRIC_CORPUS, ids=lambda c: c.split("\n", 1)[0][:30])
def test_features_metric_consistency(code):
    parsed = parse_source(code)
    f = extract_features(parsed).file_features
    fa = analyze_functions(parsed)
    funcs = fa.functions
    # Every documented file feature is present and numeric.
    assert set(FILE_FEATURE_NAMES) <= set(f)
    assert all(isinstance(f[k], (int, float)) and np.isfinite(f[k]) for k in FILE_FEATURE_NAMES)
    # Lines: docstrings are a subset of code lines; comments may share a line with code.
    assert f["docstring_lines"] <= f["code_lines"]
    assert f["code_lines"] + f["blank_lines"] <= f["total_lines"] <= f["code_lines"] + f["comment_lines"] + f["blank_lines"]
    # Loops and definitions add up.
    assert f["loop_count"] == f["for_loop_count"] + f["while_loop_count"]
    assert f["function_count"] == len(parsed.functions) == len(funcs)
    assert f["class_count"] == len(parsed.classes)
    # File-level maxima agree with the per-function analysis.
    assert f["maximum_function_cyclomatic_complexity"] == max([x.cyclomatic_complexity for x in funcs], default=0)
    assert f["maximum_function_length"] == max([x.function_length for x in funcs], default=0)
    assert f["cyclomatic_complexity"] >= max(1, f["maximum_function_cyclomatic_complexity"])
    assert f["maximum_nesting_depth"] >= f["maximum_loop_nesting_depth"]
    assert all(x.cyclomatic_complexity >= 1 and x.maximum_nesting_depth <= f["maximum_nesting_depth"] for x in funcs)
    # The extractor's per-function view and the function analyzer agree.
    by_name = {x["qualname"]: x for x in extract_features(parsed).function_features}
    for x in funcs:
        assert by_name[x.qualname]["cyclomatic_complexity"] == x.cyclomatic_complexity
        assert by_name[x.qualname]["maximum_nesting_depth"] == x.maximum_nesting_depth
    # The predictor reports the same numbers.
    s = pr.predict(code, per_function_ml=False).to_dict()["file_summary"]
    assert (s["function_count"], s["maximum_cyclomatic_complexity"], s["maximum_nesting_depth"]) == (
        f["function_count"], f["maximum_function_cyclomatic_complexity"], f["maximum_nesting_depth"])
    # Deterministic: analysing twice gives identical features.
    assert extract_features(parse_source(code)).file_features == f


# =========================================================================== #
# ALGORITHM ANALYSIS
# =========================================================================== #
def estimate(code: str, function: str) -> dict:
    a = analyze_algorithms(parse_source(dd(code)))
    assert a.success
    return next(e for e in a.estimates if e.function_id == function).to_dict()


@pytest.mark.parametrize("key, function, time", [
    ("linear_search", "linear_search", "O(n)"),
    ("binary_search", "binary_search", "O(log n)"),
    ("nested_loops", "count_pairs_with_sum", "O(n^2)"),
    ("merge_sort", "merge_sort", "O(n log n)"),
    ("recursive_fibonacci", "fibonacci", "O(2^n)"),
])
def test_algorithm_known_patterns(key, function, time):
    est = estimate(get_example(key).code, function)
    assert est["time_class"] == time and est["time_status"] == "estimated"
    assert est["evidence"] and est["explanation"]


def test_algorithm_sequential_loops_add_not_multiply():
    est = estimate('''
        def two_passes(xs):
            total = 0
            for x in xs:
                total += x
            for x in xs:
                total -= x
            return total
    ''', "two_passes")
    assert est["time_class"] == "O(n)"
    est = estimate('''
        def two_inputs(a, b):
            for x in a:
                pass
            for y in b:
                pass
    ''', "two_inputs")
    assert est["time_complexity"] == "O(n + m)" and est["time_class"] == "O(n)"


@pytest.mark.parametrize("code", [
    # Loop bound depends on runtime data the analyzer cannot see.
    '''
    def drain(queue):
        while queue:
            item = queue.pop()
            if item.children:
                queue.extend(item.children)
    ''',
    # Calls into code that is not in the file.
    '''
    def mystery(xs):
        return external_library.solve(xs)
    ''',
])
def test_algorithm_unknown_cases_are_not_guessed(code):
    est = estimate(code, "drain" if "drain" in code else "mystery")
    assert est["time_class"] == "Unknown" and est["time_status"] == "unknown"
    assert est["explanation"].startswith("Time could not be estimated")


def test_algorithm_unresolved_call_on_items_keeps_documented_assumption():
    est = estimate('''
        def each(xs):
            for x in xs:
                external_api(x)
    ''', "each")
    assert est["time_class"] == "O(n)" and est["time_confidence"] == "low"
    assert any("external_api() is not defined" in a for a in est["assumptions"])


def test_algorithm_invalid_code_is_reported_not_raised():
    a = analyze_algorithms(parse_source("def f(:\n"))
    assert not a.success and a.estimates == [] and a.error is not None


# =========================================================================== #
# ML
# =========================================================================== #
def test_ml_preprocessing_pipeline(model_dir):  # noqa: F811
    model = pr.load_model(**paths(model_dir))
    X = pd.DataFrame([{f: 1.0 for f in model.features}, {f: 0.0 for f in model.features}])
    out = model.preprocessing.transform(X)
    assert out.shape[0] == 2 and np.isfinite(np.asarray(out, dtype=float)).all()
    # The preprocessing inside the pipeline is the saved one.
    assert list(model.preprocessing.feature_names_in_) == model.features


def test_ml_feature_order(model_dir):  # noqa: F811
    model = pr.load_model(**paths(model_dir))
    assert model.features == pp.feature_columns("patterns") == list(FILE_FEATURE_NAMES) + list(PATTERN_FEATURE_NAMES)
    values = {f: float(i) for i, f in enumerate(model.features)}
    shuffled = dict(reversed(list(values.items())))
    X = pr.feature_vector(model.features, shuffled)
    assert list(X.columns) == model.features and X.iloc[0].tolist() == [float(i) for i in range(len(model.features))]
    with pytest.raises(ValueError, match="missing"):
        pr.feature_vector(model.features, {k: v for k, v in values.items() if k != model.features[3]})
    with pytest.raises(ValueError, match="non-finite"):
        pr.feature_vector(model.features, {**values, model.features[0]: float("nan")})


def test_ml_prediction(model_dir):  # noqa: F811
    r = pr.predict(get_example("merge_sort").code, **paths(model_dir))
    p = r.prediction
    assert r.success and p is not None
    assert p.big_o in pr.BIGO_OF.values() and 0 <= p.probability <= 1
    assert sum(p.probabilities.values()) == pytest.approx(1, abs=1e-3)
    assert p.probability == max(p.probabilities.values()) and p.runner_up != p.big_o
    # Deterministic.
    assert pr.predict(get_example("merge_sort").code, **paths(model_dir)).prediction == p


def test_ml_batched_function_predictions_equal_single_predictions(model_dir):  # noqa: F811
    model = pr.load_model(**paths(model_dir))
    rows = [pr._all_features(parse_source(e.code)) for e in EXAMPLES]
    batch = pr._ml_predictions_batch(model, rows + [{"bad": 1}])
    assert batch[-1] is None
    for row, b in zip(rows, batch):
        single = pr._ml_prediction(model, row, experimental=True)
        assert b.probabilities == single.probabilities and b.big_o == single.big_o


def test_ml_missing_model_keeps_static_analysis(tmp_path):
    r = pr.predict(get_example("binary_search").code, model_path=tmp_path / "none.joblib",
                   metadata_path=tmp_path / "none.json", preprocessing_path=tmp_path / "none2.joblib")
    assert r.success and r.prediction is None
    assert r.errors[0].code == "model_missing" and "python -m src" in r.errors[0].hint
    assert r.functions[0].static_estimate["time_class"] == "O(log n)"
    assert r.derived_score["highest"] is not None


def test_ml_model_loading(model_dir):  # noqa: F811
    model = pr.load_model(**paths(model_dir))
    assert model.info.available and model.info.n_features == len(model.features) == 57
    assert set(model.classes) == set(pr.BIGO_OF) and len(model.info.model_sha256) == 64
    meta = json.loads((model_dir / "model_metadata.json").read_text())
    assert set(meta["integrity"]["files"]) == {"complexity_model.joblib", "preprocessing_pipeline.joblib"}


PREDICTION_KEYS = {"label", "big_o", "probability", "probabilities", "runner_up", "margin",
                   "measured_accuracy_in_range", "experimental", "note"}
RESULT_KEYS = {"success", "errors", "warnings", "file_metrics", "file_summary", "structural_category",
               "prediction", "functions", "model", "derived_score", "recommendations", "explanation",
               "program_estimate"}
FUNCTION_KEYS = {"function_id", "name", "kind", "class_name", "start_line", "end_line", "metrics",
                 "structural_category", "static_estimate", "derived_score", "ml_prediction"}


def test_ml_output_schema(model_dir):  # noqa: F811
    d = pr.predict(get_example("nested_loops").code, **paths(model_dir), explain=True).to_dict()
    json.dumps(d)  # fully JSON-serialisable
    assert set(d) == RESULT_KEYS
    assert set(d["prediction"]) == PREDICTION_KEYS
    assert set(d["prediction"]["probabilities"]) == set(pr.BIGO_OF.values())
    for f in d["functions"]:
        assert set(f) == FUNCTION_KEYS
        assert f["ml_prediction"] is None or set(f["ml_prediction"]) == PREDICTION_KEYS
        assert {"time_complexity", "time_class", "space_complexity", "status", "assumptions"} <= set(f["static_estimate"])
    assert all(set(e) >= {"code", "message"} for e in d["errors"] + d["warnings"])
    assert d["model"]["available"] is True and d["explanation"] is not None
    failed = pr.predict("def f(:\n").to_dict()
    assert set(failed) == RESULT_KEYS and failed["success"] is False and failed["errors"][0]["code"] == "syntax"


# =========================================================================== #
# FRONTEND (logic and widgets; the full pages are exercised headlessly in test_app.py)
# =========================================================================== #
from src.ui import components as ui  # noqa: E402
from src.ui import logic  # noqa: E402


def test_frontend_empty_input():
    chk = logic.check_input("  \n\t")
    assert not chk.ok and chk.errors[0]["code"] == "empty" and chk.lines == 2


def test_frontend_invalid_code_reaches_ui_as_structured_error():
    r = pr.predict("def f(:\n    pass\n").to_dict()
    assert r["success"] is False
    ml, other = logic.split_issues(r["errors"])
    assert ml == [] and other[0]["code"] == "syntax" and "line 1" in other[0]["message"].lower()


@pytest.mark.parametrize("code", ["model_missing", "model_metadata_invalid", "model_incompatible", "model_load_failed",
                                  "model_unverified", "model_integrity_failed"])
def test_frontend_model_problems_have_friendly_titles(code):
    ml, other = logic.split_issues([{"code": code, "message": "m"}, {"code": "syntax", "message": "s"}])
    assert [e["code"] for e in ml] == [code] and [e["code"] for e in other] == ["syntax"]
    assert logic.ML_ERROR_TITLES[code].endswith(".")


def test_frontend_missing_model_is_reported(tmp_path):
    r = pr.predict("def f(x):\n    return x\n", model_path=tmp_path / "a", metadata_path=tmp_path / "b",
                   preprocessing_path=tmp_path / "c").to_dict()
    ml, _ = logic.split_issues(r["errors"])
    assert ml[0]["code"] == "model_missing" and r["functions"]  # static results are still there


@pytest.mark.parametrize("language", ["JavaScript", "Java", "C++"])
def test_frontend_unsupported_language(language):
    chk = logic.check_input("int main() { return 0; }", language)
    assert not chk.ok and chk.errors[0]["code"] == "unsupported_language"
    assert logic.LANGUAGES[language]["supported"] is False


def test_frontend_result_formatting():
    r = pr.predict(get_example("merge_sort").code, per_function_ml=False).to_dict()
    df = logic.functions_frame(r)
    assert list(df.columns) == ["Function", "Type", "Lines", "Cyclomatic", "Nesting", "Structural category",
                                "Derived score", "Time (static)", "Aux. space (static)", "Status"]
    assert df.loc[df["Function"] == "merge_sort", "Time (static)"].item() == "O(n log n)"
    assert df.loc[df["Function"] == "merge", "Lines"].item() == "1–11"
    empty = logic.functions_frame({"functions": []})
    assert empty.empty and list(empty.columns) == list(df.columns)
    o = logic.overview(r)
    assert o["function_count"] == 2 and o["structural_category"] == "Low" and isinstance(o["derived_score"], float)
    assert (ui.fmt(None), ui.fmt(3), ui.fmt(2.0), ui.fmt(1.23456, 1)) == ("—", "3", "2.00", "1.2")


def test_frontend_widgets_escape_user_text(monkeypatch):
    shown = []
    monkeypatch.setattr(ui.st, "markdown", lambda body, **k: shown.append(body))
    evil = '<img src=x onerror="alert(1)">'
    ui.callout("error", evil, evil)
    ui.cards([ui.card(evil, evil, evil)])
    ui.muted(evil)
    ui.hero(evil, evil)
    ui.recommendation({"severity": "high", "title": evil, "rule_id": evil, "target": evil, "start_line": 1,
                       "end_line": 2, "evidence": [{"metric": evil, "value": evil, "threshold": evil}],
                       "explanation": evil, "suggested_action": evil})
    html = "\n".join(shown)
    assert "<img" not in html and "&lt;img" in html
