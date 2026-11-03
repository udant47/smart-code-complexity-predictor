"""Tests for the final improvement phase.

* ``analyze_script``: static estimates for module-level (script) code, the form
  of most CodeComplex programs.
* ``src.analysis_features``: the static-estimate and extra AST features.
* One shared feature function for training and inference (same schema).
* Sorting and divide-and-conquer algorithms in the static analyzer, including
  the ones it deliberately reports as Unknown.

Expected complexities are worst-case bounds from standard algorithm analysis.
"""

import textwrap

import pandas as pd
import pytest

from src import config
from src import preprocessing as pp
from src.algorithm_analyzer import SCRIPT_ID, analyze_algorithms, analyze_script
from src.analysis_features import (
    ANALYSIS_FEATURE_NAMES,
    AST_EXTRA_FEATURE_NAMES,
    FEATURE_DESCRIPTIONS,
    STATIC_FEATURE_NAMES,
    ast_extra_features,
    extract_analysis_features,
    program_features,
    static_estimate_features,
)
from src.code_parser import parse_source
from src.dataset_loader import BIGO_FEATURE_COLUMNS, build_bigo_dataset
from src.feature_extractor import FILE_FEATURE_NAMES
from src.pattern_features import PATTERN_FEATURE_NAMES


def src(code: str) -> str:
    return textwrap.dedent(code).strip() + "\n"


def time_of(code: str, function_id: str = None) -> str:
    result = analyze_algorithms(src(code))
    assert result.success
    e = result.get(function_id) if function_id else result.estimates[0]
    return e.time_class


# --------------------------------------------------------------------------- #
# analyze_script
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("code, expected", [
    ("n = int(input())\nprint(n * 2)\n", "O(1)"),
    ("n = int(input())\na = list(map(int, input().split()))\nprint(sum(a))\n", "O(n)"),
    ("n = int(input())\na = list(map(int, input().split()))\nfor x in a:\n    for y in a:\n        print(x + y)\n", "O(n^2)"),
    ("a = list(map(int, input().split()))\na.sort()\nprint(a[0])\n", "O(n log n)"),
    ("for i in range(int(input())):\n    print(i)\n", "O(n)"),
    ("n = int(input())\nlo, hi = 0, n\nwhile lo < hi:\n    mid = (lo + hi) // 2\n    if mid * mid < n:\n        lo = mid + 1\n    else:\n        hi = mid\nprint(lo)\n",
     "O(log n)"),
])
def test_analyze_script_estimates(code, expected):
    e = analyze_script(code)
    assert e is not None and e.function_id == SCRIPT_ID
    assert e.time_class == expected, e.explanation


def test_analyze_script_none_without_module_code():
    assert analyze_script("def f(x):\n    return x\n") is None
    assert analyze_script('"""docstring only"""\nimport os\n') is None


def test_analyze_script_uses_called_functions():
    code = src("""
        def solve(a):
            return sorted(a)
        a = list(map(int, input().split()))
        print(solve(a))
    """)
    assert analyze_script(code).time_class == "O(n log n)"


def test_analyze_script_unknown_when_input_read_inside_a_function():
    code = src("""
        def main():
            n = int(input())
            for i in range(n):
                print(i)
        main()
    """)
    assert analyze_script(code).time_class == "Unknown"


def test_analyze_script_does_not_change_function_estimates():
    code = src("""
        def f(a):
            for x in a:
                print(x)
        n = int(input())
        f(list(range(n)))
    """)
    ids = [e.function_id for e in analyze_algorithms(code).estimates]
    assert SCRIPT_ID not in ids and ids == ["f"]


def test_analyze_script_handles_invalid_code():
    assert analyze_script("def broken(:\n") is None


# --------------------------------------------------------------------------- #
# analysis_features
# --------------------------------------------------------------------------- #
def test_feature_names_and_descriptions():
    assert ANALYSIS_FEATURE_NAMES == STATIC_FEATURE_NAMES + AST_EXTRA_FEATURE_NAMES
    assert len(set(ANALYSIS_FEATURE_NAMES)) == 13
    assert set(FEATURE_DESCRIPTIONS) == set(ANALYSIS_FEATURE_NAMES)
    assert not set(ANALYSIS_FEATURE_NAMES) & set(FILE_FEATURE_NAMES + PATTERN_FEATURE_NAMES)


@pytest.mark.parametrize("code, known, degree, logs, exp", [
    ("a = list(map(int, input().split()))\nfor x in a:\n    for y in a:\n        print(x, y)\n", 1, 2, 0, 0),
    ("a = list(map(int, input().split()))\nprint(sorted(a))\n", 1, 1, 1, 0),
    ("print(1 + 2)\n", 1, 0, 0, 0),
    ("def fib(n):\n    if n < 2:\n        return n\n    return fib(n - 1) + fib(n - 2)\n", 1, 0, 0, 1),
    ("import sys\nfor line in sys.stdin:\n    print(line)\n", 0, 0, 0, 0),
])
def test_static_estimate_features(code, known, degree, logs, exp):
    f = static_estimate_features(parse_source(code))
    assert (f["static_time_known"], f["static_time_degree"], f["static_time_log_power"], f["static_time_exponential"]) == \
        (known, degree, logs, exp)


def test_ast_extra_features_counts():
    tree = parse_source(src("""
        import sys
        sys.setrecursionlimit(10000)
        def f(a, b, c=1, *args, **kw):
            x = a[0]
            y = a[1:3]
            x += 1
            a.append(x)
            if x in a and y not in a:
                return max(min(abs(x), 1), 2)
    """)).tree
    f = ast_extra_features(tree)
    assert f["assignment_count"] == 2
    assert f["augmented_assignment_count"] == 1
    assert f["subscript_count"] == 1 and f["slice_count"] == 1
    assert f["collection_mutation_call_count"] == 1
    assert f["membership_test_count"] == 2
    assert f["max_call_nesting_depth"] == 3
    assert f["max_parameter_count"] == 5
    assert f["recursion_limit_set"] == 1


@pytest.mark.parametrize("code", ["", "   \n", "def broken(:\n", "x = (\n"])
def test_analysis_features_zero_for_unusable_input(code):
    assert extract_analysis_features(code) == dict.fromkeys(ANALYSIS_FEATURE_NAMES, 0)


def test_analysis_features_never_execute_code(tmp_path):
    marker = tmp_path / "ran.txt"
    code = f"open({str(marker)!r}, 'w').write('x')\nimport os\nos.system('echo hi')\n"
    extract_analysis_features(code)
    assert not marker.exists()


# --------------------------------------------------------------------------- #
# Training and inference use the same features
# --------------------------------------------------------------------------- #
def test_dataset_columns_are_program_features():
    parsed = parse_source("a = list(map(int, input().split()))\nprint(sorted(a))\n")
    assert list(program_features(parsed)) == list(BIGO_FEATURE_COLUMNS)
    assert len(BIGO_FEATURE_COLUMNS) == len(set(BIGO_FEATURE_COLUMNS)) == 70


def test_dataset_builder_and_predictor_agree():
    """The same program gets the same feature values from the dataset builder and the predictor."""
    from src.predictor import _all_features

    code = src("""
        n = int(input())
        a = list(map(int, input().split()))
        best = 0
        for i in range(n):
            for j in range(i, n):
                best = max(best, sum(a[i:j + 1]))
        print(best)
    """)
    record = {"src": code, "complexity": "cubic", "problem": "1000_A", "raw_index": 0, "tags": []}
    df, _ = build_bigo_dataset([record])
    built = df.iloc[0][list(BIGO_FEATURE_COLUMNS)].astype(float).to_dict()
    inferred = {k: float(v) for k, v in _all_features(parse_source(code)).items()}
    assert built == inferred


def test_feature_sets_are_subsets_of_dataset_columns():
    for name, cols in pp.FEATURE_SETS.items():
        assert set(cols) <= set(BIGO_FEATURE_COLUMNS), name
    assert pp.feature_columns("patterns_static") == list(FILE_FEATURE_NAMES + PATTERN_FEATURE_NAMES + STATIC_FEATURE_NAMES)


@pytest.mark.skipif(not config.MODEL_METADATA_PATH.exists(), reason="no trained model")
def test_installed_model_features_exist_at_inference():
    import json

    meta = json.loads(config.MODEL_METADATA_PATH.read_text(encoding="utf-8"))
    assert meta["features"] == pp.feature_columns(meta["feature_set"])
    values = program_features(parse_source("print(1)\n"))
    assert set(meta["features"]) <= set(values)


@pytest.mark.skipif(not config.BIGO_DATASET_CSV.exists(), reason="processed dataset not built")
def test_processed_dataset_has_every_feature_column():
    head = pd.read_csv(config.BIGO_DATASET_CSV, nrows=5, keep_default_na=False)
    assert set(BIGO_FEATURE_COLUMNS) <= set(head.columns)


# --------------------------------------------------------------------------- #
# Sorting and divide-and-conquer
# --------------------------------------------------------------------------- #
BUBBLE = """
def bubble_sort(a):
    n = len(a)
    for i in range(n):
        for j in range(0, n - i - 1):
            if a[j] > a[j + 1]:
                a[j], a[j + 1] = a[j + 1], a[j]
    return a
"""
INSERTION = """
def insertion_sort(a):
    for i in range(1, len(a)):
        key = a[i]
        j = i - 1
        while j >= 0 and a[j] > key:
            a[j + 1] = a[j]
            j -= 1
        a[j + 1] = key
    return a
"""
SELECTION = """
def selection_sort(a):
    n = len(a)
    for i in range(n):
        m = i
        for j in range(i + 1, n):
            if a[j] < a[m]:
                m = j
        a[i], a[m] = a[m], a[i]
    return a
"""
MAX_SUBARRAY = """
def max_sub(a, lo, hi):
    if lo == hi:
        return a[lo]
    mid = (lo + hi) // 2
    best_left = s = a[mid]
    for i in range(mid - 1, lo - 1, -1):
        s += a[i]
        best_left = max(best_left, s)
    best_right = s = a[mid + 1]
    for i in range(mid + 2, hi + 1):
        s += a[i]
        best_right = max(best_right, s)
    return max(max_sub(a, lo, mid), max_sub(a, mid + 1, hi), best_left + best_right)
"""
HEAPQ_SORT = """
import heapq
def heap_sort(a):
    h = list(a)
    heapq.heapify(h)
    return [heapq.heappop(h) for _ in range(len(h))]
"""


@pytest.mark.parametrize("code, expected", [
    (BUBBLE, "O(n^2)"),
    (INSERTION, "O(n^2)"),
    (SELECTION, "O(n^2)"),
    (MAX_SUBARRAY, "O(n log n)"),
    (HEAPQ_SORT, "O(n log n)"),
])
def test_sorting_and_divide_and_conquer(code, expected):
    assert time_of(code) == expected


def test_guarded_counter_bound_is_worst_case():
    e = analyze_algorithms(src(INSERTION)).estimates[0]
    assert "worst-case bound" in e.explanation or any("worst-case bound" in x for x in e.evidence)


def test_guarded_loop_without_counter_update_stays_unknown():
    code = """
    def f(a, key):
        j = len(a) - 1
        while j >= 0 and a[j] > key:
            a[j] = key
        return a
    """
    assert time_of(code) == "Unknown"


def test_plain_interval_range_unchanged():
    code = """
    def f(a, lo, hi):
        for i in range(lo, hi):
            print(a[i])
    """
    assert time_of(code) == "O(n)"


@pytest.mark.parametrize("code, reason", [
    ("""
    def quick_sort(a):
        if len(a) <= 1:
            return a
        pivot = a[len(a) // 2]
        left = [x for x in a if x < pivot]
        right = [x for x in a if x > pivot]
        return quick_sort(left) + [pivot] + quick_sort(right)
    """, "quicksort"),
    ("""
    def heapify(a, n, i):
        while True:
            largest = i
            l, r = 2 * i + 1, 2 * i + 2
            if l < n and a[l] > a[largest]:
                largest = l
            if r < n and a[r] > a[largest]:
                largest = r
            if largest == i:
                return
            a[i], a[largest] = a[largest], a[i]
            i = largest
    """, "constant condition"),
])
def test_data_dependent_algorithms_are_reported_unknown_with_a_reason(code, reason):
    e = analyze_algorithms(src(code)).estimates[0]
    assert e.time_class == "Unknown"
    assert reason in e.explanation


# --------------------------------------------------------------------------- #
# Calibration, dataset audit and improvement report
# --------------------------------------------------------------------------- #
def test_calibration_scores_perfect_and_overconfident():
    import numpy as np

    from src.evaluate_model import calibration_scores

    classes = ["a", "b"]
    y = np.array(["a", "b", "a", "b"])
    perfect = calibration_scores(y, np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]]), classes)
    assert perfect["ece_top_label"] == 0.0 and perfect["brier_multiclass"] == 0.0 and perfect["accuracy"] == 1.0
    # always 90% sure, right only half of the time -> ECE 0.4
    over = calibration_scores(y, np.array([[0.9, 0.1]] * 4), classes)
    assert over["accuracy"] == 0.5 and over["mean_top_vote_share"] == 0.9
    assert over["ece_top_label"] == pytest.approx(0.4)
    assert "not calibrated" in over["note"]


def test_analyzer_benchmark_never_claims_a_wrong_class():
    from src.improvement_report import REFERENCE_ALGORITHMS, analyzer_benchmark

    rows = analyzer_benchmark()
    assert len(rows) == len(REFERENCE_ALGORITHMS)
    assert not [r for r in rows if r["verdict"] == "different"]
    assert sum(r["verdict"] == "matches" for r in rows) >= 12
    for r in rows:
        assert (r["static estimate"] == "Unknown") == bool(r["reason when Unknown"])


def test_dataset_audit_data_checks():
    from src.dataset_audit import data_checks, render
    from tests.test_preprocessing import fixture_table

    df = fixture_table()
    split = pp.make_split(df, "problem")
    r = data_checks(df, split, pp.feature_columns("all"))
    assert r["leakage"]["problems_in_more_than_one_partition"] == 0
    assert sum(r["class_balance"]["rows_by_partition"].values()) == len(df)
    assert r["label_consistency"]["distinct_feature_vectors"] <= len(df)
    text = render({**r, "model": None})
    assert "# Dataset quality audit" in text and "Leakage" in text


def test_improvement_report_render_and_table():
    from src.improvement_report import _table, render

    assert _table([]) == ""
    assert "| a | b |" in _table([{"a": 1, "b": None}]) and "—" in _table([{"a": 1, "b": None}])
    r = {"analyzer_summary": {"cases": 2, "matches": 1, "unknown": 1, "different": 0},
         "final_model": {"model": "Random Forest", "feature_set": "patterns_static", "n_features": 61, "hyperparameters": {}},
         "before_after": [{"metric": "Macro F1 (test)", "before": 0.4, "after": 0.5, "change": 0.1}],
         "per_class": [], "feature_ablation": [], "analyzer_benchmark": [], "notes": ["n"]}
    text = render(r)
    assert "Before vs after" in text and "patterns_static" in text and "1 of 2 match" in text


@pytest.mark.parametrize("code", ["x = " + "-" * 500 + "1\n", "x = " + "a." * 500 + "b\n",
                                  "x = " + "1 if a else " * 400 + "2\n"])
def test_analyze_script_deep_expressions_are_unknown_not_errors(code):
    e = analyze_script(code)
    assert e is not None and e.time_class == "Unknown"
    assert "internal analysis limitation" in e.explanation or e.time_status == "unknown"
    assert static_estimate_features(parse_source(code))["static_time_known"] == 0


def test_comparison_chart_highlights_exactly_the_saved_model():
    from src.ui import charts
    from src.ui.styles import TOKENS

    df = pd.DataFrame({"Model": ["Random Forest", "Random Forest", "Extra Trees"],
                       "Features": ["patterns", "patterns_static", "patterns_static"],
                       "Test macro F1": [0.44, 0.46, 0.49], "CI low": [0.3] * 3, "CI high": [0.6] * 3})
    fig = charts.model_comparison(df, "Random Forest", "patterns_static")
    colors = list(fig.data[0].marker.color)
    labels = list(fig.data[0].y)
    highlighted = [lab for lab, c in zip(labels, colors) if c == TOKENS["accent"]]
    assert highlighted == ["Random Forest  [patterns_static]"]
