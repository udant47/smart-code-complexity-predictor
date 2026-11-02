"""Tests for the documented demonstration examples (Phase 17).

The documented structural values and static time classes are checked against
the analyzer, so the documentation cannot drift from what the app measures.
"""

import ast

import pytest

from src import examples as ex
from src.predictor import predict
from src.ui import samples

REQUIRED_PATTERNS = ["simple_arithmetic", "linear_search", "binary_search", "nested_loops",
                     "recursive_fibonacci", "merge_sort", "complex_branching", "deep_nesting"]


@pytest.fixture(scope="module")
def results():
    return {e.key: predict(e.code).to_dict() for e in ex.EXAMPLES}


def test_all_required_examples_exist_in_order():
    assert [e.key for e in ex.EXAMPLES] == REQUIRED_PATTERNS
    assert [e.number for e in ex.EXAMPLES] == list(range(1, 9))
    assert len({e.title for e in ex.EXAMPLES}) == len(ex.EXAMPLES)


@pytest.mark.parametrize("example", ex.EXAMPLES, ids=lambda e: e.key)
def test_example_is_valid_python_and_fully_documented(example):
    ast.parse(example.code)  # parsed only, never executed
    assert example.summary and example.pattern and example.characteristics
    assert set(example.structure) == set(ex.STRUCTURE_LABELS)
    assert example.complexity
    defined = {n.name for n in ast.walk(ast.parse(example.code)) if isinstance(n, ast.FunctionDef)}
    for fc in example.complexity:
        assert fc.function in defined
        assert fc.time.startswith(("O(", "Θ(")) and fc.space.startswith("O(")
        assert fc.assumptions, f"{fc.function}: complexity must state its assumptions"


def test_no_ml_class_is_documented():
    for e in ex.EXAMPLES:
        assert not any("ml" in k.lower() for k in e.structure)
        assert not hasattr(e, "ml_class") and not hasattr(e, "expected_ml")


@pytest.mark.parametrize("example", ex.EXAMPLES, ids=lambda e: e.key)
def test_documented_structure_matches_analyzer(example, results):
    assert ex.measured_structure(results[example.key]) == example.structure


@pytest.mark.parametrize("example", ex.EXAMPLES, ids=lambda e: e.key)
def test_documented_static_time_matches_analyzer(example, results):
    funcs = {f["function_id"]: f for f in results[example.key]["functions"]}
    for fc in example.complexity:
        est = funcs[fc.function]["static_estimate"]
        assert est["time_class"] == fc.static_time_class
        assert est["time_status"] == "estimated"


@pytest.mark.parametrize("example", ex.EXAMPLES, ids=lambda e: e.key)
def test_expected_recommendations(example, results):
    got = {r["rule_id"] for r in results[example.key]["recommendations"]["recommendations"]}
    assert got == set(example.expected_recommendations)


def test_check_against_result_rows(results):
    e = ex.get_example("deep_nesting")
    rows = ex.check_against_result(e, results["deep_nesting"])
    assert all(r.status in ("matches", "not documented") for r in rows)
    assert any(r.aspect == "Whole-file ML class" and r.documented == "not documented" for r in rows)
    # A wrong documented value is reported as "differs", not silently accepted.
    wrong = ex.Example(**{**e.__dict__, "structure": {**e.structure, "max_nesting_depth": 2}})
    bad = [r for r in ex.check_against_result(wrong, results["deep_nesting"]) if r.status == "differs"]
    assert [r.aspect for r in bad] == ["Maximum nesting depth"] and bad[0].measured == "6"


def test_lookup_helpers():
    e = ex.EXAMPLES[2]
    assert ex.get_example(e.key) is e and ex.get_example(e.title) is e
    assert ex.find_by_code(e.code + "\n\n") is e
    assert ex.find_by_code(e.code.replace("target", "goal")) is None
    with pytest.raises(KeyError):
        ex.get_example("nope")


# --------------------------------------------------------------------------- #
# Sample loading for the UI
# --------------------------------------------------------------------------- #
def test_ui_samples_come_from_documented_examples():
    assert list(samples.SAMPLES) == [e.title for e in ex.EXAMPLES]
    for e in ex.EXAMPLES:
        assert samples.SAMPLES[e.title] == e.code
    assert samples.DEFAULT_SAMPLE in samples.SAMPLES
    assert list(samples.PAIRS) == [p.title for p in ex.COMPARISON_PAIRS]
    assert (samples.COMPARE_BEFORE, samples.COMPARE_AFTER) == (ex.COMPARISON_PAIRS[0].before,
                                                               ex.COMPARISON_PAIRS[0].after)


# --------------------------------------------------------------------------- #
# Comparison pairs
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("pair", ex.COMPARISON_PAIRS, ids=lambda p: p.key)
def test_pair_is_documented_and_matches_static_estimates(pair):
    ast.parse(pair.before)
    ast.parse(pair.after)
    assert pair.before != pair.after and pair.changes and pair.lesson and pair.behaviour
    rb, ra = predict(pair.before).to_dict(), predict(pair.after).to_dict()
    for exp in pair.expected:
        fb = next(f for f in rb["functions"] if f["function_id"] == exp.function)
        fa = next(f for f in ra["functions"] if f["function_id"] == exp.function)
        assert fb["static_estimate"]["time_class"] == exp.time_before
        if "analyzer: Unknown" in exp.time_after:
            assert fa["static_estimate"]["time_class"] == "Unknown"
        else:
            assert fa["static_estimate"]["time_class"] == exp.time_after
            assert fa["static_estimate"]["space_class"] == exp.space_after
        assert fb["static_estimate"]["space_class"] == exp.space_before


def test_pairs_include_shorter_code_that_is_not_faster():
    p = ex.get_pair("shorter_but_slower")
    assert p.after.count("\n") < p.before.count("\n")
    assert p.expected[0].time_before == "O(n)" and p.expected[0].time_after == "O(n log n)"
    same = ex.get_pair("shorter_same_complexity").expected[0]
    assert same.time_before == same.time_after
