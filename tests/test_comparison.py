"""Tests for the before/after comparison calculations (Phase 17)."""

from fractions import Fraction

import pytest

from src import comparison as cmp
from src.examples import get_pair
from src.predictor import predict


def run(before: str, after: str) -> cmp.Comparison:
    return cmp.compare(predict(before).to_dict(), predict(after).to_dict())


def titles(c):
    return [o.title for o in c.observations]


# --------------------------------------------------------------------------- #
# Growth order
# --------------------------------------------------------------------------- #
def test_growth_order_is_ascending():
    order = ["O(1)", "O(log n)", "O(log^2 n)", "O(√n)", "O(n)", "O(n log n)", "O(n^2)", "O(n^2 log n)",
             "O(n^3)", "O(2^n)", "O(n·2^n)", "O(3^n)", "O(n!)"]
    keys = [cmp.growth_key(o) for o in order]
    assert None not in keys
    assert keys == sorted(keys) and len(set(keys)) == len(keys)
    assert cmp.growth_key("O(n^2.5)")[2] == Fraction(5, 2)


@pytest.mark.parametrize("bad", [None, "", "Unknown", "O(n + m)", "O(n·m)", "n^2", "O(log)"])
def test_unparseable_growth_is_not_compared(bad):
    assert cmp.growth_key(bad) is None
    assert cmp.compare_growth(bad, "O(n)") is None


def test_compare_growth():
    assert cmp.compare_growth("O(n^2)", "O(n)") == "lower"
    assert cmp.compare_growth("O(n)", "O(n log n)") == "higher"
    assert cmp.compare_growth("O(n)", "O(n)") == "same"


# --------------------------------------------------------------------------- #
# Metric calculations
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def hash_pair():
    p = get_pair("nested_loops_to_hash_map")
    rb, ra = predict(p.before).to_dict(), predict(p.after).to_dict()
    return rb, ra, cmp.compare(rb, ra)


def test_metric_changes_are_after_minus_before(hash_pair):
    rb, ra, c = hash_pair
    vb, va = cmp.summary_values(rb), cmp.summary_values(ra)
    for key, _ in cmp.METRICS:
        m = c.metric(key)
        assert (m.before, m.after) == (vb[key], va[key])
        if isinstance(m.before, (int, float)) and isinstance(m.after, (int, float)):
            assert m.change == pytest.approx(m.after - m.before)
            assert m.direction == ("same" if m.change == 0 else "lower" if m.change < 0 else "higher")
    assert c.metric("max_nesting_depth").before == rb["file_summary"]["maximum_nesting_depth"] == 3
    assert c.metric("max_nesting_depth").after == 1
    assert c.metric("derived_score").before == rb["derived_score"]["highest"]
    assert c.metric("function_count").change == 0


def test_function_level_predictions_and_big_o(hash_pair):
    rb, ra, c = hash_pair
    f = c.function("count_pairs_with_sum")
    assert f.status == "in both"
    assert (f.time_before, f.time_after, f.time_change) == ("O(n^2)", "O(n)", "lower")
    assert (f.space_before, f.space_after, f.space_change) == ("O(1)", "O(n)", "higher")
    assert (f.cyclomatic_before, f.cyclomatic_after) == (4, 2)
    fb = rb["functions"][0]
    assert f.ml_before == (fb.get("ml_prediction") or {}).get("big_o")
    assert f.ml_probability_before == (fb.get("ml_prediction") or {}).get("probability")
    assert any("auxiliary space" in t for t in titles(c))


def test_structural_category_direction():
    p = get_pair("branches_to_lookup")
    c = run(p.before, p.after)
    cat = c.metric("structural_category")
    assert (cat.before, cat.after, cat.direction) == ("High", "Medium", "lower")


# --------------------------------------------------------------------------- #
# Honest observations
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("key", ["nested_loops_to_hash_map", "recursion_to_iteration", "branches_to_lookup",
                                 "nesting_to_guard_clauses", "shorter_but_slower", "shorter_same_complexity"])
def test_observations_never_claim_improvement(key):
    p = get_pair(key)
    c = run(p.before, p.after)
    text = " ".join(f"{o.title} {o.detail}" for o in c.observations).lower()
    for word in ("improv", "better", "worse", "optimi"):
        assert word not in text


def test_line_change_always_carries_the_caveat():
    p = get_pair("shorter_same_complexity")
    c = run(p.before, p.after)
    line_obs = next(o for o in c.observations if o.title.startswith("Code lines"))
    assert line_obs.detail == cmp.LINE_CAVEAT
    assert "linear_search: static time estimate unchanged (O(n))" in titles(c)
    assert "Highest function cyclomatic complexity unchanged (3)" in titles(c)


def test_shorter_but_slower_is_flagged():
    p = get_pair("shorter_but_slower")
    c = run(p.before, p.after)
    assert c.metric("code_lines").direction == "lower"
    assert c.metric("max_cyclomatic_complexity").direction == "lower"
    assert c.function("smallest").time_change == "higher"
    cautions = [o for o in c.observations if o.kind == "caution"]
    assert "Shorter, but with a higher static time estimate" in [o.title for o in cautions]
    assert "smallest: static time estimate O(n) → O(n log n)" in [o.title for o in cautions]


def test_shorter_without_structural_change_is_flagged():
    before = "def f(a, b):\n    total = a + b\n    result = total\n    return result\n"
    after = "def f(a, b):\n    return a + b\n"
    c = run(before, after)
    assert c.metric("code_lines").direction == "lower"
    assert c.metric("max_cyclomatic_complexity").direction == "same"
    assert "Shorter, with no measured structural reduction" in titles(c)


def test_longer_but_lower_growth_is_noted():
    p = get_pair("recursion_to_iteration")
    c = run(p.before, p.after)
    assert c.metric("code_lines").direction == "higher"
    assert "Longer, with a lower static time estimate" in titles(c)


def test_unknown_estimates_are_not_compared():
    p = get_pair("branches_to_lookup")
    c = run(p.before, p.after)
    f = c.function("shipping_cost")
    assert f.time_after == "Unknown" and f.time_change is None
    assert "shipping_cost: time estimates cannot be compared" in titles(c)
    added = c.function("weight_surcharge")
    assert added.status == "added" and added.time_before is None


def test_ml_change_is_reported_as_model_output():
    p = get_pair("recursion_to_iteration")
    c = run(p.before, p.after)
    ml = c.metric("ml_class")
    obs = [o for o in c.observations if o.title.startswith("ML class")]
    if ml.before is not None and ml.after is not None and ml.before != ml.after:
        assert obs and obs[0].detail == cmp.ML_CAVEAT
    else:
        assert not obs


def test_identical_code_has_no_measured_difference():
    code = get_pair("shorter_but_slower").before
    c = run(code, code)
    assert titles(c)[0] == "No measured difference"
    assert all(m.direction in ("same", "n/a") for m in c.metrics)
    assert c.function("smallest").time_change == "same"


def test_renamed_function_is_removed_plus_added():
    c = run("def f(x):\n    return x\n", "def g(x):\n    return x\n")
    assert {f.function_id: f.status for f in c.functions} == {"f": "removed", "g": "added"}
    assert "Functions added or removed" in titles(c)


def test_to_dict_is_json_serialisable(hash_pair):
    import json
    json.dumps(hash_pair[2].to_dict(), default=str)
