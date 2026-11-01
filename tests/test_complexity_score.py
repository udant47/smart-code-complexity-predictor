"""Tests for src.complexity_score (Phase 13)."""

import math
import textwrap

import pandas as pd
import pytest

from src import complexity_score as dcs
from src import config
from src.feature_extractor import extract_features

ZERO = {c.metric: c.trivial for c in dcs.COMPONENTS}  # every component at its trivial anchor
FULL = {c.metric: c.saturation for c in dcs.COMPONENTS}  # every component saturated


def functions_of(src: str) -> dict:
    report = extract_features(textwrap.dedent(src))
    return {row["qualname"]: row for row in report.function_features}


# --------------------------------------------------------------------------- #
# Definition
# --------------------------------------------------------------------------- #
def test_weights_sum_to_one_and_no_component_dominates():
    weights = [c.weight for c in dcs.COMPONENTS]
    assert sum(weights) == pytest.approx(1.0)
    assert max(weights) <= 0.25
    assert len(dcs.COMPONENTS) >= 5  # multiple metrics
    assert all(c.saturation > c.trivial >= 0 for c in dcs.COMPONENTS)


def test_label_and_disclaimer():
    r = dcs.score_metrics(ZERO)
    assert r.label == "Derived Complexity Score"
    for phrase in ("Not an industry standard", "not a probability", "separate from the ML"):
        assert phrase in r.disclaimer


def test_bands_cover_0_to_100_without_gaps():
    edges = [(lo, hi) for _, lo, hi, _ in dcs.BANDS]
    assert edges[0][0] == 0 and edges[-1][1] > 100
    assert all(a[1] == b[0] for a, b in zip(edges, edges[1:]))
    assert [dcs.band_for(s)[0] for s in (0, 24.9, 25, 49.9, 50, 74.9, 75, 100)] == \
        ["Low", "Low", "Moderate", "Moderate", "High", "High", "Very high", "Very high"]


# --------------------------------------------------------------------------- #
# Normalization
# --------------------------------------------------------------------------- #
def test_normalize_anchors_and_log_scale():
    assert dcs.normalize(1, 1, 30) == 0.0
    assert dcs.normalize(30, 1, 30) == 1.0
    assert dcs.normalize(0, 1, 30) == 0.0  # below trivial
    mid = dcs.normalize(5, 1, 30)
    assert mid == pytest.approx((math.log1p(5) - math.log1p(1)) / (math.log1p(30) - math.log1p(1)))
    # diminishing returns: the first steps count more than later ones
    assert dcs.normalize(4, 1, 30) - dcs.normalize(2, 1, 30) > dcs.normalize(24, 1, 30) - dcs.normalize(22, 1, 30)


def test_extremes():
    assert dcs.score_metrics(ZERO).score == 0.0
    assert dcs.score_metrics(FULL).score == 100.0
    assert dcs.score_metrics(FULL).band == "Very high"


def test_unusually_large_values_are_capped():
    huge = {**ZERO, "cyclomatic_complexity": 10_000, "code_lines": 1_000_000}
    r = dcs.score_metrics(huge)
    by = {c.metric: c for c in r.components}
    assert by["cyclomatic_complexity"].contribution == pytest.approx(25.0)
    assert by["code_lines"].contribution == pytest.approx(15.0)
    assert by["code_lines"].saturated
    assert r.score == pytest.approx(40.0)


@pytest.mark.parametrize("component", dcs.COMPONENTS, ids=lambda c: c.metric)
def test_one_metric_alone_cannot_exceed_its_weight(component):
    r = dcs.score_metrics({**ZERO, component.metric: 10**9})
    assert r.score == pytest.approx(100 * component.weight)


@pytest.mark.parametrize("component", dcs.COMPONENTS, ids=lambda c: c.metric)
def test_monotonic_in_every_metric(component):
    scores = [dcs.score_metrics({**ZERO, component.metric: v}).score
              for v in (component.trivial, component.trivial + 1, (component.trivial + component.saturation) / 2,
                        component.saturation, component.saturation * 3)]
    assert scores == sorted(scores) and scores[-1] > scores[0]


def test_contributions_add_up_to_score():
    r = dcs.score_metrics({**ZERO, "cyclomatic_complexity": 8, "maximum_nesting_depth": 3, "code_lines": 40})
    assert sum(c.contribution for c in r.components) == pytest.approx(r.score, abs=0.1)
    assert [c.name for c in r.top_contributors(2)] == ["Decision paths", "Nesting"]


# --------------------------------------------------------------------------- #
# Missing metrics
# --------------------------------------------------------------------------- #
def test_missing_metric_is_excluded_and_weights_rescaled():
    partial = {k: v for k, v in FULL.items() if k != "recursive_call_count"}
    r = dcs.score_metrics(partial)
    assert r.missing == ["recursive_call_count"]
    assert r.coverage == pytest.approx(0.95)
    assert r.score == 100.0  # every available component saturated
    assert sum(c.effective_weight for c in r.components) == pytest.approx(1.0, abs=1e-3)
    assert any("Missing metrics" in w for w in r.warnings)


@pytest.mark.parametrize("bad", [None, float("nan"), float("inf"), -3, "many", True])
def test_invalid_values_count_as_missing(bad):
    r = dcs.score_metrics({**ZERO, "cyclomatic_complexity": bad})
    assert "cyclomatic_complexity" in r.missing
    assert r.score is not None


def test_too_much_missing_gives_no_score():
    r = dcs.score_metrics({"cyclomatic_complexity": 12, "parameter_count": 4})  # 35% of the weight
    assert r.score is None and r.band is None and r.percentile is None
    assert r.coverage < dcs.MIN_COVERAGE
    assert any("no score is reported" in w for w in r.warnings)
    assert dcs.score_metrics({}).score is None


def test_not_applicable_is_not_missing():
    r = dcs.score_module({**FULL})
    assert r.not_applicable == ["parameter_count"] and r.missing == [] and r.coverage == 1.0
    assert r.score == 100.0


# --------------------------------------------------------------------------- #
# Percentiles and summaries
# --------------------------------------------------------------------------- #
def test_percentile_is_monotonic_and_bounded():
    pts = [dcs.percentile_of(s) for s in (0, 5, 10, 20, 40, 70, 100)]
    assert pts == sorted(pts) and pts[0] == 0.0 and pts[-1] == 100.0
    assert dcs.percentile_of(None) is None
    assert len(dcs.REFERENCE_QUANTILES) == 21
    assert list(dcs.REFERENCE_QUANTILES) == sorted(dcs.REFERENCE_QUANTILES)


def test_summary_highest_and_line_weighted_mean():
    a, b = dcs.score_metrics(ZERO), dcs.score_metrics(FULL)
    s = dcs.summarize([("small", a, 10), ("big", b, 30)])
    assert s["highest"] == 100.0 and s["highest_function"] == "big"
    assert s["line_weighted_mean"] == pytest.approx(75.0)
    assert s["band_counts"]["Low"] == 1 and s["band_counts"]["Very high"] == 1
    empty = dcs.summarize([])
    assert empty["highest"] is None


# --------------------------------------------------------------------------- #
# Real code
# --------------------------------------------------------------------------- #
REAL = '''
def add(a, b):
    return a + b


def process(records, rules, limit, mode, verbose, strict):
    out = []
    for r in records:
        if r is None:
            continue
        for rule in rules:
            if rule.applies(r):
                if mode == "a" and strict:
                    for k in r.keys():
                        if k in rule.fields:
                            while limit > 0 and r[k]:
                                limit -= 1
                                if verbose:
                                    print(k)
                elif mode == "b" or verbose:
                    out.append(rule.apply(r))
    return process(out, rules, limit - 1, mode, verbose, strict) if limit > 0 else out
'''


def test_real_functions_are_ranked_sensibly():
    rows = functions_of(REAL)
    simple, tangled = dcs.score_function(rows["add"]), dcs.score_function(rows["process"])
    assert simple.score == 0.0 and simple.band == "Low" and simple.missing == []
    assert tangled.score > 50 and tangled.band in ("High", "Very high")
    assert tangled.percentile > 95
    names = [c.name for c in tangled.top_contributors(3)]
    assert "Nesting" in names and "Loop nesting" in names


def test_deterministic():
    rows = functions_of(REAL)
    assert dcs.score_function(rows["process"]).to_dict() == dcs.score_function(rows["process"]).to_dict()


def test_score_is_independent_of_the_ml_model(monkeypatch):
    """The score is computed from metrics only; no model is loaded."""
    import src.predictor as pr

    monkeypatch.setattr(pr, "get_model", lambda *a, **k: (_ for _ in ()).throw(AssertionError("model used")))
    assert dcs.score_function(functions_of(REAL)["process"]).score > 50


@pytest.mark.skipif(not config.STRUCTURAL_DATASET_CSV.exists(), reason="structural dataset not built")
def test_reference_quantiles_match_the_dataset():
    rows = pd.read_csv(config.STRUCTURAL_DATASET_CSV, keep_default_na=False).to_dict("records")
    fresh = dcs.reference_quantiles(rows)
    # Python 3.11 builds reproduce them exactly; newer Pythons parse ~8% more files, so allow small drift
    assert max(abs(a - b) for a, b in zip(fresh, dcs.REFERENCE_QUANTILES)) <= 3.0
