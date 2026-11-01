"""Derived Complexity Score: a 0-100 structural indicator (Phase 13).

    from src.complexity_score import score_function, score_metrics
    s = score_metrics(function_metrics_dict)     # ScoreResult
    s.score, s.band, s.components, s.percentile

What it is, and what it is not
------------------------------
The **Derived Complexity Score** is a project-specific indicator for
*relative comparison*: it ranks pieces of code by how much structure a
reader has to keep in mind (branching, nesting, size, loops, parameters,
vocabulary, recursion).

* It is **not an industry standard**. The weights and anchors below are
  this project's documented choices, partly informed by common tool
  thresholds (radon's cyclomatic-complexity ranks, pylint's default
  argument limit), not a validated model.
* It is **not a probability** of defects, bugs or poor quality, and it was
  not calibrated against any outcome data.
* It is **separate from the ML prediction.** The machine-learning model
  predicts a time-complexity class (O(1) ... O(2^n)); this score measures
  structure. A short O(2^n) function can score low, and a long O(1) function
  can score high.

Formula
-------
For each component *i* with metric value *x*, a "trivial" anchor *a_i* and a
"saturation" anchor *b_i*::

    s_i = clip( (ln(1 + x) - ln(1 + a_i)) / (ln(1 + b_i) - ln(1 + a_i)), 0, 1 )
    score = 100 * sum(w_i * s_i) / sum(w_i over available components)

* **Log scale:** going from 2 to 4 nested levels matters more than going from 20 to
  22 lines, so each metric has diminishing returns.
* **Anchors:** values at or below *a_i* contribute 0. Values at or above *b_i*
  contribute the full weight. **Unusually large values are capped** at the
  saturation anchor, so one 2,000-line function cannot push the score past 100
  or swamp the other components.
* **Weights** sum to 1, and the largest is 0.25. **No single metric can contribute
  more than its weight × 100 points**, whatever its value. Cyclomatic
  complexity, nesting and loop nesting are correlated, so their weights are
  kept moderate to limit double counting.

=================== ============================ ======= ========== ==============
Component           Metric                       Weight  Trivial    Saturation
=================== ============================ ======= ========== ==============
Decision paths      cyclomatic_complexity        0.25    1          30
Nesting             maximum_nesting_depth        0.20    1          6
Size                code_lines                   0.15    10         150
Loop nesting        maximum_loop_nesting_depth   0.15    1          4
Parameters          parameter_count              0.10    3          8
Vocabulary          unique_identifier_count      0.10    10         60
Recursion           recursive_call_count         0.05    0          2
=================== ============================ ======= ========== ==============

Anchor sources: cyclomatic complexity 30 is the boundary of radon's rank D
("more than moderate risk"); 8 parameters is above pylint's default limit of
5; the others are project choices, documented here so they can be changed.

Missing metrics
---------------
A metric that is absent, ``None``, ``NaN``, infinite or negative is
**excluded** and listed in ``missing``. The weights of the remaining components are
rescaled to sum to 1. If the available components carry less than
``MIN_COVERAGE`` (60%) of the total weight, **no score is produced**
(``score is None``) instead of a misleading one. For module-level scripts,
``parameter_count`` does not exist, so it is reported as not applicable and
the score uses the other 90% of the weight.

Interpretation
--------------
Score bands (fixed thresholds, independent of the ML model):

======== ========== =====================================================
Score    Band       Reading
======== ========== =====================================================
0–24     Low        little structure to follow
25–49    Moderate   some branching, nesting or size
50–74    High       substantial structure; worth reviewing or splitting
75–100   Very high  most components near saturation
======== ========== =====================================================

``percentile`` places a score among 3,910 real functions from
TheAlgorithms/Python (pinned commit, see ``config.THEALGORITHMS``). For
example, percentile 90 means a higher score than about 90% of those functions.
That reference is educational code, so the percentiles describe relative position in
that corpus only. On that corpus the bands split 80% Low, 18% Moderate, 2% High and
no Very high (maximum 69.8): the anchors are deliberately generous, so High and
Very high mark code that stands out.

Because small functions differ mostly in their branching, the score correlates
strongly with cyclomatic complexity on that corpus (Spearman 0.91). The other
components separate functions with similar branching but different nesting,
size, loops, parameters or recursion.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Mapping, Optional

LABEL = "Derived Complexity Score"
DISCLAIMER = (
    "Project-specific indicator for relative comparison of code structure. Not an industry standard, "
    "not a probability of defects or poor quality, and separate from the ML time-complexity prediction."
)
MIN_COVERAGE = 0.6


@dataclass(frozen=True)
class Component:
    name: str
    metric: str
    weight: float
    trivial: float
    saturation: float
    description: str


COMPONENTS: tuple[Component, ...] = (
    Component("Decision paths", "cyclomatic_complexity", 0.25, 1, 30, "independent paths through the code (McCabe)"),
    Component("Nesting", "maximum_nesting_depth", 0.20, 1, 6, "deepest level of nested blocks"),
    Component("Size", "code_lines", 0.15, 10, 150, "lines containing code"),
    Component("Loop nesting", "maximum_loop_nesting_depth", 0.15, 1, 4, "loops inside loops"),
    Component("Parameters", "parameter_count", 0.10, 3, 8, "inputs the caller must supply"),
    Component("Vocabulary", "unique_identifier_count", 0.10, 10, 60, "distinct names a reader must track"),
    Component("Recursion", "recursive_call_count", 0.05, 0, 2, "direct self-calls"),
)

BANDS: tuple[tuple[str, float, float, str], ...] = (
    ("Low", 0, 25, "little structure to follow"),
    ("Moderate", 25, 50, "some branching, nesting or size"),
    ("High", 50, 75, "substantial structure; worth reviewing or splitting"),
    ("Very high", 75, 100.0001, "most components near saturation"),
)

# Scores of 3,910 functions from TheAlgorithms/Python (pinned commit), built with
# this module's formula on Python 3.11; quantiles at 0%, 5%, ..., 100%.
# Regenerate with reference_quantiles(); a test checks they stay in sync.
REFERENCE_QUANTILES: tuple[float, ...] = (
    0.0, 0.0, 0.0, 0.0, 2.1, 3.7, 4.2, 6.0, 7.1, 8.8, 10.5,
    12.3, 14.6, 16.8, 19.1, 21.6, 24.9, 29.4, 33.8, 41.4, 69.8,
)
REFERENCE_SOURCE = "TheAlgorithms/Python functions (Phase 7 structural dataset, Python 3.11)"


@dataclass
class ComponentScore:
    name: str
    metric: str
    value: Optional[float]
    normalized: Optional[float]  # 0..1, None when missing
    weight: float  # original weight
    effective_weight: float  # after rescaling for missing components
    contribution: float  # points out of 100
    saturated: bool


@dataclass
class ScoreResult:
    label: str
    score: Optional[float]  # 0..100, None when too much is missing
    band: Optional[str]
    band_meaning: Optional[str]
    percentile: Optional[float]  # position among the reference functions
    coverage: float  # share of total weight that was available
    components: list = field(default_factory=list)
    missing: list = field(default_factory=list)
    not_applicable: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    disclaimer: str = DISCLAIMER

    def to_dict(self) -> dict:
        return asdict(self)

    def top_contributors(self, n: int = 3) -> list[ComponentScore]:
        return sorted((c for c in self.components if c.contribution > 0), key=lambda c: -c.contribution)[:n]


# --------------------------------------------------------------------------- #
# Core
# --------------------------------------------------------------------------- #
def normalize(value: float, trivial: float, saturation: float) -> float:
    """Log-scaled position of ``value`` between the two anchors, clipped to [0, 1]."""
    lo, hi = math.log1p(trivial), math.log1p(saturation)
    return min(1.0, max(0.0, (math.log1p(value) - lo) / (hi - lo)))


def _valid(v) -> Optional[float]:
    if v is None or isinstance(v, bool):
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if math.isnan(f) or math.isinf(f) or f < 0:
        return None
    return f


def band_for(score: Optional[float]) -> tuple[Optional[str], Optional[str]]:
    if score is None:
        return None, None
    for name, lo, hi, meaning in BANDS:
        if lo <= score < hi:
            return name, meaning
    raise ValueError(f"score out of range: {score}")


def percentile_of(score: Optional[float], quantiles: tuple[float, ...] = None) -> Optional[float]:
    """Approximate share (0-100) of reference functions scoring below ``score``."""
    q = REFERENCE_QUANTILES if quantiles is None else quantiles
    if score is None or not q:
        return None
    levels = [100 * i / (len(q) - 1) for i in range(len(q))]
    if score <= q[0]:
        return 0.0
    if score >= q[-1]:
        return 100.0
    # last level whose quantile is below the score, interpolating within flat runs conservatively
    for i in range(1, len(q)):
        if score < q[i]:
            lo_q, hi_q = q[i - 1], q[i]
            frac = (score - lo_q) / (hi_q - lo_q) if hi_q > lo_q else 0.0
            return round(levels[i - 1] + frac * (levels[i] - levels[i - 1]), 1)
    return 100.0


def score_metrics(metrics: Mapping, not_applicable: tuple[str, ...] = ()) -> ScoreResult:
    """Score a mapping of metric name -> value (e.g. a FunctionRecord or feature dict)."""
    parts, missing, warnings = [], [], []
    for c in COMPONENTS:
        if c.metric in not_applicable:
            continue
        raw = metrics.get(c.metric) if hasattr(metrics, "get") else None
        v = _valid(raw)
        if v is None:
            missing.append(c.metric)
            if raw is not None:
                warnings.append(f"{c.metric}={raw!r} is not a valid count and was ignored.")
        parts.append((c, v))

    available = sum(c.weight for c, v in parts if v is not None)
    applicable = sum(c.weight for c, _ in parts)
    coverage = round(available / applicable, 4) if applicable else 0.0
    result = ScoreResult(label=LABEL, score=None, band=None, band_meaning=None, percentile=None,
                         coverage=coverage, missing=missing, not_applicable=list(not_applicable), warnings=warnings)
    if available == 0 or coverage < MIN_COVERAGE:
        result.warnings.append(
            f"Only {100 * coverage:.0f}% of the score's weight could be computed (minimum {100 * MIN_COVERAGE:.0f}%); "
            "no score is reported.")
        result.components = [ComponentScore(c.name, c.metric, v, None, c.weight, 0.0, 0.0, False) for c, v in parts]
        return result

    total_points = 0.0
    for c, v in parts:
        if v is None:
            result.components.append(ComponentScore(c.name, c.metric, None, None, c.weight, 0.0, 0.0, False))
            continue
        s = normalize(v, c.trivial, c.saturation)
        w = c.weight / available
        pts = 100 * w * s
        total_points += pts
        result.components.append(ComponentScore(c.name, c.metric, v, round(s, 4), c.weight, round(w, 4),
                                                 round(pts, 2), v >= c.saturation))
    if missing:
        result.warnings.append(f"Missing metrics ({', '.join(missing)}) were left out and the other weights rescaled.")
    score = round(min(100.0, max(0.0, total_points)), 1)
    result.score = score
    result.band, result.band_meaning = band_for(score)
    result.percentile = percentile_of(score)
    return result


# --------------------------------------------------------------------------- #
# Convenience wrappers
# --------------------------------------------------------------------------- #
def score_function(row: Mapping) -> ScoreResult:
    """Score one function from its feature row (``extract_features(...).function_features``).

    Any mapping with the metric names works; metrics it lacks are handled as
    documented under "Missing metrics".
    """
    return score_metrics(row)


def score_module(file_metrics: Mapping) -> ScoreResult:
    """Score module-level code (a script with no functions) from file-level metrics.

    A script has no parameters, so that component is not applicable.
    """
    return score_metrics(file_metrics, not_applicable=("parameter_count",))


def summarize(function_scores: list[tuple[str, ScoreResult, int]]) -> dict:
    """File summary: highest score, and the mean weighted by code lines."""
    scored = [(fid, s, max(lines, 1)) for fid, s, lines in function_scores if s.score is not None]
    if not scored:
        return {"label": LABEL, "highest": None, "highest_function": None, "line_weighted_mean": None,
                "band_counts": {b[0]: 0 for b in BANDS}}
    top = max(scored, key=lambda t: t[1].score)
    weighted = sum(s.score * n for _, s, n in scored) / sum(n for *_, n in scored)
    counts = {b[0]: sum(1 for _, s, _ in scored if s.band == b[0]) for b in BANDS}
    return {"label": LABEL, "highest": top[1].score, "highest_function": top[0], "highest_band": top[1].band,
            "line_weighted_mean": round(weighted, 1), "mean_band": band_for(round(weighted, 1))[0],
            "band_counts": counts, "disclaimer": DISCLAIMER}


def reference_quantiles(function_rows) -> tuple[float, ...]:
    """Recompute REFERENCE_QUANTILES from feature rows (e.g. structural_functions.csv)."""
    import numpy as np

    scores = [score_metrics(row).score for row in function_rows]
    scores = [s for s in scores if s is not None]
    return tuple(round(float(v), 1) for v in np.quantile(scores, np.linspace(0, 1, 21)))
