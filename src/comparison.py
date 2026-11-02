"""Before/after comparison of two analysis results (Phase 17).

Pure functions over predictor result dicts (``PredictionResult.to_dict()``):
no Streamlit and no model logic, so the calculations are testable on their
own.

Principles
----------
* Only measured differences are reported. Every number shown comes from the
  two results.
* Directions are described neutrally ("lower", "higher", "same"). Nothing is
  called an improvement.
* A change in line count is always reported together with the caveat that
  line count alone says nothing about simplicity or speed. If the code got
  shorter while its static time estimate grew, or while no other measurement
  went down, that is called out explicitly.
* Big-O estimates are compared only when both sides have a static estimate
  with status "estimated". Unknown or partial estimates are reported as not
  comparable.
* A change in the ML class is reported as a change in model output, not as
  evidence: the model is a statistical estimate with limited accuracy.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from fractions import Fraction
from typing import Optional

# --------------------------------------------------------------------------- #
# Growth-order comparison of Big-O strings
# --------------------------------------------------------------------------- #
_POW = r"n\^(\d+(?:\.\d+)?)"  # plain pattern strings (the security test forbids any call named compile)
_EXP = r"(\d+)\^n"
_LOGK = r"log\^(\d+)"


def growth_key(big_o: Optional[str]) -> Optional[tuple]:
    """Sort key for a single-variable Big-O string such as ``O(n^2 log n)``.

    Returns ``(factorial, exponential_base, polynomial_power, log_power)`` or
    None when the string is not a single-variable class this function
    understands (e.g. ``Unknown`` or ``O(n + m)``). Comparing keys orders
    classes by asymptotic growth.
    """
    if not big_o:
        return None
    m = re.fullmatch(r"\s*O\((.+)\)\s*", big_o)
    if not m:
        return None
    body = m.group(1).strip()
    if body == "1":
        return (0, 0, Fraction(0), 0)
    fact, base, power, logs = 0, 0, Fraction(0), 0
    tokens = body.replace("·", " ").replace("*", " ").split()
    i = 0
    while i < len(tokens):
        t = tokens[i]
        if t == "n!":
            fact = 1
        elif re.fullmatch(_EXP, t):
            base = max(base, int(re.fullmatch(_EXP, t).group(1)))
        elif t == "log" or re.fullmatch(_LOGK, t):
            k = 1 if t == "log" else int(re.fullmatch(_LOGK, t).group(1))
            if i + 1 >= len(tokens) or tokens[i + 1] != "n":
                return None
            logs += k
            i += 1
        elif t == "n":
            power += 1
        elif t == "√n":
            power += Fraction(1, 2)
        elif re.fullmatch(_POW, t):
            power += Fraction(re.fullmatch(_POW, t).group(1))
        else:
            return None
        i += 1
    return (fact, base, power, logs)


def compare_growth(before: Optional[str], after: Optional[str]) -> Optional[str]:
    """"lower" / "higher" / "same" growth order of ``after`` relative to ``before``; None if not comparable."""
    a, b = growth_key(before), growth_key(after)
    if a is None or b is None:
        return None
    return "same" if a == b else ("lower" if b < a else "higher")


# --------------------------------------------------------------------------- #
# Data classes
# --------------------------------------------------------------------------- #
def _direction(a, b) -> str:
    if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
        return "n/a"
    return "same" if a == b else ("lower" if b < a else "higher")


def _delta(a, b):
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        d = b - a
        return round(d, 2) if isinstance(d, float) else d
    return None


@dataclass
class MetricChange:
    key: str
    label: str
    before: object
    after: object
    change: Optional[float]
    direction: str  # lower | higher | same | changed | n/a

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class FunctionChange:
    function_id: str
    status: str  # "in both" | "removed" | "added"
    lines_before: Optional[int] = None
    lines_after: Optional[int] = None
    cyclomatic_before: Optional[int] = None
    cyclomatic_after: Optional[int] = None
    nesting_before: Optional[int] = None
    nesting_after: Optional[int] = None
    score_before: Optional[float] = None
    score_after: Optional[float] = None
    category_before: Optional[str] = None
    category_after: Optional[str] = None
    time_before: Optional[str] = None
    time_after: Optional[str] = None
    time_status_before: Optional[str] = None
    time_status_after: Optional[str] = None
    time_change: Optional[str] = None  # lower | higher | same | None (not comparable)
    space_before: Optional[str] = None
    space_after: Optional[str] = None
    space_change: Optional[str] = None
    ml_before: Optional[str] = None
    ml_after: Optional[str] = None
    ml_probability_before: Optional[float] = None
    ml_probability_after: Optional[float] = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Observation:
    kind: str  # "info" | "caution"
    title: str
    detail: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Comparison:
    metrics: list[MetricChange] = field(default_factory=list)
    functions: list[FunctionChange] = field(default_factory=list)
    observations: list[Observation] = field(default_factory=list)

    def metric(self, key: str) -> MetricChange:
        return next(m for m in self.metrics if m.key == key)

    def function(self, function_id: str) -> FunctionChange:
        return next(f for f in self.functions if f.function_id == function_id)

    def to_dict(self) -> dict:
        return {"metrics": [m.to_dict() for m in self.metrics],
                "functions": [f.to_dict() for f in self.functions],
                "observations": [o.to_dict() for o in self.observations]}


LINE_CAVEAT = ("Line count alone is not evidence that code is simpler or faster; compare the structural "
               "metrics and the time estimates instead.")
ML_CAVEAT = ("ML classes are statistical estimates with limited measured accuracy (see Model evaluation). "
             "A change in model output is not evidence on its own.")


# --------------------------------------------------------------------------- #
# Extraction
# --------------------------------------------------------------------------- #
METRICS = (  # key, label
    ("code_lines", "Code lines"),
    ("function_count", "Functions"),
    ("class_count", "Classes"),
    ("loop_count", "Loops"),
    ("file_cyclomatic_complexity", "File cyclomatic complexity"),
    ("max_cyclomatic_complexity", "Highest function cyclomatic complexity"),
    ("max_nesting_depth", "Maximum nesting depth"),
    ("derived_score", "Derived Complexity Score (highest)"),
    ("recommendation_count", "Refactoring recommendations"),
)
CATEGORICAL = (
    ("structural_category", "Structural category"),
    ("derived_band", "Derived score band"),
    ("ml_class", "ML time-complexity class (whole file)"),
)
_CATEGORY_ORDER = {"Low": 0, "Medium": 1, "High": 2}


def summary_values(result: dict) -> dict:
    fs = result.get("file_summary") or {}
    fm = result.get("file_metrics") or {}
    ds = result.get("derived_score") or {}
    sc = result.get("structural_category") or {}
    recs = (result.get("recommendations") or {}).get("recommendations")
    return {
        "code_lines": fs.get("code_lines", fm.get("code_lines")),
        "function_count": fs.get("function_count", fm.get("function_count")),
        "class_count": fs.get("class_count", fm.get("class_count")),
        "loop_count": fm.get("loop_count"),
        "file_cyclomatic_complexity": fs.get("file_cyclomatic_complexity", fm.get("cyclomatic_complexity")),
        "max_cyclomatic_complexity": fs.get("maximum_cyclomatic_complexity"),
        "max_nesting_depth": fs.get("maximum_nesting_depth", fm.get("maximum_nesting_depth")),
        "derived_score": ds.get("highest"),
        "recommendation_count": None if recs is None else len(recs),
        "structural_category": sc.get("highest") or sc.get("module_level"),
        "derived_band": ds.get("highest_band"),
        "ml_class": (result.get("prediction") or {}).get("big_o"),
    }


def _function_values(f: dict) -> dict:
    m = f.get("metrics") or {}
    est = f.get("static_estimate") or {}
    ml = f.get("ml_prediction") or {}
    return {
        "lines": m.get("function_length"), "cyclomatic": m.get("cyclomatic_complexity"),
        "nesting": m.get("maximum_nesting_depth"), "score": (f.get("derived_score") or {}).get("score"),
        "category": f.get("structural_category"),
        "time": est.get("time_class"), "time_status": est.get("time_status"),
        "space": est.get("space_class"), "space_status": est.get("space_status"),
        "ml": ml.get("big_o"), "ml_probability": ml.get("probability"),
    }


def _estimated(value: Optional[str], status: Optional[str]) -> bool:
    return status == "estimated" and value not in (None, "Unknown")


def _function_changes(before: dict, after: dict) -> list[FunctionChange]:
    ia = {f["function_id"]: f for f in before.get("functions") or []}
    ib = {f["function_id"]: f for f in after.get("functions") or []}
    out = []
    for fid in list(ia) + [k for k in ib if k not in ia]:
        a = _function_values(ia[fid]) if fid in ia else {}
        b = _function_values(ib[fid]) if fid in ib else {}
        fc = FunctionChange(fid, "in both" if a and b else ("removed" if a else "added"))
        for k in ("lines", "cyclomatic", "nesting", "score", "category", "time", "time_status", "space", "ml",
                  "ml_probability"):
            setattr(fc, f"{k}_before", a.get(k))
            setattr(fc, f"{k}_after", b.get(k))
        if a and b:
            if _estimated(a["time"], a["time_status"]) and _estimated(b["time"], b["time_status"]):
                fc.time_change = compare_growth(a["time"], b["time"])
            if _estimated(a["space"], a["space_status"]) and _estimated(b["space"], b["space_status"]):
                fc.space_change = compare_growth(a["space"], b["space"])
        out.append(fc)
    return out


# --------------------------------------------------------------------------- #
# Main entry point
# --------------------------------------------------------------------------- #
def compare(before: dict, after: dict) -> Comparison:
    """Compare two successful predictor results (dicts)."""
    va, vb = summary_values(before), summary_values(after)
    comp = Comparison()
    for key, label in METRICS:
        a, b = va.get(key), vb.get(key)
        comp.metrics.append(MetricChange(key, label, a, b, _delta(a, b), _direction(a, b)))
    for key, label in CATEGORICAL:
        a, b = va.get(key), vb.get(key)
        direction = "n/a" if a is None or b is None else ("same" if a == b else "changed")
        if key == "structural_category" and a in _CATEGORY_ORDER and b in _CATEGORY_ORDER and a != b:
            direction = "lower" if _CATEGORY_ORDER[b] < _CATEGORY_ORDER[a] else "higher"
        comp.metrics.append(MetricChange(key, label, a, b, None, direction))
    comp.functions = _function_changes(before, after)
    comp.observations = _observations(comp)
    return comp


def _fmt(v) -> str:
    if isinstance(v, float):
        return f"{v:.1f}"
    return "—" if v is None else str(v)


def _band(x: MetricChange) -> str:
    return f"band unchanged: {x.after}" if x.before == x.after else f"band {x.before} → {x.after}"


def _observations(comp: Comparison) -> list[Observation]:
    obs: list[Observation] = []
    m = {x.key: x for x in comp.metrics}
    matched = [f for f in comp.functions if f.status == "in both"]
    time_higher = [f for f in matched if f.time_change == "higher"]
    time_lower = [f for f in matched if f.time_change == "lower"]

    # Static time estimates (per matched function)
    for f in matched:
        if f.time_change in ("lower", "higher"):
            meaning = ("grows more slowly for large inputs" if f.time_change == "lower"
                       else "grows faster for large inputs: slower at scale")
            obs.append(Observation("info" if f.time_change == "lower" else "caution",
                                   f"{f.function_id}: static time estimate {f.time_before} → {f.time_after}",
                                   f"{f.time_change.capitalize()} growth order: the after version {meaning}, "
                                   "under the analyzer's stated assumptions (worst case, primitive operations O(1))."))
        elif f.time_change is None and (f.time_before != f.time_after
                                        or "Unknown" in (f.time_before, f.time_after)):
            obs.append(Observation("info", f"{f.function_id}: time estimates cannot be compared",
                                   f"Static estimate {_fmt(f.time_before)} ({_fmt(f.time_status_before)}) → "
                                   f"{_fmt(f.time_after)} ({_fmt(f.time_status_after)}). An estimate that is "
                                   "Unknown or partial is not compared."))
        if f.space_change in ("lower", "higher"):
            obs.append(Observation("info", f"{f.function_id}: auxiliary space estimate {f.space_before} → {f.space_after}",
                                   "Uses less memory as the input grows." if f.space_change == "lower"
                                   else "Uses more memory as the input grows: a time saving may be paid for with memory."))

    # Static time estimate unchanged (worth saying when the code changed shape)
    for f in matched:
        if f.time_change == "same":
            obs.append(Observation("info", f"{f.function_id}: static time estimate unchanged ({f.time_after})",
                                   "Same growth order before and after."))

    # Structure
    words = {"max_cyclomatic_complexity": ("Fewer decision paths", "More decision paths"),
             "max_nesting_depth": ("Shallower nesting", "Deeper nesting"),
             "function_count": ("Fewer functions", "More functions")}
    for key, (fewer, more) in words.items():
        x = m[key]
        if x.direction in ("lower", "higher"):
            obs.append(Observation("info", f"{x.label}: {_fmt(x.before)} → {_fmt(x.after)}",
                                   f"{fewer if x.direction == 'lower' else more} ({x.change:+g})."))
    cc = m["max_cyclomatic_complexity"]
    if cc.direction == "same" and any(m[k].direction in ("lower", "higher") for k in ("max_nesting_depth", "code_lines")):
        obs.append(Observation("info", f"{cc.label} unchanged ({_fmt(cc.after)})",
                               "The same number of decision paths: the code changed shape, not the branching logic."))
    cat = m["structural_category"]
    if cat.direction in ("lower", "higher", "changed"):
        obs.append(Observation("info", f"Structural category: {cat.before} → {cat.after}",
                               "Rule-based category of the most complex function (from cyclomatic complexity)."))
    ds = m["derived_score"]
    if ds.direction in ("lower", "higher"):
        obs.append(Observation("info", f"Derived Complexity Score: {_fmt(ds.before)} → {_fmt(ds.after)}",
                               f"{ds.direction.capitalize()} by {abs(ds.change):.1f} points "
                               f"({_band(m['derived_band'])}). The score is a "
                               "weighted summary of structural metrics, not a measure of speed."))
    added = [f.function_id for f in comp.functions if f.status == "added"]
    removed = [f.function_id for f in comp.functions if f.status == "removed"]
    if added or removed:
        obs.append(Observation("info", "Functions added or removed",
                               (f"Added: {', '.join(added)}. " if added else "") +
                               (f"Removed: {', '.join(removed)}. " if removed else "") +
                               "Functions are matched by qualified name, so a renamed function appears as removed "
                               "plus added."))

    # Lines, always with the caveat, and explicit warnings when line count is misleading
    lines = m["code_lines"]
    if lines.direction in ("lower", "higher"):
        obs.append(Observation("info", f"Code lines: {_fmt(lines.before)} → {_fmt(lines.after)}", LINE_CAVEAT))
    structural_down = any(m[k].direction == "lower" for k in ("max_cyclomatic_complexity", "max_nesting_depth",
                                                             "derived_score"))
    if lines.direction == "lower" and time_higher:
        names = ", ".join(f.function_id for f in time_higher)
        obs.append(Observation("caution", "Shorter, but with a higher static time estimate",
                               f"The after version has fewer lines, yet the estimated growth order of {names} is "
                               "higher. Shorter code is not necessarily faster."))
    elif lines.direction == "lower" and not structural_down and not time_lower:
        obs.append(Observation("caution", "Shorter, with no measured structural reduction",
                               "Fewer lines, but cyclomatic complexity, nesting, the Derived Complexity Score and "
                               "the comparable time estimates did not go down."))
    elif lines.direction == "higher" and time_lower:
        obs.append(Observation("info", "Longer, with a lower static time estimate",
                               "The after version has more lines but a lower estimated growth order. Longer code "
                               "is not necessarily slower."))

    # ML output
    ml = m["ml_class"]
    if ml.direction == "changed":
        obs.append(Observation("info", f"ML class (whole file): {ml.before} → {ml.after}", ML_CAVEAT))

    unchanged = all(x.direction in ("same", "n/a") for x in comp.metrics)
    if unchanged and not any(f.time_change in ("lower", "higher") or f.time_before != f.time_after
                             for f in matched) and not added and not removed:
        obs.insert(0, Observation("info", "No measured difference",
                                  "The two versions have the same measured structure and static estimates."))
    return obs
