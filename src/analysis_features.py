"""Additional program features (final improvement phase).

Two kinds, both computed statically from the syntax tree (nothing is executed):

1. **Static-estimate features.** The rule-based analyzer's own time estimate for
   the whole program, encoded as numbers the model can use: whether an estimate
   exists, the polynomial degree, the power of log n and whether growth is
   exponential. Scripts (module-level code, as in most competitive-programming
   solutions) are analysed with ``analyze_script``; files made of functions use
   the largest per-function estimate. "Unknown" stays unknown (``known = 0``).

2. **AST counts the existing features do not cover:** assignments, augmented
   assignments, subscripts, slices, collection-mutating method calls,
   membership tests, call nesting depth, the largest parameter list and
   ``sys.setrecursionlimit``.

Which of these the final model uses is decided by grouped cross-validation, not
assumed; see ``reports/improvement/improvement_report.md``.
"""

from __future__ import annotations

import ast
from typing import Optional, Union

from src.code_parser import ParseResult, parse_source
from src.comparison import growth_key

STATIC_FEATURE_NAMES = (
    "static_time_known",
    "static_time_degree",
    "static_time_log_power",
    "static_time_exponential",
)
AST_EXTRA_FEATURE_NAMES = (
    "assignment_count",
    "augmented_assignment_count",
    "subscript_count",
    "slice_count",
    "collection_mutation_call_count",
    "membership_test_count",
    "max_call_nesting_depth",
    "max_parameter_count",
    "recursion_limit_set",
)
ANALYSIS_FEATURE_NAMES = STATIC_FEATURE_NAMES + AST_EXTRA_FEATURE_NAMES

FEATURE_DESCRIPTIONS = {
    "static_time_known": "1 if the rule-based analyzer produced a time estimate for the program, else 0.",
    "static_time_degree": "Polynomial degree of that estimate (O(n^2) = 2, O(n log n) = 1); 0 when unknown.",
    "static_time_log_power": "Power of log n in that estimate (O(log n) = 1, O(n log n) = 1); 0 when unknown.",
    "static_time_exponential": "1 if that estimate grows exponentially or factorially, else 0.",
    "assignment_count": "Assignment statements (=, annotated), including tuple unpacking.",
    "augmented_assignment_count": "Augmented assignments such as +=, -=, *=, //=.",
    "subscript_count": "Index operations a[i] (excluding slices).",
    "slice_count": "Slice operations a[i:j].",
    "collection_mutation_call_count": "Calls to append, extend, insert, pop, remove, add, update, popleft, appendleft, discard.",
    "membership_test_count": "Membership tests with in / not in.",
    "max_call_nesting_depth": "Deepest nesting of calls inside call arguments, e.g. f(g(h(x))) = 3.",
    "max_parameter_count": "Largest number of parameters of any function (0 without functions).",
    "recursion_limit_set": "1 if the code calls sys.setrecursionlimit (a hint of deep recursion), else 0.",
}

_MUTATORS = {"append", "extend", "insert", "pop", "remove", "add", "update", "popleft", "appendleft", "discard"}


def static_estimate_features(parsed: ParseResult) -> dict:
    """Encode the program-level static time estimate (script code, else the largest function estimate)."""
    from src.algorithm_analyzer import analyze_algorithms, analyze_script

    out = dict.fromkeys(STATIC_FEATURE_NAMES, 0)
    keys = []
    script = analyze_script(parsed)
    if script is not None:
        if script.time_status == "estimated":
            keys.append(growth_key(script.time_class))
    else:
        a = analyze_algorithms(parsed)
        if a.success:
            keys = [growth_key(e.time_class) for e in a.estimates if e.time_status == "estimated"]
    keys = [k for k in keys if k is not None]
    if keys:
        fact, base, power, logs = max(keys)
        out.update(static_time_known=1, static_time_degree=float(power), static_time_log_power=int(logs),
                   static_time_exponential=int(bool(fact or base)))
    return out


def _call_depth(node: ast.AST) -> int:
    """Deepest chain of calls nested in call arguments (iterative)."""
    best = 0
    stack = [(node, 0)]
    while stack:
        n, d = stack.pop()
        if isinstance(n, ast.Call):
            d += 1
            best = max(best, d)
        stack.extend((c, d) for c in ast.iter_child_nodes(n))
    return best


def ast_extra_features(tree: ast.AST) -> dict:
    out = dict.fromkeys(AST_EXTRA_FEATURE_NAMES, 0)
    for n in ast.walk(tree):
        if isinstance(n, (ast.Assign, ast.AnnAssign)):
            out["assignment_count"] += 1
        elif isinstance(n, ast.AugAssign):
            out["augmented_assignment_count"] += 1
        elif isinstance(n, ast.Subscript):
            out["slice_count" if isinstance(n.slice, ast.Slice) else "subscript_count"] += 1
        elif isinstance(n, ast.Compare):
            out["membership_test_count"] += sum(isinstance(op, (ast.In, ast.NotIn)) for op in n.ops)
        elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            a = n.args
            count = len(a.posonlyargs) + len(a.args) + len(a.kwonlyargs) + bool(a.vararg) + bool(a.kwarg)
            out["max_parameter_count"] = max(out["max_parameter_count"], count)
        if isinstance(n, ast.Call):
            f = n.func
            if isinstance(f, ast.Attribute) and f.attr in _MUTATORS:
                out["collection_mutation_call_count"] += 1
            if isinstance(f, ast.Attribute) and f.attr == "setrecursionlimit":
                out["recursion_limit_set"] = 1
    out["max_call_nesting_depth"] = _call_depth(tree)
    return out


def extract_analysis_features(source: Union[str, ParseResult]) -> dict:
    """All ANALYSIS_FEATURE_NAMES for one program (zeros for code that cannot be parsed)."""
    parsed = source if isinstance(source, ParseResult) else parse_source(source)
    if not parsed.success:
        return dict.fromkeys(ANALYSIS_FEATURE_NAMES, 0)
    return {**static_estimate_features(parsed), **ast_extra_features(parsed.tree)}


def program_features(parsed: ParseResult, file_features: Optional[dict] = None) -> dict:
    """Every model-input feature for one parsed program.

    This is the single place that builds the feature dictionary. The dataset
    builder (training) and the predictor (inference) both call it, so the two
    always see the same schema. The model then picks its own columns, in
    training order, from ``model_metadata.json``.
    """
    from src.feature_extractor import extract_features
    from src.pattern_features import extract_pattern_features

    if file_features is None:
        file_features = extract_features(parsed).file_features
    return {**file_features, **extract_pattern_features(parsed.tree), **extract_analysis_features(parsed)}
