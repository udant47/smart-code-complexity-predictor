"""Deterministic refactoring recommendations from measured code metrics (Phase 15).

**These are review guidance, not guaranteed fixes.** Each recommendation
points at a place where the measured structure suggests a second look. Code
that triggers a rule can still be the clearest way to write that logic, and
code that triggers nothing is not guaranteed to be well designed.

How recommendations are produced
--------------------------------
* No language model and no randomness: the same code always gives the same
  recommendations, in the same order. The code is parsed with ``ast`` and
  measured statically; it is never executed.
* Every rule is a small, separate function registered in ``RULES``. A rule
  fires only when a documented threshold is crossed **and** it can quote the
  evidence (metric values, line numbers, the code pattern it found). A
  metric being non-zero is never enough: a function with one loop, one ``if``
  or one recursive call gets no recommendation for that.
* Suggested actions depend on the evidence. For example, "use a guard
  clause" is only suggested when an ``if`` block that could be inverted was
  actually found; otherwise the nesting rule suggests extracting the inner
  block.
* Overlap is resolved deterministically (``_resolve``): when the high
  cyclomatic complexity rule fires for a function, the generic part of the
  many-branches rule is folded into it instead of repeating the same advice.
  All rules recommend simplifying, flattening, extracting or checking; none
  recommends the opposite (inlining, merging, adding nesting), so they cannot
  contradict each other. Each (rule, location) pair appears at most once.

Rules and thresholds
--------------------
``THRESHOLDS`` holds every number. Sources:

* Cyclomatic complexity: radon's ranks (A 1-5, B 6-10, C 11-20, D 21-30, ...),
  the same bands the project uses elsewhere. 11+ medium, 21+ high.
* Nesting depth: pylint's ``too-many-nested-blocks`` default limit is 5;
  depth 4 is reported as medium and depth above 5 as high. Depth counts
  blocks inside the function body (``for`` > ``if`` > ``if`` is depth 3).
* Function length: code lines (no blank, comment or docstring lines).
  50+ medium, 100+ high. Pylint's ``too-many-statements`` default is 50
  statements; lines are used here as a simpler, documented stand-in.
* Conditional branches: an ``if``/``elif`` chain with 5 or more arms, or 15+
  ``if``/``elif``/conditional-expression/``case`` branches in one function.
  Pylint's ``too-many-branches`` default is 12; these are project choices.
* Recursion: fires only on concrete risk evidence (see ``rule_recursion``).
* Duplication: functions whose renamed-token 5-shingle Jaccard similarity is
  at least 0.85 (both with 6+ lines of code after the docstring), and identical (after renaming
  identifiers and literals) blocks of 3+ consecutive statements spanning 5+
  lines and 25+ tokens, found 2+ times.

Limitations
-----------
Static metrics cannot see intent, naming, tests, performance needs or domain
constraints. Recursion analysis is lexical: it does not follow aliases or
mutual recursion. Duplication detection finds copy-paste-like repetition, not
semantic duplication written differently.

Public API
----------
suggest(source_or_parse_result) -> SuggestionReport
SuggestionReport.recommendations  list[Recommendation], most severe first
SuggestionReport.to_dict() / .table()
RULES, THRESHOLDS, SEVERITY_ORDER, DISCLAIMER
"""

from __future__ import annotations

import ast
import textwrap
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from typing import Callable, Iterable, Optional, Union

import pandas as pd

from src.code_parser import ParseResult, parse_bytes, parse_source
from src.dataset_loader import code_tokens, jaccard, shingles
from src.feature_extractor import extract_features
from src.function_analyzer import analyze_functions
from src.pattern_features import _divides

DISCLAIMER = ("Review guidance from static metrics, not guaranteed fixes. The code may be fine as it is; "
              "use these as prompts for a human review.")

SEVERITY_ORDER = {"high": 0, "medium": 1, "low": 2}

THRESHOLDS = {
    "cyclomatic_medium": 11,
    "cyclomatic_high": 21,
    "nesting_medium": 4,
    "nesting_high": 6,
    "length_medium": 50,
    "length_high": 100,
    "chain_arms": 5,
    "branches_total": 15,
    "recursion_limit": 1000,  # CPython's default sys.getrecursionlimit()
    "duplicate_function_similarity": 0.85,
    "duplicate_function_min_lines": 6,
    "duplicate_block_min_statements": 3,
    "duplicate_block_min_lines": 5,
    "duplicate_block_min_tokens": 25,
}

_FUNC = (ast.FunctionDef, ast.AsyncFunctionDef)
_SCOPE = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)
_LOOPS = (ast.For, ast.AsyncFor, ast.While)
_MEMO_DECORATORS = {"lru_cache", "cache", "cached"}
_MEMO_NAMES = {"memo", "cache", "dp", "seen", "visited", "lookup", "table"}


# --------------------------------------------------------------------------- #
# Data structures
# --------------------------------------------------------------------------- #
@dataclass
class Evidence:
    metric: str
    value: object
    threshold: Optional[object] = None
    detail: str = ""


@dataclass
class Recommendation:
    rule_id: str
    title: str
    severity: str  # "high" | "medium" | "low"
    target: str  # function_id, or "<module>" / "<file>"
    start_line: int
    end_line: int
    evidence: list  # list[Evidence]
    explanation: str
    suggested_action: str
    action: str  # short tag: split | flatten | simplify_conditions | check_recursion | deduplicate
    related: list = field(default_factory=list)  # notes folded in from overlapping rules

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class SuggestionReport:
    success: bool
    recommendations: list = field(default_factory=list)
    rules_checked: list = field(default_factory=list)
    functions_checked: int = 0
    disclaimer: str = DISCLAIMER
    warnings: list = field(default_factory=list)
    error: Optional[dict] = None

    @property
    def counts(self) -> dict:
        out = {s: 0 for s in SEVERITY_ORDER}
        for r in self.recommendations:
            out[r.severity] += 1
        return out

    def for_target(self, target: str) -> list:
        return [r for r in self.recommendations if r.target == target]

    def to_dict(self) -> dict:
        return {"success": self.success, "recommendations": [r.to_dict() for r in self.recommendations],
                "counts": self.counts, "rules_checked": self.rules_checked,
                "functions_checked": self.functions_checked, "disclaimer": self.disclaimer,
                "warnings": self.warnings, "error": self.error}

    def table(self) -> pd.DataFrame:
        rows = [{"Severity": r.severity, "Rule": r.rule_id, "Title": r.title, "Where": r.target,
                 "Lines": f"{r.start_line}-{r.end_line}",
                 "Evidence": "; ".join(f"{e.metric} = {e.value}" + (f" (threshold {e.threshold})" if e.threshold is not None else "")
                                       for e in r.evidence),
                 "Suggested action": r.suggested_action} for r in self.recommendations]
        return pd.DataFrame(rows, columns=["Severity", "Rule", "Title", "Where", "Lines", "Evidence", "Suggested action"])


@dataclass
class Unit:
    """One analysed unit: a function, or the module body of a script without functions."""

    target: str
    node: ast.AST
    metrics: dict
    start_line: int
    end_line: int
    is_function: bool
    kind: str = "function"  # function | method | nested_function | module


@dataclass
class Context:
    parsed: ParseResult
    units: list
    functions: list  # FunctionInfo, aligned with function units
    lines: list


@dataclass(frozen=True)
class Rule:
    rule_id: str
    title: str
    check: Callable[[Context], list]
    description: str


# --------------------------------------------------------------------------- #
# AST helpers
# --------------------------------------------------------------------------- #
def _own_walk(node: ast.AST):
    """Nodes of a function's (or module's) own code, not entering nested defs, classes or lambdas."""
    stack = list(ast.iter_child_nodes(node))
    while stack:
        n = stack.pop()
        yield n
        if not isinstance(n, _SCOPE):
            stack.extend(ast.iter_child_nodes(n))


def _segment(ctx: Context, node: ast.AST) -> str:
    return "\n".join(ctx.lines[node.lineno - 1: getattr(node, "end_lineno", node.lineno)]).strip()


def _short(text: str, n: int = 70) -> str:
    text = " ".join(text.split())
    return text if len(text) <= n else text[: n - 1] + "…"


def _sev(value: float, medium: float, high: float) -> Optional[str]:
    return "high" if value >= high else "medium" if value >= medium else None


def _exits(stmts: list) -> bool:
    """The last statement leaves the block (return / raise / continue / break)."""
    return bool(stmts) and isinstance(stmts[-1], (ast.Return, ast.Raise, ast.Continue, ast.Break))


def _has_block(stmts: list) -> bool:
    return any(isinstance(s, (ast.If, ast.For, ast.AsyncFor, ast.While, ast.With, ast.AsyncWith, ast.Try)) for s in stmts)


# --------------------------------------------------------------------------- #
# Rules
# --------------------------------------------------------------------------- #
def rule_cyclomatic(ctx: Context) -> list:
    """High McCabe cyclomatic complexity (radon C and above)."""
    out = []
    for u in ctx.units:
        cc = u.metrics.get("cyclomatic_complexity", 0)
        sev = _sev(cc, THRESHOLDS["cyclomatic_medium"], THRESHOLDS["cyclomatic_high"])
        if not sev:
            continue
        m = u.metrics
        parts = {"if": m.get("if_count", 0), "elif": m.get("elif_count", 0), "loops": m.get("loop_count", 0),
                 "and/or": m.get("logical_operator_count", 0), "except": m.get("except_count", 0),
                 "conditional expressions": m.get("conditional_expression_count", 0)}
        parts = {k: v for k, v in parts.items() if v}
        biggest = max(parts, key=parts.get) if parts else None
        out.append(Recommendation(
            "HIGH_CYCLOMATIC_COMPLEXITY", "Review branching and consider splitting complex logic", sev, u.target,
            u.start_line, u.end_line,
            [Evidence("cyclomatic_complexity", cc, THRESHOLDS["cyclomatic_medium"] if sev == "medium" else THRESHOLDS["cyclomatic_high"],
                      "1 + number of decision points"),
             Evidence("decision points by kind", parts)],
            f"{u.target} has {cc} independent paths. Each path is a case to understand and test; radon ranks "
            f"{'11-20 as C (moderate risk)' if sev == 'medium' else '21+ as D or worse (high risk)'}.",
            "Look for parts that make separate decisions and move them into well-named helper functions, so each "
            "piece can be read and tested alone." + (f" The largest contributor is {biggest} ({parts[biggest]})." if biggest else ""),
            "split"))
    return out


def _guard_candidates(body: list) -> list:
    """if-blocks that could be inverted into an early exit: (line, kind)."""
    found = []

    def visit(stmts: list, in_loop: bool, top: bool):
        for i, s in enumerate(stmts):
            last = i == len(stmts) - 1
            if isinstance(s, ast.If):
                is_elif_chain = len(s.orelse) == 1 and isinstance(s.orelse[0], ast.If)
                if last and not s.orelse and _has_block(s.body) and (top or in_loop):
                    found.append((s.lineno, "wraps the rest of the block"))
                elif s.orelse and not is_elif_chain and _exits(s.body) and _has_block(s.orelse) and len(s.body) <= 3:
                    found.append((s.lineno, "else-branch after an early exit"))
            for child in ("body", "orelse", "finalbody"):
                inner = getattr(s, child, None)
                if isinstance(inner, list) and inner and isinstance(inner[0], ast.stmt) and not isinstance(s, _SCOPE):
                    visit(inner, in_loop or isinstance(s, _LOOPS), False)
            for h in getattr(s, "handlers", []) or []:
                visit(h.body, in_loop, False)

    visit(body, False, True)
    return found


def _deepest_block(node: ast.AST) -> Optional[ast.AST]:
    """The outermost block statement on the path to the deepest nesting."""
    best = (0, None)

    def depth(n: ast.AST, d: int, top):
        nonlocal best
        for c in ast.iter_child_nodes(n):
            if isinstance(c, _SCOPE):
                continue
            block = isinstance(c, (ast.If, ast.For, ast.AsyncFor, ast.While, ast.With, ast.AsyncWith, ast.Try)) \
                or (hasattr(ast, "Match") and isinstance(c, ast.Match))
            nd = d + 1 if block else d
            t = top if top is not None or not block else c
            if nd > best[0]:
                best = (nd, t)
            depth(c, nd, t)

    depth(node, 0, None)
    return best[1]


def rule_nesting(ctx: Context) -> list:
    """Deep nesting of blocks (pylint's too-many-nested-blocks limit is 5)."""
    out = []
    for u in ctx.units:
        depth = u.metrics.get("maximum_nesting_depth", 0)
        sev = _sev(depth, THRESHOLDS["nesting_medium"], THRESHOLDS["nesting_high"])
        if not sev:
            continue
        body = u.node.body if hasattr(u.node, "body") else []
        guards = _guard_candidates(body)
        ev = [Evidence("maximum_nesting_depth", depth, THRESHOLDS["nesting_medium"] if sev == "medium" else THRESHOLDS["nesting_high"])]
        if guards:
            ev.append(Evidence("guard-clause candidates", [f"line {ln} ({k})" for ln, k in guards[:5]]))
            action = ("Invert the condition at " + ", ".join(f"line {ln}" for ln, _ in guards[:3]) +
                      " into an early return/continue (a guard clause), so the main logic moves one level left.")
        else:
            top = _deepest_block(u.node)
            where = f" (the block starting at line {top.lineno})" if top is not None else ""
            action = f"Extract the innermost logic{where} into a helper function with a descriptive name."
        out.append(Recommendation(
            "HIGH_NESTING_DEPTH", "Reduce nesting with guard clauses or extracted helpers", sev, u.target,
            u.start_line, u.end_line, ev,
            f"Code inside {depth} nested blocks requires keeping every enclosing condition in mind at once.",
            action, "flatten"))
    return out


def rule_length(ctx: Context) -> list:
    """Long functions (code lines)."""
    out = []
    for u in ctx.units:
        lines = u.metrics.get("code_lines", 0)
        sev = _sev(lines, THRESHOLDS["length_medium"], THRESHOLDS["length_high"])
        if not sev:
            continue
        body = u.node.body if hasattr(u.node, "body") else []
        sections = [s for s in body if isinstance(s, (ast.If, ast.For, ast.AsyncFor, ast.While, ast.With, ast.Try))
                    and (s.end_lineno - s.lineno + 1) >= 5]
        ev = [Evidence("code_lines", lines, THRESHOLDS["length_medium"] if sev == "medium" else THRESHOLDS["length_high"],
                       "excluding blank, comment and docstring lines")]
        if len(sections) >= 2:
            ev.append(Evidence("top-level blocks of 5+ lines", [f"lines {s.lineno}-{s.end_lineno}" for s in sections[:6]]))
            action = ("These top-level blocks are natural candidates to become separate functions if they do different "
                      "jobs: " + ", ".join(f"lines {s.lineno}-{s.end_lineno}" for s in sections[:4]) + ".")
        else:
            action = "Identify the distinct steps this code performs and give each its own function."
        out.append(Recommendation(
            "LONG_FUNCTION", "Separate distinct responsibilities", sev, u.target, u.start_line, u.end_line, ev,
            f"{u.target} has {lines} code lines. Long units usually do several things, which makes them harder to "
            "name, read and test.", action, "split"))
    return out


def _chains(node: ast.AST) -> list:
    """if/elif chains in the unit's own code: list of (first If node, arms, compared expression or None)."""
    elif_nodes = set()
    chains = []
    for n in _own_walk(node):
        if isinstance(n, ast.If) and id(n) not in elif_nodes:
            arms, cur, subjects = 1, n, [_compared(n.test)]
            while len(cur.orelse) == 1 and isinstance(cur.orelse[0], ast.If):
                cur = cur.orelse[0]
                elif_nodes.add(id(cur))
                arms += 1
                subjects.append(_compared(cur.test))
            same = subjects[0] if subjects[0] is not None and all(s == subjects[0] for s in subjects) else None
            chains.append((n, arms, same))
    return chains


def _compared(test: ast.AST) -> Optional[str]:
    """`x == const` / `x in (consts)` -> dump of x; else None."""
    if isinstance(test, ast.Compare) and len(test.ops) == 1 and isinstance(test.ops[0], (ast.Eq, ast.In)):
        right = test.comparators[0]
        if isinstance(right, ast.Constant) or (isinstance(right, (ast.Tuple, ast.List, ast.Set))
                                               and all(isinstance(e, ast.Constant) for e in right.elts)):
            return ast.dump(test.left)
    return None


def rule_branches(ctx: Context) -> list:
    """Long if/elif chains or very many branches in one unit."""
    out = []
    for u in ctx.units:
        m = u.metrics
        total = m.get("if_count", 0) + m.get("elif_count", 0) + m.get("conditional_expression_count", 0) + _case_count(u.node)
        chains = [c for c in _chains(u.node) if c[1] >= THRESHOLDS["chain_arms"]]
        if not chains and total < THRESHOLDS["branches_total"]:
            continue
        label = "conditional branches (if + elif + conditional expressions + case)"
        ev = [Evidence(label, total, THRESHOLDS["branches_total"]) if total >= THRESHOLDS["branches_total"]
              else Evidence(label, total, detail="below the total threshold; the long chain triggered this rule")]
        actions = []
        for n, arms, same in chains[:3]:
            ev.append(Evidence("if/elif chain", f"{arms} arms at line {n.lineno}", THRESHOLDS["chain_arms"],
                               "every arm compares the same value with a constant" if same else ""))
        dispatch = [c for c in chains if c[2]]
        if dispatch:
            actions.append(("The chains at " if len(dispatch) > 1 else "The chain at ") +
                           ", ".join(f"line {c[0].lineno}" for c in dispatch[:3]) +
                           (" compare" if len(dispatch) > 1 else " compares") + " one value with constants; a dictionary lookup (or match statement) may express "
                           "this mapping more directly.")
        if len(dispatch) < len(chains):
            actions.append("For the other long chains, check whether conditions overlap, can be ordered more simply, "
                           "or belong to separate cases handled by separate functions.")
        if not chains:
            actions.append("Check whether some conditions repeat or overlap, and whether related decisions can be "
                           "grouped into helper functions.")
        sev = "medium" if chains or total >= 2 * THRESHOLDS["branches_total"] else "low"
        out.append(Recommendation(
            "MANY_CONDITIONAL_BRANCHES", "Review whether the decision logic can be simplified", sev, u.target,
            u.start_line, u.end_line, ev,
            "Many branches make it hard to see which case applies and to check that every case is handled.",
            " ".join(actions), "simplify_conditions",
            related=["generic" if not chains else "chain"]))
    return out


def _case_count(node: ast.AST) -> int:
    match_case = getattr(ast, "match_case", None)
    return sum(1 for n in _own_walk(node) if match_case is not None and isinstance(n, match_case))


def _is_self_call(c: ast.Call, fn: ast.AST, is_method: bool) -> bool:
    """f(...) inside function f; self.f(...) / cls.f(...) inside method f (a bare f(...) in a method
    calls a module-level function, not the method)."""
    f = c.func
    if isinstance(f, ast.Name):
        return not is_method and f.id == fn.name
    return isinstance(f, ast.Attribute) and f.attr == fn.name and isinstance(f.value, ast.Name) \
        and f.value.id in ("self", "cls")


def _self_calls(fn: ast.AST, is_method: bool) -> list:
    return [n for n in _own_walk(fn) if isinstance(n, ast.Call) and _is_self_call(n, fn, is_method)]


def _branch_paths(fn: ast.AST) -> dict:
    """id(node) -> tuple of (id(if-node), arm) for every if/conditional-expression arm enclosing it."""
    paths = {}

    def visit(n: ast.AST, path: tuple):
        paths[id(n)] = path
        for name, value in ast.iter_fields(n):
            children = value if isinstance(value, list) else [value]
            for c in children:
                if not isinstance(c, ast.AST) or isinstance(c, _SCOPE):
                    continue
                arm = name if isinstance(n, (ast.If, ast.IfExp)) and name in ("body", "orelse") else None
                visit(c, path + ((id(n), arm),) if arm else path)

    for stmt in fn.body:
        visit(stmt, ())
    return paths


def _exclusive(a: tuple, b: tuple) -> bool:
    """Two nodes sit in different arms of the same if / conditional expression."""
    arms = dict(a)
    return any(k in arms and arms[k] != v for k, v in b)


def _co_executable(calls: list, fn: ast.AST) -> list:
    """The largest set of calls that can all run in one invocation (pairwise not in exclusive arms)."""
    paths = _branch_paths(fn)
    best = []
    for c in calls:
        group = [d for d in calls if not _exclusive(paths.get(id(c), ()), paths.get(id(d), ()))]
        group = [d for d in group if all(not _exclusive(paths.get(id(d), ()), paths.get(id(e), ())) for e in group)]
        if len(group) > len(best):
            best = group
    return best


def _unconditional_self_calls(fn: ast.AST, is_method: bool) -> list:
    """(statement index, call) for self-calls not inside any conditional, loop, try or short-circuit construct."""
    found = []

    def visit(n: ast.AST, idx: int):
        if isinstance(n, (ast.If, ast.IfExp, ast.While, ast.For, ast.AsyncFor, ast.Try, ast.BoolOp, ast.comprehension,
                          ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp) + _SCOPE) or \
                (hasattr(ast, "Match") and isinstance(n, ast.Match)):
            return
        if isinstance(n, ast.Call) and _is_self_call(n, fn, is_method):
            found.append((idx, n))
        for c in ast.iter_child_nodes(n):
            visit(c, idx)

    for i, stmt in enumerate(fn.body):
        visit(stmt, i)
    return found


def _can_exit(stmt: ast.stmt) -> bool:
    return any(isinstance(n, (ast.Return, ast.Raise)) for n in [stmt, *_own_walk(stmt)])


def _constant_step(call: ast.Call, fn: ast.AST, is_method: bool) -> Optional[str]:
    """Describe the call if it moves by a constant step: every argument is either the same parameter
    unchanged or that parameter +/- a constant (n - 1, i + 1) or sliced by a constant (xs[1:]), and
    at least one argument steps. Then the recursion depth grows linearly with the input.
    Anything else (a midpoint, a child node, a computed value) gives None: the depth is unknown."""
    params = [a.arg for a in fn.args.posonlyargs + fn.args.args]
    if is_method and params:
        params = params[1:]  # self / cls
    if any(isinstance(a, ast.Starred) for a in call.args) or len(call.args) > len(params):
        return None
    pairs = list(zip(params, call.args)) + [(k.arg, k.value) for k in call.keywords]
    steps = []
    for name, a in pairs:
        if name is None:
            return None
        if isinstance(a, ast.Name) and a.id == name:
            continue
        if isinstance(a, ast.BinOp) and isinstance(a.op, (ast.Sub, ast.Add)) and isinstance(a.left, ast.Name) \
                and a.left.id == name and isinstance(a.right, ast.Constant) and isinstance(a.right.value, int) \
                and a.right.value != 0:
            steps.append(ast.unparse(a))
            continue
        if isinstance(a, ast.Subscript) and isinstance(a.value, ast.Name) and a.value.id == name \
                and isinstance(a.slice, ast.Slice) and a.slice.step is None \
                and all(b is None or (isinstance(b, (ast.Constant, ast.UnaryOp)) and isinstance(ast.literal_eval(b), int))
                        for b in (a.slice.lower, a.slice.upper)) and (a.slice.lower or a.slice.upper):
            steps.append(ast.unparse(a))
            continue
        return None
    return ", ".join(steps) if steps else None


def _memoized(fn: ast.AST) -> bool:
    for d in fn.decorator_list:
        target = d.func if isinstance(d, ast.Call) else d
        name = target.attr if isinstance(target, ast.Attribute) else getattr(target, "id", "")
        if name in _MEMO_DECORATORS:
            return True
    params = {a.arg for a in fn.args.args + fn.args.kwonlyargs}
    names = {n.id for n in _own_walk(fn) if isinstance(n, ast.Name)}
    names |= {n.attr for n in _own_walk(fn) if isinstance(n, ast.Attribute)}
    names = {x.lower().lstrip("_") for x in params | names}
    return any(x in _MEMO_NAMES or "memo" in x or "cache" in x for x in names)


def rule_recursion(ctx: Context) -> list:
    """Recursion risks with concrete evidence; recursion by itself is never flagged.

    1. A recursive call runs unconditionally and no earlier statement can return or raise:
       no visible termination condition (high).
    2. Branching recursion: 2+ self-calls that can run in the same invocation (not in different
       arms of one if), each moving by a constant step (n - 1, col + 1), with no memoization
       (no lru_cache/cache decorator, no memo/cache/dp table): the same subproblems may be
       recomputed exponentially often (medium).
    3. Recursion depth grows linearly with the input (every argument unchanged or stepped by a
       constant, e.g. n - 1 or xs[1:]): CPython's default limit is 1000 frames, so large inputs
       raise RecursionError (low). Midpoints, child nodes or computed arguments are not flagged
       because the depth cannot be read from the code.
    """
    out = []
    file_sets_limit = any(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "setrecursionlimit"
                          for n in ast.walk(ctx.parsed.tree))
    for u in ctx.units:
        if not u.is_function:
            continue
        fn = u.node
        is_method = u.kind == "method"
        calls = _self_calls(fn, is_method)
        if not calls:
            continue
        uncond = _unconditional_self_calls(fn, is_method)
        no_exit = [(i, c) for i, c in uncond if not any(_can_exit(s) for s in fn.body[:i])]
        if no_exit:
            i, c = no_exit[0]
            out.append(Recommendation(
                "RECURSION_TERMINATION", "Check the recursion's termination condition", "high", u.target,
                u.start_line, u.end_line,
                [Evidence("recursive calls", len(calls)),
                 Evidence("unconditional recursive call", f"line {c.lineno}: {_short(ast.unparse(c))}",
                          detail="no earlier statement in the function can return or raise")],
                "Every call reaches this recursive call before any visible way to stop, so unless the stop is "
                "hidden elsewhere (for example an exception raised deeper), the recursion cannot end.",
                f"Make sure a base case is checked before line {c.lineno} (for example `if <smallest input>: return ...`).",
                "check_recursion"))
            continue
        shrinking = [(c, _constant_step(c, fn, is_method)) for c in calls]
        shrinking = [(c, d) for c, d in shrinking if d and not _divides(c)]
        together = _co_executable([c for c, _ in shrinking], fn)
        if len(together) >= 2 and not _memoized(fn):
            shrinking = [(c, d) for c, d in shrinking if c in together]
            out.append(Recommendation(
                "RECURSION_REPEATED_WORK", "Check for repeated subproblems in branching recursion", "medium", u.target,
                u.start_line, u.end_line,
                [Evidence("recursive calls that can run in the same invocation", len(together), 2),
                 Evidence("arguments shrink by a constant", [f"line {c.lineno}: {d}" for c, d in shrinking[:3]]),
                 Evidence("memoization", "none found (no lru_cache/cache decorator or memo table)")],
                "Each call makes several recursive calls on slightly smaller inputs; such recursions often solve the "
                "same subproblem many times, which can take exponential time.",
                "Check whether calls repeat the same arguments; if they do, cache results (functools.lru_cache or a "
                "dictionary) or rewrite the recursion as a bottom-up loop.",
                "check_recursion"))
        elif shrinking:
            c, d = shrinking[0]
            note = " The file calls sys.setrecursionlimit, which raises the limit but can crash the interpreter " \
                   "if the C stack runs out." if file_sets_limit else ""
            out.append(Recommendation(
                "RECURSION_DEPTH", "Check maximum recursion depth", "low", u.target, u.start_line, u.end_line,
                [Evidence("recursive argument", f"line {c.lineno}: {d}", detail="depth grows by one per element/step"),
                 Evidence("CPython default recursion limit", THRESHOLDS["recursion_limit"])],
                "Each call goes one step deeper, so the depth equals the input size. Inputs larger than about "
                f"{THRESHOLDS['recursion_limit']} would raise RecursionError.{note}",
                "Confirm the maximum input size; if it can be large, consider an iterative version (a loop or an "
                "explicit stack).",
                "check_recursion"))
    return out


def _renamed(ctx: Context, nodes: list) -> Optional[tuple]:
    text = "\n".join(textwrap.dedent(_segment(ctx, n)) for n in nodes)
    toks = code_tokens(textwrap.dedent(text) + "\n", rename=True)
    return tuple(toks) if toks else None


def _statement_lists(tree: ast.AST) -> Iterable[list]:
    for n in ast.walk(tree):
        for name in ("body", "orelse", "finalbody"):
            v = getattr(n, name, None)
            if isinstance(v, list) and v and isinstance(v[0], ast.stmt):
                yield v
        for h in getattr(n, "handlers", []) or []:
            yield h.body


def _is_docstring(s: ast.stmt) -> bool:
    return isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant) and isinstance(s.value.value, str)


def rule_duplication(ctx: Context) -> list:
    """Near-duplicate functions and repeated statement blocks (copy-paste-like repetition)."""
    out = []
    # 1. Near-duplicate functions
    def code_body(u):
        return [s for s in u.node.body if not _is_docstring(s)]

    fun_units = [u for u in ctx.units if u.is_function and code_body(u)
                 and code_body(u)[-1].end_lineno - code_body(u)[0].lineno + 1 >= THRESHOLDS["duplicate_function_min_lines"]]
    sh = {}
    for u in fun_units:
        toks = _renamed(ctx, code_body(u))
        if toks and len(toks) >= THRESHOLDS["duplicate_block_min_tokens"]:
            sh[u.target] = shingles(list(toks))
    paired_lines = []
    for i, a in enumerate(fun_units):
        for b in fun_units[i + 1:]:
            if a.target not in sh or b.target not in sh:
                continue
            sim = jaccard(sh[a.target], sh[b.target])
            if sim >= THRESHOLDS["duplicate_function_similarity"]:
                paired_lines += [(a.start_line, a.end_line), (b.start_line, b.end_line)]
                out.append(Recommendation(
                    "DUPLICATED_STRUCTURE", "Review near-duplicate functions", "medium", f"{a.target}, {b.target}",
                    min(a.start_line, b.start_line), max(a.end_line, b.end_line),
                    [Evidence("token similarity (identifiers and literals ignored)", round(sim, 3),
                              THRESHOLDS["duplicate_function_similarity"]),
                     Evidence("locations", [f"{a.target}: lines {a.start_line}-{a.end_line}",
                                            f"{b.target}: lines {b.start_line}-{b.end_line}"])],
                    "The two functions have almost the same structure with different names or values, which often "
                    "means a fix in one has to be repeated in the other.",
                    "If they implement the same idea, keep one function and pass the differing names or values as "
                    "parameters.", "deduplicate"))

    # 2. Repeated statement blocks (maximal windows first; occurrences must not overlap)
    covered: set = set()
    groups: dict = {}
    lists = list(_statement_lists(ctx.parsed.tree))
    max_len = max((len(s) for s in lists), default=0)
    for size in range(min(max_len, 12), THRESHOLDS["duplicate_block_min_statements"] - 1, -1):
        found = defaultdict(list)
        for stmts in lists:
            for i in range(len(stmts) - size + 1):
                win = stmts[i:i + size]
                if any(id(s) in covered for s in win) or any(isinstance(s, _SCOPE) or _is_docstring(s) for s in win):
                    continue
                first, last = win[0].lineno, win[-1].end_lineno
                if last - first + 1 < THRESHOLDS["duplicate_block_min_lines"]:
                    continue
                if not any(isinstance(n, ast.Call) or isinstance(n, (ast.If, ast.For, ast.While)) for s in win for n in ast.walk(s)):
                    continue
                key = _renamed(ctx, win)
                if key and len(key) >= THRESHOLDS["duplicate_block_min_tokens"]:
                    found[key].append(win)
        for key, wins in found.items():
            chosen = []
            for w in wins:  # drop overlapping occurrences
                if not any(id(s) in covered for s in w):
                    chosen.append(w)
                    covered.update(id(s) for s in w)
            if len(chosen) >= 2:
                groups[key] = chosen
            else:
                for w in chosen:
                    covered.difference_update(id(s) for s in w)
    for wins in groups.values():
        spans = [(w[0].lineno, w[-1].end_lineno) for w in wins]
        if all(any(a <= s and e <= b for a, b in paired_lines) for s, e in spans):
            continue  # already reported as near-duplicate functions
        lines_each = spans[0][1] - spans[0][0] + 1
        sev = "medium" if len(wins) >= 3 or lines_each >= 10 else "low"
        out.append(Recommendation(
            "DUPLICATED_STRUCTURE", "Review repeated code blocks", sev, "<file>", spans[0][0], spans[-1][1],
            [Evidence("identical blocks (after renaming identifiers and literals)", len(wins), 2),
             Evidence("statements per block", len(wins[0]), THRESHOLDS["duplicate_block_min_statements"]),
             Evidence("locations", [f"lines {s}-{e}" for s, e in spans])],
            "The same sequence of statements appears more than once, differing at most in names or values.",
            "If the repetition is intentional logic rather than coincidence, move it into one helper function "
            "(or a loop over the differing values) and call it from each place.", "deduplicate"))
    return out


RULES: tuple = (
    Rule("HIGH_CYCLOMATIC_COMPLEXITY", "High cyclomatic complexity", rule_cyclomatic,
         f"cyclomatic complexity >= {THRESHOLDS['cyclomatic_medium']} (medium), >= {THRESHOLDS['cyclomatic_high']} (high)"),
    Rule("HIGH_NESTING_DEPTH", "Deep nesting", rule_nesting,
         f"maximum nesting depth >= {THRESHOLDS['nesting_medium']} (medium), >= {THRESHOLDS['nesting_high']} (high)"),
    Rule("LONG_FUNCTION", "Long function", rule_length,
         f"code lines >= {THRESHOLDS['length_medium']} (medium), >= {THRESHOLDS['length_high']} (high)"),
    Rule("MANY_CONDITIONAL_BRANCHES", "Many conditional branches", rule_branches,
         f"an if/elif chain with >= {THRESHOLDS['chain_arms']} arms, or >= {THRESHOLDS['branches_total']} branches"),
    Rule("RECURSION", "Recursion risks", rule_recursion,
         "unconditional recursive call without prior exit; branching recursion on shrinking arguments without "
         "memoization; depth linear in input size"),
    Rule("DUPLICATED_STRUCTURE", "Repeated structural patterns", rule_duplication,
         f"function similarity >= {THRESHOLDS['duplicate_function_similarity']}; repeated blocks of "
         f">= {THRESHOLDS['duplicate_block_min_statements']} statements / {THRESHOLDS['duplicate_block_min_lines']} lines"),
)


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def build_context(parsed: ParseResult) -> Context:
    feats = extract_features(parsed)
    analysis = analyze_functions(parsed)
    rows = {r["lineno"]: r for r in feats.function_features}
    nodes = {n.lineno: n for n in ast.walk(parsed.tree) if isinstance(n, _FUNC)}
    units = []
    for rec in analysis.functions:
        node = nodes.get(rec.start_line)
        if node is None:
            continue
        units.append(Unit(rec.function_id, node, rows.get(rec.start_line, {}), rec.start_line, rec.end_line, True, rec.kind))
    if not units:  # a script: analyse the module body as one unit
        ff = feats.file_features or {}
        units.append(Unit("<module>", parsed.tree, ff, 1, max(parsed.line_count, 1), False, "module"))
    return Context(parsed, units, parsed.functions, parsed.source.splitlines())


def _resolve(recs: list) -> list:
    """Deterministic overlap handling: one recommendation per (rule, target); fold redundant advice."""
    by_target = defaultdict(dict)
    for r in recs:
        by_target[r.target].setdefault(r.rule_id, r)  # first wins; rules emit at most one per target
    out = []
    for target, rules in by_target.items():
        cc = rules.get("HIGH_CYCLOMATIC_COMPLEXITY")
        br = rules.get("MANY_CONDITIONAL_BRANCHES")
        if cc and br and br.related == ["generic"]:
            # the generic many-branches advice repeats the complexity advice: fold its evidence in
            cc.evidence.extend(br.evidence)
            cc.related.append("MANY_CONDITIONAL_BRANCHES folded in (same advice)")
            del rules["MANY_CONDITIONAL_BRANCHES"]
        elif br:
            br.related = []
        out.extend(rules.values())
    out.sort(key=lambda r: (SEVERITY_ORDER[r.severity], r.start_line, r.rule_id, r.target))
    return out


def suggest(source: Union[str, bytes, ParseResult], rules: Iterable[Rule] = RULES) -> SuggestionReport:
    """Run every rule on the code (never executed) and return review guidance."""
    if isinstance(source, ParseResult):
        parsed = source
    elif isinstance(source, (bytes, bytearray)):
        parsed = parse_bytes(bytes(source))
    elif isinstance(source, str):
        parsed = parse_source(source)
    else:
        return SuggestionReport(False, error={"code": "invalid_input_type",
                                              "message": f"Expected source code as text, got {type(source).__name__}."})
    if not parsed.success:
        e = parsed.error
        return SuggestionReport(False, error={"code": e.kind, "message": e.message, "hint": e.hint})
    rules = list(rules)
    ctx = build_context(parsed)
    recs = []
    for rule in rules:
        recs.extend(rule.check(ctx))
    report = SuggestionReport(True, _resolve(recs), [r.rule_id for r in rules],
                              sum(1 for u in ctx.units if u.is_function))
    if not any(u.is_function for u in ctx.units):
        report.warnings.append("No functions found: the module-level code was analysed as one unit.")
    return report


def format_text(report: SuggestionReport) -> str:
    if not report.success:
        return f"Could not analyse the code: {report.error['message']}"
    if not report.recommendations:
        return "No recommendations: no rule found evidence above its threshold.\n" + report.disclaimer
    lines = []
    for r in report.recommendations:
        lines.append(f"[{r.severity.upper()}] {r.rule_id} - {r.title} ({r.target}, lines {r.start_line}-{r.end_line})")
        for e in r.evidence:
            lines.append(f"    evidence: {e.metric} = {e.value}" + (f" (threshold {e.threshold})" if e.threshold is not None else ""))
        lines.append(f"    why: {r.explanation}")
        lines.append(f"    suggestion: {r.suggested_action}")
    lines.append(report.disclaimer)
    return "\n".join(lines)
