"""Rule-based estimation of time and auxiliary-space complexity (static only).

Nothing is executed. Each function's AST is walked once, building a
symbolic cost with ``src.asymptotics``. The result is either an estimate
with an explanation, assumptions and a confidence level, or an explicit
"Unknown" with the reason. Whenever the evidence is insufficient, the
answer is Unknown rather than a guess.

Public API
----------
analyze_algorithms(source_or_parse_result) -> AlgorithmAnalysis
    .estimates: list[ComplexityEstimate], one per function (same ids as
    src.function_analyzer). ComplexityEstimate.to_dict() gives the schema:

    {"function_id", "function_name", "time_complexity", "space_complexity",
     "time_class", "space_class", "status", "time_status", "space_status",
     "time_confidence", "space_confidence", "explanation", "assumptions",
     "evidence", "size_variables", "recurrence"}

Size variables
--------------
Costs are expressed in the function's own input sizes: ``n = len(arr)``,
``n = value of n``, ``m = len(other)``, ``k = len of each item in grid``.
Different inputs keep different letters, so a loop over ``a`` nested in a
loop over ``b`` is ``O(n·m)``, not ``O(n^2)``. ``time_class`` /
``space_class`` give the single-variable form obtained by assuming every
input size is proportional to one n (``n·m -> n^2``), for comparison with
labelled datasets. That collapse is itself an assumption.

Supported subset
----------------
* ``for`` over ``range(...)`` (1–3 arguments, constant step), over a
  parameter or local collection, ``enumerate``/``zip``/``reversed``/
  ``sorted``/``.items()``/``.keys()``/``.values()``, or a constant literal.
  ``range(i, n)`` or ``range(i)`` inside a loop over ``i`` (triangular
  loops) is counted as n per outer iteration, which is still Θ(n^2) in total.
* ``while`` loops matching one of these patterns:
  - a counter changed by a constant (``i += 1``, ``n -= 1``) on every
    iteration → linear in the bound (``i * i <= n`` gives √n);
  - a counter multiplied or divided by a constant (``n //= 2``,
    ``i *= 2``, ``n >>= 1``) on every iteration → logarithmic;
  - binary search: ``while lo <= hi`` with ``mid = (lo + hi) // 2`` and
    ``lo = mid ± 1`` / ``hi = mid ± 1`` → logarithmic;
  - two pointers: ``while i < len(a) and j < len(b)`` where every path
    advances at least one pointer → ``len(a) + len(b)``.
  Anything else (``while True``, ``while queue``, conditional-only
  updates, data-dependent conditions) → Unknown.
* Recursion (direct self-calls only):
  - every recursive call must shrink the same argument the same way:
    subtracting a constant (``n - 1``, ``arr[1:]``) or dividing by a
    constant (``n // 2``, ``arr[:mid]`` with ``mid = len(arr) // 2``,
    ``(lo, mid - 1)`` with ``mid = (lo + hi) // 2``);
  - a path without recursive calls (base case) must exist;
  - a = the maximum number of recursive calls on one execution path
    (calls in different if/else branches are alternatives, not added);
  - T(n) = a·T(n - c) + f(n): a = 1 → n·f(n); a ≥ 2 → O(a^n) (an upper
    bound; Fibonacci-style n-1/n-2 recursion is Θ(φ^n), φ ≈ 1.618);
  - T(n) = a·T(n / b) + f(n) with f(n) = n^k log^j n → master theorem;
  - a recursive call inside a loop over the shrinking input, shrinking it
    by one (permutation-style backtracking) → O(n!) times other factors;
  - ``@lru_cache`` / ``@cache`` or an explicit ``if n in memo: return
    memo[n]`` with a single shrinking argument → (number of distinct
    arguments) × (work per call).
  Data-dependent splits (quicksort partitions), several shrinking
  arguments (Euclid's gcd, Ackermann), mutual recursion and recursion
  that does not visibly shrink → Unknown.
* Calls to other functions in the same file are analysed and their cost
  substituted with the caller's argument sizes.

Cost of built-in operations (CPython, worst case unless stated)
-----------------------------------------------------------------
O(1): arithmetic on numbers, comparisons of numbers, ``len``, indexing,
``append``, ``pop()``, ``dict``/``set`` lookup, insertion and ``in``
(average case, assuming well-behaved hashing), ``deque`` operations at
either end.
O(n): ``x in list/tuple/str``, ``list.index``/``count``/``remove``/
``insert``/``pop(i)``, ``sum``/``min``/``max``/``any``/``all`` over a
collection, ``list()``/``set()``/``dict()``/``tuple()`` copies, slicing
(proportional to the slice; taken as the whole length unless both bounds
are constants), concatenation, ``str.join``, ``copy``.
O(n log n): ``sorted``, ``list.sort`` (Timsort; auxiliary space O(n)).
O(log n): ``heapq.heappush``/``heappop``, ``bisect.bisect_*``.
Membership on a parameter of unknown type is assumed to be a linear scan
(it would be O(1) for a set or dict). Type annotations are used when present.
Calls to functions that are neither built-in nor defined in the file are
assumed O(1); this is recorded as a low-confidence assumption.

Auxiliary space
---------------
Auxiliary space is memory the function allocates beyond its inputs:
local collections, copies, slices, comprehensions, the result it builds,
and the call stack of recursion. The inputs themselves are not counted
(total memory = input + auxiliary). An allocation inside a loop that is
rebound on every iteration counts once (it is a temporary); appends and
insertions into a collection inside a loop accumulate over the
iterations. Recursion counts the stack depth plus the memory alive along
one chain of calls.

Limitations
-----------
Estimates are worst-case upper bounds up to constants, derived only from
syntax. The analyzer does not know types, does not track values through
data structures, ignores early exits that would make typical cases faster,
and cannot see the cost of functions defined elsewhere. "Unknown" means
the evidence was insufficient, not that the code is wrong.
"""

from __future__ import annotations

import ast
import math
from dataclasses import asdict, dataclass, field
from fractions import Fraction
from typing import Optional, Union

from src.asymptotics import Cost, Mono
from src.code_parser import FunctionInfo, ParseError, ParseResult, parse_source
from src.function_analyzer import analyze_functions

__all__ = ["AlgorithmAnalysis", "ComplexityEstimate", "analyze_algorithms"]

TIME_CLASSES = ("O(1)", "O(log n)", "O(n)", "O(n log n)", "O(n^2)", "O(n^3)", "O(2^n)", "O(n!)")
SPACE_CLASSES = ("O(1)", "O(log n)", "O(n)", "O(n log n)", "O(n^2)")

_LEVELS = {"standard": 0, "medium": 1, "low": 2}
_CONFIDENCE = {0: "high", 1: "medium", 2: "low"}


# --------------------------------------------------------------------------- #
# Public result types
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ComplexityEstimate:
    function_id: str
    function_name: str
    qualname: str
    time_complexity: str  # e.g. "O(n·m)" or "Unknown"
    space_complexity: str
    time_class: str  # single-variable form, e.g. "O(n^2)" or "Unknown"
    space_class: str
    status: str  # "estimated" | "partial" | "unknown"
    time_status: str  # "estimated" | "unknown"
    space_status: str
    time_confidence: Optional[str]  # "high" | "medium" | "low"; None if unknown
    space_confidence: Optional[str]
    explanation: str
    assumptions: tuple[str, ...]
    evidence: tuple[str, ...]
    size_variables: dict = field(default_factory=dict)  # letter -> meaning
    recurrence: Optional[str] = None

    def to_dict(self) -> dict:
        d = asdict(self)
        d["assumptions"] = list(self.assumptions)
        d["evidence"] = list(self.evidence)
        return d


@dataclass
class AlgorithmAnalysis:
    success: bool
    estimates: list[ComplexityEstimate] = field(default_factory=list)
    error: Optional[ParseError] = None

    def get(self, function_id: str) -> Optional[ComplexityEstimate]:
        return next((e for e in self.estimates if e.function_id == function_id), None)


def analyze_algorithms(source: Union[str, ParseResult]) -> AlgorithmAnalysis:
    """Estimate time and auxiliary space for every function in ``source``."""
    parsed = source if isinstance(source, ParseResult) else parse_source(source)
    if not parsed.success:
        return AlgorithmAnalysis(success=False, error=parsed.error)
    functions = analyze_functions(parsed)
    module = _Module(parsed)
    estimates = []
    for record in functions.functions:
        info = module.info_by_line[record.start_line]
        estimates.append(module.estimate(info, record.function_id))
    return AlgorithmAnalysis(success=True, estimates=estimates)


# --------------------------------------------------------------------------- #
# Module-level bookkeeping (function lookup, caching, inter-procedural calls)
# --------------------------------------------------------------------------- #
@dataclass
class _FnResult:
    info: FunctionInfo
    time: Cost
    space: Cost
    time_reasons: list[str]
    space_reasons: list[str]
    assumptions: list[tuple[str, str]]
    evidence: list[str]
    sym_desc: dict[str, str]
    sym_order: list[str]
    recurrence: Optional[str]
    param_syms: dict[str, str]  # param name -> symbol


class _Module:
    def __init__(self, parsed: ParseResult):
        self.parsed = parsed
        self.info_by_line = {f.lineno: f for f in parsed.functions}
        self.nodes = {
            n.lineno: n
            for n in ast.walk(parsed.tree)
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        self.cache: dict[int, _FnResult] = {}
        self.in_progress: set[int] = set()

    # Lookup ------------------------------------------------------------ #
    def resolve_name(self, caller: FunctionInfo, name: str) -> Optional[FunctionInfo]:
        """Find the function a bare-name call refers to (innermost scope first)."""
        scope = caller.qualname
        while True:
            matches = [f for f in self.parsed.functions if f.parent == scope and f.name == name and f.kind != "method"]
            if len(matches) == 1:
                return matches[0]
            if len(matches) > 1:
                return None  # ambiguous (redefinitions)
            if scope is None:
                return None
            scope = scope.rsplit(".", 1)[0] if "." in scope else None

    def resolve_method(self, class_name: Optional[str], name: str) -> Optional[FunctionInfo]:
        if not class_name:
            return None
        matches = [f for f in self.parsed.functions if f.class_name == class_name and f.name == name]
        return matches[0] if len(matches) == 1 else None

    # Analysis ------------------------------------------------------------ #
    def result(self, info: FunctionInfo) -> Optional[_FnResult]:
        """Analyse a function once; None if it is already being analysed."""
        if info.lineno in self.cache:
            return self.cache[info.lineno]
        if info.lineno in self.in_progress:
            return None  # mutual recursion
        self.in_progress.add(info.lineno)
        try:
            res = _FunctionAnalyzer(self, info, self.nodes[info.lineno]).run()
        except (RecursionError, ValueError, KeyError, AttributeError, TypeError, ZeroDivisionError) as exc:
            res = _FnResult(
                info, Cost.unknown(), Cost.unknown(),
                [f"internal analysis limitation ({type(exc).__name__})"],
                [f"internal analysis limitation ({type(exc).__name__})"],
                [], [], {}, [], None, {},
            )
        finally:
            self.in_progress.discard(info.lineno)
        self.cache[info.lineno] = res
        return res

    def estimate(self, info: FunctionInfo, function_id: str) -> ComplexityEstimate:
        res = self.result(info)
        return _to_estimate(res, function_id)


# --------------------------------------------------------------------------- #
# Per-function analysis
# --------------------------------------------------------------------------- #
_CONST_BUILTINS = {
    "len", "abs", "int", "float", "bool", "chr", "ord", "hash", "id", "isinstance",
    "issubclass", "type", "range", "enumerate", "zip", "iter", "next", "divmod",
    "pow", "round", "print", "callable", "getattr", "setattr", "hasattr", "repr",
    "str", "bin", "hex", "oct", "reversed", "object", "super", "map", "filter",
}
_LINEAR_BUILTINS = {"sum", "any", "all", "list", "tuple", "set", "frozenset", "dict", "bytes", "bytearray"}
_ALLOCATING_BUILTINS = {"list", "tuple", "set", "frozenset", "dict", "sorted", "bytes", "bytearray"}

_CONST_METHODS = {
    "append", "add", "get", "discard", "keys", "values", "items", "setdefault",
    "appendleft", "popleft", "isdigit", "isalpha", "isalnum", "isspace",
    "isupper", "islower", "__len__",
}
_RECEIVER_LINEAR_METHODS = {
    "index", "count", "insert", "copy", "reverse", "clear", "find", "rfind", "split",
    "strip", "lstrip", "rstrip", "lower", "upper", "replace", "startswith", "endswith",
    "title", "capitalize", "swapcase", "splitlines",
}
_ARG_LINEAR_METHODS = {"extend", "update", "join", "union", "intersection", "difference", "issubset"}
_ALLOCATING_METHODS = {"copy", "split", "join", "union", "intersection", "difference", "splitlines",
                       "lower", "upper", "replace", "strip", "lstrip", "rstrip"}
_GROW_METHODS = {"append": "one", "add": "one", "appendleft": "one", "insert": "one",
                 "extend": "arg", "update": "arg", "setdefault": "one"}

_HASHED_KINDS = {"set", "dict", "frozenset", "Counter", "defaultdict", "OrderedDict"}
_SEQUENCE_KINDS = {"list", "tuple", "str", "deque", "bytes"}
_ANNOTATION_KINDS = {
    "set": "set", "Set": "set", "frozenset": "set", "FrozenSet": "set", "AbstractSet": "set",
    "dict": "dict", "Dict": "dict", "Mapping": "dict", "MutableMapping": "dict",
    "list": "list", "List": "list", "Sequence": "list", "tuple": "tuple", "Tuple": "tuple",
    "str": "str", "int": "int", "float": "float",
}
_MEMO_DECORATORS = {"lru_cache", "cache", "memoize", "memoized"}


class _FunctionAnalyzer:
    def __init__(self, module: _Module, info: FunctionInfo, node: ast.AST):
        self.module = module
        self.info = info
        self.node = node
        self.q = info.qualname
        self.is_method = info.kind == "method"
        params = [a.arg for a in (*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs)]
        if node.args.vararg:
            params.append(node.args.vararg.arg)
        if node.args.kwarg:
            params.append(node.args.kwarg.arg)
        self.all_params = params
        is_static = any(d.split(".")[-1] == "staticmethod" for d in info.decorators)
        self.receiver = params[0] if self.is_method and params and not is_static else None
        self.params = [p for p in params if p != self.receiver]

        self.sym_desc: dict[str, str] = {}
        self.sym_order: list[str] = []
        self.scalar: dict[str, Optional[Mono]] = {}
        self.coll: dict[str, Optional[Mono]] = {}
        self.kind: dict[str, str] = {}
        self.elem: dict[str, Mono] = {}  # loop variable bound to an item of a parameter
        self.loop_index: set[str] = set()
        self.mids: dict[str, tuple[str, str]] = {}  # mid -> (lo, hi)
        self.halves: dict[str, tuple[str, int]] = {}  # m -> (collection, divisor)
        self.reduced_alias: dict[str, ast.AST] = {}  # local name -> expression it was built from
        self.rec_results: set[str] = set()

        self.mult: Cost = Cost.const()  # product of enclosing loop iteration counts
        self.loop_depth = 0
        self.rec_calls: list[tuple[ast.Call, bool, int]] = []

        self.space_terms: list[Cost] = []
        self.time_reasons: list[str] = []
        self.space_reasons: list[str] = []
        self.assumptions: list[tuple[str, str]] = []
        self.evidence: list[str] = []

        for a in (*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs):
            if a.annotation is not None:
                k = _annotation_kind(a.annotation)
                if k:
                    self.kind[a.arg] = k

    # ------------------------------------------------------------------ #
    # Driver
    # ------------------------------------------------------------------ #
    def run(self) -> _FnResult:
        body = list(self.node.body)
        if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
            body = body[1:]  # docstring
        time = self._block(body)
        space = Cost.total(self.space_terms)
        recurrence = None

        if self.rec_calls:
            time, space, recurrence = self._solve_recursion(time, space)

        if self.time_reasons:
            time = Cost.unknown()
        if self.space_reasons:
            space = Cost.unknown()
        return _FnResult(
            self.info, time, space, self.time_reasons, self.space_reasons,
            self.assumptions, self.evidence, self.sym_desc, self.sym_order,
            recurrence, {p: self._sym_name(p) for p in self.params},
        )

    # ------------------------------------------------------------------ #
    # Bookkeeping helpers
    # ------------------------------------------------------------------ #
    def _assume(self, text: str, level: str = "standard") -> None:
        if all(text != t for t, _ in self.assumptions):
            self.assumptions.append((text, level))

    def _unknown_time(self, reason: str, first: bool = False) -> Cost:
        if reason not in self.time_reasons:
            self.time_reasons.insert(0, reason) if first else self.time_reasons.append(reason)
        return Cost.unknown()

    def _unknown_space(self, reason: str, first: bool = False) -> None:
        if reason not in self.space_reasons:
            self.space_reasons.insert(0, reason) if first else self.space_reasons.append(reason)

    def _sym_name(self, param: str) -> str:
        return f"{self.q}|{param}"

    def _param_sym(self, param: str, usage: str) -> Mono:
        sym = self._sym_name(param)
        if sym not in self.sym_desc:
            self.sym_desc[sym] = f"value of {param}" if usage == "value" else f"len({param})"
            self.sym_order.append(sym)
        return Mono.sym(sym)

    def _elem_sym(self, param: str) -> Mono:
        sym = f"{self.q}|{param}[*]"
        if sym not in self.sym_desc:
            self.sym_desc[sym] = f"len of each item in {param}"
            self.sym_order.append(sym)
        return Mono.sym(sym)

    def _interval_sym(self, lo: str, hi: str) -> Mono:
        sym = f"{self.q}|{lo}..{hi}"
        if sym not in self.sym_desc:
            self.sym_desc[sym] = f"{hi} - {lo}"
            self.sym_order.append(sym)
        return Mono.sym(sym)

    def _alloc(self, size: Optional[Mono], what: str, line: int) -> None:
        """A temporary allocation (counted once, not per loop iteration)."""
        if size is None:
            self._unknown_space(f"line {line}: size of {what} is not statically known")
        elif not size.is_const:
            self.space_terms.append(Cost.of(size))

    def _grow(self, target: str, per_op: Optional[Mono], line: int) -> None:
        """A collection that grows by ``per_op`` on every enclosing iteration."""
        if per_op is None:
            self._unknown_space(f"line {line}: size of items added to {target} is not known")
            return
        if self.mult.is_unknown:
            self._unknown_space(f"line {line}: {target} grows inside a loop with an unknown bound")
            return
        growth = self.mult.times(per_op)
        self.space_terms.append(growth)
        if growth.terms and len(growth.terms) == 1:
            (m,) = growth.terms
            prev = self.coll.get(target)
            self.coll[target] = m if prev is None or prev.dominated_by(m) else prev
        if not (len(growth.terms) == 1 and next(iter(growth.terms)).is_const):
            self.evidence.append(f"line {line}: {target} grows to {growth.render(self._names())}")

    def _names(self) -> dict[str, str]:
        return _letters(self.sym_order, self.sym_desc)

    # ------------------------------------------------------------------ #
    # Sizes of expressions
    # ------------------------------------------------------------------ #
    def size_of(self, e: ast.AST) -> Optional[Mono]:
        """Magnitude of an integer-valued expression (e.g. a loop bound)."""
        if isinstance(e, ast.Constant):
            return Mono.const()
        if isinstance(e, ast.Name):
            if e.id in self.scalar:
                return self.scalar[e.id]
            if e.id in self.coll:
                return self.coll[e.id]
            if e.id in self.elem:
                return None  # the *value* of an item is not a size
            if e.id in self.params:
                return self._param_sym(e.id, "value")
            return None
        if isinstance(e, ast.Call) and isinstance(e.func, ast.Name):
            fn = e.func.id
            if fn == "len" and e.args:
                return self.coll_size(e.args[0])
            if fn in ("int", "abs", "round") and e.args:
                return self.size_of(e.args[0])
            if fn in ("min", "max") and len(e.args) >= 2:
                sizes = [self.size_of(a) for a in e.args]
                if any(s is None for s in sizes):
                    return None
                return _max_mono(sizes)
            if fn == "isqrt" or (fn == "sqrt"):
                s = self.size_of(e.args[0]) if e.args else None
                return _pow(s, Fraction(1, 2)) if s is not None else None
        if isinstance(e, ast.UnaryOp) and isinstance(e.op, (ast.USub, ast.UAdd)):
            return self.size_of(e.operand)
        if isinstance(e, ast.BinOp):
            a, b = self.size_of(e.left), self.size_of(e.right)
            if isinstance(e.op, (ast.Add, ast.Sub)):
                if a is None or b is None:
                    return None
                return _max_mono([a, b])
            if isinstance(e.op, ast.Mult):
                return None if a is None or b is None else a * b
            if isinstance(e.op, (ast.FloorDiv, ast.Div, ast.RShift)) and _is_const(e.right):
                return a
            if isinstance(e.op, ast.Mod):
                return b
            if isinstance(e.op, ast.Pow) and _const_int(e.right) and a is not None:
                return _pow(a, Fraction(_const_int(e.right)))
        return None

    def coll_size(self, e: ast.AST) -> Optional[Mono]:
        """Number of items in a collection-valued expression."""
        if isinstance(e, ast.Constant):
            return Mono.const()
        if isinstance(e, ast.Name):
            if e.id in self.coll:
                return self.coll[e.id]
            if e.id in self.elem:
                return self.elem[e.id]
            if e.id in self.params:
                return self._param_sym(e.id, "len")
            if e.id in self.scalar:
                return self.scalar[e.id]
            return None
        if isinstance(e, (ast.List, ast.Tuple, ast.Set)):
            return None if any(isinstance(x, ast.Starred) for x in e.elts) else Mono.const()
        if isinstance(e, ast.Dict):
            return None if any(k is None for k in e.keys) else Mono.const()
        if isinstance(e, ast.Call):
            if self._is_recursive_call(e):
                arg = self._first_shrinking_arg(e)
                size = self.coll_size(arg) if arg is not None else None
                if size is not None:
                    self._assume("A recursive call is assumed to return a collection the same size as its input (as in sorting).", "medium")
                return size
            fn = _call_name(e)
            if fn == "range":
                bound, _ = self._range_bound(e)
                return bound
            if fn in ("sorted", "list", "tuple", "set", "frozenset", "reversed", "enumerate",
                      "iter", "deque", "Counter", "dict", "bytes", "bytearray") and e.args:
                return self.coll_size(e.args[0])
            if fn == "zip" and e.args:
                sizes = [self.coll_size(a) for a in e.args]
                if any(s is None for s in sizes):
                    return None
                best = _max_mono(sizes)
                if best is None:
                    self._assume("zip() over inputs of different sizes is bounded by its first argument.", "medium")
                    return sizes[0]
                return best
            if isinstance(e.func, ast.Attribute) and e.func.attr in ("keys", "values", "items", "copy"):
                return self.coll_size(e.func.value)
            if isinstance(e.func, ast.Attribute) and e.func.attr == "split":
                return self.coll_size(e.func.value)
        if isinstance(e, ast.Subscript):
            if isinstance(e.slice, ast.Slice):
                lo, hi = e.slice.lower, e.slice.upper
                if (lo is None or _is_const(lo)) and hi is not None and _is_const(hi) and _const_int(hi) and _const_int(hi) > 0:
                    return Mono.const()
                return self.coll_size(e.value)
            if isinstance(e.value, ast.Name) and e.value.id in self.params:
                return self._elem_sym(e.value.id)
            return None
        if isinstance(e, ast.BinOp):
            if isinstance(e.op, ast.Add):
                a, b = self.coll_size(e.left), self.coll_size(e.right)
                return None if a is None or b is None else _max_mono([a, b])
            if isinstance(e.op, ast.Mult):
                a = self.coll_size(e.left) if isinstance(e.left, (ast.List, ast.Tuple, ast.Constant)) else None
                b = self.size_of(e.right)
                if a is not None and b is not None:
                    return a * b
        if isinstance(e, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
            total = Mono.const()
            for gen in e.generators:
                b = self._iter_bound(gen.iter, register=False)
                if b is None:
                    return None
                total = total * b
            return total
        return None

    def _range_bound(self, call: ast.Call) -> tuple[Optional[Mono], str]:
        args = call.args
        if not args or len(args) > 3:
            return None, "range with unusual arguments"
        if len(args) == 3 and not _is_const(args[2]):
            return None, "range with a non-constant step"
        step_negative = len(args) == 3 and (_const_int(args[2]) or 0) < 0
        if len(args) == 1:
            start, stop = None, args[0]
        else:
            start, stop = args[0], args[1]
        if step_negative:
            start, stop = stop, start  # range(n, 0, -1): the size comes from the start
        interval = self._interval_range(start, stop)
        if interval is not None:  # e.g. range(mid + 1, hi) inside a divide-and-conquer on [lo, hi]
            return interval, ast.unparse(call)
        size = self.size_of(stop)
        if size is None:
            return None, f"range bound {ast.unparse(stop)} is not statically known"
        if start is not None and not _is_const(start):
            names = {n.id for n in ast.walk(start) if isinstance(n, ast.Name)}
            if names & self.loop_index:
                self._assume("Triangular loops (range starting at an outer loop variable) are counted as n per outer iteration; the total is still Θ(n^2).")
            elif not size.is_const:
                self._assume(
                    f"range({ast.unparse(start)}, {ast.unparse(stop)}) is bounded by {ast.unparse(stop)} (start assumed small in comparison).",
                    "medium",
                )
        return size, ast.unparse(call)

    def _interval_range(self, start: Optional[ast.AST], stop: Optional[ast.AST]) -> Optional[Mono]:
        """hi - lo if both ends are lo, hi or mid = (lo + hi) // 2 (each +/- a constant), lo and hi parameters.

        A range between two points of [lo, hi] has at most hi - lo + c items, so this is a safe upper bound.
        Only used when one end is the midpoint, so plain range(lo, hi) keeps its earlier treatment.
        """
        names = set()
        for e in (start, stop):
            if e is None:
                return None
            if isinstance(e, ast.BinOp) and isinstance(e.op, (ast.Add, ast.Sub)) and _const_int(e.right) is not None:
                e = e.left
            if not isinstance(e, ast.Name):
                return None
            names.add(e.id)
        for mid, (lo, hi) in self.mids.items():
            if mid in names and names <= {lo, hi, mid} and lo in self.params and hi in self.params:
                return self._interval_sym(lo, hi)
        return None

    def _iter_bound(self, it: ast.AST, register: bool = True) -> Optional[Mono]:
        if isinstance(it, ast.Call) and _call_name(it) == "range":
            bound, why = self._range_bound(it)
            if bound is None and register:
                self._unknown_time(f"line {it.lineno}: {why}")
            return bound
        size = self.coll_size(it)
        if size is None and register:
            target = ast.unparse(it)
            if isinstance(it, ast.Name) and it.id in self.rec_results:
                self._unknown_time(f"line {it.lineno}: iterates over the result of a recursive call, whose size is not known")
            else:
                self._unknown_time(f"line {it.lineno}: iterates over {target}, whose size depends on the data")
        return size

    # ------------------------------------------------------------------ #
    # Statements
    # ------------------------------------------------------------------ #
    def _block(self, stmts: list[ast.stmt]) -> Cost:
        return Cost.total(self._stmt(s) for s in stmts)

    def _stmt(self, s: ast.stmt) -> Cost:
        base = Cost.const()
        if isinstance(s, (ast.Assign, ast.AnnAssign)):
            value = s.value
            if value is None:
                return base
            cost = self._expr(value)
            targets = s.targets if isinstance(s, ast.Assign) else [s.target]
            for t in targets:
                self._bind(t, value)
            return base + cost
        if isinstance(s, ast.AugAssign):
            cost = self._expr(s.value)
            if isinstance(s.target, ast.Name):
                name = s.target.id
                is_coll = name in self.coll or self.kind.get(name) in _SEQUENCE_KINDS | {"list"}
                if is_coll and isinstance(s.op, ast.Add):
                    added = self.coll_size(s.value)
                    cost = cost + Cost.of(added) if added is not None else self._unknown_time(f"line {s.lineno}: size added to {name} is not known")
                    self._grow(name, added, s.lineno)
                elif name in self.scalar and isinstance(s.op, (ast.Mult, ast.Pow, ast.LShift)):
                    self.scalar[name] = None
            else:
                cost = cost + self._expr(s.target)
            return base + cost
        if isinstance(s, ast.Expr):
            return base + self._expr(s.value)
        if isinstance(s, (ast.Return,)):
            return base + (self._expr(s.value) if s.value is not None else Cost.zero())
        if isinstance(s, ast.If):
            test = self._expr(s.test)
            body = self._block(s.body)
            orelse = self._block(s.orelse)
            return base + test + body + orelse  # sum == max asymptotically (worst-case branch)
        if isinstance(s, (ast.For, ast.AsyncFor)):
            return base + self._for(s)
        if isinstance(s, ast.While):
            return base + self._while(s)
        if isinstance(s, (ast.Try, getattr(ast, "TryStar", ast.Try))):
            parts = [self._block(s.body), self._block(s.orelse), self._block(s.finalbody)]
            parts += [self._block(h.body) for h in s.handlers]
            return base + Cost.total(parts)
        if isinstance(s, (ast.With, ast.AsyncWith)):
            items = Cost.total(self._expr(i.context_expr) for i in s.items)
            return base + items + self._block(s.body)
        if isinstance(s, (ast.Raise, ast.Assert, ast.Delete)):
            exprs = [x for x in ast.iter_child_nodes(s) if isinstance(x, ast.expr)]
            return base + Cost.total(self._expr(x) for x in exprs)
        if isinstance(s, (ast.Pass, ast.Break, ast.Continue, ast.Global, ast.Nonlocal,
                          ast.Import, ast.ImportFrom, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            return base
        if hasattr(ast, "Match") and isinstance(s, ast.Match):
            return base + self._expr(s.subject) + Cost.total(self._block(c.body) for c in s.cases)
        return self._unknown_time(f"line {s.lineno}: unsupported statement ({type(s).__name__})")

    def _bind(self, target: ast.AST, value: ast.AST) -> None:
        """Record what an assignment tells us about sizes and kinds."""
        if isinstance(target, ast.Name):
            name = target.id
            self.scalar.pop(name, None)
            self.coll.pop(name, None)
            self.elem.pop(name, None)
            self.mids.pop(name, None)
            self.halves.pop(name, None)
            self.reduced_alias.pop(name, None)
            kind = _value_kind(value)
            if kind:
                self.kind[name] = kind
            if _is_mid(value):
                self.mids[name] = _is_mid(value)
            half = self._half_of(value)
            if half:
                self.halves[name] = half
            if self._is_recursive_call(value):
                self.rec_results.add(name)
                arg = self._first_shrinking_arg(value)
                if arg is not None and self.coll_size(arg) is not None:
                    self.coll[name] = self.coll_size(arg)
                    self._assume("A recursive call is assumed to return a collection the same size as its input (as in sorting).", "medium")
                return
            self.reduced_alias[name] = value
            if _is_collection_expr(value) or kind in _SEQUENCE_KINDS | _HASHED_KINDS:
                self.coll[name] = self.coll_size(value)
            else:
                self.scalar[name] = self.size_of(value)
        elif isinstance(target, (ast.Tuple, ast.List)):
            if isinstance(value, (ast.Tuple, ast.List)) and len(value.elts) == len(target.elts):
                for t, v in zip(target.elts, value.elts):
                    self._bind(t, v)
            else:
                for t in target.elts:
                    if isinstance(t, ast.Name):
                        self.scalar[t.id] = None
                        self.coll.pop(t.id, None)
        elif isinstance(target, ast.Subscript) and isinstance(target.value, ast.Name):
            name = target.value.id
            if self.kind.get(name) in _HASHED_KINDS:
                self._grow(name, Mono.const(), target.lineno)

    def _half_of(self, value: ast.AST) -> Optional[tuple[str, int]]:
        # m = len(x) // b   or   m = len(x) >> 1
        if isinstance(value, ast.BinOp) and isinstance(value.op, (ast.FloorDiv, ast.RShift)):
            left = value.left
            if (isinstance(left, ast.Call) and _call_name(left) == "len" and left.args
                    and isinstance(left.args[0], ast.Name) and _const_int(value.right)):
                d = _const_int(value.right)
                divisor = 2 ** d if isinstance(value.op, ast.RShift) else d
                if divisor >= 2:
                    return left.args[0].id, divisor
        return None

    # ------------------------------------------------------------------ #
    # Loops
    # ------------------------------------------------------------------ #
    def _for(self, s: ast.AST) -> Cost:
        header = self._expr(s.iter, iterating=True)
        bound = self._iter_bound(s.iter)
        desc = ast.unparse(s.iter)
        self._bind_loop_target(s.target, s.iter, bound)
        bound_cost = Cost.of(bound) if bound is not None else Cost.unknown()
        if bound is not None:
            self.evidence.append(
                f"line {s.lineno}: for loop over {desc} → {Cost.of(bound).render(self._names())} iterations"
                + (f" (nested at depth {self.loop_depth + 1})" if self.loop_depth else "")
            )
        body = self._in_loop(bound_cost, s.body)
        self._unbind_loop_target(s.target)
        orelse = self._block(s.orelse)
        return header + bound_cost * (Cost.const() + body) + orelse

    def _in_loop(self, bound: Cost, body: list[ast.stmt]) -> Cost:
        saved = self.mult
        self.mult = self.mult * bound
        self.loop_depth += 1
        try:
            return self._block(body)
        finally:
            self.mult = saved
            self.loop_depth -= 1

    def _bind_loop_target(self, target: ast.AST, it: ast.AST, bound: Optional[Mono]) -> None:
        source = it
        index_target, item_target = None, target
        if isinstance(it, ast.Call) and _call_name(it) == "enumerate" and isinstance(target, ast.Tuple) and len(target.elts) == 2:
            index_target, item_target = target.elts
            source = it.args[0] if it.args else it
        if isinstance(it, ast.Call) and _call_name(it) == "range" and isinstance(target, ast.Name):
            self.scalar[target.id] = bound
            self.loop_index.add(target.id)
            return
        if isinstance(index_target, ast.Name):
            self.scalar[index_target.id] = bound
            self.loop_index.add(index_target.id)
        if isinstance(item_target, ast.Name):
            name = item_target.id
            for d in (self.scalar, self.coll):
                d.pop(name, None)
            if isinstance(source, ast.Name) and source.id in self.params:
                self.elem[name] = self._elem_sym(source.id)
            else:
                self.elem.pop(name, None)
                self.scalar[name] = None

    def _unbind_loop_target(self, target: ast.AST) -> None:
        for t in ([target] if isinstance(target, ast.Name) else getattr(target, "elts", [])):
            if isinstance(t, ast.Name):
                self.loop_index.discard(t.id)

    def _while(self, s: ast.While) -> Cost:
        test = self._expr(s.test)
        bound, desc = self._while_bound(s)
        if bound is None:
            bound_cost = self._unknown_time(f"line {s.lineno}: {desc}")
        else:
            bound_cost = bound
            self.evidence.append(f"line {s.lineno}: while loop ({desc}) → {bound.render(self._names())} iterations")
        body = self._in_loop(bound_cost, s.body)
        orelse = self._block(s.orelse)
        return bound_cost * (Cost.const() + test + body) + orelse

    def _while_bound(self, s: ast.While) -> tuple[Optional[Cost], str]:
        test = s.test
        if isinstance(test, ast.Constant):
            return None, "while loop with a constant condition; iterations depend on when it breaks"

        # Binary search: while lo <= hi (or lo < hi, hi - lo > 1)
        pair = _lo_hi(test)
        if pair:
            lo, hi = pair
            mids = [m for m, (a, b) in _mid_assignments(s.body) if {a, b} == {lo, hi}]
            if mids and _assigns_from(s.body, lo, mids) and _assigns_from(s.body, hi, mids):
                size = self.scalar.get(hi) if hi in self.scalar else (self._param_sym(hi, "value") if hi in self.params else None)
                if size is None:
                    return None, f"binary-search loop, but the initial range {lo}..{hi} has unknown size"
                if size.is_const:
                    return Cost.const(), "binary search over a constant range"
                (sym,) = size.symbols() if len(size.symbols()) == 1 else (None,)
                if sym is None:
                    return None, "binary-search range over several sizes"
                return Cost.of(Mono.log(sym)), f"binary search halves {hi} - {lo} each iteration"

        # Two pointers: while i < A and j < B, every path advances i or j
        if isinstance(test, ast.BoolOp) and isinstance(test.op, ast.And):
            comps = [_counter_compare(v) for v in test.values]
            if all(comps):
                vars_ = [c[0] for c in comps]
                if len(set(vars_)) == len(vars_) and _always_advances(s.body, set(vars_)) and not _ever_decreases(s.body, set(vars_)):
                    sizes = [self.size_of(c[1]) for c in comps]
                    if any(sz is None for sz in sizes):
                        return None, "two-pointer loop with unknown bounds"
                    return Cost.of(*sizes), "each iteration advances one of " + ", ".join(vars_)

        # Guarded counter: while j >= 0 and a[j] > key (insertion sort). The counter condition alone bounds the
        # loop; the other conditions can only end it earlier, so the bound is a worst-case upper bound.
        if isinstance(test, ast.BoolOp) and isinstance(test.op, ast.And):
            for v in test.values:
                if _counter_compare(v) is None:
                    continue
                sub = ast.copy_location(ast.While(test=v, body=s.body, orelse=[]), s)
                bound, desc = self._while_bound(sub)
                if bound is not None:
                    return bound, f"{desc}; the other conditions can only stop it earlier (worst-case bound)"

        # Single counter
        comp = _counter_compare(test)
        if comp:
            var, other, squared = comp
            kind, unconditional = _update_kind(s.body, var)
            if kind is None:
                return None, f"loop variable {var} is not updated by a constant step"
            if not unconditional:
                return None, f"{var} is only updated in some iterations, so the iteration count depends on the data"
            if _is_const(other) or other is None:
                size = self.scalar.get(var) if var in self.scalar else (self._param_sym(var, "value") if var in self.params else None)
            else:
                size = self.size_of(other)
            if size is None:
                return None, f"bound of {var} is not statically known"
            if size.is_const:
                return Cost.const(), f"{var} moves over a constant range"
            if kind == "linear":
                if squared:
                    return Cost.of(_pow(size, Fraction(1, 2))), f"{var} grows by a constant while {var}² stays below the bound"
                return Cost.of(size), f"{var} changes by a constant step each iteration"
            if len(size.symbols()) != 1:
                return None, "logarithmic loop over a composite bound"
            (sym,) = size.symbols()
            return Cost.of(Mono.log(sym)), f"{var} is multiplied or divided by a constant each iteration"

        if isinstance(test, (ast.Name, ast.Attribute, ast.Call, ast.UnaryOp)):
            return None, f"worklist loop on {ast.unparse(test)}: the number of iterations depends on the data"
        return None, f"loop condition {ast.unparse(test)} does not match a supported pattern"

    # ------------------------------------------------------------------ #
    # Expressions
    # ------------------------------------------------------------------ #
    def _expr(self, e: Optional[ast.AST], iterating: bool = False) -> Cost:
        if e is None or isinstance(e, (ast.Constant, ast.Name, ast.Lambda)):
            return Cost.zero()
        line = getattr(e, "lineno", 0)
        if isinstance(e, ast.Call):
            return self._call(e, iterating)
        if isinstance(e, ast.Compare):
            cost = self._expr(e.left) + Cost.total(self._expr(c) for c in e.comparators)
            for op, right in zip(e.ops, e.comparators):
                if isinstance(op, (ast.In, ast.NotIn)):
                    cost = cost + self._membership(right, line)
            return cost
        if isinstance(e, ast.BinOp):
            cost = self._expr(e.left) + self._expr(e.right)
            if isinstance(e.op, ast.Add) and (_is_collection_expr(e.left) or _is_collection_expr(e.right) or self._is_coll_name(e.left)):
                parts = [self.coll_size(x) for x in _concat_operands(e)]
                if any(p is None for p in parts):
                    return cost + self._unknown_time(f"line {line}: concatenation involving {ast.unparse(e)} has unknown size")
                size = Cost.of(*parts)
                self._alloc_cost(size, "concatenation", line)
                return cost + size
            if isinstance(e.op, ast.Mult) and isinstance(e.left, (ast.List, ast.Tuple)):
                size = self.coll_size(e)
                if size is None:
                    return cost + self._unknown_time(f"line {line}: repeated list of unknown size")
                self._alloc(size, "repeated list", line)
                return cost + Cost.of(size)
            return cost + Cost.const()
        if isinstance(e, ast.Subscript):
            cost = self._expr(e.value)
            if isinstance(e.slice, ast.Slice):
                cost = cost + self._expr(e.slice.lower) + self._expr(e.slice.upper) + self._expr(e.slice.step)
                size = self.coll_size(e)
                if size is None:
                    return cost + self._unknown_time(f"line {line}: slice of {ast.unparse(e.value)}, whose size is not known")
                self._alloc(size, f"slice {ast.unparse(e)}", line)
                return cost + Cost.of(size)
            return cost + self._expr(e.slice) + Cost.const()
        if isinstance(e, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
            return self._comprehension(e)
        if isinstance(e, ast.IfExp):
            return self._expr(e.test) + self._expr(e.body) + self._expr(e.orelse)
        if isinstance(e, (ast.BoolOp, ast.UnaryOp, ast.Tuple, ast.List, ast.Set, ast.Dict,
                          ast.JoinedStr, ast.FormattedValue, ast.Attribute, ast.Starred,
                          ast.Await, ast.Yield, ast.YieldFrom, ast.NamedExpr, ast.Slice)):
            children = [c for c in ast.iter_child_nodes(e) if isinstance(c, ast.expr)]
            return Cost.const() + Cost.total(self._expr(c) for c in children)
        return self._unknown_time(f"line {line}: unsupported expression ({type(e).__name__})")

    def _is_coll_name(self, e: ast.AST) -> bool:
        return isinstance(e, ast.Name) and (e.id in self.coll or self.kind.get(e.id) in _SEQUENCE_KINDS)

    def _membership(self, container: ast.AST, line: int) -> Cost:
        kind = None
        if isinstance(container, ast.Name):
            kind = self.kind.get(container.id)
        elif isinstance(container, (ast.Set, ast.Dict, ast.SetComp, ast.DictComp)):
            kind = "set"
        elif isinstance(container, ast.Call):
            name = _call_name(container)
            if name in ("set", "frozenset", "dict", "Counter"):
                kind = "set"
            if name == "range":
                return Cost.const()
            if isinstance(container.func, ast.Attribute) and container.func.attr == "keys":
                return Cost.const()
        if kind in _HASHED_KINDS:
            return Cost.const()
        size = self.coll_size(container)
        if size is None:
            return self._unknown_time(f"line {line}: membership test on {ast.unparse(container)}, whose size is not known")
        if isinstance(container, ast.Name) and container.id in self.params and kind is None and not size.is_const:
            self._assume(f"`in {container.id}` is assumed to be a linear scan (it would be O(1) if {container.id} is a set or dict).", "medium")
        return Cost.of(size)

    def _comprehension(self, e: ast.AST) -> Cost:
        total = Cost.zero()
        mult = Cost.const()
        saved_scalar, saved_elem, saved_index = dict(self.scalar), dict(self.elem), set(self.loop_index)
        for gen in e.generators:
            total = total + mult * self._expr(gen.iter, iterating=True)
            bound = self._iter_bound(gen.iter)
            self._bind_loop_target(gen.target, gen.iter, bound)
            mult = mult * (Cost.of(bound) if bound is not None else Cost.unknown())
            for cond in gen.ifs:
                total = total + mult * self._expr(cond)
        elt = (self._expr(e.key) + self._expr(e.value)) if isinstance(e, ast.DictComp) else self._expr(e.elt)
        total = total + mult * (Cost.const() + elt)
        item = Mono.const()
        if not isinstance(e, (ast.GeneratorExp, ast.DictComp)) and _is_collection_expr(e.elt):
            item = self.coll_size(e.elt)
        self.scalar, self.elem, self.loop_index = saved_scalar, saved_elem, saved_index
        if not isinstance(e, ast.GeneratorExp):
            if item is None:
                self._unknown_space(f"line {e.lineno}: size of comprehension items is not known")
            else:
                self._alloc_cost(mult.times(item), "comprehension", e.lineno)
        return total

    def _alloc_cost(self, size: Cost, what: str, line: int) -> None:
        if size.is_unknown:
            self._unknown_space(f"line {line}: size of {what} is not statically known")
        else:
            self.space_terms.append(size)

    # Calls ------------------------------------------------------------- #
    def _call(self, e: ast.Call, iterating: bool) -> Cost:
        line = e.lineno
        args_cost = Cost.total(self._expr(a) for a in e.args) + Cost.total(self._expr(k.value) for k in e.keywords)
        func = e.func

        if self._is_recursive_call(e):
            self.rec_calls.append((e, self.loop_depth > 0, line))
            return args_cost + Cost.const()

        # Functions defined in this file
        callee = None
        receiver_offset = 0
        if isinstance(func, ast.Name):
            callee = self.module.resolve_name(self.info, func.id)
        elif (isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name)
              and self.receiver and func.value.id == self.receiver):
            callee = self.module.resolve_method(self.info.class_name, func.attr)
            receiver_offset = 1
        if callee is not None:
            return args_cost + self._local_call(e, callee, receiver_offset)

        if isinstance(func, ast.Name):
            return args_cost + self._builtin_call(func.id, e)
        if isinstance(func, ast.Attribute):
            cost = self._expr(func.value)
            module_name = func.value.id if isinstance(func.value, ast.Name) else None
            if module_name in ("math", "cmath", "random", "time", "operator", "string"):
                return args_cost + cost + Cost.const()
            if module_name == "heapq":
                return args_cost + cost + self._heapq(func.attr, e)
            if module_name == "bisect":
                size = self.coll_size(e.args[0]) if e.args else None
                if size is None:
                    return args_cost + self._unknown_time(f"line {line}: bisect on a list of unknown size")
                if func.attr.startswith("insort"):
                    return args_cost + Cost.of(size)
                sym = next(iter(size.symbols()), None)
                return args_cost + (Cost.of(Mono.log(sym)) if sym else Cost.const())
            if module_name in ("copy",) and func.attr == "deepcopy":
                size = self.coll_size(e.args[0]) if e.args else None
                self._alloc(size, "deep copy", line)
                return args_cost + (Cost.of(size) if size is not None else self._unknown_time(f"line {line}: deepcopy of unknown size"))
            return args_cost + cost + self._method_call(func, e)
        return args_cost + self._unresolved(e)

    def _builtin_call(self, name: str, e: ast.Call) -> Cost:
        line = e.lineno
        if name in ("min", "max") and len(e.args) >= 2:
            return Cost.const()
        if name == "sorted" or name in _LINEAR_BUILTINS or name in ("min", "max"):
            if not e.args:
                return Cost.const()
            size = self.coll_size(e.args[0])
            if size is None:
                return self._unknown_time(f"line {line}: {name}() over {ast.unparse(e.args[0])}, whose size is not known")
            if name in _ALLOCATING_BUILTINS:
                self._alloc(size, f"{name}(...)", line)
            if name == "sorted":
                self.evidence.append(f"line {line}: sorted() → n log n (Timsort)")
                return Cost.of(_n_log_n(size))
            return Cost.of(size)
        if name in _CONST_BUILTINS:
            return Cost.const()
        if name in ("deque", "Counter", "defaultdict", "OrderedDict", "heapify"):
            size = self.coll_size(e.args[0]) if e.args else Mono.const()
            if size is None:
                return self._unknown_time(f"line {line}: {name}() of unknown size")
            self._alloc(size, f"{name}(...)", line)
            return Cost.of(size)
        return self._unresolved(e)

    def _heapq(self, attr: str, e: ast.Call) -> Cost:
        size = self.coll_size(e.args[0]) if e.args else None
        if attr in ("heappush", "heappop", "heappushpop", "heapreplace"):
            if size is None:
                return self._unknown_time(f"line {e.lineno}: heap of unknown size")
            sym = next(iter(size.symbols()), None)
            if attr == "heappush" and isinstance(e.args[0], ast.Name):
                self._grow(e.args[0].id, Mono.const(), e.lineno)
            return Cost.of(Mono.log(sym)) if sym else Cost.const()
        if attr == "heapify":
            return Cost.of(size) if size is not None else self._unknown_time(f"line {e.lineno}: heap of unknown size")
        if attr in ("nlargest", "nsmallest") and len(e.args) >= 2:
            n = self.coll_size(e.args[1])
            return Cost.of(_n_log_n(n)) if n is not None else self._unknown_time(f"line {e.lineno}: heap of unknown size")
        return self._unresolved(e)

    def _method_call(self, func: ast.Attribute, e: ast.Call) -> Cost:
        attr, recv = func.attr, func.value
        line = e.lineno
        recv_name = recv.id if isinstance(recv, ast.Name) else None
        kind = self.kind.get(recv_name) if recv_name else None
        size = self.coll_size(recv)

        if attr in _GROW_METHODS and recv_name is not None:
            per = Mono.const() if _GROW_METHODS[attr] == "one" else (self.coll_size(e.args[0]) if e.args else Mono.const())
            if attr == "append" and e.args:
                item_size = self.coll_size(e.args[0]) if _is_collection_expr(e.args[0]) or self._is_coll_name(e.args[0]) else Mono.const()
                per = item_size if item_size is not None else None
            self._grow(recv_name, per, line)
        if attr == "insert" and kind in ("deque",):
            return Cost.const()
        if attr == "pop":
            if not e.args or (e.args and _const_int(e.args[0]) == -1) or kind in _HASHED_KINDS:
                return Cost.const()
            return Cost.of(size) if size is not None else self._recv_unknown(recv, attr, line)
        if attr == "remove" and kind in _HASHED_KINDS:
            return Cost.const()
        if attr == "sort":
            if size is None:
                return self._recv_unknown(recv, attr, line)
            self._alloc(size, ".sort() workspace", line)
            self.evidence.append(f"line {line}: .sort() → n log n (Timsort)")
            return Cost.of(_n_log_n(size))
        if attr in _CONST_METHODS:
            return Cost.const()
        if attr in _ARG_LINEAR_METHODS:
            arg = self.coll_size(e.args[0]) if e.args else Mono.const()
            if arg is None:
                return self._unknown_time(f"line {line}: {attr}() with an argument of unknown size")
            if attr in _ALLOCATING_METHODS:
                self._alloc(arg, f".{attr}()", line)
            return Cost.of(arg)
        if attr in _RECEIVER_LINEAR_METHODS or attr == "remove":
            if attr in _ALLOCATING_METHODS and size is not None:
                self._alloc(size, f".{attr}()", line)
            return Cost.of(size) if size is not None else self._recv_unknown(recv, attr, line)
        return self._unresolved(e)

    def _recv_unknown(self, recv: ast.AST, attr: str, line: int) -> Cost:
        # Operations on an item of a collection (e.g. a word in a list of words)
        self._assume(
            f"Items such as {ast.unparse(recv)} are assumed to have constant size, so .{attr}() on them is O(1).",
            "low",
        )
        return Cost.const()

    def _unresolved(self, e: ast.Call) -> Cost:
        # A call into code outside this file that receives a whole input collection could cost anything
        # (np.linalg.inv(a) is O(n^3)), so the time is Unknown rather than an assumed O(1).
        for arg in list(e.args) + [k.value for k in e.keywords]:
            if isinstance(arg, ast.Starred):
                arg = arg.value
            try:
                size = self.coll_size(arg)
            except Exception:  # defensive: size inference must never break the analysis
                size = None
            # Items of a collection (symbols ending in "[*]") are assumed constant-size, as elsewhere.
            if size is not None and any(not sym.endswith("[*]") for sym in size.symbols()):
                return self._unknown_time(
                    f"line {e.lineno}: {ast.unparse(e.func)}() is not defined in this file and receives the "
                    f"collection {ast.unparse(arg)}, so its cost cannot be bounded")
        self._assume(
            f"{ast.unparse(e.func)}() is not defined in this file and is assumed to be O(1).",
            "low",
        )
        return Cost.const()

    def _local_call(self, e: ast.Call, callee: FunctionInfo, offset: int) -> Cost:
        line = e.lineno
        res = self.module.result(callee)
        if res is None:
            return self._unknown_time(f"line {line}: mutual recursion with {callee.qualname} is not supported")
        if res.time.is_unknown:
            self._unknown_space(f"line {line}: {callee.qualname} could not be analysed")
            return self._unknown_time(f"line {line}: cost of {callee.qualname}() is unknown ({res.time_reasons[0] if res.time_reasons else 'see its estimate'})")

        needed = res.time.symbols() | (res.space.symbols() if not res.space.is_unknown else set())
        mapping: dict[str, Mono] = {}
        callee_params = [p for p in _params_of(self.module.nodes[callee.lineno])]
        if callee.kind == "method" and offset:
            callee_params = callee_params[1:]
        for sym in needed:
            param = sym.split("|", 1)[1]
            base = param[:-3] if param.endswith("[*]") else param
            if base not in callee_params:
                return self._unknown_time(f"line {line}: {callee.qualname}() cost depends on {res.sym_desc.get(sym, sym)}, which the caller cannot see")
            idx = callee_params.index(base)
            arg = None
            if idx < len(e.args) and not isinstance(e.args[idx], ast.Starred):
                arg = e.args[idx]
            for k in e.keywords:
                if k.arg == base:
                    arg = k.value
            if arg is None:
                mapping[sym] = Mono.const()  # default value: assumed constant size
                continue
            if param.endswith("[*]"):
                size = self._elem_sym(arg.id) if isinstance(arg, ast.Name) and arg.id in self.params else None
            else:
                size = self.coll_size(arg) if "len(" in res.sym_desc.get(sym, "") else self.size_of(arg)
                if size is None:
                    size = self.coll_size(arg) or self.size_of(arg)
            if size is None:
                return self._unknown_time(f"line {line}: argument {ast.unparse(arg)} to {callee.qualname}() has unknown size")
            mapping[sym] = size
        try:
            time = res.time.map_terms(lambda t: t.substitute(mapping))
            if res.space.is_unknown:
                self._unknown_space(f"line {line}: space of {callee.qualname}() is unknown")
            else:
                self.space_terms.append(res.space.map_terms(lambda t: t.substitute(mapping)))
        except ValueError:
            return self._unknown_time(f"line {line}: cannot translate the cost of {callee.qualname}() to this function's sizes")
        for text, level in res.assumptions:
            self._assume(text, level)
        self.evidence.append(
            f"line {line}: calls {callee.qualname}() → {time.render(self._names())}"
        )
        return time

    # ------------------------------------------------------------------ #
    # Recursion
    # ------------------------------------------------------------------ #
    def _is_recursive_call(self, e: ast.AST) -> bool:
        if not isinstance(e, ast.Call):
            return False
        f = e.func
        if not self.is_method:
            return isinstance(f, ast.Name) and f.id == self.info.name
        class_simple = self.info.class_name.split(".")[-1] if self.info.class_name else None
        return (
            isinstance(f, ast.Attribute) and f.attr == self.info.name and isinstance(f.value, ast.Name)
            and f.value.id in {self.receiver, class_simple} - {None}
        )

    def _call_args(self, call: ast.Call) -> dict[str, ast.AST]:
        args = {}
        for p, a in zip(self.params, call.args):
            args[p] = a
        for k in call.keywords:
            if k.arg in self.params:
                args[k.arg] = k.value
        return args

    def _first_shrinking_arg(self, call: ast.Call) -> Optional[ast.AST]:
        for p, a in self._call_args(call).items():
            if self._reduction(p, a)[0] in ("dec", "div"):
                return a
        return None

    def _reduction(self, p: str, a: ast.AST, depth: int = 0) -> tuple:
        """How argument ``a`` relates to parameter ``p``: same / dec c / div b / other."""
        if isinstance(a, ast.Name):
            if a.id == p:
                return ("same",)
            if a.id in self.reduced_alias and depth < 3:
                return self._reduction(p, self.reduced_alias[a.id], depth + 1)
            return ("other",)
        if isinstance(a, ast.BinOp) and isinstance(a.left, ast.Name) and a.left.id == p:
            c = _const_int(a.right)
            if isinstance(a.op, ast.Sub) and c and c > 0:
                return ("dec", c)
            if isinstance(a.op, (ast.FloorDiv, ast.Div)) and c and c >= 2:
                return ("div", c)
            if isinstance(a.op, ast.RShift) and c and c >= 1:
                return ("div", 2 ** c)
        if isinstance(a, ast.Subscript) and isinstance(a.value, ast.Name) and a.value.id == p and isinstance(a.slice, ast.Slice):
            lo, hi = a.slice.lower, a.slice.upper
            if a.slice.step is not None:
                return ("other",)
            if hi is None and lo is not None and (_const_int(lo) or 0) > 0:
                return ("dec", _const_int(lo))
            if lo is None and hi is not None and (_const_int(hi) or 0) < 0:
                return ("dec", -_const_int(hi))
            bound = hi if lo is None else lo if hi is None else None
            if isinstance(bound, ast.Name) and self.halves.get(bound.id, (None,))[0] == p:
                return ("div", self.halves[bound.id][1])
            if bound is not None and self._half_of(bound) and self._half_of(bound)[0] == p:
                return ("div", self._half_of(bound)[1])
        if (isinstance(a, ast.BinOp) and isinstance(a.op, ast.Add)
                and all(isinstance(x, ast.Subscript) and isinstance(x.value, ast.Name) and x.value.id == p
                        and isinstance(x.slice, ast.Slice) for x in (a.left, a.right))):
            return ("dec", 1)  # p[:i] + p[i+1:]
        return ("other",)

    def _interval_reduction(self, call: ast.Call) -> Optional[tuple[str, str]]:
        """(lo, hi) if the call recurses on one half of [lo, hi] via mid."""
        args = self._call_args(call)
        for mid, (lo, hi) in self.mids.items():
            if lo not in args or hi not in args:
                continue
            a_lo, a_hi = args[lo], args[hi]
            left_half = _is_name(a_lo, lo) and _mid_offset(a_hi, mid) is not None
            right_half = _mid_offset(a_lo, mid) is not None and _is_name(a_hi, hi)
            if left_half or right_half:
                return lo, hi
        return None

    def _solve_recursion(self, f: Cost, frame_space: Cost) -> tuple[Cost, Cost, Optional[str]]:
        calls = self.rec_calls
        lines = ", ".join(str(c[2]) for c in calls)

        # 1. What shrinks?
        measure = None  # (symbol Mono, kind, factor)
        interval = [self._interval_reduction(c) for c, _, _ in calls]
        if all(interval) and len(set(interval)) == 1:
            lo, hi = interval[0]
            measure = (self._interval_sym(lo, hi), "div", 2, f"{hi} - {lo}")
        else:
            for p in self.params:
                reds = []
                for c, _, _ in calls:
                    a = self._call_args(c).get(p)
                    reds.append(self._reduction(p, a) if a is not None else ("other",))
                kinds = {r[0] for r in reds}
                if kinds == {"dec"}:
                    usage = "value" if self._used_as_value(p) else "len"
                    measure = (self._param_sym(p, usage), "dec", max(r[1] for r in reds), p)
                    if len({r[1] for r in reds}) > 1:
                        self._assume("Recursive calls shrink the input by different constants (e.g. n-1 and n-2); O(c^n) is an upper bound (Fibonacci-style recursion is Θ(φ^n), φ ≈ 1.618).")
                    break
                if kinds == {"div"} and len({r[1] for r in reds}) == 1:
                    usage = "value" if self._used_as_value(p) else "len"
                    measure = (self._param_sym(p, usage), "div", reds[0][1], p)
                    break
        if measure is None:
            reason = f"recursive calls (line {lines}) do not all shrink the same argument by a constant step or factor"
            partition = any(
                isinstance(a, ast.Name) and isinstance(self.reduced_alias.get(a.id), (ast.ListComp, ast.Call))
                for c, _, _ in calls for a in c.args
            ) or any(isinstance(a, (ast.ListComp, ast.GeneratorExp)) for c, _, _ in calls for a in c.args)
            if partition:
                reason += " (sizes depend on the data, e.g. quicksort partitions: O(n log n) average, O(n^2) worst case)"
            self._unknown_time(reason, first=True)
            self._unknown_space(reason, first=True)
            return Cost.unknown(), Cost.unknown(), None
        sym_mono, kind, factor, label = measure
        (s,) = sym_mono.symbols()
        size_text = self.sym_desc.get(s, label)

        # 2. Base case and branching
        max_calls, min_calls = _calls_per_path(self.node.body, {id(c) for c, _, _ in calls})
        if min_calls > 0:
            reason = "no execution path without a recursive call (base case) was found"
            self._unknown_time(reason, first=True)
            self._unknown_space(reason, first=True)
            return Cost.unknown(), Cost.unknown(), None
        in_loop = any(l for _, l, _ in calls)
        a = max(max_calls, 1)
        memo = self._memoized(label)

        def per_term(fn) -> Cost:
            if f.is_unknown:
                return Cost.unknown()
            return Cost.total(Cost.of(fn(t.power_of(s), t.log_of(s)) * t.without(s)) for t in (f.terms or {Mono.const()}))

        names = None
        if memo:
            if kind != "dec":
                time = None
            else:
                time = per_term(lambda k, j: Mono(((s, k + 1),), ((s, j),) if j else ()))
                recurrence = f"memoized: each value of {label} is computed once ({size_text} distinct values) × work per call"
                self.evidence.append("memoization detected: overlapping subproblems are solved once")
        if not memo or time is None:
            if in_loop:
                loop_bound_ok = kind == "dec" and factor == 1
                if not loop_bound_ok:
                    reason = "recursive call inside a loop (backtracking) with an unsupported shape"
                    self._unknown_time(reason)
                    self._unknown_space(reason)
                    return Cost.unknown(), Cost.unknown(), None
                time = per_term(lambda k, j: Mono.factorial(s))
                recurrence = f"T(n) = n·T(n-1) + f(n) with n = {size_text} → O(n!) (polynomial factors per call are absorbed)"
                self._assume("Recursion inside a loop over the remaining input (permutation-style) is treated as n·T(n-1).", "medium")
            elif kind == "dec":
                if a == 1:
                    time = per_term(lambda k, j: Mono(((s, k + 1),), ((s, j),) if j else ()))
                    recurrence = f"T(n) = T(n-{factor}) + f(n), n = {size_text}: n levels × work per level"
                else:
                    time = per_term(lambda k, j: Mono.exp(s, a))
                    recurrence = f"T(n) = {a}·T(n-c) + f(n), n = {size_text}: up to {a}^n calls"
            else:
                b = factor
                crit = math.log(a) / math.log(b)

                def master(k, j):
                    k_f = float(k)
                    if abs(k_f - crit) < 1e-9:
                        return Mono(((s, k),) if k else (), ((s, j + 1),))
                    if k_f > crit:
                        return Mono(((s, k),) if k else (), ((s, j),) if j else ())
                    cp = Fraction(round(crit)) if abs(crit - round(crit)) < 1e-9 else Fraction(crit).limit_denominator(100)
                    return Mono(((s, cp),))

                time = per_term(master)
                recurrence = f"T(n) = {a}·T(n/{b}) + f(n), n = {size_text} (master theorem)"
            if memo and kind != "dec":
                self._assume("Memoization was detected but does not change this divide-and-conquer recurrence.", "medium")

        # 3. Space: stack depth + memory alive along one chain of calls
        if frame_space.is_unknown:
            space = Cost.unknown()
        else:
            depth = Mono.sym(s) if kind == "dec" else Mono.log(s)
            alive = Cost.zero()
            for t in frame_space.terms:
                k = t.power_of(s)
                if kind == "dec":
                    alive = alive + Cost.of(Mono(((s, k + 1),)) * t.without(s))
                else:
                    alive = alive + (Cost.of(t) if k > 0 else Cost.of(t * Mono.log(s)))
            space = Cost.of(depth) + alive
            if memo:
                space = space + Cost.of(Mono.sym(s))
        f_text = f.render(self._names()) if (f.terms or f.is_unknown) else "O(1)"
        self.evidence.append(
            f"recursion: up to {a} recursive call(s) per invocation on {size_text}"
            + (f" (lines {lines})" if lines else "")
            + f"; non-recursive work per call {f_text}"
        )
        return time, space, recurrence

    def _used_as_value(self, p: str) -> bool:
        """True if parameter p is used as a number (n - 1, range(n)) rather than a collection."""
        for n in ast.walk(self.node):
            if isinstance(n, ast.BinOp) and isinstance(n.left, ast.Name) and n.left.id == p and isinstance(n.op, (ast.Sub, ast.FloorDiv, ast.Div, ast.RShift)):
                return True
            if isinstance(n, ast.Compare) and isinstance(n.left, ast.Name) and n.left.id == p and all(_is_const(c) for c in n.comparators):
                return True
        return False

    def _memoized(self, param: str) -> bool:
        if any(d.split("(")[0].split(".")[-1] in _MEMO_DECORATORS for d in self.info.decorators):
            self._assume("@lru_cache/@cache memoization: each distinct argument value is computed once.")
            return True
        for n in ast.walk(self.node):
            if (isinstance(n, ast.If) and isinstance(n.test, ast.Compare) and len(n.test.ops) == 1
                    and isinstance(n.test.ops[0], ast.In) and _is_name(n.test.left, param)
                    and any(isinstance(b, ast.Return) for b in n.body)):
                self._assume(f"Manual memoization on {param}: each distinct value is computed once.")
                return True
        return False


# --------------------------------------------------------------------------- #
# Pattern helpers (pure functions over AST)
# --------------------------------------------------------------------------- #
def _params_of(node: ast.AST) -> list[str]:
    a = node.args
    names = [x.arg for x in (*a.posonlyargs, *a.args, *a.kwonlyargs)]
    return names


def _call_name(e: ast.Call) -> Optional[str]:
    if isinstance(e.func, ast.Name):
        return e.func.id
    if isinstance(e.func, ast.Attribute) and isinstance(e.func.value, ast.Name) and e.func.value.id in ("collections", "functools", "itertools", "heapq"):
        return e.func.attr
    return None


def _is_const(e: Optional[ast.AST]) -> bool:
    if isinstance(e, ast.Constant):
        return True
    return isinstance(e, ast.UnaryOp) and isinstance(e.operand, ast.Constant)


def _const_int(e: Optional[ast.AST]) -> Optional[int]:
    if isinstance(e, ast.Constant) and isinstance(e.value, int) and not isinstance(e.value, bool):
        return e.value
    if isinstance(e, ast.UnaryOp) and isinstance(e.op, ast.USub) and isinstance(e.operand, ast.Constant) and isinstance(e.operand.value, int):
        return -e.operand.value
    return None


def _is_name(e: Optional[ast.AST], name: str) -> bool:
    return isinstance(e, ast.Name) and e.id == name


def _pow(m: Mono, p: Fraction) -> Mono:
    out = Mono()
    for s, q in m.powers:
        out = out * Mono.sym(s, q * p)
    for s, k in m.logs:
        out = out * Mono.log(s, k)
    return out


def _max_mono(monos: list[Mono]) -> Optional[Mono]:
    best = monos[0]
    for m in monos[1:]:
        if best.dominated_by(m):
            best = m
        elif not m.dominated_by(best):
            return None  # n vs m: no single term bounds n + m
    return best


def _concat_operands(e: ast.AST) -> list[ast.AST]:
    if isinstance(e, ast.BinOp) and isinstance(e.op, ast.Add):
        return _concat_operands(e.left) + _concat_operands(e.right)
    return [e]


def _n_log_n(size: Mono) -> Mono:
    if size.is_const:
        return size
    out = size
    for s in sorted(size.symbols()):
        out = out * Mono.log(s)
        break
    return out


def _annotation_kind(ann: ast.AST) -> Optional[str]:
    base = ann.value if isinstance(ann, ast.Subscript) else ann
    name = base.id if isinstance(base, ast.Name) else base.attr if isinstance(base, ast.Attribute) else None
    return _ANNOTATION_KINDS.get(name)


def _value_kind(v: ast.AST) -> Optional[str]:
    if isinstance(v, (ast.Set, ast.SetComp)):
        return "set"
    if isinstance(v, (ast.Dict, ast.DictComp)):
        return "dict"
    if isinstance(v, (ast.List, ast.ListComp)):
        return "list"
    if isinstance(v, ast.Constant) and isinstance(v.value, str):
        return "str"
    if isinstance(v, ast.Call):
        name = _call_name(v)
        if name in ("set", "frozenset"):
            return "set"
        if name in ("dict", "Counter", "defaultdict", "OrderedDict"):
            return name if name != "frozenset" else "set"
        if name in ("list", "sorted"):
            return "list"
        if name == "deque":
            return "deque"
    return None


def _is_collection_expr(v: ast.AST) -> bool:
    if isinstance(v, (ast.List, ast.Set, ast.Dict, ast.ListComp, ast.SetComp, ast.DictComp, ast.Subscript)):
        return not isinstance(v, ast.Subscript) or isinstance(v.slice, ast.Slice)
    if isinstance(v, ast.Tuple):
        return True
    if isinstance(v, ast.Call):
        if _call_name(v) in ("list", "set", "dict", "sorted", "tuple", "frozenset", "deque", "Counter", "defaultdict", "OrderedDict", "reversed"):
            return True
        if isinstance(v.func, ast.Attribute) and v.func.attr in ("copy", "split", "keys", "values", "items"):
            return True
    if isinstance(v, ast.BinOp) and isinstance(v.op, (ast.Add, ast.Mult)):
        return _is_collection_expr(v.left) or _is_collection_expr(v.right)
    return False


def _is_mid(v: ast.AST) -> Optional[tuple[str, str]]:
    """(lo, hi) if v is (lo + hi) // 2, (lo + hi) >> 1 or lo + (hi - lo) // 2."""
    if isinstance(v, ast.BinOp) and isinstance(v.op, (ast.FloorDiv, ast.RShift)):
        if isinstance(v.left, ast.BinOp) and isinstance(v.left.op, ast.Add):
            a, b = v.left.left, v.left.right
            if isinstance(a, ast.Name) and isinstance(b, ast.Name):
                return a.id, b.id
    if isinstance(v, ast.BinOp) and isinstance(v.op, ast.Add) and isinstance(v.left, ast.Name):
        r = v.right
        if isinstance(r, ast.BinOp) and isinstance(r.op, (ast.FloorDiv, ast.RShift)) and isinstance(r.left, ast.BinOp) and isinstance(r.left.op, ast.Sub):
            hi, lo = r.left.left, r.left.right
            if isinstance(hi, ast.Name) and isinstance(lo, ast.Name) and lo.id == v.left.id:
                return lo.id, hi.id
    return None


def _mid_offset(e: ast.AST, mid: str) -> Optional[int]:
    if _is_name(e, mid):
        return 0
    if isinstance(e, ast.BinOp) and _is_name(e.left, mid) and isinstance(e.op, (ast.Add, ast.Sub)) and _const_int(e.right) is not None:
        return _const_int(e.right)
    return None


def _lo_hi(test: ast.AST) -> Optional[tuple[str, str]]:
    if isinstance(test, ast.Compare) and len(test.ops) == 1:
        left, op, right = test.left, test.ops[0], test.comparators[0]
        if isinstance(left, ast.Name) and isinstance(right, ast.Name):
            if isinstance(op, (ast.Lt, ast.LtE)):
                return left.id, right.id
            if isinstance(op, (ast.Gt, ast.GtE)):
                return right.id, left.id
        if (isinstance(left, ast.BinOp) and isinstance(left.op, ast.Sub) and isinstance(left.left, ast.Name)
                and isinstance(left.right, ast.Name) and isinstance(op, (ast.Gt, ast.GtE))):
            return left.right.id, left.left.id
    return None


def _mid_assignments(stmts: list[ast.stmt]) -> list[tuple[str, tuple[str, str]]]:
    found = []
    for n in _walk_no_nested_loops(stmts):
        if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name):
            pair = _is_mid(n.value)
            if pair:
                found.append((n.targets[0].id, pair))
    return found


def _assigns_from(stmts: list[ast.stmt], var: str, mids: list[str]) -> bool:
    for n in _walk_no_nested_loops(stmts):
        if isinstance(n, ast.Assign) and len(n.targets) == 1 and _is_name(n.targets[0], var):
            if any(_mid_offset(n.value, m) is not None for m in mids):
                return True
    return False


def _walk_no_nested_loops(stmts: list[ast.stmt]):
    stack = list(stmts)
    while stack:
        n = stack.pop()
        yield n
        if isinstance(n, (ast.For, ast.While, ast.AsyncFor, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
            continue
        stack.extend(ast.iter_child_nodes(n))


def _counter_compare(test: ast.AST) -> Optional[tuple[str, Optional[ast.AST], bool]]:
    """(var, other side, squared) for `var OP other` / `other OP var` / `var*var <= n`."""
    if not (isinstance(test, ast.Compare) and len(test.ops) == 1):
        return None
    if not isinstance(test.ops[0], (ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.NotEq)):
        return None
    left, right = test.left, test.comparators[0]
    sq = _squared_var(left)
    if sq:
        return sq, right, True
    if isinstance(left, ast.Name):
        return left.id, right, False
    if isinstance(right, ast.Name) and not isinstance(left, ast.Name):
        return right.id, left, False
    return None


def _squared_var(e: ast.AST) -> Optional[str]:
    if isinstance(e, ast.BinOp) and isinstance(e.op, ast.Mult) and _is_name(e.left, getattr(e.right, "id", None)):
        return e.left.id
    if isinstance(e, ast.BinOp) and isinstance(e.op, ast.Pow) and isinstance(e.left, ast.Name) and _const_int(e.right) == 2:
        return e.left.id
    return None


def _step_kind(stmt: ast.AST, var: str) -> Optional[str]:
    """'linear' / 'log' / 'other' for a statement that assigns ``var``; None if it doesn't."""
    if isinstance(stmt, ast.AugAssign) and _is_name(stmt.target, var):
        c = _const_int(stmt.value)
        if isinstance(stmt.op, (ast.Add, ast.Sub)) and c is not None and c != 0:
            return "linear"
        if isinstance(stmt.op, (ast.Mult, ast.FloorDiv, ast.Div)) and c is not None and abs(c) >= 2:
            return "log"
        if isinstance(stmt.op, (ast.RShift, ast.LShift)) and c is not None and c >= 1:
            return "log"
        return "other"
    if isinstance(stmt, ast.Assign) and any(_is_name(t, var) for t in stmt.targets):
        v = stmt.value
        if isinstance(v, ast.BinOp) and _is_name(v.left, var):
            c = _const_int(v.right)
            if isinstance(v.op, (ast.Add, ast.Sub)) and c:
                return "linear"
            if isinstance(v.op, (ast.Mult, ast.FloorDiv, ast.Div)) and c is not None and abs(c) >= 2:
                return "log"
            if isinstance(v.op, (ast.RShift, ast.LShift)) and c:
                return "log"
        return "other"
    return None


def _update_kind(body: list[ast.stmt], var: str) -> tuple[Optional[str], bool]:
    """Kind of all updates to ``var`` in a loop body, and whether one is unconditional."""
    kinds = set()
    for n in _walk_no_nested_loops(body):
        k = _step_kind(n, var)
        if k:
            kinds.add(k)
    for n in ast.walk(ast.Module(body=body, type_ignores=[])):
        if isinstance(n, (ast.For, ast.While)) and any(_step_kind(x, var) for x in ast.walk(n) if x is not n):
            return None, False  # updated inside an inner loop
    if len(kinds) != 1 or "other" in kinds:
        return None, False
    unconditional = any(_step_kind(s, var) for s in body)
    return kinds.pop(), unconditional


def _always_advances(stmts: list[ast.stmt], vars_: set[str]) -> bool:
    for s in stmts:
        if any(_step_kind(s, v) == "linear" and _is_increment(s) for v in vars_):
            return True
        if isinstance(s, (ast.Break, ast.Return, ast.Raise)):
            return True
        if isinstance(s, ast.If) and s.orelse and _always_advances(s.body, vars_) and _always_advances(s.orelse, vars_):
            return True
    return False


def _is_increment(s: ast.AST) -> bool:
    if isinstance(s, ast.AugAssign):
        return isinstance(s.op, ast.Add) and (_const_int(s.value) or 0) > 0
    if isinstance(s, ast.Assign) and isinstance(s.value, ast.BinOp):
        return isinstance(s.value.op, ast.Add) and (_const_int(s.value.right) or 0) > 0
    return False


def _ever_decreases(stmts: list[ast.stmt], vars_: set[str]) -> bool:
    for n in _walk_no_nested_loops(stmts):
        for v in vars_:
            if _step_kind(n, v) and not _is_increment(n):
                return True
    return False


def _calls_per_path(stmts: list[ast.stmt], call_ids: set[int]) -> tuple[int, int]:
    """(max, min) number of recursive calls along one execution path."""

    def count(e: Optional[ast.AST]) -> tuple[int, int]:
        if e is None:
            return 0, 0
        if isinstance(e, ast.IfExp):
            t = count(e.test)
            b, o = count(e.body), count(e.orelse)
            return t[0] + max(b[0], o[0]), t[1] + min(b[1], o[1])
        if isinstance(e, ast.BoolOp):
            parts = [count(v) for v in e.values]
            return sum(p[0] for p in parts), parts[0][1]
        if isinstance(e, (ast.Lambda,)):
            return 0, 0
        own = 1 if id(e) in call_ids else 0
        hi = lo = own
        for c in ast.iter_child_nodes(e):
            ch = count(c)
            hi += ch[0]
            lo += ch[1]
        return hi, lo

    NEG = None

    def block(body: list[ast.stmt]) -> tuple[Optional[tuple[int, int]], Optional[tuple[int, int]]]:
        cont = (0, 0)
        term = NEG
        for s in body:
            c, t = stmt(s)
            if t is not None:
                tt = (cont[0] + t[0], cont[1] + t[1])
                term = tt if term is None else (max(term[0], tt[0]), min(term[1], tt[1]))
            if c is None:
                return None, term
            cont = (cont[0] + c[0], cont[1] + c[1])
        return cont, term

    def merge(a, b):
        if a is None:
            return b
        if b is None:
            return a
        return (max(a[0], b[0]), min(a[1], b[1]))

    def stmt(s: ast.stmt):
        if isinstance(s, (ast.Return,)):
            return None, count(s.value)
        if isinstance(s, ast.Raise):
            return None, count(s.exc)
        if isinstance(s, ast.If):
            t = count(s.test)
            bc, bt = block(s.body)
            oc, ot = block(s.orelse)
            add = lambda x: None if x is None else (x[0] + t[0], x[1] + t[1])
            return add(merge(bc, oc)), add(merge(bt, ot))
        if isinstance(s, (ast.For, ast.While, ast.AsyncFor)):
            header = count(s.iter if not isinstance(s, ast.While) else s.test)
            bc, bt = block(s.body)
            body_max = max((bc or (0, 0))[0], (bt or (0, 0))[0])
            # the loop may run zero times: min contribution 0
            return (header[0] + body_max, header[1]), None
        if isinstance(s, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            return (0, 0), None
        if isinstance(s, (ast.Try, getattr(ast, "TryStar", ast.Try), ast.With, ast.AsyncWith)):
            inner = list(getattr(s, "body", [])) + list(getattr(s, "orelse", [])) + list(getattr(s, "finalbody", []))
            return block(inner)
        return count(s), None

    cont, term = block(stmts)
    both = merge(cont, term) or (0, 0)
    return both


def _letters(order: list[str], desc: dict[str, str]) -> dict[str, str]:
    """Map internal symbols to short letters: n, m, k, p, q, ..."""
    pool = ["n", "m", "k", "p", "q", "r", "s", "t"]
    names: dict[str, str] = {}
    used: set[str] = set()
    for sym in order:
        param = sym.split("|", 1)[1]
        preferred = param if len(param) == 1 and param.isalpha() and param not in used and desc.get(sym, "").startswith("value") else None
        letter = preferred or next((x for x in pool if x not in used), sym)
        names[sym] = letter
        used.add(letter)
    return names


# --------------------------------------------------------------------------- #
# Result assembly
# --------------------------------------------------------------------------- #
def _to_estimate(res: _FnResult, function_id: str) -> ComplexityEstimate:
    used = (res.time.symbols() if not res.time.is_unknown else set()) | (
        res.space.symbols() if not res.space.is_unknown else set()
    )
    # Letters follow parameter order (then any other sizes in order of appearance)
    param_order = list(res.param_syms.values())
    order = sorted(
        (s for s in res.sym_order if s in used),
        key=lambda s: (param_order.index(s.replace("[*]", "")) if s.replace("[*]", "") in param_order else len(param_order),
                       res.sym_order.index(s)),
    )
    names = _letters(order, res.sym_desc)
    time_str = res.time.render(names)
    space_str = res.space.render(names)
    time_class = "Unknown" if res.time.is_unknown else res.time.collapse().render({"n": "n"})
    space_class = "Unknown" if res.space.is_unknown else res.space.collapse().render({"n": "n"})

    max_level = max((_LEVELS[l] for _, l in res.assumptions), default=0)
    if len(order) > 1 and time_class != time_str:
        max_level = max(max_level, 1)
    time_status = "unknown" if res.time.is_unknown else "estimated"
    space_status = "unknown" if res.space.is_unknown else "estimated"
    status = (
        "estimated" if time_status == space_status == "estimated"
        else "unknown" if time_status == space_status == "unknown"
        else "partial"
    )
    confidence = _CONFIDENCE[max_level]

    assumptions = [t for t, _ in res.assumptions]
    if len(order) > 1:
        assumptions.append(
            "time_class/space_class assume all input sizes ("
            + ", ".join(names[s] for s in order)
            + ") grow proportionally to one n."
        )
    if time_status == "estimated" and any("for loop" in e or "while loop" in e for e in res.evidence):
        assumptions.append("Worst case: loops are assumed to run to completion (no early exit).")

    if res.time.is_unknown:
        explanation = "Time could not be estimated: " + (res.time_reasons[0] if res.time_reasons else "unsupported construct") + "."
    elif res.recurrence:
        explanation = f"Time {time_str} from the recurrence {res.recurrence}."
    elif res.evidence:
        explanation = f"Time {time_str} from: " + "; ".join(res.evidence[:4]) + "."
    else:
        explanation = f"Time {time_str}: no loops, recursion or size-dependent operations."
    if res.space.is_unknown:
        explanation += " Space could not be estimated: " + (res.space_reasons[0] if res.space_reasons else "unsupported construct") + "."
    else:
        explanation += f" Auxiliary space {space_str}" + (
            " (recursion stack and memory alive along one chain of calls)." if res.recurrence else
            " (inputs are not counted)."
        )

    return ComplexityEstimate(
        function_id=function_id,
        function_name=res.info.name,
        qualname=res.info.qualname,
        time_complexity=time_str,
        space_complexity=space_str,
        time_class=time_class,
        space_class=space_class,
        status=status,
        time_status=time_status,
        space_status=space_status,
        time_confidence=None if res.time.is_unknown else confidence,
        space_confidence=None if res.space.is_unknown else confidence,
        explanation=explanation,
        assumptions=tuple(assumptions),
        evidence=tuple(res.evidence),
        size_variables={names[s]: res.sym_desc[s] for s in order},
        recurrence=res.recurrence,
    )


# --------------------------------------------------------------------------- #
# Module-level scripts (competitive-programming style code with no functions)
# --------------------------------------------------------------------------- #
SCRIPT_ID = "<module>"
_SCRIPT_LINE = 10 ** 9  # synthetic line key; real code never reaches it
_STDIN_ATTRS = {"readline", "read", "readlines", "buffer"}


def _input_aliases(statements: list) -> set[str]:
    """Names bound to an input reader, e.g. ``input = sys.stdin.readline`` or ``r = lambda: ...readline()``."""
    names = {"input"}
    for st in statements:
        if isinstance(st, ast.Assign) and len(st.targets) == 1 and isinstance(st.targets[0], ast.Name):
            v = st.value
            if isinstance(v, ast.Lambda):
                v = v.body
            if any(isinstance(n, ast.Attribute) and n.attr in _STDIN_ATTRS for n in ast.walk(v)):
                names.add(st.targets[0].id)
    return names


def _is_input_call(node: ast.AST, aliases: set[str]) -> bool:
    if not isinstance(node, ast.Call):
        return False
    f = node.func
    if isinstance(f, ast.Name):
        return f.id in aliases
    if isinstance(f, ast.Attribute) and f.attr in _STDIN_ATTRS:
        return True
    return False


class _ScriptInputs(ast.NodeTransformer):
    """Turn values read from standard input into size parameters of a synthetic function.

    ``n = int(input())`` and ``a = list(map(int, input().split()))`` make ``n`` and ``a`` parameters (the
    statement is dropped); ``a = sorted(map(int, input().split()))`` keeps its sorting cost as ``a = sorted(a)``.
    An input read used directly inside an expression (``range(int(input()))``) becomes a fresh parameter.
    """

    def __init__(self, aliases: set[str]):
        self.aliases = aliases
        self.params: list[str] = []

    def _add(self, name: str) -> None:
        if name not in self.params:
            self.params.append(name)

    def _reads_input(self, node: ast.AST) -> bool:
        return any(_is_input_call(n, self.aliases) for n in ast.walk(node))

    def _targets(self, targets) -> list[str]:
        out = []
        for t in targets:
            for n in ast.walk(t):
                if isinstance(n, ast.Name):
                    out.append(n.id)
        return out

    def visit_FunctionDef(self, node):  # function bodies are analysed on their own
        return node

    visit_AsyncFunctionDef = visit_FunctionDef
    visit_ClassDef = visit_FunctionDef
    visit_Lambda = visit_FunctionDef

    def visit_Assign(self, node):
        if node.value is not None and self._reads_input(node.value):
            names = self._targets(node.targets)
            if not names:
                return ast.copy_location(ast.Pass(), node)
            for n in names:
                self._add(n)
            sorts = any(isinstance(c, ast.Call) and isinstance(c.func, ast.Name) and c.func.id == "sorted"
                        for c in ast.walk(node.value))
            if sorts and len(names) == 1:
                new = ast.Assign(targets=[ast.Name(id=names[0], ctx=ast.Store())],
                                 value=ast.Call(func=ast.Name(id="sorted", ctx=ast.Load()),
                                                args=[ast.Name(id=names[0], ctx=ast.Load())], keywords=[]))
                return ast.fix_missing_locations(ast.copy_location(new, node))
            return ast.copy_location(ast.Pass(), node)
        return self.generic_visit(node)

    def visit_AnnAssign(self, node):
        if node.value is not None and self._reads_input(node.value):
            for n in self._targets([node.target]):
                self._add(n)
            return ast.copy_location(ast.Pass(), node)
        return self.generic_visit(node)

    def visit_Expr(self, node):
        if self._reads_input(node.value) and isinstance(node.value, ast.Call) and _is_input_call(node.value, self.aliases):
            return ast.copy_location(ast.Pass(), node)  # a line read and discarded
        return self.generic_visit(node)

    def visit_Call(self, node):
        # int(input()) / input() used inside an expression -> a fresh size parameter
        inner = node.args[0] if (isinstance(node.func, ast.Name) and node.func.id in ("int", "float") and node.args) else None
        if _is_input_call(node, self.aliases) or (inner is not None and _is_input_call(inner, self.aliases)):
            name = f"input_{len(self.params) + 1}"
            self._add(name)
            return ast.copy_location(ast.Name(id=name, ctx=ast.Load()), node)
        return self.generic_visit(node)


def analyze_script(source: Union[str, ParseResult]) -> Optional[ComplexityEstimate]:
    """Estimate the module-level code of a script as one unit, or None if the file has no such code.

    Values read from standard input are the input sizes; calls to functions defined in the file are
    costed by the per-function analysis. The same limits apply: anything outside the supported
    patterns is reported as Unknown, never guessed.
    """
    parsed = source if isinstance(source, ParseResult) else parse_source(source)
    if not parsed.success:
        return None
    body = [s for s in parsed.tree.body
            if not isinstance(s, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Import, ast.ImportFrom))]
    body = [s for s in body if not (isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant))]  # docstrings
    if not body:
        return None
    import copy

    rewriter = _ScriptInputs(_input_aliases(parsed.tree.body))
    try:
        new_body = [rewriter.visit(copy.deepcopy(s)) for s in body]
    except RecursionError:  # pathologically deep expressions: report Unknown, never guess
        new_body, rewriter.params = [ast.Pass()], []
        too_deep = True
    else:
        too_deep = False
    args = ast.arguments(posonlyargs=[], args=[ast.arg(arg=p) for p in rewriter.params], vararg=None,
                         kwonlyargs=[], kw_defaults=[], kwarg=None, defaults=[])
    fields = dict(name=SCRIPT_ID, args=args, body=new_body, decorator_list=[], returns=None, type_comment=None)
    if "type_params" in ast.FunctionDef._fields:
        fields["type_params"] = []
    node = ast.FunctionDef(**fields)
    first, last = body[0], body[-1]
    node.lineno, node.col_offset = _SCRIPT_LINE, 0
    node.end_lineno, node.end_col_offset = _SCRIPT_LINE, 0
    ast.fix_missing_locations(node)
    info = FunctionInfo(name=SCRIPT_ID, qualname=SCRIPT_ID, kind="function", is_async=False, lineno=_SCRIPT_LINE,
                        end_lineno=_SCRIPT_LINE, col=0, end_col=0, depth=0, parent=None, class_name=None,
                        decorators=(), parameters=tuple(rewriter.params), source="")
    module = _Module(parsed)
    module.nodes[_SCRIPT_LINE] = node
    module.info_by_line[_SCRIPT_LINE] = info
    if too_deep:
        reason = ["internal analysis limitation (RecursionError): module-level code is nested too deeply"]
        module.cache[_SCRIPT_LINE] = _FnResult(info, Cost.unknown(), Cost.unknown(), reason, list(reason),
                                               [], [], {}, [], None, {})
    return module.estimate(info, SCRIPT_ID)
