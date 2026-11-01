"""Algorithm-pattern features for Big-O classification (Phase 10, improvement round).

The structural features of ``feature_extractor`` count syntax (loops,
branches, calls). The classes the Phase 10 model confused most differ in
things those counts cannot see:

* O(n) vs O(n log n): a sort, a heap or a binary search inside a loop;
* O(1) / O(log n): a loop that halves or doubles its variable;
* O(n^3) vs O(2^n) / O(n!): a loop over ``range(1 << n)``, ``itertools``
  combinatorics, branching recursion;
* constant-size inner loops (``for d in range(4)``) that add nesting without
  adding a factor of n.

Each feature below is a documented, static AST count over the whole
program. Nothing is executed. They are chosen from algorithm theory
(which patterns change a complexity class); they are heuristics, not
proofs, and are meant as additional model inputs.

Public API
----------
PATTERN_FEATURE_NAMES        fixed column order
extract_pattern_features(source_or_tree) -> dict[str, int]
"""

from __future__ import annotations

import ast
from typing import Union

PATTERN_FEATURE_NAMES: tuple[str, ...] = (
    "sort_call_count",
    "heap_call_count",
    "bisect_call_count",
    "hash_container_count",
    "math_call_count",
    "halving_loop_count",
    "binary_search_mid_count",
    "constant_bound_loop_count",
    "max_input_loop_depth",
    "dependent_inner_loop_count",
    "while_true_count",
    "exponential_range_loop_count",
    "power_of_two_expr_count",
    "combinatoric_call_count",
    "recursive_function_count",
    "max_recursive_branching",
    "dividing_recursion_count",
    "memoized_function_count",
)

PATTERN_DEFINITIONS: dict[str, str] = {
    "sort_call_count": "sorted(...) and x.sort(...) calls.",
    "heap_call_count": "heapq calls (heappush, heappop, heapify, heapreplace, heappushpop, nlargest, nsmallest).",
    "bisect_call_count": "bisect module calls (bisect, bisect_left/right, insort...).",
    "hash_container_count": "set/dict/frozenset/Counter/defaultdict constructions (calls, displays, comprehensions).",
    "math_call_count": "math.* calls plus gcd/isqrt/pow called by name (number-theory shortcuts).",
    "halving_loop_count": "while loops whose loop variable is multiplied or divided by a constant (//=, >>=, *=, <<=, x = x // c ...).",
    "binary_search_mid_count": "assignments of the form m = (a + b) // 2, (a + b) >> 1 or a + (b - a) // 2.",
    "constant_bound_loop_count": "for loops over range(<constants>) or a literal list/tuple/string.",
    "max_input_loop_depth": "deepest nesting of loops, not counting constant-bound for loops.",
    "dependent_inner_loop_count": "for-range loops whose bounds use the variable of an enclosing loop (e.g. range(i, n)).",
    "while_true_count": "while True / while 1 loops.",
    "exponential_range_loop_count": "for loops over range(...) whose bound contains 1 << x, 2 ** x or pow(2, x).",
    "power_of_two_expr_count": "expressions 1 << x, 2 ** x and pow(2, x) anywhere.",
    "combinatoric_call_count": "itertools permutations/combinations/product/combinations_with_replacement calls.",
    "recursive_function_count": "functions that call themselves directly (f(...) or self.f(...)).",
    "max_recursive_branching": "largest number of direct self-calls written in one function.",
    "dividing_recursion_count": "self-calls with an argument that divides the input (x // c, x >> c, x / c, or a slice bound by a midpoint).",
    "memoized_function_count": "functions decorated with lru_cache / cache.",
}

_HEAP = {"heappush", "heappop", "heapify", "heapreplace", "heappushpop", "nlargest", "nsmallest"}
_BISECT = {"bisect", "bisect_left", "bisect_right", "insort", "insort_left", "insort_right"}
_HASH = {"set", "dict", "frozenset", "Counter", "defaultdict", "OrderedDict"}
_COMBI = {"permutations", "combinations", "product", "combinations_with_replacement"}
_MATH_NAMES = {"gcd", "isqrt", "pow", "lcm", "comb", "factorial"}
_LOOPS = (ast.For, ast.AsyncFor, ast.While)
_FUNCS = (ast.FunctionDef, ast.AsyncFunctionDef)


def extract_pattern_features(source: Union[str, ast.AST]) -> dict[str, int]:
    tree = ast.parse(source) if isinstance(source, str) else source
    counts = {name: 0 for name in PATTERN_FEATURE_NAMES}

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = _call_name(node)
            owner = _call_owner(node)
            if name == "sorted" or (name == "sort" and owner is not None):
                counts["sort_call_count"] += 1
            if name in _HEAP:
                counts["heap_call_count"] += 1
            if name in _BISECT and (owner in (None, "bisect")):
                counts["bisect_call_count"] += 1
            if name in _HASH and owner in (None, "collections"):
                counts["hash_container_count"] += 1
            if owner == "math" or (owner is None and name in _MATH_NAMES):
                counts["math_call_count"] += 1
            if name in _COMBI and owner in (None, "itertools"):
                counts["combinatoric_call_count"] += 1
        elif isinstance(node, (ast.Set, ast.Dict, ast.SetComp, ast.DictComp)):
            counts["hash_container_count"] += 1
        elif isinstance(node, ast.Assign) and _is_mid(node.value):
            counts["binary_search_mid_count"] += 1
        elif isinstance(node, ast.While):
            if isinstance(node.test, ast.Constant) and node.test.value:
                counts["while_true_count"] += 1
            if _is_halving_loop(node):
                counts["halving_loop_count"] += 1
        elif isinstance(node, (ast.For, ast.AsyncFor)):
            if _is_constant_bound(node.iter):
                counts["constant_bound_loop_count"] += 1
            if _is_range(node.iter) and any(_is_power_of_two(n) for n in ast.walk(node.iter)):
                counts["exponential_range_loop_count"] += 1
        if _is_power_of_two(node):
            counts["power_of_two_expr_count"] += 1
        if isinstance(node, _FUNCS):
            calls = [c for c in ast.walk(node) if isinstance(c, ast.Call) and _is_self_call(c, node)]
            if calls:
                counts["recursive_function_count"] += 1
                counts["max_recursive_branching"] = max(counts["max_recursive_branching"], len(calls))
                counts["dividing_recursion_count"] += sum(_divides(c) for c in calls)
            if any(_decorator_name(d) in ("lru_cache", "cache") for d in node.decorator_list):
                counts["memoized_function_count"] += 1

    depth, dependent = _loop_structure(tree)
    counts["max_input_loop_depth"] = depth
    counts["dependent_inner_loop_count"] = dependent
    return counts


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _call_name(c: ast.Call):
    f = c.func
    return f.id if isinstance(f, ast.Name) else f.attr if isinstance(f, ast.Attribute) else None


def _call_owner(c: ast.Call):
    f = c.func
    if isinstance(f, ast.Attribute):
        return f.value.id if isinstance(f.value, ast.Name) else "<expr>"
    return None


def _decorator_name(d: ast.AST):
    if isinstance(d, ast.Call):
        d = d.func
    return d.attr if isinstance(d, ast.Attribute) else d.id if isinstance(d, ast.Name) else None


def _const_int(e):
    return e.value if isinstance(e, ast.Constant) and isinstance(e.value, int) and not isinstance(e.value, bool) else None


def _is_range(e) -> bool:
    return isinstance(e, ast.Call) and isinstance(e.func, ast.Name) and e.func.id == "range"


def _is_constant_bound(it) -> bool:
    if _is_range(it):
        return bool(it.args) and all(_const_int(a) is not None or (isinstance(a, ast.UnaryOp) and _const_int(a.operand) is not None) for a in it.args)
    if isinstance(it, (ast.List, ast.Tuple, ast.Set)):
        return all(not isinstance(x, ast.Starred) for x in it.elts)
    return isinstance(it, ast.Constant) and isinstance(it.value, str)


def _is_power_of_two(n) -> bool:
    if isinstance(n, ast.BinOp):
        if isinstance(n.op, ast.LShift) and _const_int(n.left) == 1:
            return True
        if isinstance(n.op, ast.Pow) and _const_int(n.left) == 2 and _const_int(n.right) is None:
            return True
    if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "pow" and len(n.args) == 2:
        return _const_int(n.args[0]) == 2 and _const_int(n.args[1]) is None
    return False


def _is_mid(v) -> bool:
    if isinstance(v, ast.BinOp) and isinstance(v.op, (ast.FloorDiv, ast.RShift)) and _const_int(v.right) in (1, 2):
        return isinstance(v.left, ast.BinOp) and isinstance(v.left.op, ast.Add)
    if isinstance(v, ast.BinOp) and isinstance(v.op, ast.Add) and isinstance(v.right, ast.BinOp):
        r = v.right
        return isinstance(r.op, (ast.FloorDiv, ast.RShift)) and isinstance(r.left, ast.BinOp) and isinstance(r.left.op, ast.Sub)
    return False


def _scales(stmt, names: set) -> bool:
    """True if stmt multiplies/divides one of ``names`` by a constant >= 2 (or shifts it)."""
    if isinstance(stmt, ast.AugAssign) and isinstance(stmt.target, ast.Name) and stmt.target.id in names:
        c = _const_int(stmt.value)
        if isinstance(stmt.op, (ast.FloorDiv, ast.Div, ast.Mult)) and c is not None and abs(c) >= 2:
            return True
        return isinstance(stmt.op, (ast.RShift, ast.LShift)) and c is not None and c >= 1
    if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name):
        name, v = stmt.targets[0].id, stmt.value
        if name in names and isinstance(v, ast.BinOp) and isinstance(v.left, ast.Name) and v.left.id == name:
            c = _const_int(v.right)
            if isinstance(v.op, (ast.FloorDiv, ast.Div, ast.Mult)) and c is not None and abs(c) >= 2:
                return True
            return isinstance(v.op, (ast.RShift, ast.LShift)) and c is not None and c >= 1
    return False


def _is_halving_loop(w: ast.While) -> bool:
    names = {n.id for n in ast.walk(w.test) if isinstance(n, ast.Name)}
    if not names:
        return False
    body = [n for s in w.body for n in ast.walk(s)]
    if any(_scales(n, names) for n in body):
        return True
    # binary search: a midpoint assigned, and a compared variable reassigned from it
    mids = {n.targets[0].id for n in body if isinstance(n, ast.Assign) and len(n.targets) == 1
            and isinstance(n.targets[0], ast.Name) and _is_mid(n.value)}
    return bool(mids) and any(
        isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id in names for t in n.targets)
        and any(isinstance(x, ast.Name) and x.id in mids for x in ast.walk(n.value))
        for n in body
    )


def _is_self_call(c: ast.Call, fn) -> bool:
    f = c.func
    if isinstance(f, ast.Name):
        return f.id == fn.name
    return isinstance(f, ast.Attribute) and f.attr == fn.name and isinstance(f.value, ast.Name) and f.value.id in ("self", "cls")


def _divides(c: ast.Call) -> bool:
    for a in list(c.args) + [k.value for k in c.keywords]:
        for n in ast.walk(a):
            if isinstance(n, ast.BinOp) and isinstance(n.op, (ast.FloorDiv, ast.Div, ast.RShift)) and _const_int(n.right):
                return True
            if isinstance(n, ast.Subscript) and isinstance(n.slice, ast.Slice):
                bound = n.slice.upper if n.slice.lower is None else n.slice.lower if n.slice.upper is None else None
                if isinstance(bound, ast.Name) and "mid" in bound.id.lower():
                    return True
                if isinstance(bound, ast.BinOp) and isinstance(bound.op, (ast.FloorDiv, ast.RShift)):
                    return True
    return False


def _loop_structure(tree: ast.AST) -> tuple[int, int]:
    """(max depth of non-constant loops, count of loops bounded by an enclosing loop variable)."""
    best, dependent = 0, 0
    # stack items: node, depth of input-dependent loops, enclosing loop variable names
    stack = [(tree, 0, frozenset())]
    while stack:
        node, depth, outer_vars = stack.pop()
        new_depth, new_vars = depth, outer_vars
        if isinstance(node, _LOOPS):
            constant = isinstance(node, (ast.For, ast.AsyncFor)) and _is_constant_bound(node.iter)
            if not constant:
                new_depth = depth + 1
                best = max(best, new_depth)
            if isinstance(node, (ast.For, ast.AsyncFor)):
                if _is_range(node.iter) and outer_vars & {n.id for n in ast.walk(node.iter) if isinstance(n, ast.Name)}:
                    dependent += 1
                new_vars = outer_vars | {n.id for n in ast.walk(node.target) if isinstance(n, ast.Name)}
        if isinstance(node, _FUNCS):
            new_depth, new_vars = 0, frozenset()  # loops restart inside a new function
        for child in ast.iter_child_nodes(node):
            stack.append((child, new_depth, new_vars))
    return best, dependent
