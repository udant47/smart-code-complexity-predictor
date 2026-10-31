"""Static structural features for a whole file and for each function.

Nothing here executes the submitted code: input goes through
``src.code_parser`` (``ast.parse``) and the standard ``tokenize`` module.

Public API
----------
extract_features(source_or_parse_result) -> FeatureReport
    .file_features      dict, keys exactly FILE_FEATURE_NAMES (or None on error)
    .function_features  list of dicts, keys FUNCTION_METADATA_KEYS + FUNCTION_FEATURE_NAMES
METRIC_DEFINITIONS      name -> precise one-line definition (also for UI tooltips)

Scope: file level vs function level
-----------------------------------
* **File level** counts every construct anywhere in the file, including
  module-level code, class bodies, methods and nested functions.
* **Function level** counts only the function's *own* code: its parameters
  and its body, but **not** the bodies of functions or classes defined inside
  it. Those nested functions get their own rows. What a nested ``def`` or
  ``class`` statement evaluates at definition time (decorators, default
  values, annotations, base classes) runs in the enclosing function, so it is
  counted there. Lambdas and comprehensions are expressions, so they count
  toward the function that contains them.
* **Line metrics are the exception.** For a function they describe its
  physical extent, from the ``def`` line (decorators excluded) to its last
  line, nested definitions included. Structural metrics describe logic, line
  metrics describe size on screen.
* Because each AST node belongs to exactly one scope, the additive counts
  (loops, branches, calls, operators, ...) of a file made only of
  functions equal the sum of its functions' counts. The tests check this.

Cyclomatic complexity convention
--------------------------------
McCabe's definition for a single-entry, single-exit unit:
``CC = 1 + number of decision points``. Decision points (+1 each):

* each ``if`` and each ``elif``
* each conditional expression ``a if c else b``
* each ``for``, ``async for`` and ``while`` loop
* each ``except`` / ``except*`` handler
* each ``for`` clause and each ``if`` filter inside a comprehension
* each ``and`` / ``or`` keyword (``a and b and c`` adds 2)
* each ``case`` of a ``match`` statement, except an unguarded ``case _:``

Not decision points: ``else``, ``finally``, ``with``, ``assert``, ``return``,
``break``, ``continue``, loop ``else`` and ``try`` ``else`` clauses.

Relation to radon: on the 4,651 functions of TheAlgorithms/Python, 92.8%
get the same value as radon. Every difference comes from radon also
counting each ``assert`` and each loop/``try`` ``else`` clause as +1, while
ignoring ``and``/``or`` and comprehensions inside an ``assert``.
File-level CC treats the whole file as one unit: ``1 +`` every decision
point in the file.

Limitations
-----------
These are size and control-flow metrics. They do **not** measure everything
about maintainability: naming quality, cohesion, coupling between modules,
test coverage, documentation quality and algorithmic efficiency are outside
their scope. Recursion detection is lexical: it does not see aliasing,
shadowing, mutual recursion or recursion through ``super()``.
"""

from __future__ import annotations

import ast
import io
import tokenize
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional, Union

from src.code_parser import FunctionInfo, ParseError, ParseResult, parse_source

__all__ = [
    "FILE_FEATURE_NAMES",
    "FUNCTION_FEATURE_NAMES",
    "FUNCTION_METADATA_KEYS",
    "METRIC_DEFINITIONS",
    "FeatureReport",
    "extract_features",
]

# --------------------------------------------------------------------------- #
# Feature schema
# --------------------------------------------------------------------------- #
_LINE_METRICS = (
    "code_lines",
    "blank_lines",
    "comment_lines",
    "docstring_lines",
    "comment_ratio",
)
_CONTROL_FLOW = (
    "if_count",
    "elif_count",
    "else_count",
    "for_loop_count",
    "while_loop_count",
    "loop_count",
    "try_count",
    "except_count",
    "conditional_expression_count",
    "comprehension_count",
)
_COMPLEXITY = (
    "cyclomatic_complexity",
    "maximum_nesting_depth",
    "average_nesting_depth",
    "nested_loop_count",
    "maximum_loop_nesting_depth",
    "recursive_call_count",
)
_OTHER = (
    "import_count",
    "identifier_count",
    "unique_identifier_count",
    "operator_count",
    "arithmetic_operator_count",
    "comparison_operator_count",
    "logical_operator_count",
    "bitwise_operator_count",
    "function_call_count",
    "collection_literal_count",
)

FILE_FEATURE_NAMES: tuple[str, ...] = (
    "total_lines",
    *_LINE_METRICS,
    "function_count",
    "class_count",
    "average_function_length",
    "maximum_function_length",
    "return_count",
    *_CONTROL_FLOW,
    "cyclomatic_complexity",
    "average_function_cyclomatic_complexity",
    "maximum_function_cyclomatic_complexity",
    *_COMPLEXITY[1:],
    *_OTHER,
)

FUNCTION_METADATA_KEYS: tuple[str, ...] = ("qualname", "name", "kind", "lineno", "end_lineno")

FUNCTION_FEATURE_NAMES: tuple[str, ...] = (
    "function_length",
    *_LINE_METRICS,
    "parameter_count",
    "return_count",
    "function_count",
    "class_count",
    *_CONTROL_FLOW,
    *_COMPLEXITY,
    *_OTHER,
)

METRIC_DEFINITIONS: dict[str, str] = {
    # Lines (computed from tokens, not from the AST)
    "total_lines": "Physical lines in the file; a final newline does not start an extra line.",
    "function_length": "Physical lines from the def line to the function's last line, decorators excluded, nested definitions included.",
    "code_lines": "Lines containing at least one code token. Docstrings and every line of a multi-line string count as code.",
    "blank_lines": "Lines with no token at all (only whitespace), and not inside a multi-line string.",
    "comment_lines": "Lines containing a # comment, whether alone on the line or after code.",
    "docstring_lines": "Lines occupied by module, class or function docstrings (a subset of code_lines).",
    "comment_ratio": "comment_lines / non-blank lines (code lines + comment-only lines); 0.0 when there are none. Docstrings are not comments.",
    # Definitions
    "function_count": "File: every def / async def (functions, methods, nested). Function: defs directly in its own scope.",
    "class_count": "File: every class statement. Function: class statements directly in its own scope.",
    "average_function_length": "Mean function_length over all functions in the file; 0.0 when there are none.",
    "maximum_function_length": "Largest function_length in the file; 0 when there are no functions.",
    "parameter_count": "All parameters: positional-only, regular, *args, keyword-only and **kwargs, including self/cls.",
    "return_count": "return statements in scope.",
    # Control flow
    "if_count": "if statements, not counting elif branches.",
    "elif_count": "elif branches (detected by the 'elif' keyword, so 'else: if' is an else plus an if).",
    "else_count": "else clauses of if/elif chains, for loops, while loops and try statements.",
    "for_loop_count": "for and async for statements (comprehension clauses excluded).",
    "while_loop_count": "while statements.",
    "loop_count": "for_loop_count + while_loop_count.",
    "try_count": "try statements (try/except, try/finally, try/except*).",
    "except_count": "except and except* handlers.",
    "conditional_expression_count": "Conditional expressions: a if condition else b.",
    "comprehension_count": "List, set and dict comprehensions and generator expressions.",
    # Complexity
    "cyclomatic_complexity": "1 + decision points (see module docstring for the exact convention).",
    "average_function_cyclomatic_complexity": "Mean cyclomatic_complexity over all functions in the file; 0.0 when none.",
    "maximum_function_cyclomatic_complexity": "Largest function cyclomatic_complexity in the file; 0 when none.",
    "maximum_nesting_depth": "Deepest statement nesting inside if/elif/else, for, while, try/except/finally, with and match/case blocks. A statement directly in a function body or at module level has depth 0; elif does not add a level; each def/class body restarts at 0.",
    "average_nesting_depth": "Mean nesting depth over all statements in scope; 0.0 when there are none.",
    "nested_loop_count": "Loops lexically inside the body of another loop in the same scope (a loop in a loop's else clause is not nested).",
    "maximum_loop_nesting_depth": "Deepest chain of loops inside loops (1 = a single loop, 0 = no loops).",
    "recursive_call_count": "Direct self-calls: f(...) inside function f; self.m(...), cls.m(...) or ClassName.m(...) inside method m. File: the sum over all functions.",
    # Other
    "import_count": "Imported names: 'import os, sys' = 2, 'from a import b as c' = 1, 'from a import *' = 1.",
    "identifier_count": "Name occurrences: variable names (load/store/delete), parameter names (including lambda parameters) and attribute names in a.b.",
    "unique_identifier_count": "Distinct identifier strings among those occurrences.",
    "operator_count": "arithmetic + comparison + logical + bitwise operators.",
    "arithmetic_operator_count": "+ - * / // % ** @ in binary and augmented assignments (+=, ...), plus unary + and -.",
    "comparison_operator_count": "Each operator in a comparison: == != < <= > >= in, not in, is, is not ('a < b < c' = 2).",
    "logical_operator_count": "Each and/or keyword ('a and b and c' = 2) plus each not.",
    "bitwise_operator_count": "& | ^ << >> in binary and augmented assignments, plus unary ~.",
    "function_call_count": "Call expressions of any kind (functions, methods, constructors), including those inside lambdas and comprehensions.",
    "collection_literal_count": "List, tuple, set and dict displays, excluding unpacking targets ('a, b = x' has none) and comprehensions.",
}

# --------------------------------------------------------------------------- #
# Result
# --------------------------------------------------------------------------- #
@dataclass
class FeatureReport:
    success: bool
    file_features: Optional[dict] = None
    function_features: list[dict] = field(default_factory=list)
    error: Optional[ParseError] = None
    warnings: list[str] = field(default_factory=list)
    parse_result: Optional[ParseResult] = field(default=None, repr=False)


def extract_features(source: Union[str, ParseResult]) -> FeatureReport:
    """Compute file-level and function-level features.

    Accepts source text or an existing ``ParseResult`` (to avoid parsing
    twice). Invalid input never raises: ``success`` is False, ``error``
    explains why, and no feature values are produced (no zero-filled rows
    that could be mistaken for real measurements).
    """
    parsed = source if isinstance(source, ParseResult) else parse_source(source)
    if not parsed.success or parsed.tree is None:
        return FeatureReport(success=False, error=parsed.error, parse_result=parsed)

    tree = parsed.tree
    line_bytes = [ln.encode("utf-8") for ln in parsed.source.split("\n")]
    lines, line_warning = _LineTable.build(parsed.source, tree)
    warnings = list(parsed.warnings)
    if line_warning:
        warnings.append(line_warning)

    nodes_by_line = {
        n.lineno: n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
    }

    function_rows = []
    for info in parsed.functions:
        node = nodes_by_line[info.lineno]
        function_rows.append(_function_features(info, node, lines, line_bytes))

    file_row = _file_features(tree, lines, line_bytes, function_rows)
    return FeatureReport(
        success=True,
        file_features=file_row,
        function_features=function_rows,
        warnings=warnings,
        parse_result=parsed,
    )


# --------------------------------------------------------------------------- #
# Line metrics (tokenize)
# --------------------------------------------------------------------------- #
_NON_CODE_TOKENS = {
    tokenize.NL,
    tokenize.NEWLINE,
    tokenize.INDENT,
    tokenize.DEDENT,
    tokenize.ENDMARKER,
    tokenize.ENCODING,
    tokenize.COMMENT,
}


@dataclass(frozen=True)
class _LineTable:
    """Per-line flags, 1-based (index 0 unused)."""

    total: int
    code: tuple[bool, ...]
    comment: tuple[bool, ...]
    docstring: tuple[bool, ...]

    @classmethod
    def build(cls, source: str, tree: ast.Module) -> tuple["_LineTable", Optional[str]]:
        physical = source.split("\n")
        total = len(physical) - 1 if source.endswith("\n") else len(physical)
        size = len(physical) + 2
        code, comment, doc = [False] * size, [False] * size, [False] * size
        warning = None

        try:
            for tok in tokenize.generate_tokens(io.StringIO(source).readline):
                if tok.type == tokenize.COMMENT:
                    comment[tok.start[0]] = True
                elif tok.type not in _NON_CODE_TOKENS:
                    for row in range(tok.start[0], tok.end[0] + 1):
                        code[row] = True
        except (tokenize.TokenError, SyntaxError):  # pragma: no cover - ast accepted it
            code, comment = [False] * size, [False] * size
            for i, text in enumerate(physical, start=1):
                stripped = text.strip()
                if stripped.startswith("#"):
                    comment[i] = True
                elif stripped:
                    code[i] = True
            warning = "Line metrics used a simplified fallback (tokenizer failed)."

        for node in ast.walk(tree):
            if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                body = node.body
                if (
                    body
                    and isinstance(body[0], ast.Expr)
                    and isinstance(body[0].value, ast.Constant)
                    and isinstance(body[0].value.value, str)
                ):
                    first = body[0]
                    for row in range(first.lineno, (first.end_lineno or first.lineno) + 1):
                        doc[row] = True

        return cls(total, tuple(code), tuple(comment), tuple(doc)), warning

    def summarise(self, start: int, end: int) -> dict:
        code = comment = doc = blank = comment_only = 0
        for row in range(start, end + 1):
            has_code, has_comment = self.code[row], self.comment[row]
            code += has_code
            comment += has_comment
            doc += self.docstring[row]
            if not has_code and not has_comment:
                blank += 1
            elif has_comment and not has_code:
                comment_only += 1
        non_blank = code + comment_only
        return {
            "code_lines": code,
            "blank_lines": blank,
            "comment_lines": comment,
            "docstring_lines": doc,
            "comment_ratio": round(comment / non_blank, 4) if non_blank else 0.0,
        }


# --------------------------------------------------------------------------- #
# Feature assembly
# --------------------------------------------------------------------------- #
def _function_features(
    info: FunctionInfo, node: ast.AST, lines: _LineTable, line_bytes: list[bytes]
) -> dict:
    target = _RecursionTarget.for_function(info)
    roots = [(a, 0, 0, False) for a in _arg_nodes(node.args)]
    roots += [(stmt, 0, 0, False) for stmt in node.body]
    m = _walk(roots, include_nested=False, line_bytes=line_bytes, recursion=target)

    values = {
        "function_length": info.end_lineno - info.lineno + 1,
        **lines.summarise(info.lineno, info.end_lineno),
        "parameter_count": len(info.parameters),
        **m,
    }
    row = {
        "qualname": info.qualname,
        "name": info.name,
        "kind": info.kind,
        "lineno": info.lineno,
        "end_lineno": info.end_lineno,
    }
    row.update({k: values[k] for k in FUNCTION_FEATURE_NAMES})
    return row


def _file_features(
    tree: ast.Module, lines: _LineTable, line_bytes: list[bytes], function_rows: list[dict]
) -> dict:
    roots = [(stmt, 0, 0, False) for stmt in tree.body]
    m = _walk(roots, include_nested=True, line_bytes=line_bytes, recursion=None)

    lengths = [r["function_length"] for r in function_rows]
    ccs = [r["cyclomatic_complexity"] for r in function_rows]
    values = {
        "total_lines": lines.total,
        **lines.summarise(1, lines.total),
        **m,
        "average_function_length": _mean(lengths),
        "maximum_function_length": max(lengths, default=0),
        "average_function_cyclomatic_complexity": _mean(ccs),
        "maximum_function_cyclomatic_complexity": max(ccs, default=0),
        "recursive_call_count": sum(r["recursive_call_count"] for r in function_rows),
    }
    return {k: values[k] for k in FILE_FEATURE_NAMES}


def _mean(values: list[int]) -> float:
    return round(sum(values) / len(values), 4) if values else 0.0


# --------------------------------------------------------------------------- #
# AST walk
# --------------------------------------------------------------------------- #
_FUNC_TYPES = (ast.FunctionDef, ast.AsyncFunctionDef)
_LOOP_TYPES = (ast.For, ast.AsyncFor, ast.While)
_TRY_TYPES = tuple(t for t in (getattr(ast, "Try", None), getattr(ast, "TryStar", None)) if t)
_MATCH = getattr(ast, "Match", None)
_MATCH_CASE = getattr(ast, "match_case", None)
_COMPREHENSION_TYPES = (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)
_ARITHMETIC_OPS = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow, ast.MatMult)
_BITWISE_OPS = (ast.LShift, ast.RShift, ast.BitOr, ast.BitXor, ast.BitAnd)

# Stack item: (node, nesting depth, loop depth, is_elif)
_Item = tuple[ast.AST, int, int, bool]


@dataclass(frozen=True)
class _RecursionTarget:
    name: str
    is_method: bool
    receivers: frozenset[str]

    @classmethod
    def for_function(cls, info: FunctionInfo) -> "_RecursionTarget":
        if info.kind != "method":
            return cls(info.name, False, frozenset())
        receivers = {info.class_name.split(".")[-1]} if info.class_name else set()
        is_static = any(d.split(".")[-1] == "staticmethod" for d in info.decorators)
        if info.parameters and not is_static and not info.parameters[0].startswith("*"):
            receivers.add(info.parameters[0])
        return cls(info.name, True, frozenset(receivers))

    def matches(self, call: ast.Call) -> bool:
        f = call.func
        if not self.is_method:
            return isinstance(f, ast.Name) and f.id == self.name
        return (
            isinstance(f, ast.Attribute)
            and f.attr == self.name
            and isinstance(f.value, ast.Name)
            and f.value.id in self.receivers
        )


def _arg_nodes(args: ast.arguments) -> list[ast.arg]:
    nodes = [*args.posonlyargs, *args.args]
    if args.vararg:
        nodes.append(args.vararg)
    nodes.extend(args.kwonlyargs)
    if args.kwarg:
        nodes.append(args.kwarg)
    return nodes


def _definition_time_exprs(node: ast.AST) -> list[ast.AST]:
    """Expressions a def/class statement evaluates in the *enclosing* scope."""
    exprs: list[ast.AST] = list(node.decorator_list)
    if isinstance(node, _FUNC_TYPES):
        a = node.args
        exprs += a.defaults
        exprs += [d for d in a.kw_defaults if d is not None]
        exprs += [p.annotation for p in _arg_nodes(a) if p.annotation is not None]
        if node.returns is not None:
            exprs.append(node.returns)
    else:  # ClassDef
        exprs += node.bases
        exprs += node.keywords
    return exprs


def _is_elif(orelse: list[ast.stmt], line_bytes: list[bytes]) -> bool:
    if len(orelse) != 1 or not isinstance(orelse[0], ast.If):
        return False
    node = orelse[0]
    line = line_bytes[node.lineno - 1] if node.lineno - 1 < len(line_bytes) else b""
    return line[node.col_offset : node.col_offset + 4] == b"elif"


def _is_wildcard_case(case: ast.AST) -> bool:
    p = case.pattern
    return (
        case.guard is None
        and isinstance(p, getattr(ast, "MatchAs", ()))
        and p.pattern is None
        and p.name is None
    )


def _walk(
    roots: list[_Item],
    *,
    include_nested: bool,
    line_bytes: list[bytes],
    recursion: Optional[_RecursionTarget],
) -> dict:
    """Iterative traversal (no Python recursion, so deep trees are safe).

    include_nested=False: stop at nested def/class bodies (function level).
    include_nested=True:  descend into them, restarting depth at 0 (file level).
    Each node is pushed exactly once, so nothing is counted twice.
    """
    c: Counter = Counter()
    statement_depths: list[int] = []
    identifiers: set[str] = set()
    max_loop_depth = 0
    decisions = 0

    stack: list[_Item] = list(reversed(roots))
    while stack:
        node, depth, loop_depth, is_elif = stack.pop()
        children: list[_Item] = []

        if isinstance(node, ast.stmt):
            statement_depths.append(depth)

        # ---- definitions ------------------------------------------------ #
        if isinstance(node, (*_FUNC_TYPES, ast.ClassDef)):
            c["function_count" if isinstance(node, _FUNC_TYPES) else "class_count"] += 1
            children += [(e, depth, loop_depth, False) for e in _definition_time_exprs(node)]
            if include_nested:
                if isinstance(node, _FUNC_TYPES):
                    children += [(a, 0, 0, False) for a in _arg_nodes(node.args)]
                children += [(s, 0, 0, False) for s in node.body]

        # ---- control flow ----------------------------------------------- #
        elif isinstance(node, ast.If):
            c["elif_count" if is_elif else "if_count"] += 1
            decisions += 1
            children.append((node.test, depth, loop_depth, False))
            children += [(s, depth + 1, loop_depth, False) for s in node.body]
            if _is_elif(node.orelse, line_bytes):
                children.append((node.orelse[0], depth, loop_depth, True))
            elif node.orelse:
                c["else_count"] += 1
                children += [(s, depth + 1, loop_depth, False) for s in node.orelse]

        elif isinstance(node, _LOOP_TYPES):
            c["while_loop_count" if isinstance(node, ast.While) else "for_loop_count"] += 1
            decisions += 1
            if loop_depth > 0:
                c["nested_loop_count"] += 1
            max_loop_depth = max(max_loop_depth, loop_depth + 1)
            if isinstance(node, ast.While):
                children.append((node.test, depth, loop_depth, False))
            else:
                children += [(node.target, depth, loop_depth, False), (node.iter, depth, loop_depth, False)]
            children += [(s, depth + 1, loop_depth + 1, False) for s in node.body]
            if node.orelse:
                c["else_count"] += 1
                children += [(s, depth + 1, loop_depth, False) for s in node.orelse]

        elif isinstance(node, _TRY_TYPES):
            c["try_count"] += 1
            children += [(s, depth + 1, loop_depth, False) for s in node.body]
            children += [(h, depth, loop_depth, False) for h in node.handlers]
            if node.orelse:
                c["else_count"] += 1
            children += [(s, depth + 1, loop_depth, False) for s in node.orelse]
            children += [(s, depth + 1, loop_depth, False) for s in node.finalbody]

        elif isinstance(node, ast.ExceptHandler):
            c["except_count"] += 1
            decisions += 1
            if node.type is not None:
                children.append((node.type, depth, loop_depth, False))
            children += [(s, depth + 1, loop_depth, False) for s in node.body]

        elif isinstance(node, (ast.With, ast.AsyncWith)):
            children += [(item, depth, loop_depth, False) for item in node.items]
            children += [(s, depth + 1, loop_depth, False) for s in node.body]

        elif _MATCH is not None and isinstance(node, _MATCH):
            children.append((node.subject, depth, loop_depth, False))
            children += [(case, depth, loop_depth, False) for case in node.cases]

        elif _MATCH_CASE is not None and isinstance(node, _MATCH_CASE):
            if not _is_wildcard_case(node):
                decisions += 1
            children.append((node.pattern, depth, loop_depth, False))
            if node.guard is not None:
                children.append((node.guard, depth, loop_depth, False))
            children += [(s, depth + 1, loop_depth, False) for s in node.body]

        elif isinstance(node, ast.arg):
            c["identifier_count"] += 1
            identifiers.add(node.arg)
            # annotation is evaluated by the def statement (enclosing scope)

        # ---- everything else: count, then visit all children -------------- #
        else:
            decisions += _count_simple(node, c, identifiers, recursion)
            children = [(ch, depth, loop_depth, False) for ch in ast.iter_child_nodes(node)]

        stack.extend(reversed(children))

    c["loop_count"] = c["for_loop_count"] + c["while_loop_count"]
    c["operator_count"] = (
        c["arithmetic_operator_count"]
        + c["comparison_operator_count"]
        + c["logical_operator_count"]
        + c["bitwise_operator_count"]
    )
    result = {name: c[name] for name in _COUNTED}
    result.update(
        cyclomatic_complexity=1 + decisions,
        maximum_nesting_depth=max(statement_depths, default=0),
        average_nesting_depth=_mean(statement_depths),
        maximum_loop_nesting_depth=max_loop_depth,
        unique_identifier_count=len(identifiers),
    )
    return result


_COUNTED = (
    "function_count",
    "class_count",
    "return_count",
    "if_count",
    "elif_count",
    "else_count",
    "for_loop_count",
    "while_loop_count",
    "loop_count",
    "try_count",
    "except_count",
    "conditional_expression_count",
    "comprehension_count",
    "nested_loop_count",
    "recursive_call_count",
    "import_count",
    "identifier_count",
    "operator_count",
    "arithmetic_operator_count",
    "comparison_operator_count",
    "logical_operator_count",
    "bitwise_operator_count",
    "function_call_count",
    "collection_literal_count",
)


def _count_simple(
    node: ast.AST, c: Counter, identifiers: set[str], recursion: Optional[_RecursionTarget]
) -> int:
    """Count one non-compound node. Returns the decision points it adds."""
    if isinstance(node, ast.Name):
        c["identifier_count"] += 1
        identifiers.add(node.id)
    elif isinstance(node, ast.Attribute):
        c["identifier_count"] += 1
        identifiers.add(node.attr)
    elif isinstance(node, ast.Call):
        c["function_call_count"] += 1
        if recursion is not None and recursion.matches(node):
            c["recursive_call_count"] += 1
    elif isinstance(node, (ast.BinOp, ast.AugAssign)):
        if isinstance(node.op, _ARITHMETIC_OPS):
            c["arithmetic_operator_count"] += 1
        elif isinstance(node.op, _BITWISE_OPS):
            c["bitwise_operator_count"] += 1
    elif isinstance(node, ast.UnaryOp):
        if isinstance(node.op, ast.Not):
            c["logical_operator_count"] += 1
        elif isinstance(node.op, ast.Invert):
            c["bitwise_operator_count"] += 1
        else:  # UAdd, USub
            c["arithmetic_operator_count"] += 1
    elif isinstance(node, ast.BoolOp):
        k = len(node.values) - 1
        c["logical_operator_count"] += k
        return k
    elif isinstance(node, ast.Compare):
        c["comparison_operator_count"] += len(node.ops)
    elif isinstance(node, ast.IfExp):
        c["conditional_expression_count"] += 1
        return 1
    elif isinstance(node, ast.comprehension):
        return 1 + len(node.ifs)
    elif isinstance(node, _COMPREHENSION_TYPES):
        c["comprehension_count"] += 1
    elif isinstance(node, (ast.List, ast.Tuple)):
        if isinstance(node.ctx, ast.Load):
            c["collection_literal_count"] += 1
    elif isinstance(node, (ast.Set, ast.Dict)):
        c["collection_literal_count"] += 1
    elif isinstance(node, ast.Return):
        c["return_count"] += 1
    elif isinstance(node, (ast.Import, ast.ImportFrom)):
        c["import_count"] += len(node.names)
    return 0
