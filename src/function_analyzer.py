"""Function-level structural analysis: one record per function, plus a file summary.

This is the presentation layer over ``src.feature_extractor``. It turns the
full feature rows into compact, dashboard-ready records, gives every
function a stable identifier, and summarises the file.

**These are structural metrics, not Big-O predictions.** Two nested loops
raise the loop count, nesting depth and cyclomatic complexity, but this
module never concludes "O(n^2)" from that. Loop bounds, early exits and
data structures decide running time, which is the job of
``src.algorithm_analyzer`` and the ML model in later phases.

Public API
----------
analyze_functions(source_or_parse_result) -> FunctionAnalysis
    .functions  list[FunctionRecord], in source order
    .summary    FileSummary (None when the code could not be parsed)
records_to_rows(records, labels=True) -> list[dict]   # table rows
to_dataframe(analysis)                -> pandas.DataFrame
format_text_report(analysis)          -> str
DISPLAY_COLUMNS                        # field -> column label

Scope rules (inherited from feature_extractor)
----------------------------------------------
A function's metrics cover only its own body. Code inside a nested
function or nested class body is counted in the nested function's own
record, never in the enclosing one. Decorators and default values of a
nested ``def`` run in the enclosing function and are counted there.

Stable identifiers
------------------
``function_id`` is the dotted qualified name (``Graph.bfs``,
``outer.inner``). It does not change when unrelated code above it is
edited, unlike a line number. When several definitions share a qualified
name (a redefinition, a property getter and setter, or variants defined in
if/else branches), the first keeps the plain name and later ones get
``#2``, ``#3``, ... in source order.
"""

from __future__ import annotations

import ast
from dataclasses import asdict, dataclass, field
from typing import Optional, Union

from src.code_parser import ParseError, ParseResult
from src.feature_extractor import extract_features

__all__ = [
    "DISPLAY_COLUMNS",
    "FileSummary",
    "FunctionAnalysis",
    "FunctionRecord",
    "analyze_functions",
    "format_text_report",
    "records_to_rows",
    "to_dataframe",
]


@dataclass(frozen=True)
class FunctionRecord:
    """Structural metrics for one function, method or nested function."""

    function_id: str  # stable, unique within the file
    function_name: str
    qualname: str
    kind: str  # "function" | "method" | "nested_function"
    class_name: Optional[str]  # enclosing class for methods, else None
    parent: Optional[str]  # qualname of the enclosing def/class, if any
    is_async: bool
    is_empty: bool  # body is only a docstring, pass and/or ...
    start_line: int  # the def line (decorators not included)
    end_line: int
    function_length: int
    parameter_count: int
    loop_count: int  # for + async for + while statements
    conditional_count: int  # if + elif + conditional expressions
    cyclomatic_complexity: int
    maximum_nesting_depth: int
    recursive_call_count: int
    function_call_count: int

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class FileSummary:
    """File-wide totals and extremes, ready for metric cards."""

    total_lines: int
    code_lines: int
    comment_ratio: float
    class_count: int
    function_count: int  # all functions, methods and nested functions
    top_level_function_count: int
    method_count: int
    nested_function_count: int
    empty_function_count: int
    recursive_function_count: int  # functions with at least one self-call
    file_cyclomatic_complexity: int  # whole file treated as one unit
    average_cyclomatic_complexity: float  # mean over functions
    maximum_cyclomatic_complexity: int
    most_complex_function: Optional[str]  # function_id; first in source order on ties
    average_function_length: float
    longest_function: Optional[str]
    maximum_nesting_depth: int  # deepest nesting inside any function
    deepest_function: Optional[str]

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class FunctionAnalysis:
    success: bool
    functions: list[FunctionRecord] = field(default_factory=list)
    summary: Optional[FileSummary] = None
    error: Optional[ParseError] = None
    warnings: list[str] = field(default_factory=list)

    def get(self, function_id: str) -> Optional[FunctionRecord]:
        return next((f for f in self.functions if f.function_id == function_id), None)

    def by_name(self, name: str) -> list[FunctionRecord]:
        return [f for f in self.functions if f.function_name == name]


# Field -> column label for dashboard tables, in display order.
DISPLAY_COLUMNS: dict[str, str] = {
    "function_id": "Function",
    "kind": "Type",
    "class_name": "Class",
    "start_line": "Start line",
    "end_line": "End line",
    "function_length": "Lines",
    "parameter_count": "Parameters",
    "loop_count": "Loops",
    "conditional_count": "Conditionals",
    "cyclomatic_complexity": "Cyclomatic complexity",
    "maximum_nesting_depth": "Max nesting depth",
    "recursive_call_count": "Recursive calls",
    "function_call_count": "Function calls",
}

_KIND_LABELS = {"function": "Function", "method": "Method", "nested_function": "Nested function"}


# --------------------------------------------------------------------------- #
# Analysis
# --------------------------------------------------------------------------- #
def analyze_functions(source: Union[str, ParseResult]) -> FunctionAnalysis:
    """Analyse every function in ``source`` (text or a ParseResult).

    Never raises for bad input; check ``success`` and ``error``.
    """
    report = extract_features(source)
    if not report.success:
        return FunctionAnalysis(success=False, error=report.error, warnings=report.warnings)

    parsed = report.parse_result
    infos = {f.lineno: f for f in parsed.functions}
    nodes = {
        n.lineno: n
        for n in ast.walk(parsed.tree)
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
    }

    seen: dict[str, int] = {}
    records: list[FunctionRecord] = []
    for row in report.function_features:
        info = infos[row["lineno"]]
        qualname = row["qualname"]
        seen[qualname] = seen.get(qualname, 0) + 1
        function_id = qualname if seen[qualname] == 1 else f"{qualname}#{seen[qualname]}"

        records.append(
            FunctionRecord(
                function_id=function_id,
                function_name=row["name"],
                qualname=qualname,
                kind=row["kind"],
                class_name=info.class_name,
                parent=info.parent,
                is_async=info.is_async,
                is_empty=_is_empty_body(nodes[row["lineno"]]),
                start_line=row["lineno"],
                end_line=row["end_lineno"],
                function_length=row["function_length"],
                parameter_count=row["parameter_count"],
                loop_count=row["loop_count"],
                conditional_count=(
                    row["if_count"] + row["elif_count"] + row["conditional_expression_count"]
                ),
                cyclomatic_complexity=row["cyclomatic_complexity"],
                maximum_nesting_depth=row["maximum_nesting_depth"],
                recursive_call_count=row["recursive_call_count"],
                function_call_count=row["function_call_count"],
            )
        )

    return FunctionAnalysis(
        success=True,
        functions=records,
        summary=_summarise(records, report.file_features),
        warnings=report.warnings,
    )


def _is_empty_body(node: ast.AST) -> bool:
    body = list(node.body)
    if body and _is_str_constant(body[0]):
        body = body[1:]  # docstring
    return all(
        isinstance(s, ast.Pass)
        or (isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant) and s.value.value is Ellipsis)
        for s in body
    )


def _is_str_constant(stmt: ast.stmt) -> bool:
    return (
        isinstance(stmt, ast.Expr)
        and isinstance(stmt.value, ast.Constant)
        and isinstance(stmt.value.value, str)
    )


def _summarise(records: list[FunctionRecord], file_features: dict) -> FileSummary:
    def top(key: str) -> Optional[str]:
        # max() keeps the first maximum, i.e. the earliest in source order
        return max(records, key=lambda r: getattr(r, key)).function_id if records else None

    def mean(key: str) -> float:
        return round(sum(getattr(r, key) for r in records) / len(records), 4) if records else 0.0

    def peak(key: str) -> int:
        return max((getattr(r, key) for r in records), default=0)

    return FileSummary(
        total_lines=file_features["total_lines"],
        code_lines=file_features["code_lines"],
        comment_ratio=file_features["comment_ratio"],
        class_count=file_features["class_count"],
        function_count=len(records),
        top_level_function_count=sum(r.kind == "function" for r in records),
        method_count=sum(r.kind == "method" for r in records),
        nested_function_count=sum(r.kind == "nested_function" for r in records),
        empty_function_count=sum(r.is_empty for r in records),
        recursive_function_count=sum(r.recursive_call_count > 0 for r in records),
        file_cyclomatic_complexity=file_features["cyclomatic_complexity"],
        average_cyclomatic_complexity=mean("cyclomatic_complexity"),
        maximum_cyclomatic_complexity=peak("cyclomatic_complexity"),
        most_complex_function=top("cyclomatic_complexity"),
        average_function_length=mean("function_length"),
        longest_function=top("function_length"),
        maximum_nesting_depth=peak("maximum_nesting_depth"),
        deepest_function=top("maximum_nesting_depth"),
    )


# --------------------------------------------------------------------------- #
# Output helpers
# --------------------------------------------------------------------------- #
def records_to_rows(records: list[FunctionRecord], labels: bool = True) -> list[dict]:
    """Rows with exactly the DISPLAY_COLUMNS fields, in that order.

    ``labels=True`` uses human-readable column names and kind labels;
    ``labels=False`` keeps the field names (for CSV/JSON export).
    """
    rows = []
    for r in records:
        d = r.to_dict()
        if labels:
            d["kind"] = _KIND_LABELS.get(d["kind"], d["kind"])
            d["class_name"] = d["class_name"] or ""
            rows.append({DISPLAY_COLUMNS[k]: d[k] for k in DISPLAY_COLUMNS})
        else:
            rows.append({k: d[k] for k in DISPLAY_COLUMNS})
    return rows


def to_dataframe(analysis: FunctionAnalysis, labels: bool = True):
    """A pandas DataFrame for ``st.dataframe``; empty, with columns, if none."""
    import pandas as pd  # imported lazily so the analyser itself has no pandas dependency

    columns = list(DISPLAY_COLUMNS.values()) if labels else list(DISPLAY_COLUMNS)
    return pd.DataFrame(records_to_rows(analysis.functions, labels=labels), columns=columns)


def format_text_report(analysis: FunctionAnalysis) -> str:
    """Plain-text report (CLI, logs, plain-text export)."""
    if not analysis.success:
        return analysis.error.format() if analysis.error else "Analysis failed."
    if not analysis.functions:
        return "No functions found."
    blocks = []
    for r in analysis.functions:
        blocks.append(
            "\n".join(
                [
                    f"Function: {r.function_id}",
                    f"Cyclomatic complexity: {r.cyclomatic_complexity}",
                    f"Maximum nesting depth: {r.maximum_nesting_depth}",
                ]
            )
        )
    return "\n\n".join(blocks)
