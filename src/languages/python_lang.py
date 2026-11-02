"""Python mapped onto the common schema.

A thin adapter over the existing, unchanged Python pipeline: it reads the
values that ``feature_extractor`` and ``algorithm_analyzer`` already compute.
The full Python analysis (ML, Derived Complexity Score, recommendations,
explanations) stays in ``src.predictor``; this adapter exists so repository
summaries can treat Python and other languages alike.
"""

from __future__ import annotations

from src.code_parser import parse_bytes, parse_source
from src.languages.base import Capabilities, FunctionMetrics, Issue, LanguageAnalyzer, LanguageReport, failure

FILE_FIELD_SOURCE = {  # common field -> feature_extractor file feature
    "total_lines": "total_lines", "code_lines": "code_lines", "comment_lines": "comment_lines",
    "blank_lines": "blank_lines", "function_count": "function_count", "loop_count": "loop_count",
    "cyclomatic_complexity": "cyclomatic_complexity",
    "maximum_function_cyclomatic_complexity": "maximum_function_cyclomatic_complexity",
    "maximum_nesting_depth": "maximum_nesting_depth", "maximum_loop_nesting_depth": "maximum_loop_nesting_depth",
    "recursive_call_count": "recursive_call_count", "function_call_count": "function_call_count",
    "logical_operator_count": "logical_operator_count",
}


def _conditionals(row: dict) -> int:
    return row["if_count"] + row["elif_count"] + row["conditional_expression_count"]


class PythonAnalyzer(LanguageAnalyzer):
    language_id = "python"
    display_name = "Python"
    extensions = (".py",)
    capabilities = Capabilities(
        ml_prediction=True, static_big_o=True, derived_score=True, recommendations=True,
        notes=("Full analysis: structural metrics, rule-based Big-O, ML class, Derived Complexity Score and "
               "refactoring recommendations.",))

    def __init__(self, include_big_o: bool = True):
        self.include_big_o = include_big_o

    def analyze(self, source: str) -> LanguageReport:
        return self._from_parse(parse_source(source))

    def analyze_bytes(self, data: bytes) -> LanguageReport:
        return self._from_parse(parse_bytes(data))

    def _from_parse(self, parsed) -> LanguageReport:
        from src.feature_extractor import extract_features

        if not parsed.success:
            e = parsed.error
            return failure(self.language_id, Issue(e.kind, e.message, e.hint, e.lineno, e.col), self.capabilities)
        report = extract_features(parsed)
        if not report.success:  # pragma: no cover - extract_features only fails on parse errors
            return failure(self.language_id, Issue("analysis_failed", "Features could not be extracted."),
                           self.capabilities)
        ff = report.file_features
        file_metrics = {k: ff[src] for k, src in FILE_FIELD_SOURCE.items()}
        file_metrics["conditional_count"] = _conditionals(ff)
        times = {}
        if self.include_big_o:
            from src.algorithm_analyzer import analyze_algorithms

            a = analyze_algorithms(parsed)
            if a.success:
                times = {e.function_id: e.time_class for e in a.estimates}
        functions = []
        for row in report.function_features:
            functions.append(FunctionMetrics(
                name=row["name"], qualname=row["qualname"], kind=row["kind"], start_line=row["lineno"],
                end_line=row["end_lineno"], function_length=row["function_length"],
                parameter_count=row["parameter_count"], loop_count=row["loop_count"],
                conditional_count=_conditionals(row), cyclomatic_complexity=row["cyclomatic_complexity"],
                maximum_nesting_depth=row["maximum_nesting_depth"],
                maximum_loop_nesting_depth=row["maximum_loop_nesting_depth"],
                recursive_call_count=row["recursive_call_count"], function_call_count=row["function_call_count"],
                return_count=row["return_count"], static_time=times.get(row["qualname"]),
            ))
        warnings = [Issue("parser_warning", str(w)) for w in parsed.warnings or []]
        return LanguageReport(self.language_id, True, file_metrics, functions, [], warnings, self.capabilities)
