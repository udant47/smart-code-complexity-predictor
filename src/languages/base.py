"""Common, language-neutral schema and the interface every language analyzer implements (Phase 19).

The Python pipeline (code_parser → feature_extractor → function_analyzer →
algorithm_analyzer → predictor) is unchanged. This package sits beside it:
each language has its own parser and maps the constructs it supports onto the
same field names, with the same meaning, so that reports and repository
summaries can compare functions across languages.

Only *structural* metrics are shared. ML prediction, static Big-O, the
Derived Complexity Score and the refactoring rules were built and calibrated
on Python, so a language declares which of them it supports in
``Capabilities`` and the UI shows only those.

Adding a language
-----------------
1. Write ``src/languages/<name>_lang.py`` with a subclass of
   ``LanguageAnalyzer`` that parses with a real parser for that language (never
   Python's ``ast``) and never executes or compiles the code.
2. Return a ``LanguageReport``: on any parse problem set ``success=False`` with
   an ``Issue`` (``empty``, ``too_large``, ``encoding``, ``invalid_characters``,
   ``syntax`` or ``unavailable``) instead of returning partial metrics.
3. Map each construct to the ``COMMON_FUNCTION_FIELDS`` definitions below and
   document any differences in ``docs/EXTENSIBILITY.md``.
4. Register it in ``src/languages/registry.py`` and add tests that check the
   documented values on small programs.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import Optional

from src import config
from src.dataset_loader import structural_band

# --------------------------------------------------------------------------- #
# Field definitions (the contract every language implements)
# --------------------------------------------------------------------------- #
COMMON_FUNCTION_FIELDS: dict[str, str] = {
    "name": "Function name as written.",
    "qualname": "Name qualified by enclosing classes/functions where the language has them (Python), else the name.",
    "kind": "function | method | nested_function.",
    "start_line": "1-based line of the definition (Python: the def line; C: the first line of the declaration).",
    "end_line": "1-based last line of the definition.",
    "function_length": "end_line - start_line + 1.",
    "parameter_count": "Declared parameters, including variadic ones (*args, **kwargs, ...). C `(void)` is 0.",
    "loop_count": "Loop statements in the function's own body (Python for/while; C for/while/do-while).",
    "conditional_count": "if statements, else-if/elif branches and conditional expressions (a if c else b, c ? a : b).",
    "cyclomatic_complexity": "1 + decision points: if, else-if/elif, loops, conditional expressions, each && / || "
                             "(and / or), each non-default case, each except handler, comprehension clauses (Python).",
    "maximum_nesting_depth": "Deepest nesting of control blocks (if, loops, switch/match, try, with). Statements "
                             "directly in the body are at 0; else-if/elif does not add a level; bare { } blocks "
                             "do not count.",
    "maximum_loop_nesting_depth": "Deepest chain of loops inside loops (1 = one loop, 0 = none).",
    "recursive_call_count": "Direct calls to the function itself by name (methods: self.m / cls.m in Python).",
    "function_call_count": "Call expressions. In C, invocations of function-like macros look like calls and are counted.",
    "return_count": "return statements.",
}

COMMON_FILE_FIELDS: dict[str, str] = {
    "total_lines": "Physical lines; a final newline does not start an extra line.",
    "code_lines": "Lines with at least one code token (C preprocessor directives count as code).",
    "comment_lines": "Lines containing a comment, alone or after code.",
    "blank_lines": "Lines with no token, outside comments and multi-line strings.",
    "function_count": "Function definitions in the file (C: function bodies, not prototypes).",
    "loop_count": "All loops in the file.",
    "conditional_count": "All conditionals in the file (definition as for functions).",
    "cyclomatic_complexity": "The whole file as one unit: 1 + every decision point.",
    "maximum_function_cyclomatic_complexity": "Highest function cyclomatic complexity (0 without functions).",
    "maximum_nesting_depth": "Deepest nesting anywhere in the file.",
    "maximum_loop_nesting_depth": "Deepest loop chain anywhere in the file.",
    "recursive_call_count": "Sum of recursive calls over all functions.",
    "function_call_count": "All call expressions in the file.",
    "logical_operator_count": "&&, ||, ! (C) or and, or, not (Python).",
}


@dataclass(frozen=True)
class Capabilities:
    structural_metrics: bool = True
    structural_category: bool = True  # Low/Medium/High from McCabe CC (radon thresholds), language-neutral
    ml_prediction: bool = False
    static_big_o: bool = False
    derived_score: bool = False
    recommendations: bool = False
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Issue:
    code: str
    message: str
    hint: Optional[str] = None
    line: Optional[int] = None
    column: Optional[int] = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class FunctionMetrics:
    name: str
    qualname: str
    kind: str
    start_line: int
    end_line: int
    function_length: int
    parameter_count: int
    loop_count: int
    conditional_count: int
    cyclomatic_complexity: int
    maximum_nesting_depth: int
    maximum_loop_nesting_depth: int
    recursive_call_count: int
    function_call_count: int
    return_count: int
    structural_category: str = ""
    static_time: Optional[str] = None  # only for languages with static_big_o
    extra: dict = field(default_factory=dict)  # language-specific counts (e.g. C switch/case/goto)

    def __post_init__(self):
        if not self.structural_category:
            self.structural_category = structural_band(self.cyclomatic_complexity)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class LanguageReport:
    language: str
    success: bool
    file_metrics: Optional[dict] = None  # keys exactly COMMON_FILE_FIELDS
    functions: list = field(default_factory=list)  # FunctionMetrics
    errors: list = field(default_factory=list)  # Issue
    warnings: list = field(default_factory=list)  # Issue
    capabilities: Capabilities = field(default_factory=Capabilities)

    @property
    def structural_category(self) -> Optional[str]:
        if not self.success:
            return None
        cc = max([f.cyclomatic_complexity for f in self.functions], default=None)
        return structural_band(cc if cc is not None else self.file_metrics["cyclomatic_complexity"])

    def to_dict(self) -> dict:
        return {"language": self.language, "success": self.success, "file_metrics": self.file_metrics,
                "structural_category": self.structural_category,
                "functions": [f.to_dict() for f in self.functions],
                "errors": [e.to_dict() for e in self.errors], "warnings": [w.to_dict() for w in self.warnings],
                "capabilities": self.capabilities.to_dict()}


def failure(language: str, issue: Issue, capabilities: Capabilities) -> LanguageReport:
    return LanguageReport(language=language, success=False, errors=[issue], capabilities=capabilities)


def check_text_limits(language: str, source: str, capabilities: Capabilities) -> Optional[LanguageReport]:
    """Shared input checks (same limits as the Python parser). None when the text is acceptable."""
    if not isinstance(source, str):
        raise TypeError("source must be str")
    if not source.strip():
        return failure(language, Issue("empty", "No code provided.", "Paste code or choose an example."), capabilities)
    try:
        size = len(source.encode("utf-8"))
    except UnicodeEncodeError as exc:
        line = source.count("\n", 0, exc.start) + 1
        return failure(language, Issue("encoding", f"The code contains an invalid character on line {line}.",
                                       "Re-copy the code from its original file.", line), capabilities)
    if size > config.MAX_SOURCE_BYTES:
        return failure(language, Issue("too_large", f"Input is too large: {size:,} bytes "
                                       f"(limit {config.MAX_SOURCE_BYTES:,} bytes).",
                                       "Analyse one file or a smaller excerpt at a time."), capabilities)
    lines = source.count("\n") + 1
    if lines > config.MAX_SOURCE_LINES:
        return failure(language, Issue("too_large", f"Input is too long: {lines:,} lines "
                                       f"(limit {config.MAX_SOURCE_LINES:,} lines).",
                                       "Analyse one file or a smaller excerpt at a time."), capabilities)
    if "\0" in source:
        line = source.count("\n", 0, source.index("\0")) + 1
        return failure(language, Issue("invalid_characters", f"The code contains a null byte on line {line}.",
                                       "Remove the null character.", line), capabilities)
    return None


class LanguageAnalyzer(ABC):
    """Interface for a language: parse with a real parser for that language, never execute anything."""

    language_id: str = ""
    display_name: str = ""
    extensions: tuple[str, ...] = ()
    capabilities: Capabilities = Capabilities()

    @classmethod
    def available(cls) -> tuple[bool, str]:
        """(True, "") when the parser can be used; otherwise (False, reason with an install hint)."""
        return True, ""

    @abstractmethod
    def analyze(self, source: str) -> LanguageReport:
        """Structural analysis of one file's text. Must not raise for bad input."""

    def analyze_bytes(self, data: bytes) -> LanguageReport:
        """Decode file bytes (UTF-8, optional BOM) and analyse them."""
        if data.startswith(b"\xef\xbb\xbf"):
            data = data[3:]
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError as exc:
            line = data.count(b"\n", 0, exc.start) + 1
            return failure(self.language_id, Issue("encoding", f"The file is not valid UTF-8 (line {line}).",
                                                   "Save the file as UTF-8.", line), self.capabilities)
        return self.analyze(text.replace("\r\n", "\n").replace("\r", "\n"))
