"""Safe, static parsing of Python source code.

This module turns Python source text into an abstract syntax tree (AST) and
an inventory of the functions and classes it defines. It is the first stage
of the analysis pipeline and has no Streamlit dependency.

Safety
------
Submitted code is **never executed**. ``ast.parse`` only tokenises and
parses the text into a tree of node objects; it does not import, run or
evaluate anything. No ``exec``, ``eval``, ``compile``-and-run, imports of
the submitted code, subprocesses or shells are used here (a test in
``tests/test_code_parser.py`` enforces this for the whole ``src`` package).

Public API
----------
parse_source(source)      -> ParseResult    # for text (pasted code)
parse_bytes(data)         -> ParseResult    # for uploaded .py files
ParseResult.to_dict()     -> dict           # JSON-friendly summary

Positions
---------
``lineno`` / ``end_lineno`` are 1-based. ``col`` / ``end_col`` are 0-based
*character* offsets. (The AST itself stores UTF-8 *byte* offsets, which are
wrong for display when a line contains non-ASCII text, so they are converted.)

Named ``code_parser`` rather than ``parser`` to avoid confusion with the
removed standard-library ``parser`` module.
"""

from __future__ import annotations

import ast
import io
import sys
import tokenize
import warnings
from dataclasses import asdict, dataclass, field
from typing import Optional, Union

from src.config import MAX_SOURCE_BYTES, MAX_SOURCE_LINES

__all__ = [
    "ParseError",
    "FunctionInfo",
    "ClassInfo",
    "ParseResult",
    "parse_source",
    "parse_bytes",
]

# Error kinds returned in ParseError.kind - stable strings the UI can switch on.
ERR_EMPTY = "empty"
ERR_TOO_LARGE = "too_large"
ERR_SYNTAX = "syntax"
ERR_INDENTATION = "indentation"
ERR_INVALID_CHARACTERS = "invalid_characters"
ERR_ENCODING = "encoding"
ERR_TOO_COMPLEX = "too_complex"

_FuncNode = Union[ast.FunctionDef, ast.AsyncFunctionDef]


# --------------------------------------------------------------------------- #
# Result data structures
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ParseError:
    """A user-facing explanation of why the code could not be analysed."""

    kind: str
    message: str
    lineno: Optional[int] = None
    col: Optional[int] = None  # 1-based column for display, as editors show it
    line_text: Optional[str] = None
    hint: Optional[str] = None

    def format(self) -> str:
        """Multi-line message with the offending line and a caret."""
        parts = [self.message]
        if self.line_text is not None:
            parts.append(f"    {self.line_text}")
            if self.col and self.col > 0:
                parts.append("    " + " " * (self.col - 1) + "^")
        if self.hint:
            parts.append(f"Hint: {self.hint}")
        return "\n".join(parts)


@dataclass(frozen=True)
class FunctionInfo:
    """One ``def`` or ``async def`` found in the source."""

    name: str
    qualname: str  # dotted path, e.g. "Graph.bfs" or "outer.inner"
    kind: str  # "function" | "method" | "nested_function"
    is_async: bool
    lineno: int
    end_lineno: int
    col: int
    end_col: int
    depth: int  # number of enclosing defs/classes; 0 = module level
    parent: Optional[str]  # qualname of the enclosing def/class, if any
    class_name: Optional[str]  # enclosing class for methods
    decorators: tuple[str, ...]
    parameters: tuple[str, ...]
    source: str

    @property
    def line_count(self) -> int:
        return self.end_lineno - self.lineno + 1

    @property
    def is_top_level(self) -> bool:
        return self.depth == 0


@dataclass(frozen=True)
class ClassInfo:
    """One ``class`` found in the source."""

    name: str
    qualname: str
    lineno: int
    end_lineno: int
    col: int
    end_col: int
    depth: int
    parent: Optional[str]
    bases: tuple[str, ...]
    methods: tuple[str, ...]  # names of methods defined directly in the body

    @property
    def is_top_level(self) -> bool:
        return self.depth == 0


@dataclass
class ParseResult:
    """Everything later pipeline stages need from the parsing step."""

    success: bool
    source: str = ""
    tree: Optional[ast.Module] = field(default=None, repr=False)
    functions: list[FunctionInfo] = field(default_factory=list)
    classes: list[ClassInfo] = field(default_factory=list)
    error: Optional[ParseError] = None
    warnings: list[str] = field(default_factory=list)
    line_count: int = 0
    char_count: int = 0

    # Convenience views ---------------------------------------------------- #
    @property
    def top_level_functions(self) -> list[FunctionInfo]:
        return [f for f in self.functions if f.kind == "function"]

    @property
    def methods(self) -> list[FunctionInfo]:
        return [f for f in self.functions if f.kind == "method"]

    @property
    def nested_functions(self) -> list[FunctionInfo]:
        return [f for f in self.functions if f.kind == "nested_function"]

    @property
    def top_level_classes(self) -> list[ClassInfo]:
        return [c for c in self.classes if c.is_top_level]

    def get_function(self, qualname: str) -> Optional[FunctionInfo]:
        return next((f for f in self.functions if f.qualname == qualname), None)

    def to_dict(self) -> dict:
        """JSON-serialisable summary (no AST, no full source)."""
        return {
            "success": self.success,
            "line_count": self.line_count,
            "char_count": self.char_count,
            "error": asdict(self.error) if self.error else None,
            "warnings": list(self.warnings),
            "classes": [asdict(c) for c in self.classes],
            "functions": [
                {k: v for k, v in asdict(f).items() if k != "source"}
                for f in self.functions
            ],
        }


# --------------------------------------------------------------------------- #
# Public functions
# --------------------------------------------------------------------------- #
def parse_source(
    source: str,
    *,
    filename: str = "<submitted>",
    max_bytes: int = MAX_SOURCE_BYTES,
    max_lines: int = MAX_SOURCE_LINES,
) -> ParseResult:
    """Parse Python source text without executing it.

    Never raises for bad *input*: empty, oversized, or syntactically invalid
    code is reported through ``ParseResult.error``. Passing a non-string is a
    programming error and raises ``TypeError``.
    """
    if not isinstance(source, str):
        raise TypeError(
            f"source must be str, not {type(source).__name__}; "
            "use parse_bytes() for raw file contents"
        )

    source = _normalise(source)

    # 1. Empty input ------------------------------------------------------- #
    if not source.strip():
        return _failure(
            source,
            ParseError(
                kind=ERR_EMPTY,
                message="No code provided.",
                hint="Paste Python code or upload a .py file.",
            ),
        )

    # 2. Size limits - checked BEFORE parsing so huge input costs nothing ---- #
    try:
        size = len(source.encode("utf-8"))
    except UnicodeEncodeError as exc:  # lone surrogates (e.g. from a broken paste) are not valid text
        line = source.count("\n", 0, exc.start) + 1
        return _failure(
            source,
            ParseError(
                kind=ERR_ENCODING,
                message=f"The code contains an invalid character (unpaired surrogate) on line {line}.",
                hint="Re-copy the code from its original file, or delete the broken character.",
                lineno=line,
            ),
        )
    if size > max_bytes:
        return _failure(
            source,
            ParseError(
                kind=ERR_TOO_LARGE,
                message=(
                    f"Input is too large: {size:,} bytes "
                    f"(limit {max_bytes:,} bytes)."
                ),
                hint="Analyse one module or a smaller excerpt at a time.",
            ),
        )
    line_count = source.count("\n") + 1
    if line_count > max_lines:
        return _failure(
            source,
            ParseError(
                kind=ERR_TOO_LARGE,
                message=(
                    f"Input is too long: {line_count:,} lines "
                    f"(limit {max_lines:,} lines)."
                ),
                hint="Analyse one module or a smaller excerpt at a time.",
            ),
        )

    # 3. Parse (static only) ---------------------------------------------- #
    lines = source.split("\n")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            tree = ast.parse(source, filename=filename, mode="exec")
        except SyntaxError as exc:  # includes IndentationError, TabError
            return _failure(source, _syntax_error(exc, lines))
        except ValueError as exc:  # null bytes on Python < 3.12
            return _failure(source, _invalid_characters(str(exc)))
        except (RecursionError, MemoryError):
            return _failure(source, _too_complex())

    parse_warnings = [_format_warning(w) for w in caught]

    # 4. Inventory functions and classes ---------------------------------- #
    try:
        functions, classes = _collect_definitions(tree, source, lines)
    except RecursionError:
        return _failure(source, _too_complex())

    if not functions and not classes:
        parse_warnings.append("No functions or classes were found in this code.")

    return ParseResult(
        success=True,
        source=source,
        tree=tree,
        functions=functions,
        classes=classes,
        warnings=parse_warnings,
        line_count=line_count,
        char_count=len(source),
    )


def parse_bytes(
    data: bytes,
    *,
    filename: str = "<uploaded>",
    max_bytes: int = MAX_SOURCE_BYTES,
    max_lines: int = MAX_SOURCE_LINES,
) -> ParseResult:
    """Decode raw file bytes (e.g. a Streamlit upload) and parse them.

    Encoding follows Python's own rules (PEP 263): UTF-8 by default, a BOM or
    a ``# -*- coding: ... -*-`` comment is honoured.
    """
    if not isinstance(data, (bytes, bytearray)):
        raise TypeError(f"data must be bytes, not {type(data).__name__}")

    if len(data) > max_bytes:
        return _failure(
            "",
            ParseError(
                kind=ERR_TOO_LARGE,
                message=(
                    f"File is too large: {len(data):,} bytes "
                    f"(limit {max_bytes:,} bytes)."
                ),
                hint="Analyse one module or a smaller excerpt at a time.",
            ),
        )

    try:
        encoding, _ = tokenize.detect_encoding(io.BytesIO(bytes(data)).readline)
        text = bytes(data).decode(encoding)
    except (SyntaxError, LookupError, UnicodeDecodeError) as exc:
        return _failure(
            "",
            ParseError(
                kind=ERR_ENCODING,
                message=f"Could not read the file as text: {exc}",
                hint="Save the file as UTF-8 and upload it again.",
            ),
        )

    return parse_source(
        text, filename=filename, max_bytes=max_bytes, max_lines=max_lines
    )


# --------------------------------------------------------------------------- #
# Definition collection
# --------------------------------------------------------------------------- #
def _collect_definitions(
    tree: ast.Module, source: str, lines: list[str]
) -> tuple[list[FunctionInfo], list[ClassInfo]]:
    """Walk the tree once, tracking the chain of enclosing defs/classes.

    Iterative (explicit stack) so very deep trees cannot overflow Python's
    recursion limit inside our own code.
    """
    functions: list[FunctionInfo] = []
    classes: list[ClassInfo] = []
    # Lines with their endings, computed once (ast.get_source_segment re-splits the whole
    # source on every call, which is quadratic for files with many functions).
    kept = [ln + "\n" for ln in lines[:-1]] + lines[-1:]

    # Stack items: (node, scope) where scope is a tuple of enclosing
    # (kind, qualname) pairs; kind is "class" or "function".
    stack: list[tuple[ast.AST, tuple[tuple[str, str], ...]]] = [
        (child, ()) for child in reversed(tree.body)
    ]

    while stack:
        node, scope = stack.pop()
        parent_kind, parent_qualname = scope[-1] if scope else (None, None)

        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            qualname = _join(parent_qualname, node.name)
            if parent_kind == "class":
                kind = "method"
                class_name = parent_qualname
            elif parent_kind == "function":
                kind = "nested_function"
                class_name = None
            else:
                kind = "function"
                class_name = None

            col, end_col = _char_cols(node, lines)
            functions.append(
                FunctionInfo(
                    name=node.name,
                    qualname=qualname,
                    kind=kind,
                    is_async=isinstance(node, ast.AsyncFunctionDef),
                    lineno=node.lineno,
                    end_lineno=node.end_lineno or node.lineno,
                    col=col,
                    end_col=end_col,
                    depth=len(scope),
                    parent=parent_qualname,
                    class_name=class_name,
                    decorators=tuple(_unparse(d) for d in node.decorator_list),
                    parameters=_parameter_names(node),
                    source=_segment(kept, node),
                )
            )
            new_scope = scope + (("function", qualname),)

        elif isinstance(node, ast.ClassDef):
            qualname = _join(parent_qualname, node.name)
            col, end_col = _char_cols(node, lines)
            classes.append(
                ClassInfo(
                    name=node.name,
                    qualname=qualname,
                    lineno=node.lineno,
                    end_lineno=node.end_lineno or node.lineno,
                    col=col,
                    end_col=end_col,
                    depth=len(scope),
                    parent=parent_qualname,
                    bases=tuple(_unparse(b) for b in node.bases),
                    methods=tuple(
                        b.name
                        for b in node.body
                        if isinstance(b, (ast.FunctionDef, ast.AsyncFunctionDef))
                    ),
                )
            )
            new_scope = scope + (("class", qualname),)
        else:
            new_scope = scope  # if/for/try/with blocks do not create scopes

        # Children in source order (stack is LIFO, so push reversed).
        stack.extend((child, new_scope) for child in reversed(_children(node)))

    functions.sort(key=lambda f: (f.lineno, f.col))
    classes.sort(key=lambda c: (c.lineno, c.col))
    return functions, classes


def _children(node: ast.AST) -> list[ast.AST]:
    return list(ast.iter_child_nodes(node))


def _join(parent: Optional[str], name: str) -> str:
    return f"{parent}.{name}" if parent else name


def _parameter_names(node: _FuncNode) -> tuple[str, ...]:
    a = node.args
    names = [p.arg for p in (*a.posonlyargs, *a.args)]
    if a.vararg:
        names.append("*" + a.vararg.arg)
    names.extend(p.arg for p in a.kwonlyargs)
    if a.kwarg:
        names.append("**" + a.kwarg.arg)
    return tuple(names)


def _unparse(node: ast.AST) -> str:
    try:
        return ast.unparse(node)
    except Exception:  # pragma: no cover - defensive; unparse is pure
        return type(node).__name__


def _char_cols(node: ast.AST, lines: list[str]) -> tuple[int, int]:
    """Convert the AST's UTF-8 byte columns into character columns."""
    return (
        _byte_to_char(lines, node.lineno, node.col_offset),
        _byte_to_char(
            lines, node.end_lineno or node.lineno, node.end_col_offset or 0
        ),
    )


def _byte_to_char(lines: list[str], lineno: int, byte_offset: int) -> int:
    if not 1 <= lineno <= len(lines):
        return byte_offset
    prefix = lines[lineno - 1].encode("utf-8")[:byte_offset]
    return len(prefix.decode("utf-8", errors="ignore"))


# --------------------------------------------------------------------------- #
# Errors and helpers
# --------------------------------------------------------------------------- #
def _segment(kept: list[str], node: ast.AST) -> str:
    """Same result as ``ast.get_source_segment(source, node)`` for '\\n'-normalised source,
    using lines split once (``kept`` keeps the line endings)."""
    try:
        lineno, end_lineno = node.lineno - 1, node.end_lineno - 1
        col, end_col = node.col_offset, node.end_col_offset
    except (AttributeError, TypeError):
        return ""
    if end_lineno is None or end_col is None or end_lineno >= len(kept):
        return ""
    if lineno == end_lineno:
        return kept[lineno].encode()[col:end_col].decode()
    first = kept[lineno].encode()[col:].decode()
    last = kept[end_lineno].encode()[:end_col].decode()
    return first + "".join(kept[lineno + 1:end_lineno]) + last


def _normalise(source: str) -> str:
    """Strip a leading BOM and use '\\n' line endings (Windows files use \\r\\n)."""
    if source.startswith("﻿"):
        source = source[1:]
    return source.replace("\r\n", "\n").replace("\r", "\n")


def _failure(source: str, error: ParseError) -> ParseResult:
    return ParseResult(
        success=False,
        source=source,
        error=error,
        line_count=(source.count("\n") + 1) if source else 0,
        char_count=len(source),
    )


def _syntax_error(exc: SyntaxError, lines: list[str]) -> ParseError:
    msg = exc.msg or "invalid syntax"
    if "null bytes" in msg:  # Python 3.12+ reports this as SyntaxError
        return _invalid_characters(msg)

    lineno = exc.lineno
    line_text = None
    if lineno and 1 <= lineno <= len(lines):
        line_text = lines[lineno - 1].rstrip()
    elif exc.text:
        line_text = exc.text.rstrip("\n")

    if isinstance(exc, IndentationError):  # TabError is a subclass
        kind = ERR_INDENTATION
        label = "Indentation error"
    else:
        kind = ERR_SYNTAX
        label = "Syntax error"

    where = f" on line {lineno}" if lineno else ""
    return ParseError(
        kind=kind,
        message=f"{label}{where}: {msg}",
        lineno=lineno,
        col=exc.offset,  # already 1-based characters
        line_text=line_text,
        hint=_hint_for(msg, kind),
    )


def _hint_for(msg: str, kind: str) -> Optional[str]:
    m = msg.lower()
    if ("print" in m and "parentheses" in m) or "multiple exception types" in m:
        return "This looks like Python 2 code. Only Python 3 is supported."
    if "tab" in m:
        return "Mixing tabs and spaces is not allowed. Use 4 spaces per level."
    if kind == ERR_INDENTATION:
        return "Check that each block is indented consistently (4 spaces per level)."
    if "never closed" in m or "unexpected eof" in m or "unmatched" in m:
        return "Check for an unclosed or extra bracket, parenthesis or quote."
    if "unterminated" in m:
        return "A string literal is missing its closing quote."
    if "expected ':'" in m:
        return "Statements like def, if, for, while and class end with a colon."
    if "too many nested" in m or "too many levels" in m:
        return "The code is nested too deeply to analyse."
    if kind == ERR_SYNTAX:
        return (
            "If this code runs on a newer Python, it may use syntax that this "
            f"app's Python {sys.version_info.major}.{sys.version_info.minor} "
            "does not support yet."
        )
    return None


def _invalid_characters(msg: str) -> ParseError:
    return ParseError(
        kind=ERR_INVALID_CHARACTERS,
        message=f"The code contains characters Python cannot parse ({msg}).",
        hint="Remove null bytes or binary content; upload a plain-text .py file.",
    )


def _too_complex() -> ParseError:
    return ParseError(
        kind=ERR_TOO_COMPLEX,
        message="The code is nested too deeply to analyse safely.",
        hint="Split deeply nested expressions or blocks into smaller pieces.",
    )


def _format_warning(w: warnings.WarningMessage) -> str:
    where = f"line {w.lineno}: " if w.lineno else ""
    return f"{w.category.__name__} ({where}{w.message})"
