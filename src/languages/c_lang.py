"""C structural analysis with tree-sitter (Phase 19, experimental).

Parser choice
-------------
* Python's ``ast`` cannot parse C and is never used for it.
* ``pycparser`` needs fully preprocessed input. That means running an external
  C preprocessor (a subprocess, forbidden by this project's security policy)
  and having the system headers available.
* ``libclang`` needs a native Clang installation, the headers and compile
  flags, which is heavy and fragile on Windows.
* Regular expressions cannot parse C.
* **tree-sitter** with the ``tree-sitter-c`` grammar is a real, tested C
  grammar. It ships as prebuilt wheels (Windows included), parses without
  preprocessing and in linear time, and marks every part of the input it could
  not parse with ``ERROR`` or ``MISSING`` nodes.

Reliability policy
------------------
tree-sitter recovers from errors, but metrics from a partly understood file
would be wrong without warning. So **any** ``ERROR`` or ``MISSING`` node makes
the whole file fail with the location, and no metrics are returned. Common
causes are macros that change syntax (``FOR(i, n) { ... }``), compiler
extensions, and C++ code.

The preprocessor is not run. ``#include``/``#define`` are read as directives,
function-like macro invocations look like function calls, and the code in every
``#if``/``#ifdef`` branch is counted. A warning says so when this applies.

Supported: structural metrics in the common schema and the structural
category. Not supported for C: ML class, static Big-O, Derived Complexity
Score and refactoring recommendations, which were built and calibrated on Python.

Nothing is compiled or executed: tree-sitter only builds a syntax tree.
"""

from __future__ import annotations

import re
from typing import Optional

from src.languages.base import (Capabilities, FunctionMetrics, Issue, LanguageAnalyzer, LanguageReport,
                                check_text_limits, failure)

NESTING = {"if_statement", "for_statement", "while_statement", "do_statement", "switch_statement"}
LOOPS = {"for_statement", "while_statement", "do_statement"}
DECLARATOR_WRAPPERS = {"pointer_declarator", "parenthesized_declarator", "attributed_declarator",
                       "function_declarator"}
PREPROC_CONDITIONALS = {"preproc_if", "preproc_ifdef", "preproc_elif", "preproc_elifdef", "preproc_else"}
# Constructs that cannot appear in C but are common in C++. Such input is refused rather than measured as C.
CPP_PATTERNS = (
    (r"\bnamespace\s+\w*\s*\{", "namespace"),
    (r"\btemplate\s*<", "template"),
    (r"\bstd::", "std::"),
    (r"^\s*#\s*include\s*<(iostream|vector|string|map|memory|algorithm|unordered_map)>", "a C++ standard header"),
    (r"\bclass\s+\w+\s*(:\s*(public|private|protected)\b|\{)", "class"),
    (r"^\s*(public|private|protected)\s*:", "an access specifier"),
    (r"\busing\s+namespace\b", "using namespace"),
)

_LANGUAGE = None


def _language():
    """The tree-sitter C Language object (loaded once); raises ImportError when not installed."""
    global _LANGUAGE
    if _LANGUAGE is None:
        import tree_sitter
        import tree_sitter_c

        _LANGUAGE = tree_sitter.Language(tree_sitter_c.language())
    return _LANGUAGE


def _parser():
    import tree_sitter

    return tree_sitter.Parser(_language())  # a new parser per call: parsers are not thread-safe


def _text(node) -> str:
    return node.text.decode("utf-8", errors="replace")


def _iter(node, stop_at: Optional[set] = None):
    """Pre-order traversal with an explicit stack (no recursion limit). Children of nodes whose type is in
    ``stop_at`` are not visited (the node itself is)."""
    stack = [node]
    while stack:
        n = stack.pop()
        yield n
        if stop_at and n.type in stop_at and n is not node:
            continue
        stack.extend(reversed(n.children))


def _strip_comments_and_strings(source: str) -> str:
    """Blank out comments and string/char literals (keeping newlines) so markers inside them are ignored."""
    pattern = r'//[^\n]*|/\*.*?\*/|"(?:\\.|[^"\\\n])*"|\'(?:\\.|[^\'\\\n])*\''
    return re.sub(pattern, lambda m: re.sub(r"[^\n]", " ", m.group(0)), source, flags=re.DOTALL)


def _first_problem(root):
    for n in _iter(root):
        if n.is_error or n.is_missing:
            return n
    return None


def _function_declarator(defn):
    """The innermost function_declarator whose declarator is the function's name (handles pointers to
    functions, attributes and functions returning function pointers)."""
    top = defn.child_by_field_name("declarator")
    stack = [top] if top is not None else []
    while stack:
        n = stack.pop()
        if n.type == "function_declarator":
            inner = n.child_by_field_name("declarator")
            if inner is not None and inner.type in ("identifier", "field_identifier"):
                return n, _text(inner)
        if n.type in DECLARATOR_WRAPPERS:
            stack.extend(reversed(n.named_children))
    return None, None


def _parameter_count(fdecl) -> int:
    params = fdecl.child_by_field_name("parameters")
    if params is None:
        return 0
    items = [c for c in params.named_children if c.type in ("parameter_declaration", "variadic_parameter",
                                                            "optional_parameter_declaration")]
    if len(items) == 1 and items[0].type == "parameter_declaration" and \
            items[0].child_by_field_name("declarator") is None and _text(items[0]).strip() == "void":
        return 0  # f(void) declares no parameters
    # A bare "..." may appear as an anonymous token rather than a node.
    variadic = sum(1 for c in params.children if not c.is_named and c.type == "...")
    return len(items) + variadic


def _is_decision(n) -> bool:
    t = n.type
    if t in ("if_statement", "for_statement", "while_statement", "do_statement", "conditional_expression"):
        return True
    if t == "case_statement":
        return n.child_by_field_name("value") is not None  # default: is not a decision
    if t == "binary_expression":
        op = n.child_by_field_name("operator")
        return op is not None and op.type in ("&&", "||")
    return False


def _is_logical_operator(n) -> bool:
    if n.type == "binary_expression":
        op = n.child_by_field_name("operator")
        return op is not None and op.type in ("&&", "||")
    if n.type == "unary_expression":
        op = n.child_by_field_name("operator")
        return op is not None and op.type == "!"
    return False


def _nesting(body) -> tuple[int, int]:
    """(maximum nesting depth, maximum loop nesting depth) inside a function body.

    Depth of a control construct = 1 + the number of control constructs around it. An ``if`` that is the
    whole ``else`` branch of another ``if`` (else-if) stays at that ``if``'s depth, like Python's elif.
    """
    max_depth = max_loop = 0
    stack = [(c, 0, 0) for c in body.children]
    while stack:
        n, depth, loops = stack.pop()
        if n.type == "function_definition":
            continue
        if n.type in NESTING:
            else_if = n.type == "if_statement" and n.parent is not None and n.parent.type == "else_clause"
            depth = depth if else_if else depth + 1
            max_depth = max(max_depth, depth)
        if n.type in LOOPS:
            loops += 1
            max_loop = max(max_loop, loops)
        stack.extend((c, depth, loops) for c in n.children)
    return max_depth, max_loop


def _line_metrics(root, source: str) -> dict:
    total = source.count("\n") + (0 if source.endswith("\n") else 1)
    code, comment = set(), set()
    for n in _iter(root):
        if n.child_count and n.type != "comment":
            continue
        rows = range(n.start_point[0], n.end_point[0] + 1)
        if n.type == "comment":
            comment.update(rows)
        elif n.type != "translation_unit":
            text = _text(n)
            if text.strip():
                code.update(rows)
    return {"total_lines": total, "code_lines": len(code), "comment_lines": len(comment),
            "blank_lines": total - len(code | comment)}


class CAnalyzer(LanguageAnalyzer):
    language_id = "c"
    display_name = "C"
    extensions = (".c", ".h")
    capabilities = Capabilities(
        ml_prediction=False, static_big_o=False, derived_score=False, recommendations=False,
        notes=("Experimental: structural metrics only (tree-sitter-c parser).",
               "The ML model, the static Big-O analyzer, the Derived Complexity Score and the refactoring rules "
               "were built and calibrated on Python, so they are not applied to C.",
               "The preprocessor is not run: macro invocations look like function calls, control flow inside "
               "macros is not counted, and code in every #if/#ifdef branch is counted."))

    @classmethod
    def available(cls) -> tuple[bool, str]:
        try:
            _language()
            return True, ""
        except ImportError:
            return False, ("The C parser is not installed. Install it with: "
                           "pip install tree-sitter==0.25.2 tree-sitter-c==0.24.1")
        except Exception as exc:  # incompatible versions
            return False, f"The C parser could not be loaded ({type(exc).__name__}). Reinstall the pinned versions."

    def analyze(self, source: str) -> LanguageReport:
        bad = check_text_limits(self.language_id, source, self.capabilities)
        if bad:
            return bad
        ok, reason = self.available()
        if not ok:
            return failure(self.language_id, Issue("unavailable", reason), self.capabilities)
        source = source.lstrip("\ufeff").replace("\r\n", "\n").replace("\r", "\n")
        cpp = self._cpp_marker(source)
        if cpp:
            return failure(self.language_id, Issue(
                "unsupported_language", f"This looks like C++ (it uses {cpp[0]} on line {cpp[1]}), which SmartCode "
                "does not support.", "Only C is supported. C++ needs its own parser and is not analysed.",
                cpp[1]), self.capabilities)
        tree = _parser().parse(source.encode("utf-8"))
        root = tree.root_node
        if root.has_error:
            return failure(self.language_id, self._syntax_issue(root, source), self.capabilities)
        return self._report(root, source)

    # ------------------------------------------------------------------ #
    @staticmethod
    def _cpp_marker(source: str) -> Optional[tuple[str, int]]:
        code = _strip_comments_and_strings(source)
        for pattern, label in CPP_PATTERNS:
            m = re.search(pattern, code, flags=re.MULTILINE)
            if m:
                return label, code.count("\n", 0, m.start()) + 1
        return None

    def _syntax_issue(self, root, source: str) -> Issue:
        n = _first_problem(root) or root
        line, col = n.start_point[0] + 1, n.start_point[1] + 1
        what = f"missing '{n.type}'" if n.is_missing else "unexpected code"
        hint = ("SmartCode only reports metrics for C it can parse completely. Macros that change the syntax "
                    "(for example a FOR(i, n) loop macro), compiler extensions or a real syntax error can cause "
                    "this. No partial metrics are shown, because they would be unreliable.")
        return Issue("syntax", f"The C code could not be parsed: {what} at line {line}, column {col}.", hint,
                     line, col)

    def _report(self, root, source: str) -> LanguageReport:
        macros = {_text(n.child_by_field_name("name")) for n in _iter(root)
                  if n.type == "preproc_function_def" and n.child_by_field_name("name") is not None}
        functions, macro_calls = [], set()
        for defn in (n for n in _iter(root) if n.type == "function_definition"):
            fdecl, name = _function_declarator(defn)
            body = defn.child_by_field_name("body")
            if fdecl is None or body is None:
                continue
            counts = {"loops": 0, "conditionals": 0, "decisions": 0, "calls": 0, "recursive": 0, "returns": 0,
                      "switch": 0, "case": 0, "goto": 0, "do_while": 0}
            for n in _iter(body, stop_at={"function_definition"}):
                t = n.type
                if t == "function_definition" and n is not body:
                    continue
                counts["decisions"] += _is_decision(n)
                if t in LOOPS:
                    counts["loops"] += 1
                    counts["do_while"] += t == "do_statement"
                elif t in ("if_statement", "conditional_expression"):
                    counts["conditionals"] += 1
                elif t == "call_expression":
                    counts["calls"] += 1
                    fn = n.child_by_field_name("function")
                    if fn is not None and fn.type == "identifier":
                        callee = _text(fn)
                        counts["recursive"] += callee == name
                        if callee in macros:
                            macro_calls.add(callee)
                elif t == "return_statement":
                    counts["returns"] += 1
                elif t == "switch_statement":
                    counts["switch"] += 1
                elif t == "case_statement":
                    counts["case"] += n.child_by_field_name("value") is not None
                elif t == "goto_statement":
                    counts["goto"] += 1
            depth, loop_depth = _nesting(body)
            start, end = defn.start_point[0] + 1, defn.end_point[0] + 1
            storage = [_text(c) for c in defn.children if c.type == "storage_class_specifier"]
            functions.append(FunctionMetrics(
                name=name, qualname=name, kind="function", start_line=start, end_line=end,
                function_length=end - start + 1, parameter_count=_parameter_count(fdecl),
                loop_count=counts["loops"], conditional_count=counts["conditionals"],
                cyclomatic_complexity=1 + counts["decisions"], maximum_nesting_depth=depth,
                maximum_loop_nesting_depth=loop_depth, recursive_call_count=counts["recursive"],
                function_call_count=counts["calls"], return_count=counts["returns"],
                extra={"switch_count": counts["switch"], "case_count": counts["case"], "goto_count": counts["goto"],
                       "do_while_count": counts["do_while"], "storage": storage},
            ))

        all_nodes = list(_iter(root))
        file_metrics = _line_metrics(root, source)
        file_metrics.update({
            "function_count": len(functions),
            "loop_count": sum(1 for n in all_nodes if n.type in LOOPS),
            "conditional_count": sum(1 for n in all_nodes if n.type in ("if_statement", "conditional_expression")),
            "cyclomatic_complexity": 1 + sum(1 for n in all_nodes if _is_decision(n)),
            "maximum_function_cyclomatic_complexity": max([f.cyclomatic_complexity for f in functions], default=0),
            "maximum_nesting_depth": max([f.maximum_nesting_depth for f in functions], default=0),
            "maximum_loop_nesting_depth": max([f.maximum_loop_nesting_depth for f in functions], default=0),
            "recursive_call_count": sum(f.recursive_call_count for f in functions),
            "function_call_count": sum(1 for n in all_nodes if n.type == "call_expression"),
            "logical_operator_count": sum(1 for n in all_nodes if _is_logical_operator(n)),
        })

        warnings = []
        if macro_calls:
            names = ", ".join(sorted(macro_calls)[:5])
            warnings.append(Issue("macro_calls", f"Function-like macros defined in this file are used ({names}). "
                                  "They are counted as calls; loops or branches inside them are not counted."))
        if any(n.type in PREPROC_CONDITIONALS for n in all_nodes):
            warnings.append(Issue("conditional_compilation", "#if/#ifdef blocks are not evaluated: the code in "
                                  "every branch is counted."))
        if not functions:
            warnings.append(Issue("no_functions", "No function definitions found (prototypes and declarations "
                                  "are not functions)."))
        return LanguageReport(self.language_id, True, file_metrics, functions, [], warnings, self.capabilities)
