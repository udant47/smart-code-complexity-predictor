"""Tests for src.code_parser (Phase 3)."""

import ast
import json
import textwrap
from pathlib import Path

import pytest

from src.code_parser import (
    ERR_EMPTY,
    ERR_ENCODING,
    ERR_INDENTATION,
    ERR_INVALID_CHARACTERS,
    ERR_SYNTAX,
    ERR_TOO_LARGE,
    parse_bytes,
    parse_source,
)
from src.config import MAX_SOURCE_LINES, PROJECT_ROOT


def code(s: str) -> str:
    return textwrap.dedent(s).lstrip("\n")


# --------------------------------------------------------------------------- #
# Valid simple Python
# --------------------------------------------------------------------------- #
def test_valid_simple_function():
    result = parse_source(code("""
        def add(a, b):
            return a + b
    """))
    assert result.success
    assert result.error is None
    assert isinstance(result.tree, ast.Module)
    assert len(result.functions) == 1

    f = result.functions[0]
    assert f.name == "add"
    assert f.qualname == "add"
    assert f.kind == "function"
    assert f.is_top_level
    assert (f.lineno, f.end_lineno, f.col) == (1, 2, 0)
    assert f.line_count == 2
    assert f.parameters == ("a", "b")
    assert f.source.startswith("def add(a, b):")
    assert result.line_count == 3  # trailing newline counts as a line


def test_function_source_segment_and_parameters():
    result = parse_source(code("""
        @staticmethod
        def f(a, /, b, *args, c, d=1, **kwargs):
            pass
    """))
    f = result.functions[0]
    assert f.parameters == ("a", "b", "*args", "c", "d", "**kwargs")
    assert f.decorators == ("staticmethod",)
    assert f.lineno == 2  # def line, not decorator line


# --------------------------------------------------------------------------- #
# Multiple functions
# --------------------------------------------------------------------------- #
def test_multiple_functions_in_source_order():
    result = parse_source(code("""
        def first():
            return 1

        async def second():
            return 2

        if True:
            def third():  # inside a block, still module scope
                return 3
    """))
    assert result.success
    assert [f.name for f in result.functions] == ["first", "second", "third"]
    assert [f.lineno for f in result.functions] == [1, 4, 8]
    assert all(f.kind == "function" for f in result.functions)
    assert result.functions[1].is_async
    assert len(result.top_level_functions) == 3


# --------------------------------------------------------------------------- #
# Classes and methods
# --------------------------------------------------------------------------- #
def test_classes_and_methods():
    result = parse_source(code("""
        class Base:
            pass

        class Stack(Base):
            def __init__(self):
                self.items = []

            @property
            def size(self):
                return len(self.items)

            async def drain(self):
                pass

        def helper():
            pass
    """))
    assert result.success
    assert [c.name for c in result.classes] == ["Base", "Stack"]

    stack = result.classes[1]
    assert stack.bases == ("Base",)
    assert stack.methods == ("__init__", "size", "drain")
    assert stack.is_top_level and (stack.lineno, stack.end_lineno) == (4, 13)

    methods = result.methods
    assert [m.qualname for m in methods] == ["Stack.__init__", "Stack.size", "Stack.drain"]
    assert all(m.class_name == "Stack" and m.depth == 1 for m in methods)
    assert result.get_function("Stack.size").decorators == ("property",)
    assert [f.name for f in result.top_level_functions] == ["helper"]


def test_nested_class_methods():
    result = parse_source(code("""
        class Outer:
            class Inner:
                def m(self):
                    pass
    """))
    assert [c.qualname for c in result.classes] == ["Outer", "Outer.Inner"]
    assert result.classes[1].parent == "Outer"
    m = result.functions[0]
    assert (m.qualname, m.kind, m.class_name, m.depth) == ("Outer.Inner.m", "method", "Outer.Inner", 2)


# --------------------------------------------------------------------------- #
# Nested functions
# --------------------------------------------------------------------------- #
def test_nested_functions():
    result = parse_source(code("""
        def outer(n):
            def inner(x):
                def innermost():
                    return x
                return innermost()
            return inner(n)

        class Tree:
            def walk(self):
                def visit(node):
                    return node
                return visit(self)
    """))
    by_name = {f.qualname: f for f in result.functions}
    assert set(by_name) == {
        "outer", "outer.inner", "outer.inner.innermost", "Tree.walk", "Tree.walk.visit"
    }
    assert by_name["outer"].kind == "function"
    assert by_name["outer.inner"].kind == "nested_function"
    assert by_name["outer.inner"].parent == "outer"
    assert by_name["outer.inner.innermost"].depth == 2
    visit = by_name["Tree.walk.visit"]
    assert (visit.kind, visit.parent, visit.class_name) == ("nested_function", "Tree.walk", None)
    assert visit.lineno == 10 and visit.col == 8
    assert len(result.nested_functions) == 3


def test_class_inside_function_has_methods():
    result = parse_source(code("""
        def factory():
            class Local:
                def m(self):
                    pass
            return Local
    """))
    m = result.get_function("factory.Local.m")
    assert m is not None and m.kind == "method"
    assert result.classes[0].parent == "factory"


# --------------------------------------------------------------------------- #
# Empty input
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("src", ["", "   ", "\n\n\t\n", "\ufeff", "\r\n"])
def test_empty_input(src):
    result = parse_source(src)
    assert not result.success
    assert result.error.kind == ERR_EMPTY
    assert result.tree is None and result.functions == []
    assert result.error.hint


def test_comments_only_is_valid_but_warns():
    result = parse_source("# just a comment\nx = 1\n")
    assert result.success
    assert result.functions == [] and result.classes == []
    assert any("No functions or classes" in w for w in result.warnings)


# --------------------------------------------------------------------------- #
# Invalid syntax
# --------------------------------------------------------------------------- #
def test_invalid_syntax_reports_location():
    result = parse_source(code("""
        def ok():
            return 1

        def broken(:
            pass
    """))
    assert not result.success
    err = result.error
    assert err.kind == ERR_SYNTAX
    assert err.lineno == 4
    assert err.line_text == "def broken(:"
    assert "line 4" in err.message
    assert "^" in err.format()


def test_missing_colon_has_hint():
    result = parse_source("def f()\n    return 1\n")
    assert result.error.kind == ERR_SYNTAX
    assert result.error.hint and "colon" in result.error.hint


def test_unclosed_bracket_hint():
    result = parse_source("x = [1, 2,\n")
    assert result.error.kind == ERR_SYNTAX
    assert "bracket" in (result.error.hint or "")


def test_indentation_error():
    result = parse_source("def f():\nreturn 1\n")
    assert result.error.kind == ERR_INDENTATION
    assert result.error.lineno == 2


def test_python2_print_hint():
    result = parse_source('print "hello"\n')
    assert result.error.kind == ERR_SYNTAX
    assert "Python 2" in (result.error.hint or "")


def test_python2_except_hint():
    result = parse_source("try:\n    pass\nexcept ValueError, e:\n    pass\n")
    assert result.error.kind == ERR_SYNTAX
    assert "Python 2" in (result.error.hint or "")


def test_generic_syntax_error_mentions_python_version():
    import sys

    result = parse_source("x = = 1\n")
    assert result.error.kind == ERR_SYNTAX
    assert f"{sys.version_info.major}.{sys.version_info.minor}" in (result.error.hint or "")


def test_null_bytes_rejected():
    result = parse_source("x = 1\x00\n")
    assert not result.success
    assert result.error.kind == ERR_INVALID_CHARACTERS


# --------------------------------------------------------------------------- #
# Unicode identifiers
# --------------------------------------------------------------------------- #
def test_unicode_identifiers():
    result = parse_source(code("""
        def calcular_área(radio):
            return 3.14 * radio ** 2

        class Größe:
            def größe(self):
                return 1

        def 计算(数据):
            return 数据
    """))
    assert result.success
    assert [f.qualname for f in result.functions] == ["calcular_área", "Größe.größe", "计算"]
    assert result.functions[2].parameters == ("数据",)


def test_unicode_columns_are_characters_not_bytes():
    line = 'def ñandú(): return "äöü"'
    result = parse_source(line)
    f = result.functions[0]
    assert f.col == 0
    assert f.end_col == len(line)  # bytes would give a larger number


def test_windows_line_endings_and_bom():
    result = parse_source("\ufeffdef f():\r\n    return 1\r\n")
    assert result.success
    assert result.functions[0].end_lineno == 2


# --------------------------------------------------------------------------- #
# Large input handling
# --------------------------------------------------------------------------- #
def test_too_many_bytes_rejected_before_parsing():
    result = parse_source("x = 1\n" * 100, max_bytes=50)
    assert not result.success
    assert result.error.kind == ERR_TOO_LARGE
    assert result.tree is None


def test_too_many_lines_rejected_with_default_limit():
    result = parse_source("\n" * MAX_SOURCE_LINES + "x = 1")
    assert result.error.kind == ERR_TOO_LARGE
    assert "lines" in result.error.message


def test_input_exactly_at_limit_is_accepted():
    src = "x = 1\n"
    result = parse_source(src, max_bytes=len(src.encode()), max_lines=2)
    assert result.success


def test_deeply_nested_expression_fails_cleanly():
    result = parse_source("x = " + "(" * 5000 + "1" + ")" * 5000)
    assert not result.success
    assert result.error.kind in {ERR_SYNTAX, "too_complex"}


def test_deeply_nested_blocks_fail_cleanly():
    src = "".join("    " * i + "if x:\n" for i in range(150)) + "    " * 150 + "pass\n"
    result = parse_source(src)
    assert not result.success
    assert result.error.kind in {ERR_INDENTATION, ERR_SYNTAX, "too_complex"}


def test_moderately_deep_nested_functions_supported():
    depth = 40
    src = "".join("    " * i + f"def f{i}():\n" for i in range(depth)) + "    " * depth + "pass\n"
    result = parse_source(src)
    assert result.success
    assert len(result.functions) == depth
    assert result.functions[-1].depth == depth - 1


# --------------------------------------------------------------------------- #
# Bytes input (uploads)
# --------------------------------------------------------------------------- #
def test_parse_bytes_utf8_and_bom():
    assert parse_bytes("def é(): pass\n".encode("utf-8")).functions[0].name == "é"
    assert parse_bytes(b"\xef\xbb\xbfdef f(): pass\n").success


def test_parse_bytes_honours_coding_cookie():
    data = "# -*- coding: latin-1 -*-\ns = 'café'\n".encode("latin-1")
    assert parse_bytes(data).success


def test_parse_bytes_invalid_encoding():
    result = parse_bytes(b"x = '\xff\xfe'\n")
    assert result.error.kind == ERR_ENCODING


def test_parse_bytes_size_limit():
    assert parse_bytes(b"x = 1\n" * 10, max_bytes=5).error.kind == ERR_TOO_LARGE


def test_wrong_types_raise():
    with pytest.raises(TypeError):
        parse_source(b"x = 1")
    with pytest.raises(TypeError):
        parse_bytes("x = 1")


# --------------------------------------------------------------------------- #
# Results are reusable outside Streamlit
# --------------------------------------------------------------------------- #
def test_to_dict_is_json_serialisable():
    result = parse_source("class A:\n    def m(self): pass\n")
    data = json.loads(json.dumps(result.to_dict()))
    assert data["success"] is True
    assert data["functions"][0]["qualname"] == "A.m"
    assert "source" not in data["functions"][0]


def test_syntax_warnings_are_captured_not_printed():
    result = parse_source('def f():\n    return "\\d"\n')
    assert result.success
    # DeprecationWarning on Python 3.11, SyntaxWarning on 3.12+
    assert any("invalid escape sequence" in w for w in result.warnings)


# --------------------------------------------------------------------------- #
# Safety: submitted code is never executed
# --------------------------------------------------------------------------- #
def test_submitted_code_is_not_executed(tmp_path: Path):
    marker = tmp_path / "executed.txt"
    payload = f"open({str(marker)!r}, 'w').write('ran')"
    src = code(f"""
        import os
        {payload}

        def f(x={payload}):
            return x

        @(lambda fn: {payload})
        def g():
            pass

        class C({payload} or object):
            pass

        raise SystemExit(1)
    """)
    result = parse_source(src)
    assert result.success
    assert not marker.exists()


FORBIDDEN_CALLS = {"exec", "eval", "compile", "__import__", "execfile"}
FORBIDDEN_MODULES = {"subprocess", "pty", "runpy", "importlib"}


def test_src_package_has_no_execution_primitives():
    offenders = []
    for path in (PROJECT_ROOT / "src").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                fn = node.func
                name = fn.id if isinstance(fn, ast.Name) else getattr(fn, "attr", None)
                if name in FORBIDDEN_CALLS or name in {"system", "popen"}:
                    offenders.append(f"{path.name}:{node.lineno} {name}()")
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                mods = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module or ""]
                for m in mods:
                    if m.split(".")[0] in FORBIDDEN_MODULES:
                        offenders.append(f"{path.name}:{node.lineno} import {m}")
    assert offenders == []
