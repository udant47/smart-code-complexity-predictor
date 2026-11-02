"""Robustness: hostile, malformed and very large inputs never crash the app (Phase 18).

Every input here returns a structured result (success, or a known error
code), within a reasonable time. The size limits are enforced exactly at
their boundaries.
"""

import time

import pytest

from src import config
from src import predictor as pr
from src.code_parser import parse_bytes, parse_source
from src.ui import logic

KNOWN_ERRORS = {"empty", "too_large", "syntax", "indentation", "invalid_characters", "encoding", "too_complex"}

ADVERSARIAL = {
    # parser limits (CPython refuses these; we report them)
    "deep_parentheses": "x = " + "(" * 5000 + "1" + ")" * 5000,
    "deep_lists": "x = " + "[" * 2000 + "]" * 2000,
    "deep_indentation": "".join("    " * i + f"if a{i}:\n" for i in range(150)) + "    " * 150 + "pass\n",
    "long_binop": "x = " + "+".join(["1"] * 60000),
    "deep_lambda": "f = " + "lambda: " * 3000 + "1",
    "deep_unary": "x = " + "-" * 3000 + "1\n",
    "deep_attributes": "x = a" + ".b" * 3000 + "\n",
    # deep but valid trees just below the parser's limits
    "unary_500": "x = " + "-" * 500 + "1\n",
    "attributes_500": "x = a" + ".b" * 500 + "\n",
    "ifexp_2000": "x = " + "a if b else " * 2000 + "c\n",
    "elif_chain_2000": "def f(a):\n    if a == 0:\n        return 0\n" + "".join(
        f"    elif a == {i}:\n        return {i}\n" for i in range(1, 2000)),
    "deep_loops_90": "".join("    " * i + f"for i{i} in range(n):\n" for i in range(90)) + "    " * 90 + "pass\n",
    "mutual_recursion": "".join(f"def g{i}(n):\n    return g{(i + 1) % 300}(n - 1) + g{(i + 2) % 300}(n // 2)\n"
                                for i in range(300)),
    # characters and encodings
    "null_byte": "x = 1\0\n",
    "lone_surrogate": "x = '\udcff'\n",
    "bom": "\ufeffx = 1\n",
    "crlf": "def f(x):\r\n    return x\r\n",
    "cr_only": "def f(x):\r    return x\r",
    "tabs_and_spaces": "def f():\n\tif x:\n        pass\n",
    "unicode_identifiers": "def café(ñ):\n    return ñ\n",
    "emoji_strings": "def f():\n    return '🚀' * 3  # ünïcödé\n",
    "huge_string_literal": "s = '" + "a" * 190_000 + "'",
    "only_comments": "# just a comment\n",
    # modern syntax
    "match_statement": "def f(x):\n    match x:\n        case [a, *b]:\n            return a\n        case _:\n            return 0\n",
    "async_constructs": "async def f(xs):\n    async for x in xs:\n        await g(x)\n    async with a as b:\n        pass\n",
    "walrus_and_fstrings": "def f(xs):\n    if (n := len(xs)) > 1:\n        return f'{n!r:>{n}}'\n",
    "decorated_class": "class A(B, metaclass=M):\n    @property\n    @staticmethod\n    def f(self, *a, **k) -> int: ...\n",
    "global_nonlocal": "def f():\n    x = 0\n    def g():\n        nonlocal x\n        global y\n        x += 1\n    return g\n",
    "star_args_defaults": "def f(a, /, b=[], *c, d=lambda: 0, **e):\n    return a\n",
    "type_alias_like": "from typing import TypeAlias\nVector: TypeAlias = list[float]\n",
}


@pytest.mark.parametrize("name", list(ADVERSARIAL))
def test_adversarial_input_returns_structured_result(name):
    t = time.time()
    r = pr.predict(ADVERSARIAL[name])
    elapsed = time.time() - t
    d = r.to_dict()
    assert isinstance(d["success"], bool)
    if not d["success"]:
        assert d["errors"] and d["errors"][0]["code"] in KNOWN_ERRORS and d["errors"][0]["message"]
        assert d["prediction"] is None and d["functions"] == []
    assert elapsed < 60, f"{name} took {elapsed:.1f}s"


@pytest.mark.parametrize("name, code", [
    ("deep_parentheses", "syntax"), ("deep_indentation", "indentation"), ("long_binop", "too_complex"),
    ("deep_lambda", "too_complex"), ("null_byte", "invalid_characters"), ("lone_surrogate", "encoding"),
    ("tabs_and_spaces", "indentation"),
])
def test_adversarial_errors_are_specific(name, code):
    r = pr.predict(ADVERSARIAL[name])
    assert not r.success and r.errors[0].code == code and r.errors[0].hint


def test_lone_surrogate_reports_the_line():
    r = parse_source("x = 1\ny = '\udcff'\n")
    assert r.error.kind == "encoding" and r.error.lineno == 2


# --------------------------------------------------------------------------- #
# Size limits at their exact boundaries
# --------------------------------------------------------------------------- #
def test_line_limit_boundary():
    at_limit = "x = 1\n" * (config.MAX_SOURCE_LINES - 1) + "x = 1"
    assert at_limit.count("\n") + 1 == config.MAX_SOURCE_LINES
    assert parse_source(at_limit).success
    over = at_limit + "\nx = 1"
    r = parse_source(over)
    assert not r.success and r.error.kind == "too_large" and "lines" in r.error.message
    assert logic.check_input(at_limit).ok and not logic.check_input(over).ok


def test_byte_limit_boundary():
    prefix = "x = '"
    at_limit = prefix + "a" * (config.MAX_SOURCE_BYTES - len(prefix) - 1) + "'"
    assert len(at_limit.encode()) == config.MAX_SOURCE_BYTES
    assert parse_source(at_limit).success
    over = prefix + "a" * (config.MAX_SOURCE_BYTES - len(prefix)) + "'"
    r = parse_source(over)
    assert not r.success and r.error.kind == "too_large" and "bytes" in r.error.message


def test_multibyte_characters_count_as_bytes():
    text = "s = '" + "é" * (config.MAX_SOURCE_BYTES // 2) + "'"  # under the limit in characters, over in bytes
    assert len(text) < config.MAX_SOURCE_BYTES < len(text.encode())
    assert parse_source(text).error.kind == "too_large"


def test_oversized_input_is_rejected_before_parsing(monkeypatch):
    import ast

    monkeypatch.setattr(ast, "parse", lambda *a, **k: pytest.fail("ast.parse called on oversized input"))
    assert parse_source("x = 1\n" * (config.MAX_SOURCE_LINES + 10)).error.kind == "too_large"


def test_large_valid_input_finishes_in_reasonable_time():
    code = "def f{i}(xs):\n    t = 0\n    for x in xs:\n        if x > 0:\n            t += x\n    return t\n\n"
    source = "".join(code.format(i=i) for i in range(700))  # 4,900 lines, 700 functions
    assert source.count("\n") < config.MAX_SOURCE_LINES
    t = time.time()
    r = pr.predict(source)
    elapsed = time.time() - t
    assert r.success and len(r.functions) == 700
    assert elapsed < 30, f"took {elapsed:.1f}s; per-function work may have become quadratic"


# --------------------------------------------------------------------------- #
# Uploaded bytes
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("data, ok, kind", [
    (b"x = '\xff'\n", False, "encoding"),
    ("# -*- coding: latin-1 -*-\nx = 'caf\xe9'\n".encode("latin-1"), True, None),
    (b"\xef\xbb\xbfx = 1\n", True, None),
    ("x = 1\n".encode("utf-16"), False, "encoding"),
    (b"x = 1\r\ny = 2\r\n", True, None),
    (b"", False, "empty"),
    (b"x = 1\n" * (config.MAX_SOURCE_LINES + 1), False, "too_large"),
])
def test_parse_bytes(data, ok, kind):
    r = parse_bytes(data)
    assert r.success is ok and (r.error.kind if r.error else None) == kind


# --------------------------------------------------------------------------- #
# The UI service layer never lets an exception through
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("exc", [RuntimeError("boom"), RecursionError(), MemoryError(), ValueError("x"),
                                 KeyError("k"), ZeroDivisionError()])
def test_service_turns_any_internal_error_into_a_message(monkeypatch, exc):
    from src.ui import services

    def boom(*a, **k):
        raise exc

    monkeypatch.setattr(pr, "predict", boom)
    services._analyze_cached.clear()
    out = services.analyze(f"def unique_{type(exc).__name__}():\n    return 1\n")
    services._analyze_cached.clear()
    assert out["ok"] is False and out["result"] is None
    assert out["error"]["code"] == "analysis_failed" and "reference" in out["error"]["hint"]
    assert "boom" not in out["error"]["message"]


@pytest.mark.parametrize("name", ["deep_parentheses", "lone_surrogate", "long_binop", "null_byte"])
def test_service_reports_parser_failures(name):
    from src.ui import services

    out = services.analyze(ADVERSARIAL[name])
    assert out["ok"] and out["result"]["success"] is False and out["result"]["errors"][0]["code"] in KNOWN_ERRORS
