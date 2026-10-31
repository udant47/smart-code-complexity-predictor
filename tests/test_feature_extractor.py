"""Tests for src.feature_extractor (Phase 4).

Expected values are worked out by hand from METRIC_DEFINITIONS, so each test
also documents the convention it checks.
"""

import textwrap

import pytest

from src.code_parser import parse_source
from src.config import PROJECT_ROOT
from src.feature_extractor import (
    FILE_FEATURE_NAMES,
    FUNCTION_FEATURE_NAMES,
    FUNCTION_METADATA_KEYS,
    METRIC_DEFINITIONS,
    extract_features,
)


def code(s: str) -> str:
    return textwrap.dedent(s).lstrip("\n")


def file_of(src: str) -> dict:
    report = extract_features(code(src))
    assert report.success, report.error
    return report.file_features


def func(src: str, qualname: str) -> dict:
    report = extract_features(code(src))
    assert report.success, report.error
    return next(f for f in report.function_features if f["qualname"] == qualname)


# --------------------------------------------------------------------------- #
# Schema, consistency, determinism
# --------------------------------------------------------------------------- #
SAMPLE = code('''
    """Sample module."""
    import os, sys  # two imports

    def fact(n):
        # base case
        if n <= 1:
            return 1
        return n * fact(n - 1)


    class Graph:
        def bfs(self, start):
            seen = {start}
            for node in self.adj[start]:
                for nxt in self.adj[node]:
                    if nxt not in seen and nxt != start:
                        seen.add(nxt)
            return [x for x in seen if x]
''')


def test_every_feature_has_a_definition():
    names = set(FILE_FEATURE_NAMES) | set(FUNCTION_FEATURE_NAMES)
    assert names <= set(METRIC_DEFINITIONS)
    assert set(METRIC_DEFINITIONS) <= names  # no stale definitions


def test_no_duplicate_feature_names():
    assert len(FILE_FEATURE_NAMES) == len(set(FILE_FEATURE_NAMES))
    assert len(FUNCTION_FEATURE_NAMES) == len(set(FUNCTION_FEATURE_NAMES))


def test_dicts_have_consistent_keys_order_and_types():
    report = extract_features(SAMPLE)
    assert list(report.file_features) == list(FILE_FEATURE_NAMES)
    expected = list(FUNCTION_METADATA_KEYS) + list(FUNCTION_FEATURE_NAMES)
    for row in report.function_features:
        assert list(row) == expected
        for name in FUNCTION_FEATURE_NAMES:
            assert type(row[name]) in (int, float), name
    for name in FILE_FEATURE_NAMES:
        assert type(report.file_features[name]) in (int, float), name


def test_ratio_and_average_metrics_are_floats_and_counts_are_ints():
    f = extract_features(SAMPLE).file_features
    assert isinstance(f["comment_ratio"], float)
    assert isinstance(f["average_nesting_depth"], float)
    assert isinstance(f["average_function_length"], float)
    assert isinstance(f["if_count"], int) and not isinstance(f["if_count"], bool)


def test_deterministic():
    a = extract_features(SAMPLE)
    b = extract_features(SAMPLE)
    assert a.file_features == b.file_features
    assert a.function_features == b.function_features


def test_accepts_existing_parse_result():
    parsed = parse_source(SAMPLE)
    assert extract_features(parsed).file_features == extract_features(SAMPLE).file_features


def test_full_sample_values():
    report = extract_features(SAMPLE)
    f = report.file_features
    assert f["total_lines"] == 18
    assert (f["code_lines"], f["blank_lines"], f["comment_lines"]) == (14, 3, 2)
    assert f["comment_ratio"] == round(2 / 15, 4)
    assert (f["function_count"], f["class_count"]) == (2, 1)
    assert f["average_function_length"] == 6.0 and f["maximum_function_length"] == 7
    assert f["cyclomatic_complexity"] == 8
    assert f["maximum_function_cyclomatic_complexity"] == 7
    assert f["recursive_call_count"] == 1

    bfs = next(r for r in report.function_features if r["qualname"] == "Graph.bfs")
    assert bfs["kind"] == "method"
    assert bfs["cyclomatic_complexity"] == 7  # for, for, if, and, comprehension for + if
    assert bfs["identifier_count"] == 23 and bfs["unique_identifier_count"] == 8
    assert (bfs["maximum_nesting_depth"], bfs["average_nesting_depth"]) == (3, 1.0)


# --------------------------------------------------------------------------- #
# Error handling
# --------------------------------------------------------------------------- #
def test_syntax_error_is_reported_not_raised():
    report = extract_features("def broken(:\n    pass\n")
    assert not report.success
    assert report.error.kind == "syntax"
    assert report.file_features is None
    assert report.function_features == []


def test_empty_input_is_reported():
    report = extract_features("   \n")
    assert not report.success and report.error.kind == "empty"


def test_file_without_functions():
    f = file_of("x = 1\n")
    assert f["function_count"] == 0
    assert f["average_function_length"] == 0.0 and f["maximum_function_length"] == 0
    assert f["average_function_cyclomatic_complexity"] == 0.0
    assert f["cyclomatic_complexity"] == 1


def test_very_long_expression_does_not_recurse():
    src = "x = " + " + ".join(["1"] * 3000) + "\n"
    report = extract_features(src)
    if report.success:  # the parser may reject it first; either way no crash
        assert report.file_features["arithmetic_operator_count"] == 2999


# --------------------------------------------------------------------------- #
# Line metrics (tokenize-based)
# --------------------------------------------------------------------------- #
def test_line_classification():
    f = file_of('''
        # header comment

        x = 1  # inline comment
        s = """
        not a comment # inside string

        """
        y = "# also not a comment"
    ''')
    # lines: 1 comment-only, 2 blank, 3 code+comment, 4-7 string (code, incl. blank line 6), 8 code
    assert f["total_lines"] == 8
    assert f["comment_lines"] == 2
    assert f["blank_lines"] == 1
    assert f["code_lines"] == 6
    assert f["comment_ratio"] == round(2 / 7, 4)


def test_line_invariant_code_plus_blank_plus_comment_only():
    f = file_of(SAMPLE)
    comment_only = 1  # line 5 "# base case"
    assert f["code_lines"] + f["blank_lines"] + comment_only == f["total_lines"]


def test_docstring_lines_are_code_not_comments():
    f = func('''
        def f():
            """Line one.

            Line three.
            """
            return 1
    ''', "f")
    assert f["docstring_lines"] == 4
    assert f["comment_lines"] == 0 and f["comment_ratio"] == 0.0
    assert f["code_lines"] == 6 and f["blank_lines"] == 0


def test_total_lines_with_and_without_final_newline():
    assert file_of("a = 1\nb = 2\n")["total_lines"] == 2
    assert extract_features("a = 1\nb = 2").file_features["total_lines"] == 2
    assert extract_features("a = 1\r\nb = 2\r\n").file_features["total_lines"] == 2


def test_function_length_excludes_decorators_includes_nested():
    f = func('''
        @decorator
        def outer():
            def inner():
                pass
            return inner
    ''', "outer")
    assert f["function_length"] == 4  # def outer .. return inner


# --------------------------------------------------------------------------- #
# Control flow
# --------------------------------------------------------------------------- #
def test_if_elif_else_chain():
    f = func('''
        def f(x):
            if x > 2:
                return 2
            elif x > 1:
                return 1
            elif x > 0:
                return 0
            else:
                return -1
    ''', "f")
    assert (f["if_count"], f["elif_count"], f["else_count"]) == (1, 2, 1)
    assert f["cyclomatic_complexity"] == 4
    assert f["return_count"] == 4
    assert f["maximum_nesting_depth"] == 1  # elif does not add depth


def test_else_if_is_not_elif():
    f = func('''
        def f(x):
            if x:
                pass
            else:
                if x is None:
                    pass
    ''', "f")
    assert (f["if_count"], f["elif_count"], f["else_count"]) == (2, 0, 1)
    assert f["maximum_nesting_depth"] == 2


def test_loops_and_loop_else():
    f = func('''
        def f(items):
            for i in items:
                pass
            else:
                while True:
                    break
            async_like = [j for j in items]
    ''', "f")
    assert (f["for_loop_count"], f["while_loop_count"], f["loop_count"]) == (1, 1, 2)
    assert f["else_count"] == 1
    assert f["nested_loop_count"] == 0  # loop in else-clause is not inside the loop body
    assert f["maximum_loop_nesting_depth"] == 1
    assert f["comprehension_count"] == 1  # not a loop statement


def test_async_for_counts_as_for():
    f = func('''
        async def f(stream):
            async for x in stream:
                pass
    ''', "f")
    assert f["for_loop_count"] == 1 and f["cyclomatic_complexity"] == 2


def test_try_except_else_finally():
    f = func('''
        def f():
            try:
                pass
            except ValueError:
                pass
            except (TypeError, KeyError) as e:
                pass
            else:
                pass
            finally:
                pass
    ''', "f")
    assert (f["try_count"], f["except_count"], f["else_count"]) == (1, 2, 1)
    assert f["cyclomatic_complexity"] == 3  # two handlers; else/finally are not decisions
    assert f["maximum_nesting_depth"] == 1


def test_except_star():
    f = func('''
        def f():
            try:
                pass
            except* ValueError:
                pass
            except* TypeError:
                pass
    ''', "f")
    assert (f["try_count"], f["except_count"]) == (1, 2)


def test_conditional_expressions():
    f = func('''
        def f(a, b):
            return a if a > b else (b if b else 0)
    ''', "f")
    assert f["conditional_expression_count"] == 2
    assert f["cyclomatic_complexity"] == 3


# --------------------------------------------------------------------------- #
# Cyclomatic complexity convention
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "body, expected",
    [
        ("return 1", 1),
        ("return a and b", 2),
        ("return a and b and c", 3),
        ("return (a and b) or c", 3),
        ("return not a", 1),
        ("return [x for x in a]", 2),
        ("return [x for x in a if x if x > 1]", 4),
        ("return [x for x in a for y in b]", 3),
        ("with a:\n        pass", 1),
        ("assert a", 1),
        ("while a:\n        break", 2),
    ],
)
def test_cyclomatic_complexity_convention(body, expected):
    f = func(f"def f(a, b, c):\n    {body}\n", "f")
    assert f["cyclomatic_complexity"] == expected


def test_match_statement_cases():
    f = func('''
        def f(cmd):
            match cmd:
                case "go":
                    return 1
                case "stop" if cmd:
                    return 2
                case _:
                    return 0
    ''', "f")
    assert f["cyclomatic_complexity"] == 3  # two cases; unguarded wildcard adds nothing
    assert f["maximum_nesting_depth"] == 1


# --------------------------------------------------------------------------- #
# Nesting depth
# --------------------------------------------------------------------------- #
def test_nesting_depths():
    f = func('''
        def f(xs):
            total = 0
            for x in xs:
                if x:
                    with open(x) as fh:
                        total += 1
            return total
    ''', "f")
    # statement depths: total=0 (0), for (0), if (1), with (2), total+=1 (3), return (0)
    assert f["maximum_nesting_depth"] == 3
    assert f["average_nesting_depth"] == round(6 / 6, 4)


def test_nested_loops():
    f = func('''
        def f(n):
            for i in range(n):
                for j in range(n):
                    while j:
                        j -= 1
            for k in range(n):
                pass
    ''', "f")
    assert f["loop_count"] == 4
    assert f["nested_loop_count"] == 2
    assert f["maximum_loop_nesting_depth"] == 3


def test_file_nesting_restarts_in_each_function():
    f = file_of('''
        if True:
            def g():
                return 1
    ''')
    # module: if (0), def (1); g body restarts: return (0)
    assert f["maximum_nesting_depth"] == 1


# --------------------------------------------------------------------------- #
# Recursion
# --------------------------------------------------------------------------- #
def test_recursion_functions():
    src = '''
        def fib(n):
            return n if n < 2 else fib(n - 1) + fib(n - 2)

        def not_recursive(n):
            return fib(n)
    '''
    assert func(src, "fib")["recursive_call_count"] == 2
    assert func(src, "not_recursive")["recursive_call_count"] == 0


def test_recursion_methods():
    src = '''
        class Tree:
            def walk(self, node):
                for child in node.children:
                    self.walk(child)
                    child.walk(child)      # different receiver: not counted
                    walk(child)            # plain name: not this method

            @classmethod
            def build(cls, n):
                return cls.build(n - 1) if n else None

            @staticmethod
            def count(n):
                return Tree.count(n - 1) if n else 0
    '''
    assert func(src, "Tree.walk")["recursive_call_count"] == 1
    assert func(src, "Tree.build")["recursive_call_count"] == 1
    assert func(src, "Tree.count")["recursive_call_count"] == 1
    assert file_of(src)["recursive_call_count"] == 3


# --------------------------------------------------------------------------- #
# Scope isolation: nested functions are not attributed to their parent
# --------------------------------------------------------------------------- #
NESTED = '''
    def outer(items):
        @decorate(level=1)
        def inner(x=default()):
            for i in x:
                if i:
                    return i
            return inner(x)
        key = lambda v: v * 2
        return sorted(items, key=key)
'''


def test_nested_function_body_excluded_from_parent():
    outer = func(NESTED, "outer")
    assert outer["for_loop_count"] == 0 and outer["if_count"] == 0
    assert outer["function_count"] == 1
    assert outer["cyclomatic_complexity"] == 1
    # inner's decorator and default are evaluated in outer: decorate(), default(), sorted()
    assert outer["function_call_count"] == 3
    assert outer["arithmetic_operator_count"] == 1  # inside the lambda, which belongs to outer
    assert outer["return_count"] == 1


def test_nested_function_has_its_own_row():
    inner = func(NESTED, "outer.inner")
    assert inner["kind"] == "nested_function"
    assert (inner["for_loop_count"], inner["if_count"], inner["return_count"]) == (1, 1, 2)
    assert inner["cyclomatic_complexity"] == 3
    assert inner["recursive_call_count"] == 1
    assert inner["function_call_count"] == 1  # decorate() and default() are not inner's
    assert inner["parameter_count"] == 1


def test_file_level_includes_nested():
    f = file_of(NESTED)
    assert f["function_count"] == 2
    assert f["for_loop_count"] == 1
    assert f["function_call_count"] == 4


def test_nested_class_body_not_attributed_to_function():
    f = func('''
        def factory():
            class Local(Base):
                def m(self):
                    while True:
                        pass
            return Local
    ''', "factory")
    assert f["class_count"] == 1
    assert f["while_loop_count"] == 0


ADDITIVE = (
    "return_count", "if_count", "elif_count", "else_count", "for_loop_count",
    "while_loop_count", "loop_count", "try_count", "except_count",
    "conditional_expression_count", "comprehension_count", "nested_loop_count",
    "recursive_call_count", "import_count", "identifier_count", "operator_count",
    "arithmetic_operator_count", "comparison_operator_count", "logical_operator_count",
    "bitwise_operator_count", "function_call_count", "collection_literal_count",
)


@pytest.mark.parametrize("src", [NESTED, SAMPLE.split("class Graph")[0].split("# two imports")[1]])
def test_file_counts_equal_sum_of_function_counts(src):
    """Every AST node belongs to exactly one scope: no double counting, no gaps."""
    report = extract_features(code(src))
    rows = report.function_features
    for name in ADDITIVE:
        assert report.file_features[name] == sum(r[name] for r in rows), name
    assert report.file_features["cyclomatic_complexity"] - 1 == sum(
        r["cyclomatic_complexity"] - 1 for r in rows
    )


# --------------------------------------------------------------------------- #
# Other metrics
# --------------------------------------------------------------------------- #
def test_parameter_count_all_kinds():
    f = func("def f(a, /, b, *args, c, d=1, **kw):\n    pass\n", "f")
    assert f["parameter_count"] == 6


def test_identifiers():
    f = func("def f(a, b):\n    return a.x + b\n", "f")
    assert f["identifier_count"] == 5  # params a, b; names a, b; attribute x
    assert f["unique_identifier_count"] == 3


def test_operator_categories():
    f = func('''
        def f(a, b, c, d, e):
            x = -a + b * c ** 2 // d % e
            y = a & b | ~c
            z = not a and b or c
            w = a < b <= c
            x += 1
            y <<= 1
            return x is not None and y in (1, 2)
    ''', "f")
    assert f["arithmetic_operator_count"] == 7  # -, +, *, **, //, %, +=
    assert f["bitwise_operator_count"] == 4  # &, |, ~, <<=
    assert f["comparison_operator_count"] == 4  # <, <=, is not, in
    assert f["logical_operator_count"] == 4  # not, and, or, and
    assert f["operator_count"] == 19


def test_function_calls():
    f = func('''
        def f(xs):
            print(len(xs))
            obj.method().chain()
            return [str(x) for x in xs]
    ''', "f")
    assert f["function_call_count"] == 5


def test_collection_literals():
    f = func('''
        def f():
            a = [1, 2]
            b = (1, 2)
            c = {1}
            d = {}
            e, g = 1, 2
            [h, i] = a
            return [x for x in a], {k: v for k, v in d.items()}
    ''', "f")
    # [1,2], (1,2), {1}, {}, rhs tuple 1, 2, returned tuple -> 6; unpacking targets excluded
    assert f["collection_literal_count"] == 6
    assert f["comprehension_count"] == 2


def test_imports_count_names():
    f = file_of('''
        import os, sys
        from collections import Counter as C, deque
        from math import *

        def g():
            import json
    ''')
    assert f["import_count"] == 6
    assert func("def g():\n    import json\n", "g")["import_count"] == 1


# --------------------------------------------------------------------------- #
# Real-world smoke test: the project's own source
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("path", sorted((PROJECT_ROOT / "src").rglob("*.py")), ids=lambda p: p.name)
def test_project_sources_satisfy_invariants(path):
    report = extract_features(path.read_text(encoding="utf-8"))
    assert report.success
    f = report.file_features
    assert f["loop_count"] == f["for_loop_count"] + f["while_loop_count"]
    assert f["operator_count"] == sum(
        f[k] for k in ("arithmetic_operator_count", "comparison_operator_count",
                       "logical_operator_count", "bitwise_operator_count")
    )
    assert f["code_lines"] + f["blank_lines"] <= f["total_lines"]
    assert 0.0 <= f["comment_ratio"] <= 1.0
    assert f["function_count"] == len(report.function_features)
    for row in report.function_features:
        assert row["cyclomatic_complexity"] >= 1
        assert row["maximum_nesting_depth"] >= row["average_nesting_depth"] >= 0
        assert row["function_length"] >= row["code_lines"]
