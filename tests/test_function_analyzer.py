"""Tests for src.function_analyzer (Phase 5)."""

import textwrap

import pytest

from src.code_parser import parse_source
from src.function_analyzer import (
    DISPLAY_COLUMNS,
    FunctionRecord,
    analyze_functions,
    format_text_report,
    records_to_rows,
    to_dataframe,
)


def code(s: str) -> str:
    return textwrap.dedent(s).lstrip("\n")


def analyse(src: str):
    result = analyze_functions(code(src))
    assert result.success, result.error
    return result


ALGORITHMS = '''
    def merge_sort(items):
        if len(items) <= 1:
            return items
        mid = len(items) // 2
        left = merge_sort(items[:mid])
        right = merge_sort(items[mid:])
        return merge(left, right)


    def merge(left, right):
        result = []
        i = j = 0
        while i < len(left) and j < len(right):
            if left[i] <= right[j]:
                result.append(left[i])
                i += 1
            else:
                result.append(right[j])
                j += 1
        return result + left[i:] + right[j:]


    def triple_sum(nums, target):
        n = len(nums)
        count = 0
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    if nums[i] + nums[j] + nums[k] == target:
                        count += 1
        return count
'''


# --------------------------------------------------------------------------- #
# Multiple functions
# --------------------------------------------------------------------------- #
def test_multiple_functions_in_source_order():
    result = analyse(ALGORITHMS)
    assert [f.function_id for f in result.functions] == ["merge_sort", "merge", "triple_sum"]
    assert all(f.kind == "function" and f.class_name is None for f in result.functions)
    ms = result.get("merge_sort")
    assert (ms.start_line, ms.end_line, ms.function_length) == (1, 7, 7)
    assert ms.parameter_count == 1


def test_record_has_every_requested_field():
    required = {
        "function_name", "class_name", "start_line", "end_line", "function_length",
        "parameter_count", "loop_count", "conditional_count", "cyclomatic_complexity",
        "maximum_nesting_depth", "recursive_call_count", "function_call_count",
    }
    record = analyse(ALGORITHMS).functions[0]
    assert required <= set(record.to_dict())


# --------------------------------------------------------------------------- #
# Loops and conditionals
# --------------------------------------------------------------------------- #
def test_loops_and_conditionals():
    result = analyse(ALGORITHMS)
    merge = result.get("merge")
    assert (merge.loop_count, merge.conditional_count) == (1, 1)
    assert merge.cyclomatic_complexity == 4  # while, and, if
    assert merge.maximum_nesting_depth == 2  # while > if > statement

    triple = result.get("triple_sum")
    assert (triple.loop_count, triple.conditional_count) == (3, 1)
    assert triple.cyclomatic_complexity == 5  # three loops + one if
    assert triple.maximum_nesting_depth == 4  # count += 1 sits inside 4 blocks


def test_conditional_count_includes_elif_and_ternary():
    f = analyse('''
        def grade(score):
            if score > 90:
                return "A"
            elif score > 75:
                return "B"
            return "C" if score > 50 else "F"
    ''').functions[0]
    assert f.conditional_count == 3
    assert f.loop_count == 0


# --------------------------------------------------------------------------- #
# Recursion
# --------------------------------------------------------------------------- #
def test_recursive_functions():
    result = analyse(ALGORITHMS)
    assert result.get("merge_sort").recursive_call_count == 2
    assert result.get("merge").recursive_call_count == 0
    assert result.summary.recursive_function_count == 1


def test_recursive_method():
    result = analyse('''
        class Node:
            def depth(self):
                if not self.children:
                    return 1
                return 1 + max(c.depth() for c in self.children)

            def size(self):
                return 1 + sum(self.size_of(c) for c in self.children)
    ''')
    # c.depth() is called on a child, not on self: lexically not a self-call
    assert result.get("Node.depth").recursive_call_count == 0
    assert result.get("Node.size").recursive_call_count == 0


def test_recursive_method_via_self():
    f = analyse('''
        class Tree:
            def height(self, node):
                if node is None:
                    return 0
                return 1 + max(self.height(node.left), self.height(node.right))
    ''').get("Tree.height")
    assert f.recursive_call_count == 2
    assert f.kind == "method" and f.class_name == "Tree"


# --------------------------------------------------------------------------- #
# Methods and identical names
# --------------------------------------------------------------------------- #
CLASSES = '''
    class Stack:
        def push(self, x):
            self.items.append(x)

        def size(self):
            return len(self.items)


    class Queue:
        def push(self, x):
            self.items.insert(0, x)

        def size(self):
            return len(self.items)


    def size(container):
        return container.size()
'''


def test_methods_inside_classes():
    result = analyse(CLASSES)
    methods = [f for f in result.functions if f.kind == "method"]
    assert [m.function_id for m in methods] == ["Stack.push", "Stack.size", "Queue.push", "Queue.size"]
    assert all(m.parameter_count >= 1 for m in methods)  # self is a parameter
    assert result.summary.method_count == 4 and result.summary.class_count == 2


def test_identical_names_in_different_classes_get_distinct_ids():
    result = analyse(CLASSES)
    sizes = result.by_name("size")
    assert [s.function_id for s in sizes] == ["Stack.size", "Queue.size", "size"]
    assert [s.class_name for s in sizes] == ["Stack", "Queue", None]
    ids = [f.function_id for f in result.functions]
    assert len(ids) == len(set(ids))


def test_redefinitions_and_property_setters_get_suffixes():
    result = analyse('''
        class Temperature:
            @property
            def celsius(self):
                return self._c

            @celsius.setter
            def celsius(self, value):
                self._c = value

        if FAST:
            def solve(x):
                return x
        else:
            def solve(x):
                return x * 2
    ''')
    assert [f.function_id for f in result.functions] == [
        "Temperature.celsius", "Temperature.celsius#2", "solve", "solve#2"
    ]
    assert result.get("Temperature.celsius#2").qualname == "Temperature.celsius"


def test_function_ids_are_stable_when_code_above_changes():
    before = analyse(CLASSES)
    after = analyse("import os\n\n\n# a new comment\n" + code(CLASSES))
    assert [f.function_id for f in before.functions] == [f.function_id for f in after.functions]
    assert after.get("Queue.size").start_line == before.get("Queue.size").start_line + 4


# --------------------------------------------------------------------------- #
# Nested functions: no double counting
# --------------------------------------------------------------------------- #
NESTED = '''
    def process(rows):
        def clean(row):
            for cell in row:
                if cell is None:
                    return False
            return True

        valid = [r for r in rows if clean(r)]
        return len(valid)
'''


def test_nested_function_has_own_record():
    result = analyse(NESTED)
    assert [f.function_id for f in result.functions] == ["process", "process.clean"]
    clean = result.get("process.clean")
    assert clean.kind == "nested_function" and clean.parent == "process"
    assert clean.class_name is None
    assert (clean.loop_count, clean.conditional_count, clean.cyclomatic_complexity) == (1, 1, 3)
    assert clean.maximum_nesting_depth == 2


def test_enclosing_function_excludes_nested_complexity():
    process = analyse(NESTED).get("process")
    assert process.loop_count == 0
    assert process.conditional_count == 0
    # 1 + comprehension for + its if filter; clean's loop and if are not counted here
    assert process.cyclomatic_complexity == 3
    assert process.maximum_nesting_depth == 0
    assert process.function_call_count == 2  # clean(r) and len(valid)
    assert process.function_length == 9  # line span still includes the nested def


def test_nested_inside_method():
    result = analyse('''
        class Parser:
            def parse(self, text):
                def tokens():
                    while text:
                        yield text
                return list(tokens())
    ''')
    inner = result.get("Parser.parse.tokens")
    assert inner.kind == "nested_function" and inner.parent == "Parser.parse"
    assert result.get("Parser.parse").loop_count == 0
    assert inner.loop_count == 1
    assert result.summary.nested_function_count == 1


# --------------------------------------------------------------------------- #
# Empty functions
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "body",
    ["pass", "...", '"""Only a docstring."""', '"""Doc."""\n    pass', "pass\n    ..."],
)
def test_empty_functions(body):
    f = analyse(f"def stub(a, b):\n    {body}\n").functions[0]
    assert f.is_empty
    assert f.cyclomatic_complexity == 1
    assert (f.loop_count, f.conditional_count, f.maximum_nesting_depth) == (0, 0, 0)
    assert (f.recursive_call_count, f.function_call_count) == (0, 0)
    assert f.parameter_count == 2


def test_non_empty_function_is_not_empty():
    assert not analyse("def f():\n    return 1\n").functions[0].is_empty
    assert not analyse("def f():\n    x = 1\n").functions[0].is_empty


def test_abstract_methods_count_as_empty():
    result = analyse('''
        class Shape:
            def area(self):
                raise NotImplementedError

            async def draw(self):
                ...
    ''')
    assert not result.get("Shape.area").is_empty  # raises: has behaviour
    draw = result.get("Shape.draw")
    assert draw.is_empty and draw.is_async
    assert result.summary.empty_function_count == 1


# --------------------------------------------------------------------------- #
# File-level summary
# --------------------------------------------------------------------------- #
def test_file_summary():
    s = analyse(ALGORITHMS).summary
    assert s.function_count == 3 and s.top_level_function_count == 3
    assert (s.method_count, s.nested_function_count, s.class_count) == (0, 0, 0)
    assert s.maximum_cyclomatic_complexity == 5
    assert s.most_complex_function == "triple_sum"
    assert s.average_cyclomatic_complexity == round((2 + 4 + 5) / 3, 4)
    assert s.longest_function == "merge"
    assert s.deepest_function == "triple_sum" and s.maximum_nesting_depth == 4
    assert s.file_cyclomatic_complexity == 1 + (2 - 1) + (4 - 1) + (5 - 1)


def test_summary_ties_resolve_to_first_in_source_order():
    s = analyse("def a():\n    pass\n\ndef b():\n    pass\n").summary
    assert s.most_complex_function == "a" and s.longest_function == "a"


def test_summary_for_file_without_functions():
    result = analyse("x = 1\n")
    assert result.functions == []
    s = result.summary
    assert s.function_count == 0
    assert s.most_complex_function is None and s.longest_function is None
    assert s.average_cyclomatic_complexity == 0.0 and s.maximum_cyclomatic_complexity == 0


# --------------------------------------------------------------------------- #
# Errors
# --------------------------------------------------------------------------- #
def test_syntax_error_is_reported_not_raised():
    result = analyze_functions("def f(:\n    pass\n")
    assert not result.success
    assert result.error.kind == "syntax"
    assert result.functions == [] and result.summary is None
    assert "Syntax error" in format_text_report(result)


def test_accepts_parse_result():
    parsed = parse_source(code(ALGORITHMS))
    assert analyze_functions(parsed).functions == analyse(ALGORITHMS).functions


def test_deterministic():
    assert analyse(ALGORITHMS).functions == analyse(ALGORITHMS).functions


# --------------------------------------------------------------------------- #
# Dashboard output
# --------------------------------------------------------------------------- #
def test_rows_with_labels():
    rows = records_to_rows(analyse(CLASSES).functions)
    assert list(rows[0]) == list(DISPLAY_COLUMNS.values())
    assert rows[0]["Function"] == "Stack.push"
    assert rows[0]["Type"] == "Method" and rows[0]["Class"] == "Stack"
    assert rows[-1]["Type"] == "Function" and rows[-1]["Class"] == ""


def test_rows_without_labels_use_field_names():
    rows = records_to_rows(analyse(CLASSES).functions, labels=False)
    assert list(rows[0]) == list(DISPLAY_COLUMNS)
    assert rows[-1]["class_name"] is None


def test_display_columns_are_record_fields():
    assert set(DISPLAY_COLUMNS) <= set(FunctionRecord.__dataclass_fields__)


def test_dataframe():
    df = to_dataframe(analyse(ALGORITHMS))
    assert list(df.columns) == list(DISPLAY_COLUMNS.values())
    assert len(df) == 3
    assert df.loc[df["Function"] == "triple_sum", "Cyclomatic complexity"].item() == 5


def test_dataframe_empty_keeps_columns():
    df = to_dataframe(analyse("x = 1\n"))
    assert df.empty and list(df.columns) == list(DISPLAY_COLUMNS.values())


def test_text_report_format():
    text = format_text_report(analyse(ALGORITHMS))
    assert text.split("\n\n")[0] == (
        "Function: merge_sort\nCyclomatic complexity: 2\nMaximum nesting depth: 1"
    )
    assert "Function: triple_sum\nCyclomatic complexity: 5\nMaximum nesting depth: 4" in text


def test_no_big_o_claims_in_structural_output():
    """Structural metrics must not be presented as running-time predictions."""
    result = analyse(ALGORITHMS)
    text = format_text_report(result) + str(records_to_rows(result.functions))
    text += str(result.summary.to_dict())
    assert "O(" not in text and "big_o" not in text.lower()
