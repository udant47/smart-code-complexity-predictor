"""Tests for the language layer (Phase 19): common schema, registry, Python adapter and the C analyzer."""

import json
from dataclasses import fields

import pytest

from src.examples import EXAMPLES, get_example
from src.feature_extractor import extract_features
from src.function_analyzer import analyze_functions
from src.code_parser import parse_source
from src.languages import COMMON_FILE_FIELDS, COMMON_FUNCTION_FIELDS, FunctionMetrics, registry
from src.languages.c_examples import C_EXAMPLES
from src.languages.python_lang import PythonAnalyzer

C_AVAILABLE = registry.get("c").usable
needs_c = pytest.mark.skipif(not C_AVAILABLE, reason="tree-sitter C parser not installed")


def c(code: str):
    from src.languages.c_lang import CAnalyzer

    return CAnalyzer().analyze(code)


def fn(report, name):
    return next(f for f in report.functions if f.name == name)


# --------------------------------------------------------------------------- #
# Registry: honest status
# --------------------------------------------------------------------------- #
def test_registry_statuses_are_honest():
    status = {k: v.status for k, v in registry.LANGUAGES.items()}
    assert status == {"python": "full", "c": "structural", "cpp": "unsupported", "java": "unsupported",
                      "javascript": "unsupported"}
    for lang in ("cpp", "java", "javascript"):
        info = registry.get(lang)
        assert info.analyzer is None and not info.usable and "Not supported" in info.note
        with pytest.raises(ValueError, match="not supported"):
            registry.analyzer_for(lang)
    caps = registry.get("c").analyzer.capabilities
    assert caps.structural_metrics and not (caps.ml_prediction or caps.static_big_o or caps.derived_score
                                            or caps.recommendations)
    assert PythonAnalyzer.capabilities.ml_prediction and PythonAnalyzer.capabilities.static_big_o


@pytest.mark.parametrize("path, lang", [("a.py", "python"), ("x/B.PY", "python"), ("m.c", "c"), ("m.h", "c"),
                                        ("x.cpp", "cpp"), ("x.hpp", "cpp"), ("A.java", "java"), ("a.js", "javascript"),
                                        ("README.md", None), ("Makefile", None)])
def test_language_for_path(path, lang):
    info = registry.language_for_path(path)
    assert (info.language_id if info else None) == lang


def test_supported_languages_reflect_installed_parsers():
    ids = [i.language_id for i in registry.supported_languages()]
    assert ids[0] == "python" and ("c" in ids) == C_AVAILABLE
    assert not {"cpp", "java", "javascript"} & set(ids)


def test_c_unavailable_is_reported_not_faked(monkeypatch):
    from src.languages import c_lang

    def missing():
        raise ImportError("no tree_sitter")

    monkeypatch.setattr(c_lang, "_language", missing)
    ok, reason = c_lang.CAnalyzer.available()
    assert not ok and "pip install tree-sitter" in reason
    r = c_lang.CAnalyzer().analyze("int f(void) { return 0; }")
    assert not r.success and r.errors[0].code == "unavailable" and r.file_metrics is None
    assert "c" not in [i.language_id for i in registry.supported_languages()]


# --------------------------------------------------------------------------- #
# Common schema
# --------------------------------------------------------------------------- #
def test_function_metrics_fields_match_the_documented_schema():
    names = [f.name for f in fields(FunctionMetrics)]
    assert names[:len(COMMON_FUNCTION_FIELDS)] == list(COMMON_FUNCTION_FIELDS)
    assert set(names) - set(COMMON_FUNCTION_FIELDS) == {"structural_category", "static_time", "extra"}


@pytest.mark.parametrize("example", EXAMPLES, ids=lambda e: e.key)
def test_python_adapter_reports_the_existing_pipeline_values(example):
    rep = PythonAnalyzer().analyze(example.code)
    assert rep.success and set(rep.file_metrics) == set(COMMON_FILE_FIELDS)
    parsed = parse_source(example.code)
    ff = extract_features(parsed).file_features
    fa = {f.qualname: f for f in analyze_functions(parsed).functions}
    for k in ("total_lines", "code_lines", "function_count", "loop_count", "cyclomatic_complexity",
              "maximum_nesting_depth", "recursive_call_count"):
        assert rep.file_metrics[k] == ff[k]
    for f in rep.functions:
        ref = fa[f.qualname]
        assert (f.cyclomatic_complexity, f.maximum_nesting_depth, f.loop_count, f.conditional_count,
                f.parameter_count, f.recursive_call_count, f.function_length) == (
            ref.cyclomatic_complexity, ref.maximum_nesting_depth, ref.loop_count, ref.conditional_count,
            ref.parameter_count, ref.recursive_call_count, ref.function_length)
    expected = {fc.function: fc.static_time_class for fc in example.complexity}
    assert {f.qualname: f.static_time for f in rep.functions if f.qualname in expected} == expected
    json.dumps(rep.to_dict())


def test_python_adapter_errors_use_the_parser_codes():
    assert PythonAnalyzer().analyze("def f(:\n").errors[0].code == "syntax"
    assert PythonAnalyzer().analyze("").errors[0].code == "empty"
    assert PythonAnalyzer().analyze_bytes(b"x = '\xff'\n").errors[0].code == "encoding"


# --------------------------------------------------------------------------- #
# C analyzer: construct mapping
# --------------------------------------------------------------------------- #
FULL = r'''#include <stdio.h>
#define MAX(a, b) ((a) > (b) ? (a) : (b))

/* Binary search
   over a sorted array */
int binary_search(const int *a, int n, int target) {
    int lo = 0, hi = n - 1;   // inclusive
    while (lo <= hi) {
        int mid = lo + (hi - lo) / 2;
        if (a[mid] == target) {
            return mid;
        } else if (a[mid] < target) {
            lo = mid + 1;
        } else {
            hi = mid - 1;
        }
    }
    return -1;
}

static long fib(int n) { return n < 2 ? n : fib(n - 1) + fib(n - 2); }

int classify(int x, ...) {
    switch (x) {
        case 1: return 10;
        case 2:
        case 3: if (x && !0 || x > 5) return MAX(x, 3);
        default: break;
    }
    do { x--; } while (x > 0);
    for (int i = 0; i < x; i++) for (int j = 0; j < i; j++) goto out;
out:
    return 0;
}
int (*getf(void))(int);
int noargs(void) { return 0; }
'''


@needs_c
def test_c_full_mapping():
    r = c(FULL)
    assert r.success and set(r.file_metrics) == set(COMMON_FILE_FIELDS)
    assert r.file_metrics == {
        "total_lines": 36, "code_lines": 31, "comment_lines": 3, "blank_lines": 3, "function_count": 4,
        "loop_count": 4, "conditional_count": 4, "cyclomatic_complexity": 14,
        "maximum_function_cyclomatic_complexity": 10, "maximum_nesting_depth": 2, "maximum_loop_nesting_depth": 2,
        "recursive_call_count": 2, "function_call_count": 3, "logical_operator_count": 3}
    assert [f.name for f in r.functions] == ["binary_search", "fib", "classify", "noargs"]  # prototypes are not functions
    bs = fn(r, "binary_search")
    assert (bs.cyclomatic_complexity, bs.maximum_nesting_depth, bs.conditional_count, bs.return_count,
            bs.start_line, bs.end_line, bs.parameter_count) == (4, 2, 2, 2, 6, 19, 3)
    fib = fn(r, "fib")
    assert (fib.cyclomatic_complexity, fib.recursive_call_count, fib.extra["storage"]) == (2, 2, ["static"])
    cl = fn(r, "classify")
    # case 1, case 2, case 3 (default is not a decision), if, &&, ||, do-while, for, for
    assert cl.cyclomatic_complexity == 10 and cl.parameter_count == 2  # x and the variadic ...
    assert (cl.loop_count, cl.maximum_loop_nesting_depth, cl.maximum_nesting_depth) == (3, 2, 2)
    assert cl.extra == {"switch_count": 1, "case_count": 3, "goto_count": 1, "do_while_count": 1, "storage": []}
    assert fn(r, "noargs").parameter_count == 0  # (void)
    assert [w.code for w in r.warnings] == ["macro_calls"] and "MAX" in r.warnings[0].message


@needs_c
@pytest.mark.parametrize("code, cc, depth", [
    ("int f(int a) { if (a) { if (a) { return 1; } } return 0; }", 3, 2),
    ("int f(int a) { if (a) return 1; else if (a > 1) return 2; else if (a > 2) { if (a) return 3; } return 0; }", 5, 2),
    ("int f(int a) { { { return a; } } }", 1, 0),  # bare blocks do not nest
    ("int f(int a) { while (a) { switch (a) { case 1: if (a) a--; break; } } return a; }", 4, 3),
    ("int f(int a, int b) { return a && b || !a ? 1 : 0; }", 4, 0),
])
def test_c_cyclomatic_and_nesting_rules(code, cc, depth):
    f = c(code).functions[0]
    assert (f.cyclomatic_complexity, f.maximum_nesting_depth) == (cc, depth)


@needs_c
def test_c_function_pointer_declarators_and_names():
    r = c("int (*pick(int which))(int) { return 0; }\nstatic inline unsigned *buf(void) { return 0; }\n")
    assert [(f.name, f.parameter_count) for f in r.functions] == [("pick", 1), ("buf", 0)]


@needs_c
def test_c_conditional_compilation_warning():
    r = c("#ifdef FAST\nint f(void) { return 1; }\n#else\nint f(void) { for (;;) {} }\n#endif\n")
    assert r.success and len(r.functions) == 2 and "conditional_compilation" in [w.code for w in r.warnings]


@needs_c
@pytest.mark.parametrize("code, error", [
    ("", "empty"),
    ("int f( { return 1; }", "syntax"),
    ("int f() { FOR(i, n) { x++; } }", "syntax"),  # a macro that changes syntax
    ("int x = 1\n", "syntax"),  # missing ';'
    ("int f(void) { return 0; }\0", "invalid_characters"),
    ("int f(void) { return '\udcff'; }", "encoding"),
    ("int x;\n" * 5001, "too_large"),
    ("class A { public: int x; };", "unsupported_language"),
    ("#include <vector>\nint main(void) { return 0; }", "unsupported_language"),
    ("namespace n { int f(void) { return 0; } }", "unsupported_language"),
    ("template <typename T> T id(T x) { return x; }", "unsupported_language"),
])
def test_c_errors_never_return_partial_metrics(code, error):
    r = c(code)
    assert not r.success and r.errors[0].code == error and r.file_metrics is None and r.functions == []


@needs_c
def test_c_syntax_error_has_location_and_honest_hint():
    e = c("int f(void) {\n    int x = ;\n    return x;\n}\n").errors[0]
    assert e.line == 2 and e.column is not None and "No partial metrics" in e.hint


@needs_c
def test_cpp_markers_inside_comments_and_strings_are_ignored():
    r = c('/* class Foo { */\nint f(void) { const char *s = "std::string"; return 0; }\nint class_(int class_x) '
          '{ return class_x; }\n')
    assert r.success and [f.name for f in r.functions] == ["f", "class_"]


@needs_c
def test_c_bytes_and_line_endings():
    from src.languages.c_lang import CAnalyzer

    r = CAnalyzer().analyze_bytes(b"\xef\xbb\xbfint f(void)\r\n{\r\n    return 0;\r\n}\r\n")
    assert r.success and r.functions[0].start_line == 1 and r.functions[0].end_line == 4
    assert CAnalyzer().analyze_bytes(b"int f(void) { return '\xff'; }").errors[0].code == "encoding"


@needs_c
def test_c_deep_nesting_does_not_hit_recursion_limits():
    depth = 1500
    code = "int f(int a) {\n" + "if (a) {\n" * depth + "a++;\n" + "}\n" * depth + "return a; }\n"
    r = c(code)
    assert r.success and r.functions[0].maximum_nesting_depth == depth
    expr = "int f(int a) { return " + "-" * 5000 + "a; }"
    assert c(expr).success


@needs_c
def test_c_analysis_never_executes_or_compiles(monkeypatch, tmp_path):
    import builtins
    import os
    import subprocess

    def forbid(*a, **k):
        raise AssertionError("execution attempted")

    for mod, name in ((builtins, "exec"), (builtins, "eval"), (os, "system"), (subprocess, "Popen"),
                      (subprocess, "run")):
        monkeypatch.setattr(mod, name, forbid)
    marker = tmp_path / "ran"
    r = c(f'#include <stdlib.h>\nint main(void) {{ system("touch {marker.as_posix()}"); return 0; }}\n')
    assert r.success and not marker.exists()


# --------------------------------------------------------------------------- #
# C examples: documented values, and cross-language comparison
# --------------------------------------------------------------------------- #
@needs_c
@pytest.mark.parametrize("example", C_EXAMPLES, ids=lambda e: e.key)
def test_c_example_documented_structure(example):
    r = c(example.code)
    assert r.success, r.errors
    fm = r.file_metrics
    measured = {"function_count": fm["function_count"], "max_cyclomatic_complexity":
                fm["maximum_function_cyclomatic_complexity"], "max_nesting_depth": fm["maximum_nesting_depth"],
                "loop_count": fm["loop_count"],
                "recursive_function_count": sum(1 for f in r.functions if f.recursive_call_count),
                "structural_category": r.structural_category}
    assert measured == example.structure
    assert r.warnings == []


@needs_c
@pytest.mark.parametrize("example", [e for e in C_EXAMPLES if e.key != "merge_sort"], ids=lambda e: e.key)
def test_c_and_python_versions_agree_where_documented(example):
    py = get_example(example.python_counterpart)
    rc = c(example.code)
    rp = PythonAnalyzer().analyze(py.code)
    main_c = max(rc.functions, key=lambda f: f.cyclomatic_complexity)
    main_p = next(f for f in rp.functions if f.name == main_c.name)
    assert (main_c.cyclomatic_complexity, main_c.maximum_nesting_depth, main_c.recursive_call_count) == (
        main_p.cyclomatic_complexity, main_p.maximum_nesting_depth, main_p.recursive_call_count)


@needs_c
def test_c_merge_sort_difference_is_documented_and_real():
    e = next(x for x in C_EXAMPLES if x.key == "merge_sort")
    assert "differs" in e.comparison_note.lower()
    assert fn(c(e.code), "merge").cyclomatic_complexity == 6
    assert fn(PythonAnalyzer().analyze(get_example("merge_sort").code), "merge").cyclomatic_complexity == 4


@needs_c
def test_c_performance_at_the_size_limit():
    import time

    body = "int f{i}(int *a, int n) {{ int s = 0; for (int i = 0; i < n; i++) {{ if (a[i] > 0) s += a[i]; }} return s; }}\n"
    code = "".join(body.format(i=i) for i in range(1800))
    assert len(code.encode()) < 200_000
    t = time.time()
    r = c(code)
    assert r.success and len(r.functions) == 1800
    assert time.time() - t < 30
