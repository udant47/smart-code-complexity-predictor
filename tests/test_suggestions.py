"""Tests for src.suggestions (Phase 15): deterministic, evidence-based review guidance."""

import ast
import json
import textwrap

import pytest

from src import suggestions as sg
from src.code_parser import parse_source


def run(src: str):
    return sg.suggest(textwrap.dedent(src))


def ids(report, target=None):
    return sorted(r.rule_id for r in report.recommendations if target is None or r.target == target)


SIMPLE = '''
def add(a, b):
    return a + b


def total(xs):
    s = 0
    for x in xs:
        if x > 0:
            s += x
    return s


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


def merge_sort(items):
    if len(items) <= 1:
        return items
    mid = len(items) // 2
    return merge(merge_sort(items[:mid]), merge_sort(items[mid:]))


def binary_search(xs, target, lo, hi):
    if lo > hi:
        return -1
    mid = (lo + hi) // 2
    if xs[mid] == target:
        return mid
    elif xs[mid] < target:
        return binary_search(xs, target, mid + 1, hi)
    else:
        return binary_search(xs, target, lo, mid - 1)


def walk(node):
    if node is None:
        return 0
    return 1 + walk(node.left) + walk(node.right)
'''

TANGLED = '''
def process(records, rules, limit, mode, verbose, strict):
    out = []
    for r in records:
        if r is not None:
            for rule in rules:
                if rule.applies(r):
                    if mode == "a" and strict:
                        for k in r.keys():
                            if k in rule.fields:
                                while limit > 0 and r[k]:
                                    limit -= 1
                                    if verbose:
                                        print(k)
                    elif mode == "b" or verbose:
                        out.append(rule.apply(r))
    return out
'''

CHAIN = '''
def code(c):
    if c == "a":
        return 1
    elif c == "b":
        return 2
    elif c == "c":
        return 3
    elif c == "d":
        return 4
    elif c == "e":
        return 5
    return 0
'''


# --------------------------------------------------------------------------- #
# Output format
# --------------------------------------------------------------------------- #
def test_recommendation_has_all_required_fields():
    rep = run(TANGLED)
    assert rep.success and rep.recommendations
    for r in rep.recommendations:
        d = r.to_dict()
        for key in ("rule_id", "title", "severity", "evidence", "explanation", "suggested_action", "target",
                    "start_line", "end_line", "action"):
            assert d[key] not in (None, "", [])
        assert r.severity in sg.SEVERITY_ORDER
        assert all(isinstance(e, sg.Evidence) and e.metric for e in r.evidence)


def test_report_dict_table_and_text():
    rep = run(TANGLED)
    d = rep.to_dict()
    assert set(d) == {"success", "recommendations", "counts", "rules_checked", "functions_checked", "disclaimer",
                      "warnings", "error"}
    assert d["rules_checked"] == [r.rule_id for r in sg.RULES]
    assert sum(d["counts"].values()) == len(d["recommendations"])
    assert "not guaranteed fixes" in d["disclaimer"]
    json.dumps(d)
    t = rep.table()
    assert list(t.columns) == ["Severity", "Rule", "Title", "Where", "Lines", "Evidence", "Suggested action"]
    assert len(t) == len(rep.recommendations)
    assert "not guaranteed fixes" in sg.format_text(rep)


def test_sorted_by_severity_then_line():
    rep = run(TANGLED + CHAIN)
    keys = [(sg.SEVERITY_ORDER[r.severity], r.start_line) for r in rep.recommendations]
    assert keys == sorted(keys)


def test_deterministic():
    a, b = run(TANGLED + CHAIN), run(TANGLED + CHAIN)
    assert a.to_dict() == b.to_dict()


def test_rules_are_modular():
    assert len({r.rule_id for r in sg.RULES}) == len(sg.RULES) >= 6
    only = [r for r in sg.RULES if r.rule_id == "HIGH_NESTING_DEPTH"]
    rep = sg.suggest(textwrap.dedent(TANGLED), rules=only)
    assert ids(rep) == ["HIGH_NESTING_DEPTH"] and rep.rules_checked == ["HIGH_NESTING_DEPTH"]
    for rule in sg.RULES:  # every rule runs alone without errors
        sg.suggest(textwrap.dedent(SIMPLE + TANGLED), rules=[rule])


def test_errors_are_reported():
    bad = sg.suggest("def f(:\n    pass\n")
    assert not bad.success and bad.error["code"] == "syntax" and bad.recommendations == []
    assert sg.suggest(42).error["code"] == "invalid_input_type"
    assert "Could not analyse" in sg.format_text(bad)


# --------------------------------------------------------------------------- #
# No unsupported suggestions
# --------------------------------------------------------------------------- #
def test_ordinary_code_gets_no_recommendations():
    """One loop, one if, recursion with a base case, divide and conquer: nothing to say."""
    rep = run(SIMPLE)
    assert rep.success and rep.recommendations == []
    assert "No recommendations" in sg.format_text(rep)


def test_nonzero_metrics_alone_never_trigger():
    rep = run('''
        def f(xs, n):
            if n < 0:
                raise ValueError("n")
            for x in xs:
                print(x)
            while n:
                n -= 1
            return g(xs) if xs else None
    ''')
    assert rep.recommendations == []


def test_just_below_thresholds():
    body = "\n".join(f"    if x == {i}:\n        y += {i}" for i in range(9))  # cc 10
    rep = sg.suggest(f"def f(x):\n    y = 0\n{body}\n    return y\n")
    assert "HIGH_CYCLOMATIC_COMPLEXITY" not in ids(rep)
    body = "\n".join(f"    if x == {i}:\n        y += {i}" for i in range(10))  # cc 11
    rep = sg.suggest(f"def f(x):\n    y = 0\n{body}\n    return y\n")
    assert "HIGH_CYCLOMATIC_COMPLEXITY" in ids(rep)


# --------------------------------------------------------------------------- #
# Individual rules
# --------------------------------------------------------------------------- #
def test_cyclomatic_complexity_evidence():
    rep = run(TANGLED)
    r = next(r for r in rep.recommendations if r.rule_id == "HIGH_CYCLOMATIC_COMPLEXITY")
    assert r.severity == "medium" and r.target == "process"
    cc = next(e for e in r.evidence if e.metric == "cyclomatic_complexity")
    assert cc.value == 14 and cc.threshold == 11
    parts = next(e for e in r.evidence if e.metric == "decision points by kind").value
    assert parts["if"] == 5 and parts["loops"] == 4


def test_cyclomatic_high_severity():
    body = "\n".join(f"    if x == {i}:\n        y += {i}" for i in range(22))
    rep = sg.suggest(f"def f(x):\n    y = 0\n{body}\n    return y\n")
    r = next(r for r in rep.recommendations if r.rule_id == "HIGH_CYCLOMATIC_COMPLEXITY")
    assert r.severity == "high"


def test_nesting_suggests_guard_clauses_only_with_evidence():
    r = next(r for r in run(TANGLED).recommendations if r.rule_id == "HIGH_NESTING_DEPTH")
    assert r.severity == "high"
    assert any(e.metric == "guard-clause candidates" for e in r.evidence)
    assert "guard clause" in r.suggested_action and "line 5" in r.suggested_action

    no_guard = run('''
        def f(grid):
            total = 0
            for row in grid:
                for cell in row:
                    for part in cell:
                        for v in part:
                            total += v
                    total -= 1
                total += 2
            return total
    ''')
    r = next(r for r in no_guard.recommendations if r.rule_id == "HIGH_NESTING_DEPTH")
    assert r.severity == "medium"
    assert "guard clause" not in r.suggested_action and "Extract" in r.suggested_action
    assert not any(e.metric == "guard-clause candidates" for e in r.evidence)


def test_long_function():
    body = "\n".join(f"    v{i} = data[{i}] * 2" for i in range(55))
    rep = sg.suggest(f"def f(data):\n{body}\n    return v0\n")
    r = next(r for r in rep.recommendations if r.rule_id == "LONG_FUNCTION")
    assert r.severity == "medium" and r.evidence[0].value >= 50
    short = "\n".join(f"    v{i} = data[{i}] * 2" for i in range(40))
    assert "LONG_FUNCTION" not in ids(sg.suggest(f"def f(data):\n{short}\n    return v0\n"))


def test_long_function_names_candidate_sections():
    blocks = "\n".join(f"    for i in range(n):\n        a = i\n        b = a + {k}\n        c = b * 2\n        out.append(c)\n"
                       f"    x{k} = 0\n    y{k} = 1\n    z{k} = 2\n" for k in range(7))
    rep = sg.suggest(f"def f(n):\n    out = []\n{blocks}    return out\n")
    r = next(r for r in rep.recommendations if r.rule_id == "LONG_FUNCTION")
    assert any(e.metric == "top-level blocks of 5+ lines" for e in r.evidence)
    assert "lines 3-7" in r.suggested_action


def test_branch_chain_suggests_lookup_for_same_value():
    rep = run(CHAIN)
    r = next(r for r in rep.recommendations if r.rule_id == "MANY_CONDITIONAL_BRANCHES")
    assert "dictionary lookup" in r.suggested_action
    assert any(e.metric == "if/elif chain" and "5 arms" in e.value for e in r.evidence)


def test_branch_chain_on_different_conditions_does_not_suggest_lookup():
    rep = run('''
        def grade(score, bonus, late):
            if score > 90 and not late:
                return "A"
            elif score + bonus > 80:
                return "B"
            elif late:
                return "C"
            elif bonus:
                return "D"
            elif score < 0:
                raise ValueError(score)
            return "F"
    ''')
    r = next(r for r in rep.recommendations if r.rule_id == "MANY_CONDITIONAL_BRANCHES")
    assert "dictionary lookup" not in r.suggested_action


def test_four_arms_is_fine():
    rep = run(CHAIN.replace('    elif c == "e":\n        return 5\n', ""))
    assert "MANY_CONDITIONAL_BRANCHES" not in ids(rep)


def test_recursion_without_termination():
    rep = run('''
        def countdown(n):
            print(n)
            return countdown(n - 1)
    ''')
    r = next(r for r in rep.recommendations if r.rule_id == "RECURSION_TERMINATION")
    assert r.severity == "high" and "line 4" in r.suggested_action


def test_recursion_with_base_case_in_conditional_expression_is_fine():
    rep = run('''
        def fact(n):
            return 1 if n == 0 else n * fact(n - 1)
    ''')
    assert "RECURSION_TERMINATION" not in ids(rep)
    assert ids(rep) == ["RECURSION_DEPTH"]  # depth equals n, which is a real limit
    r = rep.recommendations[0]
    assert r.severity == "low" and "1000" in r.explanation


def test_branching_recursion_without_memo():
    rep = run('''
        def fib(n):
            if n < 2:
                return n
            return fib(n - 1) + fib(n - 2)
    ''')
    assert ids(rep) == ["RECURSION_REPEATED_WORK"]
    assert "lru_cache" in rep.recommendations[0].suggested_action


def test_branching_recursion_with_memo_is_fine():
    for src in ('''
        from functools import lru_cache
        @lru_cache(maxsize=None)
        def fib(n):
            if n < 2:
                return n
            return fib(n - 1) + fib(n - 2)
    ''', '''
        def fib(n, memo):
            if n in memo:
                return memo[n]
            if n < 2:
                return n
            memo[n] = fib(n - 1, memo) + fib(n - 2, memo)
            return memo[n]
    '''):
        assert "RECURSION_REPEATED_WORK" not in ids(run(src))


def test_exclusive_recursive_calls_are_not_branching():
    rep = run('''
        def count(n):
            if n <= 0:
                return 0
            if n % 2:
                return count(n - 1)
            else:
                return 1 + count(n - 2)
    ''')
    assert "RECURSION_REPEATED_WORK" not in ids(rep)


def test_method_calling_module_function_of_same_name_is_not_recursion():
    rep = run('''
        def height(node):
            return 0 if node is None else 1 + max(height(node.left), height(node.right))

        class Tree:
            def height(self):
                return height(self.root)
    ''')
    assert not [r for r in rep.recommendations if r.target == "Tree.height"]


def test_method_recursion_via_self():
    rep = run('''
        class C:
            def loop(self, n):
                return self.loop(n - 1)
    ''')
    assert ids(rep) == ["RECURSION_TERMINATION"]


def test_recursion_limit_note():
    rep = run('''
        import sys
        sys.setrecursionlimit(10 ** 6)
        def depth(xs):
            if not xs:
                return 0
            return 1 + depth(xs[1:])
    ''')
    r = next(r for r in rep.recommendations if r.rule_id == "RECURSION_DEPTH")
    assert "setrecursionlimit" in r.explanation


DUPLICATES = '''
def total_price(items):
    total = 0
    for item in items:
        if item.price > 0:
            total += item.price * item.qty
    print("total", total)
    return total


def total_weight(parcels):
    weight = 0
    for p in parcels:
        if p.mass > 0:
            weight += p.mass * p.count
    print("weight", weight)
    return weight


def report(a, b):
    x = load(a)
    x = clean(x)
    if x:
        save(x, "out_a")
    log("done a")
    y = load(b)
    y = clean(y)
    if y:
        save(y, "out_b")
    log("done b")
'''


def test_near_duplicate_functions():
    rep = sg.suggest(DUPLICATES)
    r = next(r for r in rep.recommendations if r.title == "Review near-duplicate functions")
    assert r.target == "total_price, total_weight" and r.evidence[0].value >= 0.85


def test_repeated_blocks():
    rep = sg.suggest(DUPLICATES)
    r = next(r for r in rep.recommendations if r.title == "Review repeated code blocks")
    assert r.target == "<file>"
    assert next(e for e in r.evidence if e.metric == "locations").value == ["lines 21-25", "lines 26-30"]


def test_no_duplication_for_different_code():
    assert "DUPLICATED_STRUCTURE" not in ids(run(SIMPLE))
    rep = run('''
        def a(x):
            x = 1
            y = 2
            z = 3
            w = 4
            v = 5
        def b(x):
            for i in range(x):
                if i % 3 == 0:
                    print(i)
                else:
                    continue
            return x
    ''')
    assert "DUPLICATED_STRUCTURE" not in ids(rep)


# --------------------------------------------------------------------------- #
# Consistency
# --------------------------------------------------------------------------- #
def test_one_recommendation_per_rule_and_target():
    rep = run(TANGLED + CHAIN + DUPLICATES)
    keys = [(r.rule_id, r.target, r.title) for r in rep.recommendations]
    assert len(keys) == len(set(keys))


def test_no_contradictory_actions():
    """Rules only ever simplify/flatten/split/check/deduplicate; nothing asks for the opposite."""
    allowed = {"split", "flatten", "simplify_conditions", "check_recursion", "deduplicate"}
    rep = run(TANGLED + CHAIN + DUPLICATES)
    assert {r.action for r in rep.recommendations} <= allowed
    for word in ("inline", "add nesting", "merge these branches into one function"):
        assert all(word not in r.suggested_action.lower() for r in rep.recommendations)


def test_generic_branch_advice_is_folded_into_complexity():
    body = "\n".join(f"    if x == {i}:\n        y += 1\n    if z > {i}:\n        y -= 1" for i in range(8))
    rep = sg.suggest(f"def f(x, z):\n    y = 0\n{body}\n    return y\n")  # 16 ifs, no chain
    assert "HIGH_CYCLOMATIC_COMPLEXITY" in ids(rep) and "MANY_CONDITIONAL_BRANCHES" not in ids(rep)
    r = next(r for r in rep.recommendations if r.rule_id == "HIGH_CYCLOMATIC_COMPLEXITY")
    assert any("folded in" in n for n in r.related)
    assert any("conditional branches" in e.metric for e in r.evidence)


def test_script_without_functions_is_analysed_as_module():
    body = "\n".join(f"if x == {i}:\n    y += 1" for i in range(12))
    rep = sg.suggest(f"x = int(input())\ny = 0\n{body}\nprint(y)\n")
    assert rep.functions_checked == 0 and any("module-level" in w for w in rep.warnings)
    assert {r.target for r in rep.recommendations} <= {"<module>", "<file>"}
    assert "HIGH_CYCLOMATIC_COMPLEXITY" in ids(rep)


def test_accepts_parse_result_and_bytes():
    a = sg.suggest(parse_source(textwrap.dedent(TANGLED)))
    b = sg.suggest(textwrap.dedent(TANGLED).encode("utf-8"))
    assert a.to_dict() == b.to_dict()


def test_code_is_never_executed(tmp_path):
    marker = tmp_path / "ran.txt"
    src = f"open({str(marker)!r}, 'w').write('x')\n" + textwrap.dedent(TANGLED)
    sg.suggest(src)
    assert not marker.exists()


def test_predictor_includes_recommendations(monkeypatch):
    from src import predictor as pr

    def missing(*a, **k):
        raise pr.ModelUnavailable("model_missing", "no model")

    monkeypatch.setattr(pr, "get_model", missing)
    r = pr.predict(textwrap.dedent(TANGLED))
    assert r.recommendations["success"] and r.recommendations["recommendations"]
    assert r.recommendations == sg.suggest(textwrap.dedent(TANGLED)).to_dict()  # works without the model
