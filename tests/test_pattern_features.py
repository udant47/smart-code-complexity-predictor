"""Tests for src.pattern_features (Phase 10 improvement round)."""

import textwrap

import pytest

from src.code_parser import parse_source
from src.pattern_features import PATTERN_DEFINITIONS, PATTERN_FEATURE_NAMES, extract_pattern_features


def feats(src: str) -> dict:
    return extract_pattern_features(textwrap.dedent(src))


def test_every_feature_defined_and_returned_in_order():
    assert set(PATTERN_DEFINITIONS) == set(PATTERN_FEATURE_NAMES)
    out = feats("x = 1\n")
    assert list(out) == list(PATTERN_FEATURE_NAMES)
    assert all(v == 0 for v in out.values())


def test_sorting_heaps_bisect_hashing():
    f = feats("""
        import heapq, bisect
        from collections import Counter
        a = sorted(xs)
        xs.sort()
        heapq.heappush(h, 1)
        heapq.heappop(h)
        i = bisect.bisect_left(a, 3)
        s = set(a); d = {}; c = Counter(a)
    """)
    assert f["sort_call_count"] == 2
    assert f["heap_call_count"] == 2
    assert f["bisect_call_count"] == 1
    assert f["hash_container_count"] == 3


def test_halving_loops_and_binary_search():
    f = feats("""
        while n > 1:
            n //= 2
        i = 1
        while i < n:
            i *= 2
        lo, hi = 0, len(a) - 1
        while lo <= hi:
            mid = (lo + hi) // 2
            if a[mid] < x:
                lo = mid + 1
            else:
                hi = mid - 1
        while k > 0:
            k -= 1
    """)
    assert f["halving_loop_count"] == 3  # n //= 2, i *= 2, binary search; not k -= 1
    assert f["binary_search_mid_count"] == 1


def test_loop_depth_ignores_constant_loops_and_detects_dependent_bounds():
    f = feats("""
        for i in range(n):
            for d in range(4):
                for j in range(i + 1, n):
                    pass
    """)
    assert f["constant_bound_loop_count"] == 1
    assert f["max_input_loop_depth"] == 2  # range(4) adds no factor of n
    assert f["dependent_inner_loop_count"] == 1


def test_loop_depth_restarts_inside_functions():
    f = feats("""
        for i in range(n):
            def g(m):
                for j in range(m):
                    pass
    """)
    assert f["max_input_loop_depth"] == 1


def test_exponential_patterns():
    f = feats("""
        from itertools import permutations
        for mask in range(1 << n):
            pass
        total = 2 ** n + pow(2, k)
        for p in permutations(items):
            pass
        while True:
            break
    """)
    assert f["exponential_range_loop_count"] == 1
    assert f["power_of_two_expr_count"] == 3
    assert f["combinatoric_call_count"] == 1
    assert f["while_true_count"] == 1


def test_recursion_patterns():
    f = feats("""
        from functools import lru_cache

        @lru_cache(None)
        def fib(n):
            return n if n < 2 else fib(n - 1) + fib(n - 2)

        def msort(a):
            if len(a) < 2:
                return a
            mid = len(a) // 2
            return merge(msort(a[:mid]), msort(a[mid:]))

        def search(lo, hi):
            return search(lo, (lo + hi) // 2)
    """)
    assert f["recursive_function_count"] == 3
    assert f["max_recursive_branching"] == 2
    assert f["dividing_recursion_count"] == 3  # two msort halves + one search half
    assert f["memoized_function_count"] == 1


def test_accepts_tree_and_never_executes(tmp_path):
    marker = tmp_path / "ran"
    src = f"open({str(marker)!r}, 'w')\nfor i in range(1 << 3):\n    pass\n"
    assert extract_pattern_features(parse_source(src).tree)["exponential_range_loop_count"] == 1
    assert not marker.exists()
