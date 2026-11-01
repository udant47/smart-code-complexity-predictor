"""Tests for src.algorithm_analyzer (Phase 6).

Each example states the expected Big-O and the assumption behind it.
Expectations are worst-case bounds from standard algorithm analysis, not
values read back from the implementation.
"""

import textwrap

import pytest

from src.algorithm_analyzer import analyze_algorithms
from src.asymptotics import Cost, Mono


def estimate(src: str, function_id: str = None):
    result = analyze_algorithms(textwrap.dedent(src))
    assert result.success, result.error
    return result.get(function_id) if function_id else result.estimates[0]


def assert_estimate(e, time, space, time_class=None, space_class=None):
    assert e.time_complexity == time, e.explanation
    assert e.space_complexity == space, e.explanation
    assert e.time_class == (time_class or time)
    assert e.space_class == (space_class or space)
    assert e.status == "estimated" and e.time_status == e.space_status == "estimated"


def assert_unknown_time(e, reason_fragment):
    assert e.time_complexity == "Unknown" and e.time_class == "Unknown"
    assert e.time_status == "unknown" and e.time_confidence is None
    assert e.status in ("unknown", "partial")
    assert reason_fragment in e.explanation, e.explanation


# --------------------------------------------------------------------------- #
# Required examples
# --------------------------------------------------------------------------- #
def test_constant_time_arithmetic():
    e = estimate("""
        def area(w, h):
            return w * h
    """)
    assert_estimate(e, "O(1)", "O(1)")
    assert e.time_confidence == "high"
    assert "no loops" in e.explanation


def test_linear_search():
    # Worst case: the target is absent, so every element is compared.
    e = estimate("""
        def linear_search(arr, target):
            for i, x in enumerate(arr):
                if x == target:
                    return i
            return -1
    """)
    assert_estimate(e, "O(n)", "O(1)")
    assert e.size_variables == {"n": "len(arr)"}
    assert any("Worst case" in a for a in e.assumptions)


def test_binary_search_iterative():
    e = estimate("""
        def binary_search(arr, target):
            lo, hi = 0, len(arr) - 1
            while lo <= hi:
                mid = (lo + hi) // 2
                if arr[mid] == target:
                    return mid
                elif arr[mid] < target:
                    lo = mid + 1
                else:
                    hi = mid - 1
            return -1
    """)
    assert_estimate(e, "O(log n)", "O(1)")
    assert any("binary search" in ev for ev in e.evidence)


def test_binary_search_recursive():
    # Halves hi - lo per call; the call stack is log n deep.
    e = estimate("""
        def bsearch(arr, target, lo, hi):
            if lo > hi:
                return -1
            mid = (lo + hi) // 2
            if arr[mid] == target:
                return mid
            if arr[mid] < target:
                return bsearch(arr, target, mid + 1, hi)
            return bsearch(arr, target, lo, mid - 1)
    """)
    assert_estimate(e, "O(log n)", "O(log n)")
    assert e.size_variables == {"n": "hi - lo"}
    assert "T(n) = 1·T(n/2)" in e.recurrence


def test_simple_nested_loops():
    # Triangular loop: n(n-1)/2 comparisons = Θ(n^2).
    e = estimate("""
        def has_duplicate_pair(a):
            for i in range(len(a)):
                for j in range(i + 1, len(a)):
                    if a[i] == a[j]:
                        return True
            return False
    """)
    assert_estimate(e, "O(n^2)", "O(1)")
    assert any("Triangular" in a for a in e.assumptions)


MERGE_SORT = """
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
        result.extend(left[i:])
        result.extend(right[j:])
        return result


    def merge_sort(items):
        if len(items) <= 1:
            return items
        mid = len(items) // 2
        left = merge_sort(items[:mid])
        right = merge_sort(items[mid:])
        return merge(left, right)
"""


def test_merge_helper_two_pointers():
    e = estimate(MERGE_SORT, "merge")
    assert_estimate(e, "O(n + m)", "O(n + m)", "O(n)", "O(n)")
    assert e.size_variables == {"n": "len(left)", "m": "len(right)"}


def test_merge_sort():
    # T(n) = 2T(n/2) + O(n) -> O(n log n); slices and merged lists alive
    # along one chain of calls sum to O(n).
    e = estimate(MERGE_SORT, "merge_sort")
    assert_estimate(e, "O(n log n)", "O(n)")
    assert e.recurrence.startswith("T(n) = 2·T(n/2)")
    assert any("calls merge()" in ev for ev in e.evidence)
    # Depends on assuming the recursive call returns a list of the same size.
    assert e.time_confidence == "medium"


def test_recursive_fibonacci():
    # Two calls shrinking by 1 and 2: Θ(φ^n), reported as the O(2^n) upper bound.
    e = estimate("""
        def fib(n):
            if n < 2:
                return n
            return fib(n - 1) + fib(n - 2)
    """)
    assert_estimate(e, "O(2^n)", "O(n)")
    assert any("φ" in a for a in e.assumptions)
    assert e.size_variables == {"n": "value of n"}


def test_two_sequential_loops_are_linear_not_quadratic():
    e = estimate("""
        def min_and_max(values):
            lo = values[0]
            for v in values:
                lo = min(lo, v)
            hi = values[0]
            for v in values:
                hi = max(hi, v)
            return lo, hi
    """)
    assert_estimate(e, "O(n)", "O(1)")


@pytest.mark.parametrize(
    "src, reason",
    [
        ("""
        def repeat_first(arr):
            total = 0
            for i in range(arr[0]):
                total += i
            return total
        """, "range bound arr[0] is not statically known"),
        ("""
        def collatz_steps(n):
            steps = 0
            while n != 1:
                n = n // 2 if n % 2 == 0 else 3 * n + 1
                steps += 1
            return steps
        """, "not updated by a constant step"),
        ("""
        def bfs(graph, start):
            seen = {start}
            queue = [start]
            while queue:
                node = queue.pop()
                for nb in graph[node]:
                    if nb not in seen:
                        seen.add(nb)
                        queue.append(nb)
            return seen
        """, "depends on the data"),
        ("""
        def first_negative(xs):
            i = 0
            while i < len(xs):
                if xs[i] >= 0:
                    i += 1
            return i
        """, "only updated in some iterations"),
    ],
    ids=["value-dependent-range", "collatz", "bfs-worklist", "conditional-update"],
)
def test_data_dependent_bounds_are_unknown(src, reason):
    assert_unknown_time(estimate(src), reason)


# --------------------------------------------------------------------------- #
# Nested loops do not always mean O(n^2)
# --------------------------------------------------------------------------- #
def test_nested_loop_with_constant_inner_bound_is_linear():
    e = estimate("""
        def neighbours(cells):
            total = 0
            for c in cells:
                for d in range(4):
                    total += d
            return total
    """)
    assert_estimate(e, "O(n)", "O(1)")


def test_nested_loops_over_different_inputs():
    e = estimate("""
        def pairs(xs, ys):
            out = []
            for x in xs:
                for y in ys:
                    out.append((x, y))
            return out
    """)
    assert_estimate(e, "O(n·m)", "O(n·m)", "O(n^2)", "O(n^2)")
    assert e.size_variables == {"n": "len(xs)", "m": "len(ys)"}
    assert e.time_confidence == "medium"
    assert any("grow proportionally" in a for a in e.assumptions)


def test_loop_with_inner_logarithmic_loop():
    e = estimate("""
        def bits_total(nums, n):
            total = 0
            for x in nums:
                k = n
                while k > 1:
                    k //= 2
                    total += 1
            return total
    """)
    assert e.time_complexity == "O(m·log n)" or e.time_complexity == "O(n·log m)"
    assert e.time_class == "O(n log n)"


def test_triple_loop_matrix_multiplication():
    e = estimate("""
        def matmul(A, B, n):
            C = [[0] * n for _ in range(n)]
            for i in range(n):
                for j in range(n):
                    for k in range(n):
                        C[i][j] += A[i][k] * B[k][j]
            return C
    """)
    assert_estimate(e, "O(n^3)", "O(n^2)")


# --------------------------------------------------------------------------- #
# Logarithmic and sub-linear loops
# --------------------------------------------------------------------------- #
def test_halving_loop():
    e = estimate("""
        def halvings(n):
            count = 0
            while n > 1:
                n //= 2
                count += 1
            return count
    """)
    assert_estimate(e, "O(log n)", "O(1)")


def test_doubling_loop():
    e = estimate("""
        def powers_below(n):
            i = 1
            while i < n:
                i *= 2
            return i
    """)
    assert_estimate(e, "O(log n)", "O(1)")


def test_square_root_loop():
    e = estimate("""
        def is_prime(n):
            i = 2
            while i * i <= n:
                if n % i == 0:
                    return False
                i += 1
            return True
    """)
    assert_estimate(e, "O(√n)", "O(1)")


# --------------------------------------------------------------------------- #
# Recursion
# --------------------------------------------------------------------------- #
def test_linear_recursion():
    e = estimate("""
        def factorial(n):
            return 1 if n <= 1 else n * factorial(n - 1)
    """)
    assert_estimate(e, "O(n)", "O(n)")  # O(n) stack frames


def test_recursion_with_slicing_is_quadratic():
    # Each call copies arr[1:] (O(n)); n calls -> O(n^2) time, and the
    # copies alive along the call chain total O(n^2) space.
    e = estimate("""
        def rsum(arr):
            if not arr:
                return 0
            return arr[0] + rsum(arr[1:])
    """)
    assert_estimate(e, "O(n^2)", "O(n^2)")


def test_towers_of_hanoi():
    e = estimate("""
        def hanoi(n, src, dst, via):
            if n == 0:
                return
            hanoi(n - 1, src, via, dst)
            print(src, dst)
            hanoi(n - 1, via, dst, src)
    """)
    assert_estimate(e, "O(2^n)", "O(n)")


def test_calls_in_different_branches_are_not_added():
    e = estimate("""
        def walk(n):
            if n <= 0:
                return 0
            if n % 2:
                return walk(n - 1)
            return walk(n - 1)
    """)
    assert_estimate(e, "O(n)", "O(n)")


def test_memoized_fibonacci():
    e = estimate("""
        from functools import lru_cache

        @lru_cache(maxsize=None)
        def fib(n):
            if n < 2:
                return n
            return fib(n - 1) + fib(n - 2)
    """)
    assert_estimate(e, "O(n)", "O(n)")
    assert any("memoization" in a.lower() for a in e.assumptions)


MANUAL_MEMO = """
    def ways(n, memo{annotation}):
        if n in memo:
            return memo[n]
        if n <= 1:
            return 1
        memo[n] = ways(n - 1, memo) + ways(n - 2, memo)
        return memo[n]
"""


def test_manual_memoization():
    e = estimate(MANUAL_MEMO.format(annotation=": dict"))
    assert e.time_complexity == "O(n)"
    assert any("Manual memoization" in a for a in e.assumptions)


def test_manual_memoization_without_type_stays_conservative():
    # Without a type, `n in memo` may be a list scan: the estimate says so.
    e = estimate(MANUAL_MEMO.format(annotation=""))
    assert e.time_complexity == "O(n·m)"
    assert any("linear scan" in a for a in e.assumptions)


def test_permutation_backtracking_is_factorial():
    e = estimate("""
        def permute(prefix, rest, out):
            if not rest:
                out.append(prefix)
                return
            for i in range(len(rest)):
                permute(prefix + [rest[i]], rest[:i] + rest[i + 1:], out)
    """)
    assert "m!" in e.time_complexity and e.size_variables["m"] == "len(rest)"
    assert e.time_class == "O(n·n!)"
    assert e.time_confidence == "medium"


def test_quicksort_partitions_are_unknown():
    e = estimate("""
        def quicksort(arr):
            if len(arr) <= 1:
                return arr
            pivot = arr[0]
            left = [x for x in arr[1:] if x < pivot]
            right = [x for x in arr[1:] if x >= pivot]
            return quicksort(left) + [pivot] + quicksort(right)
    """)
    assert_unknown_time(e, "quicksort partitions")


def test_euclid_gcd_is_unknown():
    # Euclid is O(log min(a, b)), but proving that needs number theory,
    # not syntax: the arguments swap and shrink by a remainder.
    e = estimate("""
        def gcd(a, b):
            return a if b == 0 else gcd(b, a % b)
    """)
    assert_unknown_time(e, "do not all shrink the same argument")


def test_recursion_without_base_case_is_unknown():
    e = estimate("""
        def forever(n):
            return forever(n - 1)
    """)
    assert_unknown_time(e, "base case")


def test_recursion_on_unchanged_input_is_unknown():
    e = estimate("""
        def loop(xs):
            if not xs:
                return 0
            return loop(xs)
    """)
    assert_unknown_time(e, "do not all shrink")


def test_recursive_method_via_self():
    e = estimate("""
        class Tree:
            def depth(self, n):
                if n <= 0:
                    return 0
                return 1 + self.depth(n // 2)
    """, "Tree.depth")
    assert_estimate(e, "O(log n)", "O(log n)")


# --------------------------------------------------------------------------- #
# Built-in operation costs
# --------------------------------------------------------------------------- #
def test_sorting_is_n_log_n_with_linear_space():
    e = estimate("""
        def max_gap(xs):
            ys = sorted(xs)
            best = 0
            for i in range(1, len(ys)):
                best = max(best, ys[i] - ys[i - 1])
            return best
    """)
    assert_estimate(e, "O(n log n)", "O(n)")


def test_in_place_sort():
    e = estimate("""
        def sort_in_place(xs):
            xs.sort()
            return xs[0]
    """)
    assert_estimate(e, "O(n log n)", "O(n)")  # Timsort uses up to n/2 extra slots


def test_membership_in_list_vs_set():
    slow = estimate("""
        def common(a, b):
            return [x for x in a if x in b]
    """)
    assert slow.time_complexity == "O(n·m)"
    assert any("linear scan" in a for a in slow.assumptions)

    fast = estimate("""
        def common_fast(a, b):
            seen = set(b)
            return [x for x in a if x in seen]
    """)
    assert_estimate(fast, "O(n + m)", "O(n + m)", "O(n)", "O(n)")


def test_type_annotation_makes_membership_constant():
    e = estimate("""
        def common(a, b: set):
            return [x for x in a if x in b]
    """)
    assert e.time_complexity == "O(n)"


def test_list_pop_front_is_linear():
    e = estimate("""
        def drain(xs):
            for _ in range(len(xs)):
                xs.pop(0)
    """)
    assert e.time_class == "O(n^2)"


def test_dict_counting():
    e = estimate("""
        def count_items(xs):
            counts = {}
            for x in xs:
                counts[x] = counts.get(x, 0) + 1
            return counts
    """)
    assert_estimate(e, "O(n)", "O(n)")


def test_heap_operations():
    e = estimate("""
        import heapq

        def k_smallest(xs, k):
            heap = []
            for x in xs:
                heapq.heappush(heap, x)
            return [heapq.heappop(heap) for _ in range(k)]
    """)
    assert e.time_status == "estimated"
    assert "log" in e.time_complexity


# --------------------------------------------------------------------------- #
# Auxiliary space
# --------------------------------------------------------------------------- #
def test_input_is_not_counted_as_auxiliary_space():
    e = estimate("""
        def total(xs):
            s = 0
            for x in xs:
                s += x
            return s
    """)
    assert e.space_complexity == "O(1)"


def test_temporaries_in_a_loop_count_once():
    e = estimate("""
        def rows(n):
            best = 0
            for i in range(n):
                row = [0] * n
                best = max(best, len(row))
            return best
    """)
    assert e.space_complexity == "O(n)"  # not n^2: each row is discarded
    assert e.time_complexity == "O(n^2)"


def test_appends_in_a_loop_accumulate():
    e = estimate("""
        def squares(n):
            out = []
            for i in range(n):
                out.append(i * i)
            return out
    """)
    assert_estimate(e, "O(n)", "O(n)")


def test_two_dimensional_allocation():
    e = estimate("""
        def grid(rows, cols):
            return [[0] * cols for _ in range(rows)]
    """)
    assert e.space_complexity == "O(n·m)"


# --------------------------------------------------------------------------- #
# Calls, confidence and status
# --------------------------------------------------------------------------- #
def test_unresolved_call_lowers_confidence():
    e = estimate("""
        def uses_helper(xs):
            for x in xs:
                external_api(x)
    """)
    assert e.time_complexity == "O(n)"
    assert e.time_confidence == "low"
    assert any("external_api() is not defined" in a for a in e.assumptions)


def test_local_helper_cost_is_substituted():
    e = estimate("""
        def contains(xs, v):
            for x in xs:
                if x == v:
                    return True
            return False

        def count_present(xs, queries):
            return sum(1 for q in queries if contains(xs, q))
    """, "count_present")
    assert e.time_complexity == "O(n·m)"
    assert e.size_variables == {"n": "len(xs)", "m": "len(queries)"}


def test_mutual_recursion_is_unknown():
    e = estimate("""
        def is_even(n):
            return True if n == 0 else is_odd(n - 1)

        def is_odd(n):
            return False if n == 0 else is_even(n - 1)
    """, "is_even")
    assert e.time_status == "unknown"


def test_infinite_loop_is_unknown():
    assert_unknown_time(estimate("""
        def spin():
            while True:
                pass
    """), "constant condition")


def test_partial_status_when_only_space_is_known():
    e = estimate("""
        def collatz(n):
            while n != 1:
                n = n // 2 if n % 2 == 0 else 3 * n + 1
    """)
    assert e.status == "partial"
    assert e.space_complexity == "O(1)" and e.space_status == "estimated"


# --------------------------------------------------------------------------- #
# Schema, determinism, errors, safety
# --------------------------------------------------------------------------- #
SCHEMA_KEYS = {
    "function_id", "function_name", "qualname", "time_complexity", "space_complexity",
    "time_class", "space_class", "status", "time_status", "space_status",
    "time_confidence", "space_confidence", "explanation", "assumptions", "evidence",
    "size_variables", "recurrence",
}


def test_every_estimate_has_schema_explanation_and_status():
    result = analyze_algorithms(textwrap.dedent(MERGE_SORT) + "\n\ndef f():\n    while True:\n        pass\n")
    assert len(result.estimates) == 3
    for e in result.estimates:
        d = e.to_dict()
        assert set(d) == SCHEMA_KEYS
        assert d["explanation"]
        assert d["status"] in ("estimated", "partial", "unknown")
        assert isinstance(d["assumptions"], list)


def test_ids_match_function_analyzer():
    result = analyze_algorithms("class A:\n    def f(self):\n        pass\n\nclass B:\n    def f(self):\n        pass\n")
    assert [e.function_id for e in result.estimates] == ["A.f", "B.f"]


def test_deterministic():
    src = textwrap.dedent(MERGE_SORT)
    assert analyze_algorithms(src).estimates == analyze_algorithms(src).estimates


def test_syntax_error():
    result = analyze_algorithms("def f(:\n")
    assert not result.success and result.error.kind == "syntax"


def test_analysis_never_executes_code(tmp_path):
    marker = tmp_path / "ran.txt"
    src = f"""
def f(n):
    open({str(marker)!r}, "w").write("x")
    return f(n - 1) if n else 0
"""
    analyze_algorithms(src)
    assert not marker.exists()


# --------------------------------------------------------------------------- #
# Asymptotic algebra
# --------------------------------------------------------------------------- #
def test_algebra_drops_dominated_terms_but_keeps_incomparable_ones():
    n, m = Mono.sym("n"), Mono.sym("m")
    assert Cost.of(n * n, n).render({"n": "n"}) == "O(n^2)"
    assert Cost.of(n, m).render({"n": "n", "m": "m"}) == "O(n + m)"
    assert Cost.of(n * m, n).render({"n": "n", "m": "m"}) == "O(n·m)"
    assert Cost.of(Mono.exp("n", 2), n * n * n).render({"n": "n"}) == "O(2^n)"
    assert Cost.of(n * Mono.log("n")).render({"n": "n"}) == "O(n log n)"
    assert (Cost.of(n) + Cost.unknown()).is_unknown


def test_collapse_to_single_variable():
    n, m = Mono.sym("a"), Mono.sym("b")
    assert Cost.of(n * m).collapse().render({"n": "n"}) == "O(n^2)"
    assert Cost.of(n, m).collapse().render({"n": "n"}) == "O(n)"


def test_merge_sort_with_inline_recursive_calls():
    # merge(merge_sort(left half), merge_sort(right half)) without temporary variables
    src = MERGE_SORT.replace(
        "    left = merge_sort(items[:mid])\n        right = merge_sort(items[mid:])\n        return merge(left, right)",
        "    return merge(merge_sort(items[:mid]), merge_sort(items[mid:]))",
    )
    assert "merge(merge_sort(" in src
    e = estimate(src, "merge_sort")
    assert_estimate(e, "O(n log n)", "O(n)")
