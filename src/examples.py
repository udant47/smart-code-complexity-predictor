"""Documented demonstration examples (Phase 17).

Each example is a small, valid Python program representing one structural
pattern. For every example we document:

* the expected *structural* characteristics (function count, highest
  cyclomatic complexity, maximum nesting depth, loops, recursion). These are
  deterministic properties of the source, and ``tests/test_examples.py``
  checks every documented value against the analyzer, so the documentation
  cannot drift from what the app actually measures;
* the expected asymptotic time and auxiliary space of each function, with the
  assumptions that make the statement true (input-size variable, cost of
  primitive operations, worst/average case);
* the growth class the *static analyzer* is expected to report, which may be
  a looser bound than the tight one (e.g. ``O(2^n)`` for naive Fibonacci,
  whose tight bound is Θ(φ^n)), or ``None`` where the analyzer is expected
  to return Unknown.

No expected **ML** class is documented anywhere. The ML model is a
statistical classifier with a measured test macro F1 of about 0.44; its
output for any example may differ from the documented complexity, and the
app shows that disagreement rather than hiding it.

The code is only ever parsed, never executed.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Optional

# --------------------------------------------------------------------------- #
# Data model
# --------------------------------------------------------------------------- #
STRUCTURE_LABELS = {
    "function_count": "Functions",
    "max_cyclomatic_complexity": "Highest cyclomatic complexity",
    "max_nesting_depth": "Maximum nesting depth",
    "loop_count": "Loops",
    "recursive_function_count": "Recursive functions",
    "structural_category": "Structural category",
}


@dataclass(frozen=True)
class FunctionComplexity:
    """Expected asymptotic behaviour of one function, under explicit assumptions."""

    function: str
    time: str  # human statement, e.g. "O(n) worst case, O(1) best case"
    space: str  # auxiliary space
    assumptions: tuple[str, ...]
    static_time_class: Optional[str]  # single-variable class the static analyzer should report; None = Unknown
    note: str = ""


@dataclass(frozen=True)
class Example:
    key: str
    number: int
    title: str
    pattern: str
    summary: str
    code: str
    structure: dict  # keys of STRUCTURE_LABELS -> expected value
    characteristics: tuple[str, ...]
    complexity: tuple[FunctionComplexity, ...]
    expected_recommendations: tuple[str, ...] = ()  # rule ids the suggestion engine should raise

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class ExpectedChange:
    function: str
    time_before: str
    time_after: str
    space_before: str
    space_after: str


@dataclass(frozen=True)
class ComparisonPair:
    key: str
    title: str
    summary: str
    before: str
    after: str
    behaviour: str  # whether both versions compute the same thing, and any caveat
    changes: tuple[str, ...]  # what is expected to differ in the measurements
    unchanged: tuple[str, ...]  # what is expected to stay the same
    expected: tuple[ExpectedChange, ...]
    lesson: str

    def to_dict(self) -> dict:
        return asdict(self)


COMMON_ASSUMPTIONS = (
    "Arithmetic, comparisons, indexing and attribute access cost O(1).",
    "Integers are treated as fixed size (Python's big integers grow, which matters for very large values).",
)

# --------------------------------------------------------------------------- #
# Examples
# --------------------------------------------------------------------------- #
EXAMPLES: tuple[Example, ...] = (
    Example(
        key="simple_arithmetic",
        number=1,
        title="1. Simple arithmetic",
        pattern="Straight-line code",
        summary="Three small functions with no loops, branches or calls to other functions.",
        code='''def celsius_to_fahrenheit(celsius):
    return celsius * 9 / 5 + 32


def circle_area(radius):
    pi = 3.141592653589793
    return pi * radius * radius


def average_of_three(a, b, c):
    total = a + b + c
    return total / 3
''',
        structure={"function_count": 3, "max_cyclomatic_complexity": 1, "max_nesting_depth": 0,
                   "loop_count": 0, "recursive_function_count": 0, "structural_category": "Low"},
        characteristics=(
            "Every function has exactly one path through it (cyclomatic complexity 1).",
            "No nesting, loops or recursion.",
        ),
        complexity=tuple(
            FunctionComplexity(f, "O(1)", "O(1)", COMMON_ASSUMPTIONS, "O(1)")
            for f in ("celsius_to_fahrenheit", "circle_area", "average_of_three")
        ),
    ),
    Example(
        key="linear_search",
        number=2,
        title="2. Linear search",
        pattern="Single loop with early exit",
        summary="Scans a sequence from the start until it finds the target.",
        code='''def linear_search(items, target):
    for index, value in enumerate(items):
        if value == target:
            return index
    return -1
''',
        structure={"function_count": 1, "max_cyclomatic_complexity": 3, "max_nesting_depth": 2,
                   "loop_count": 1, "recursive_function_count": 0, "structural_category": "Low"},
        characteristics=(
            "One loop containing one condition: cyclomatic complexity 3 (1 + loop + if).",
            "Nesting depth 2 (the if inside the for).",
        ),
        complexity=(FunctionComplexity(
            "linear_search", "O(n) worst case (target absent or last); O(1) best case (target first)", "O(1)",
            ("n = len(items).", "Equality comparison costs O(1).",
             "The static analyzer reports the worst case: it assumes the loop runs to completion."),
            "O(n)"),),
    ),
    Example(
        key="binary_search",
        number=3,
        title="3. Binary search",
        pattern="Halving loop",
        summary="Repeatedly halves the search range of a sorted sequence.",
        code='''def binary_search(sorted_items, target):
    low, high = 0, len(sorted_items) - 1
    while low <= high:
        mid = (low + high) // 2
        if sorted_items[mid] == target:
            return mid
        if sorted_items[mid] < target:
            low = mid + 1
        else:
            high = mid - 1
    return -1
''',
        structure={"function_count": 1, "max_cyclomatic_complexity": 4, "max_nesting_depth": 2,
                   "loop_count": 1, "recursive_function_count": 0, "structural_category": "Low"},
        characteristics=(
            "One while loop with two decisions inside: cyclomatic complexity 4.",
            "The midpoint (low + high) // 2 halves the remaining range each iteration.",
        ),
        complexity=(FunctionComplexity(
            "binary_search", "O(log n) worst case; O(1) best case", "O(1)",
            ("n = len(sorted_items).", "The input is sorted in ascending order. On unsorted input the "
             "function still runs in O(log n) but can return a wrong answer.",
             "Indexing a list costs O(1)."),
            "O(log n)"),),
    ),
    Example(
        key="nested_loops",
        number=4,
        title="4. Nested-loop processing",
        pattern="Nested loops",
        summary="All-pairs counting and a row-by-row matrix total, two common nested-loop shapes.",
        code='''def count_pairs_with_sum(values, target):
    count = 0
    n = len(values)
    for i in range(n):
        for j in range(i + 1, n):
            if values[i] + values[j] == target:
                count += 1
    return count


def row_totals(matrix):
    totals = []
    for row in matrix:
        total = 0
        for cell in row:
            total += cell
        totals.append(total)
    return totals
''',
        structure={"function_count": 2, "max_cyclomatic_complexity": 4, "max_nesting_depth": 3,
                   "loop_count": 4, "recursive_function_count": 0, "structural_category": "Low"},
        characteristics=(
            "Two loops per function, one nested inside the other.",
            "count_pairs_with_sum reaches nesting depth 3 (for → for → if).",
            "Low cyclomatic complexity does not mean low running time: the pair count is quadratic.",
        ),
        complexity=(
            FunctionComplexity(
                "count_pairs_with_sum", "O(n²): exactly n(n−1)/2 inner iterations", "O(1)",
                ("n = len(values).", "Addition and comparison cost O(1)."), "O(n^2)"),
            FunctionComplexity(
                "row_totals", "O(r·c) for r rows of c cells (O(n²) for an n×n matrix)",
                "O(r) for the list of totals",
                ("r = number of rows, c = cells per row (rows may differ; then it is the total cell count).",),
                "O(n^2)", note="The analyzer reports O(n·m) with two size variables; collapsed to one size it is O(n^2)."),
        ),
    ),
    Example(
        key="recursive_fibonacci",
        number=5,
        title="5. Recursive Fibonacci",
        pattern="Branching recursion",
        summary="The textbook two-call recursive definition, without memoization.",
        code='''def fibonacci(n):
    if n < 2:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)
''',
        structure={"function_count": 1, "max_cyclomatic_complexity": 2, "max_nesting_depth": 1,
                   "loop_count": 0, "recursive_function_count": 1, "structural_category": "Low"},
        characteristics=(
            "Structurally tiny: one condition, cyclomatic complexity 2.",
            "Two recursive calls per invocation, so the call tree branches.",
            "A clear case where structural metrics and running time disagree.",
        ),
        complexity=(FunctionComplexity(
            "fibonacci", "Θ(φⁿ) ≈ Θ(1.618ⁿ); O(2ⁿ) is a valid but looser upper bound",
            "O(n) for the recursion stack",
            ("n is the integer argument (its value, not an input length).", "Addition costs O(1).",
             "No memoization or caching."),
            "O(2^n)", note="The static analyzer reports the upper bound O(2^n)."),),
        expected_recommendations=("RECURSION_REPEATED_WORK",),
    ),
    Example(
        key="merge_sort",
        number=6,
        title="6. Merge sort",
        pattern="Divide and conquer",
        summary="Splits the list in half, sorts each half recursively, and merges the results.",
        code='''def merge(left, right):
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
''',
        structure={"function_count": 2, "max_cyclomatic_complexity": 4, "max_nesting_depth": 2,
                   "loop_count": 1, "recursive_function_count": 1, "structural_category": "Low"},
        characteristics=(
            "merge has one loop with a compound condition (and) and an if/else: cyclomatic complexity 4.",
            "merge_sort makes two recursive calls on halves of its input.",
        ),
        complexity=(
            FunctionComplexity(
                "merge", "O(a + b) for inputs of lengths a and b", "O(a + b) for the result list",
                ("list.append is amortised O(1).", "Slicing copies, so the final concatenation is O(a + b)."),
                "O(n)", note="The analyzer reports O(n + m); collapsed to one size it is O(n)."),
            FunctionComplexity(
                "merge_sort", "O(n log n) in every case", "O(n) auxiliary (slices and merged lists), "
                "O(log n) recursion depth",
                ("n = len(items).", "Recurrence T(n) = 2T(n/2) + O(n), solved by the master theorem."),
                "O(n log n)"),
        ),
    ),
    Example(
        key="complex_branching",
        number=7,
        title="7. Complex branching",
        pattern="Many decision paths",
        summary="A pricing function with several if/elif chains and compound conditions.",
        code='''def shipping_cost(weight, zone, express, fragile, member):
    if weight <= 0:
        raise ValueError("weight must be positive")
    if zone == "local":
        cost = 5.0
    elif zone == "regional":
        cost = 9.0
    elif zone == "national":
        cost = 14.0
    elif zone == "international":
        cost = 30.0
    else:
        raise ValueError("unknown zone")
    if weight > 20:
        cost += 25.0
    elif weight > 5:
        cost += 10.0
    elif weight > 1:
        cost += 3.0
    if express and zone != "international":
        cost *= 1.5
    elif express:
        cost *= 2.0
    if fragile:
        cost += 4.0
    if member and cost > 20:
        cost -= 5.0
    return round(cost, 2)
''',
        structure={"function_count": 1, "max_cyclomatic_complexity": 15, "max_nesting_depth": 1,
                   "loop_count": 0, "recursive_function_count": 0, "structural_category": "High"},
        characteristics=(
            "Twelve if/elif decisions plus two boolean 'and' operators: cyclomatic complexity 15.",
            "Shallow (nesting depth 1) but with many independent paths to test.",
            "High structural complexity with constant running time.",
        ),
        complexity=(FunctionComplexity(
            "shipping_cost", "O(1)", "O(1)", COMMON_ASSUMPTIONS + ("The number of branches is fixed by the code, "
                                                                    "not by the input.",), "O(1)"),),
        expected_recommendations=("HIGH_CYCLOMATIC_COMPLEXITY",),
    ),
    Example(
        key="deep_nesting",
        number=8,
        title="8. Deeply nested conditions",
        pattern="Arrow-shaped nesting",
        summary="A validation function where each check is nested inside the previous one.",
        code='''def approve_loan(applicant):
    if applicant is not None:
        if applicant.age >= 18:
            if applicant.income > 0:
                if applicant.credit_score >= 650:
                    if applicant.debt / applicant.income < 0.4:
                        if not applicant.has_defaults:
                            return "approved"
                        return "manual review"
                    return "debt too high"
                return "credit score too low"
            return "no income"
        return "under age"
    return "no applicant"
''',
        structure={"function_count": 1, "max_cyclomatic_complexity": 7, "max_nesting_depth": 6,
                   "loop_count": 0, "recursive_function_count": 0, "structural_category": "Medium"},
        characteristics=(
            "Six nested ifs: nesting depth 6, cyclomatic complexity 7.",
            "Each early exit sits at a different indentation level, far from the condition it belongs to.",
        ),
        complexity=(FunctionComplexity("approve_loan", "O(1)", "O(1)", COMMON_ASSUMPTIONS, "O(1)"),),
        expected_recommendations=("HIGH_NESTING_DEPTH",),
    ),
)

EXAMPLES_BY_KEY = {e.key: e for e in EXAMPLES}
EXAMPLES_BY_TITLE = {e.title: e for e in EXAMPLES}


def get_example(key_or_title: str) -> Example:
    """Look up an example by key or title (KeyError if unknown)."""
    if key_or_title in EXAMPLES_BY_KEY:
        return EXAMPLES_BY_KEY[key_or_title]
    return EXAMPLES_BY_TITLE[key_or_title]


def find_by_code(code: str) -> Optional[Example]:
    """The example whose code is exactly ``code`` (ignoring trailing whitespace), if any."""
    stripped = (code or "").rstrip()
    return next((e for e in EXAMPLES if e.code.rstrip() == stripped), None)


# --------------------------------------------------------------------------- #
# Before / after pairs
# --------------------------------------------------------------------------- #
_EX = EXAMPLES_BY_KEY

COMPARISON_PAIRS: tuple[ComparisonPair, ...] = (
    ComparisonPair(
        key="nested_loops_to_hash_map",
        title="Nested loops → hash map",
        summary="Count pairs that add up to a target with one pass and a dictionary instead of all pairs.",
        before=_EX["nested_loops"].code.split("\n\n\ndef row_totals")[0] + "\n",
        after='''def count_pairs_with_sum(values, target):
    seen = {}
    count = 0
    for value in values:
        count += seen.get(target - value, 0)
        seen[value] = seen.get(value, 0) + 1
    return count
''',
        behaviour="Both return the number of index pairs i < j with values[i] + values[j] == target "
                  "(for hashable values).",
        changes=("One loop instead of two nested loops (nesting 3 → 1, cyclomatic complexity 4 → 2).",
                 "Static time estimate O(n^2) → O(n).",
                 "Auxiliary space O(1) → O(n): the speed-up is paid for with memory."),
        unchanged=("The line count barely changes, so lines are not what changed the running time.",),
        expected=(ExpectedChange("count_pairs_with_sum", "O(n^2)", "O(n)", "O(1)", "O(n)"),),
        lesson="The faster version trades memory for time, and dictionary operations are O(1) on "
               "average, not in the worst case.",
    ),
    ComparisonPair(
        key="recursion_to_iteration",
        title="Naive recursion → iteration",
        summary="The same Fibonacci numbers with a loop instead of two recursive calls.",
        before=_EX["recursive_fibonacci"].code,
        after='''def fibonacci(n):
    previous, current = 0, 1
    for _ in range(n):
        previous, current = current, previous + current
    return previous
''',
        behaviour="Both return the n-th Fibonacci number for integers n ≥ 0. For negative n the recursive "
                  "version returns n and the loop version returns 0.",
        changes=("Static time estimate O(2^n) → O(n).", "Auxiliary space O(n) (recursion stack) → O(1).",
                 "The repeated-work recursion recommendation disappears."),
        unchanged=("Cyclomatic complexity stays 2.",),
        expected=(ExpectedChange("fibonacci", "O(2^n)", "O(n)", "O(n)", "O(1)"),),
        lesson="The after version is one line longer and exponentially faster: line count says nothing "
               "about growth rate.",
    ),
    ComparisonPair(
        key="branches_to_lookup",
        title="Branch chain → lookup table",
        summary="Move the zone prices and weight brackets into data and simplify the conditions.",
        before=_EX["complex_branching"].code,
        after='''ZONE_BASE = {"local": 5.0, "regional": 9.0, "national": 14.0, "international": 30.0}
WEIGHT_SURCHARGE = ((20, 25.0), (5, 10.0), (1, 3.0))


def weight_surcharge(weight):
    for limit, extra in WEIGHT_SURCHARGE:
        if weight > limit:
            return extra
    return 0.0


def shipping_cost(weight, zone, express, fragile, member):
    if weight <= 0:
        raise ValueError("weight must be positive")
    if zone not in ZONE_BASE:
        raise ValueError("unknown zone")
    cost = ZONE_BASE[zone] + weight_surcharge(weight)
    if express:
        cost *= 2.0 if zone == "international" else 1.5
    if fragile:
        cost += 4.0
    if member and cost > 20:
        cost -= 5.0
    return round(cost, 2)
''',
        behaviour="Both compute the same price for every input (the error messages are unchanged).",
        changes=("Highest cyclomatic complexity 15 → 8; the work is split across two functions.",
                 "Structural category High → Medium.",
                 "The static time estimate becomes Unknown: the analyzer cannot see that WEIGHT_SURCHARGE "
                 "always has three entries."),
        unchanged=("By inspection the running time is still O(1): the table size is fixed in the code.",),
        expected=(ExpectedChange("shipping_cost", "O(1)", "O(1) by inspection (analyzer: Unknown)", "O(1)", "O(1)"),),
        lesson="Restructuring can lower decision complexity without changing running time, and the static "
               "analyzer says Unknown rather than guessing when it cannot bound a loop.",
    ),
    ComparisonPair(
        key="nesting_to_guard_clauses",
        title="Deep nesting → guard clauses",
        summary="Invert each check and return early, so the happy path is not buried six levels deep.",
        before=_EX["deep_nesting"].code,
        after='''def approve_loan(applicant):
    if applicant is None:
        return "no applicant"
    if applicant.age < 18:
        return "under age"
    if applicant.income <= 0:
        return "no income"
    if applicant.credit_score < 650:
        return "credit score too low"
    if applicant.debt / applicant.income >= 0.4:
        return "debt too high"
    if applicant.has_defaults:
        return "manual review"
    return "approved"
''',
        behaviour="Both return the same result for ordinary numeric values (each condition is the exact "
                  "negation of the original; NaN values would compare differently).",
        changes=("Maximum nesting depth 6 → 1.", "The nesting recommendation disappears."),
        unchanged=("Cyclomatic complexity stays 7: the same seven paths exist.",
                   "Structural category stays Medium (it follows cyclomatic complexity).",
                   "Time and space stay O(1)."),
        expected=(ExpectedChange("approve_loan", "O(1)", "O(1)", "O(1)", "O(1)"),),
        lesson="Readability can change while the number of decision paths stays exactly the same.",
    ),
    ComparisonPair(
        key="shorter_but_slower",
        title="Shorter but slower: sorting to find the minimum",
        summary="Replace a scan for the smallest value with sorted(items)[0].",
        before='''def smallest(items):
    best = items[0]
    for value in items[1:]:
        if value < best:
            best = value
    return best
''',
        after='''def smallest(items):
    return sorted(items)[0]
''',
        behaviour="Both return the smallest element of a non-empty list.",
        changes=("Code lines 6 → 2; cyclomatic complexity 3 → 1; nesting 2 → 0; Derived Complexity Score drops.",
                 "Static time estimate O(n) → O(n log n): the shorter version does more work for large inputs.",),
        unchanged=("Auxiliary space stays O(n): the before version copies items[1:], the after version builds "
                   "a sorted copy.",),
        expected=(ExpectedChange("smallest", "O(n)", "O(n log n)", "O(n)", "O(n)"),),
        lesson="Every structural metric went down and the growth order went up. A shorter or structurally "
               "simpler version is not automatically faster.",
    ),
    ComparisonPair(
        key="shorter_same_complexity",
        title="Shorter, same complexity: one-line linear search",
        summary="Rewrite the linear search as a generator expression passed to next().",
        before=_EX["linear_search"].code,
        after='''def linear_search(items, target):
    return next((index for index, value in enumerate(items) if value == target), -1)
''',
        behaviour="Both return the first matching index, or -1.",
        changes=("Code lines 5 → 2 and nesting 2 → 0.",),
        unchanged=("Cyclomatic complexity stays 3: the loop and the condition moved into the comprehension.",
                   "Static time estimate stays O(n)."),
        expected=(ExpectedChange("linear_search", "O(n)", "O(n)", "O(1)", "O(1)"),),
        lesson="Fewer lines, same decision paths, same growth order.",
    ),
)

PAIRS_BY_KEY = {p.key: p for p in COMPARISON_PAIRS}
PAIRS_BY_TITLE = {p.title: p for p in COMPARISON_PAIRS}


def get_pair(key_or_title: str) -> ComparisonPair:
    if key_or_title in PAIRS_BY_KEY:
        return PAIRS_BY_KEY[key_or_title]
    return PAIRS_BY_TITLE[key_or_title]


# --------------------------------------------------------------------------- #
# Checking documentation against an analysis result
# --------------------------------------------------------------------------- #
def measured_structure(result: dict) -> dict:
    """The STRUCTURE_LABELS values as measured in a predictor result dict."""
    fs = result.get("file_summary") or {}
    fm = result.get("file_metrics") or {}
    return {
        "function_count": fs.get("function_count"),
        "max_cyclomatic_complexity": fs.get("maximum_cyclomatic_complexity"),
        "max_nesting_depth": fs.get("maximum_nesting_depth"),
        "loop_count": fm.get("loop_count"),
        "recursive_function_count": fs.get("recursive_function_count"),
        "structural_category": (result.get("structural_category") or {}).get("highest")
        or (result.get("structural_category") or {}).get("module_level"),
    }


@dataclass
class CheckRow:
    aspect: str
    documented: str
    measured: str
    status: str  # "matches" | "differs" | "not estimated" | "not documented"

    def to_dict(self) -> dict:
        return asdict(self)


def check_against_result(example: Example, result: dict) -> list[CheckRow]:
    """Compare the documented expectations with an actual analysis result.

    ML output is reported as "not documented": no ML class is expected.
    """
    rows: list[CheckRow] = []
    measured = measured_structure(result)
    for k, label in STRUCTURE_LABELS.items():
        d, m = example.structure.get(k), measured.get(k)
        rows.append(CheckRow(label, str(d), "—" if m is None else str(m), "matches" if d == m else "differs"))
    funcs = {f.get("function_id"): f for f in result.get("functions") or []}
    for fc in example.complexity:
        f = funcs.get(fc.function) or {}
        est = f.get("static_estimate") or {}
        got = est.get("time_class")
        status_ok = est.get("time_status") == "estimated" and got not in (None, "Unknown")
        if fc.static_time_class is None:
            status = "matches" if not status_ok else "differs"
        elif not status_ok:
            status = "not estimated"
        else:
            status = "matches" if got == fc.static_time_class else "differs"
        rows.append(CheckRow(f"Time of {fc.function} (static estimate)",
                             fc.static_time_class or "Unknown", got or "Unknown", status))
        ml = (f.get("ml_prediction") or {}).get("big_o")
        if ml:
            rows.append(CheckRow(f"Time of {fc.function} (ML, experimental)", "not documented", ml, "not documented"))
    file_ml = (result.get("prediction") or {}).get("big_o")
    if file_ml:
        rows.append(CheckRow("Whole-file ML class", "not documented", file_ml, "not documented"))
    return rows
