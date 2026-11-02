"""Documented C examples for the experimental C analyzer (Phase 19).

Each mirrors one of the Python examples in ``src.examples`` so that the two
languages can be compared. The structural values are checked against the C
analyzer by ``tests/test_languages.py``. No complexity class or ML output is
claimed for C: those are not computed for C.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CExample:
    key: str
    title: str
    code: str
    python_counterpart: str  # key in src.examples.EXAMPLES_BY_KEY
    structure: dict  # same keys as src.examples.STRUCTURE_LABELS
    comparison_note: str


C_EXAMPLES: tuple[CExample, ...] = (
    CExample(
        key='linear_search',
        title='C1. Linear search',
        code=r"""int linear_search(const int *items, int n, int target) {
    for (int i = 0; i < n; i++) {
        if (items[i] == target) {
            return i;
        }
    }
    return -1;
}
""",
        python_counterpart='linear_search',
        structure={'function_count': 1, 'max_cyclomatic_complexity': 3, 'max_nesting_depth': 2, 'loop_count': 1, 'recursive_function_count': 0, 'structural_category': 'Low'},
        comparison_note='Same structure as the Python version: one for loop containing one if.',
    ),
    CExample(
        key='binary_search',
        title='C2. Binary search',
        code=r"""int binary_search(const int *sorted_items, int n, int target) {
    int low = 0, high = n - 1;
    while (low <= high) {
        int mid = low + (high - low) / 2;
        if (sorted_items[mid] == target) {
            return mid;
        }
        if (sorted_items[mid] < target) {
            low = mid + 1;
        } else {
            high = mid - 1;
        }
    }
    return -1;
}
""",
        python_counterpart='binary_search',
        structure={'function_count': 1, 'max_cyclomatic_complexity': 4, 'max_nesting_depth': 2, 'loop_count': 1, 'recursive_function_count': 0, 'structural_category': 'Low'},
        comparison_note='Same structure as the Python version. The midpoint is written lo + (hi - lo) / 2 to avoid integer overflow, which Python does not need.',
    ),
    CExample(
        key='nested_loops',
        title='C3. Nested-loop processing',
        code=r"""int count_pairs_with_sum(const int *values, int n, int target) {
    int count = 0;
    for (int i = 0; i < n; i++) {
        for (int j = i + 1; j < n; j++) {
            if (values[i] + values[j] == target) {
                count++;
            }
        }
    }
    return count;
}
""",
        python_counterpart='nested_loops',
        structure={'function_count': 1, 'max_cyclomatic_complexity': 4, 'max_nesting_depth': 3, 'loop_count': 2, 'recursive_function_count': 0, 'structural_category': 'Low'},
        comparison_note='Matches the Python count_pairs_with_sum (cyclomatic 4, nesting 3). The array length n is an explicit parameter in C.',
    ),
    CExample(
        key='recursive_fibonacci',
        title='C4. Recursive Fibonacci',
        code=r"""long fibonacci(int n) {
    if (n < 2) {
        return n;
    }
    return fibonacci(n - 1) + fibonacci(n - 2);
}
""",
        python_counterpart='recursive_fibonacci',
        structure={'function_count': 1, 'max_cyclomatic_complexity': 2, 'max_nesting_depth': 1, 'loop_count': 0, 'recursive_function_count': 1, 'structural_category': 'Low'},
        comparison_note='Same structure as the Python version: one if, two direct recursive calls.',
    ),
    CExample(
        key='merge_sort',
        title='C5. Merge sort',
        code=r"""#include <string.h>

static void merge(int *a, int *tmp, int lo, int mid, int hi) {
    int i = lo, j = mid, k = lo;
    while (i < mid && j < hi) {
        if (a[i] <= a[j]) {
            tmp[k++] = a[i++];
        } else {
            tmp[k++] = a[j++];
        }
    }
    while (i < mid) tmp[k++] = a[i++];
    while (j < hi) tmp[k++] = a[j++];
    memcpy(a + lo, tmp + lo, (size_t)(hi - lo) * sizeof *a);
}

void merge_sort(int *a, int *tmp, int lo, int hi) {
    if (hi - lo < 2) {
        return;
    }
    int mid = lo + (hi - lo) / 2;
    merge_sort(a, tmp, lo, mid);
    merge_sort(a, tmp, mid, hi);
    merge(a, tmp, lo, mid, hi);
}
""",
        python_counterpart='merge_sort',
        structure={'function_count': 2, 'max_cyclomatic_complexity': 6, 'max_nesting_depth': 2, 'loop_count': 3, 'recursive_function_count': 1, 'structural_category': 'Medium'},
        comparison_note="Differs from Python: C's merge needs two extra while loops to copy the leftover elements, which Python does with slices (left[i:]). So merge has cyclomatic complexity 6 in C and 4 in Python, and the file has 3 loops instead of 1.",
    ),
    CExample(
        key='complex_branching',
        title='C6. Complex branching (switch)',
        code=r"""/* Shipping price in cents; returns -1 for invalid input. */
int shipping_cost(int weight, char zone, int express, int fragile, int member) {
    int cost;
    if (weight <= 0) {
        return -1;
    }
    switch (zone) {
        case 'L': cost = 500; break;
        case 'R': cost = 900; break;
        case 'N': cost = 1400; break;
        case 'I': cost = 3000; break;
        default: return -1;
    }
    if (weight > 20) {
        cost += 2500;
    } else if (weight > 5) {
        cost += 1000;
    } else if (weight > 1) {
        cost += 300;
    }
    if (express && zone != 'I') {
        cost = cost * 3 / 2;
    } else if (express) {
        cost *= 2;
    }
    if (fragile) {
        cost += 400;
    }
    if (member && cost > 2000) {
        cost -= 500;
    }
    return cost;
}
""",
        python_counterpart='complex_branching',
        structure={'function_count': 1, 'max_cyclomatic_complexity': 15, 'max_nesting_depth': 1, 'loop_count': 0, 'recursive_function_count': 0, 'structural_category': 'High'},
        comparison_note="A switch with four cases replaces Python's if/elif chain on the zone. Each non-default case is a decision point, just as each if/elif is, so both versions have cyclomatic complexity 15.",
    ),
    CExample(
        key='deep_nesting',
        title='C7. Deeply nested conditions',
        code=r"""struct applicant { int age; int income; int credit_score; int debt; int has_defaults; };

const char *approve_loan(const struct applicant *a) {
    if (a != NULL) {
        if (a->age >= 18) {
            if (a->income > 0) {
                if (a->credit_score >= 650) {
                    if (a->debt * 10 < a->income * 4) {
                        if (!a->has_defaults) {
                            return "approved";
                        }
                        return "manual review";
                    }
                    return "debt too high";
                }
                return "credit score too low";
            }
            return "no income";
        }
        return "under age";
    }
    return "no applicant";
}
""",
        python_counterpart='deep_nesting',
        structure={'function_count': 1, 'max_cyclomatic_complexity': 7, 'max_nesting_depth': 6, 'loop_count': 0, 'recursive_function_count': 0, 'structural_category': 'Medium'},
        comparison_note='Same arrow shape as the Python version: six nested ifs, nesting depth 6, cyclomatic complexity 7.',
    ),
)

C_EXAMPLES_BY_TITLE = {e.title: e for e in C_EXAMPLES}
C_EXAMPLES_BY_KEY = {e.key: e for e in C_EXAMPLES}


def find_c_example(code: str):
    stripped = (code or "").rstrip()
    return next((e for e in C_EXAMPLES if e.code.rstrip() == stripped), None)
