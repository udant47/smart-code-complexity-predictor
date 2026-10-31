"""Heuristic Big-O estimation (time and auxiliary space) where feasible.

Responsibility:
    Rule-based reasoning over the AST, independent of the ML model:
    nested loops over the same input -> O(n^k); loop halving/doubling ->
    O(log n); single recursion with n-1 -> O(n); branching recursion ->
    O(2^n); divide-and-conquer recursion -> O(n log n) / O(log n);
    calls to ``sorted``/``.sort`` -> O(n log n); allocations that scale
    with input -> auxiliary space O(n), etc.

    Returns "Unknown" with a reason when a confident estimate is not
    possible - it must never pretend to certainty.

Planned public API (Phase 2):
    estimate_complexity(unit: FunctionUnit) -> ComplexityEstimate
        (time, space, confidence, reasons)
"""
