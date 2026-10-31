"""Rule-based refactoring suggestions.

Responsibility:
    Map findings to actionable advice, e.g. deep nesting -> early returns,
    O(n^2) membership test in a loop -> use a set, long function -> extract
    helpers, high parameter count -> group into a dataclass.

Planned public API (Phase 2):
    suggest(features, estimate) -> list[Suggestion]
"""
