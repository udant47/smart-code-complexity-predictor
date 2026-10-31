"""Inference facade used by the Streamlit app.

Responsibility:
    Load the saved pipeline once, and for a source string return per-function
    results: features, ML class + probabilities, heuristic Big-O estimate,
    complexity score. Works (heuristics only) when no model file exists yet.

Planned public API (Phase 2):
    analyze_source(source: str) -> AnalysisReport
"""
