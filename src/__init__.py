"""Smart Code Complexity Predictor - core package.

Pipeline (each step is its own module):
    code_parser -> function_analyzer -> feature_extractor
        -> algorithm_analyzer (heuristic Big-O)
        -> preprocessing -> predictor (ML) -> explainability / suggestions

Submitted source code is only ever parsed with ``ast``; it is never executed.
"""

__version__ = "0.1.0"
