# Repository complexity report

Folder: `C:\Users\udant gogoi\Downloads\smart-code-complexity-predictor\src`  
Analysed in 4.7 s. Structural metrics only (no ML predictions); nothing was executed.

## Summary

| | |
|---|---|
| Files analysed | 47 |
| Files that could not be parsed | 0 |
| Files skipped | 5 |
| Code lines | 12,905 |
| Functions | 630 |
| Functions flagged (cyclomatic ≥ 11, nesting ≥ 4 or ≥ 50 lines) | 98 |

## Languages

| Language | Files | Code lines | Functions | Mean CC | Median CC | CC ≥ 11 |
|---|---|---|---|---|---|---|
| Python | 47 | 12,905 | 630 | 5.97 | 3.0 | 92 |

## Cyclomatic categories (functions)

Low (1–5): 427 · Medium (6–10): 111 · High (11+): 92

## Most complex functions

| Function | File | Lines | CC | Nesting | Length | Static time | Flags |
|---|---|---|---|---|---|---|---|
| `_observations` | `comparison.py` | 298–393 | 53 | 2 | 96 | Unknown | cyclomatic ≥ 11, length ≥ 50 lines |
| `_FunctionAnalyzer._solve_recursion` | `algorithm_analyzer.py` | 1245–1368 | 51 | 4 | 124 | Unknown | cyclomatic ≥ 11, nesting ≥ 4, length ≥ 50 lines |
| `_FunctionAnalyzer.coll_size` | `algorithm_analyzer.py` | 484–555 | 51 | 3 | 72 | Unknown | cyclomatic ≥ 11, length ≥ 50 lines |
| `_walk` | `feature_extractor.py` | 482–604 | 46 | 4 | 123 | Unknown | cyclomatic ≥ 11, nesting ≥ 4, length ≥ 50 lines |
| `rule_duplication` | `suggestions.py` | 648–727 | 46 | 4 | 80 | Unknown | cyclomatic ≥ 11, nesting ≥ 4, length ≥ 50 lines |
| `_FunctionAnalyzer._reduction` | `algorithm_analyzer.py` | 1197–1230 | 44 | 2 | 34 | Unknown | cyclomatic ≥ 11 |
| `CAnalyzer._report` | `languages/c_lang.py` | 264–341 | 42 | 5 | 78 | Unknown | cyclomatic ≥ 11, nesting ≥ 4, length ≥ 50 lines |
| `extract_pattern_features` | `pattern_features.py` | 81–129 | 39 | 3 | 49 | Unknown | cyclomatic ≥ 11 |
| `_FunctionAnalyzer._while_bound` | `algorithm_analyzer.py` | 785–845 | 38 | 4 | 61 | Unknown | cyclomatic ≥ 11, nesting ≥ 4, length ≥ 50 lines |
| `_FunctionAnalyzer.size_of` | `algorithm_analyzer.py` | 438–482 | 37 | 3 | 45 | Unknown | cyclomatic ≥ 11 |
| `_FunctionAnalyzer._method_call` | `algorithm_analyzer.py` | 1043–1083 | 34 | 2 | 41 | Unknown | cyclomatic ≥ 11 |
| `scan` | `repository/scanner.py` | 96–185 | 32 | 4 | 90 | Unknown | cyclomatic ≥ 11, nesting ≥ 4, length ≥ 50 lines |
| `_to_estimate` | `algorithm_analyzer.py` | 1755–1828 | 32 | 1 | 74 | Unknown | cyclomatic ≥ 11, length ≥ 50 lines |
| `predict` | `predictor.py` | 362–482 | 31 | 3 | 121 | Unknown | cyclomatic ≥ 11, length ≥ 50 lines |
| `_FunctionAnalyzer._stmt` | `algorithm_analyzer.py` | 605–658 | 31 | 3 | 54 | Unknown | cyclomatic ≥ 11, length ≥ 50 lines |
| `_constant_step` | `suggestions.py` | 511–540 | 31 | 2 | 30 | Unknown | cyclomatic ≥ 11 |
| `explain_features` | `explainability.py` | 662–750 | 29 | 2 | 89 | Unknown | cyclomatic ≥ 11, length ≥ 50 lines |
| `_FunctionAnalyzer._expr` | `algorithm_analyzer.py` | 850–897 | 28 | 3 | 48 | Unknown | cyclomatic ≥ 11 |
| `run` | `train_model.py` | 269–367 | 27 | 4 | 99 | Unknown | cyclomatic ≥ 11, nesting ≥ 4, length ≥ 50 lines |
| `_ml` | `ui/views/results.py` | 182–268 | 26 | 3 | 87 | Unknown | cyclomatic ≥ 11, length ≥ 50 lines |

## Skipped

- ignored_folder: 5

Files with possible secrets are listed with the rule and line only; the values are never copied into this report.

## Notes

- Cyclomatic complexity and nesting measure control flow, not quality or speed.
- C support is experimental and structural only. C files that rely on syntax-changing macros cannot be parsed and are listed above instead of being measured approximately.
- Static time is the rule-based Big-O estimate for Python functions; it is blank where unknown or not available for the language.
