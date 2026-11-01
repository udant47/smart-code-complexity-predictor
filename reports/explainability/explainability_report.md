# Model explainability report (Phase 14)

Model: Random Forest, 57 features. Test rows: 814 (grouped by problem, unseen in training). Training rows: 3644. Permutation repeats: 10, seed 42.
Baseline test macro F1 0.442 (matches the saved model metadata).

**Read this first.** Importance shows which inputs this model relies on. It is association, not causation, and the global ranking is not the reason for any single prediction.

## 1. Permutation importance (held-out test set)

Drop in test macro F1 when one feature is shuffled (mean ± std over repeats). `+ repeats` = how many repeats lowered the score.

![Permutation importance](permutation_importance.png)

| rank | feature | group | F1 drop | std | + repeats | strongly correlated with (ρ ≥ 0.8) |
|---|---|---|---|---|---|---|
| 1 | `sort_call_count` | Library calls and data structures | 0.0925 | 0.0081 | 10/10 | - |
| 2 | `power_of_two_expr_count` | Exponential patterns | 0.0246 | 0.0091 | 10/10 | - |
| 3 | `halving_loop_count` | Halving and binary search | 0.0223 | 0.0021 | 10/10 | - |
| 4 | `max_input_loop_depth` | Loops | 0.0158 | 0.0051 | 10/10 | maximum_loop_nesting_depth, nested_loop_count, loop_count |
| 5 | `for_loop_count` | Loops | 0.0127 | 0.0043 | 10/10 | loop_count, maximum_loop_nesting_depth |
| 6 | `bitwise_operator_count` | Exponential patterns | 0.0070 | 0.0029 | 10/10 | - |
| 7 | `nested_loop_count` | Loops | 0.0052 | 0.0047 | 9/10 | maximum_loop_nesting_depth, max_input_loop_depth |
| 8 | `while_loop_count` | Loops | 0.0051 | 0.0049 | 7/10 | - |
| 9 | `maximum_loop_nesting_depth` | Loops | 0.0038 | 0.0036 | 9/10 | max_input_loop_depth, nested_loop_count, loop_count |
| 10 | `comprehension_count` | Loops | 0.0024 | 0.0026 | 8/10 | - |
| 11 | `if_count` | Branching | 0.0021 | 0.0019 | 8/10 | comparison_operator_count, cyclomatic_complexity, code_lines |
| 12 | `math_call_count` | Library calls and data structures | 0.0018 | 0.0027 | 7/10 | - |
| 13 | `hash_container_count` | Library calls and data structures | 0.0017 | 0.0021 | 8/10 | - |
| 14 | `dependent_inner_loop_count` | Loops | 0.0015 | 0.0015 | 7/10 | - |
| 15 | `class_count` | Size | 0.0014 | 0.0010 | 9/10 | - |

31 of 57 features change macro F1 by less than 0.002 when shuffled on their own. That does **not** prove they are useless: many have strongly correlated partners that carry the same information.

27 features have a negative mean drop (shuffling them helped slightly on this test set); 9 of them are within one standard deviation of zero. The model gains nothing measurable from these features on unseen problems; for the rest, its use of them seems to hurt slightly.

## 2. Grouped permutation importance

All features of a group shuffled together (same row order), which keeps their correlation inside the group.

![Group importance](group_importance.png)

| group | features | F1 drop | std | + repeats |
|---|---|---|---|---|
| Loops | 10 | 0.1854 | 0.0097 | 10/10 |
| Library calls and data structures | 6 | 0.1051 | 0.0063 | 10/10 |
| Exponential patterns | 4 | 0.0543 | 0.0056 | 10/10 |
| Halving and binary search | 3 | 0.0288 | 0.0055 | 10/10 |
| Recursion | 5 | 0.0034 | 0.0023 | 9/10 |
| Branching | 12 | -0.0011 | 0.0052 | 3/10 |
| Size | 13 | -0.0039 | 0.0057 | 2/10 |
| Nesting | 2 | -0.0045 | 0.0040 | 1/10 |
| Operators | 2 | -0.0080 | 0.0065 | 1/10 |

Groups with no measurable benefit on unseen problems (mean drop ≤ 0): Branching, Size, Nesting, Operators. The forest still splits on these features (see native importance), so they can appear in per-prediction explanations, which flag them. This suggests those splits fit the training problems rather than general patterns. It is a finding about this model, not proof that the features are irrelevant to complexity.

## 3. Which features matter for which class

Mean drop in that class's F1 on the test set when the feature is shuffled (top 3 per class).

| class | test F1 | top features (F1 drop) |
|---|---|---|
| O(1) | 0.443 | `sort_call_count` (0.066), `collection_literal_count` (0.012), `halving_loop_count` (0.010) |
| O(log n) | 0.433 | `halving_loop_count` (0.147), `for_loop_count` (0.122), `sort_call_count` (0.083) |
| O(n) | 0.439 | `sort_call_count` (0.120), `power_of_two_expr_count` (0.055), `for_loop_count` (0.024) |
| O(n log n) | 0.629 | `sort_call_count` (0.372), `max_input_loop_depth` (0.018), `nested_loop_count` (0.014) |
| O(n^2) | 0.370 | `power_of_two_expr_count` (0.032), `sort_call_count` (0.025), `arithmetic_operator_count` (0.015) |
| O(n^3) | 0.377 | `max_input_loop_depth` (0.073), `maximum_loop_nesting_depth` (0.042), `arithmetic_operator_count` (0.034) |
| O(2^n) | 0.405 | `power_of_two_expr_count` (0.123), `bitwise_operator_count` (0.044), `arithmetic_operator_count` (0.014) |

## 4. Native importance (mean decrease in impurity)

From the fitted forest, on the training data. Shown for comparison; it favours features with many distinct values (line and token counts) and is not measured on unseen problems.

| rank | feature | MDI | std across trees |
|---|---|---|---|
| 1 | `sort_call_count` | 0.0690 | 0.0183 |
| 2 | `max_input_loop_depth` | 0.0614 | 0.0435 |
| 3 | `power_of_two_expr_count` | 0.0499 | 0.0274 |
| 4 | `maximum_loop_nesting_depth` | 0.0477 | 0.0378 |
| 5 | `for_loop_count` | 0.0445 | 0.0314 |
| 6 | `arithmetic_operator_count` | 0.0437 | 0.0159 |
| 7 | `function_call_count` | 0.0423 | 0.0225 |
| 8 | `identifier_count` | 0.0393 | 0.0189 |
| 9 | `unique_identifier_count` | 0.0382 | 0.0191 |
| 10 | `operator_count` | 0.0371 | 0.0145 |

## 5. Aggregated per-prediction contributions

Mean absolute tree-path contribution toward each test program's predicted class, in vote-share points.

| rank | feature | mean abs. contribution (points) |
|---|---|---|
| 1 | `sort_call_count` | 8.00 |
| 2 | `max_input_loop_depth` | 5.11 |
| 3 | `maximum_loop_nesting_depth` | 3.91 |
| 4 | `power_of_two_expr_count` | 3.68 |
| 5 | `for_loop_count` | 3.10 |
| 6 | `arithmetic_operator_count` | 2.43 |
| 7 | `nested_loop_count` | 2.38 |
| 8 | `function_call_count` | 2.14 |
| 9 | `while_loop_count` | 2.09 |
| 10 | `operator_count` | 1.97 |

## 6. Do the methods agree?

Spearman correlation between the feature rankings:

| pair | Spearman (all 57) | shared top-10 features |
|---|---|---|
| native vs permutation | 0.077 | 5/10 |
| native vs aggregated tree-path | 0.965 | 8/10 |
| permutation vs aggregated tree-path | 0.142 | 7/10 |

The Spearman values over all 57 features are dominated by the many features whose permutation importance is near zero (their order is noise), so the top-10 overlap is the more useful comparison. Native importance and tree-path contributions both describe how the trees *use* features (largely on training data); permutation importance describes what *helps on unseen problems*. When they disagree, the second is the safer guide.

SHAP was not installed when this report was made, so the per-prediction method was not compared with TreeSHAP.

## 7. Per-prediction explanations

`explain_source(code)` / `explain_features(values)` return, for one program, the base vote share, each feature's contribution toward the explained class (default: the predicted class) and plain-language sentences. Contributions add up exactly to the model's vote share (checked in the tests).

Limitations:

- They explain the model's vote, which is often wrong (test macro F1 about 0.44) and not calibrated.
- The tree-path method credits features by split order; features near the root of many trees tend to get more credit. TreeSHAP (optional) avoids this but needs the `shap` package.
- Correlated features can trade credit between them; read groups (also returned) rather than single features.
- A large contribution is not a cause: changing that feature in the code would not change the code's complexity.

## Caveats

- Feature importance shows which inputs this model relies on; it is association, not causation.
- Global importance is an average over the test set, not the reason for any single prediction.
- Correlated features share importance: a feature can look unimportant because another one carries the same information.
- Native (MDI) importance is computed on training data and favours features with many distinct values.

Software: Python 3.11.9, scikit-learn 1.7.2.
