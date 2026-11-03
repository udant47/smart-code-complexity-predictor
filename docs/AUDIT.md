# SmartCode: final audit and improvement report

This document records the audit of the completed project (after Phase 20), the weaknesses it found, what was changed,
and the measured effect of each change. Every number comes from a file the pipeline writes:

| Numbers | Command that regenerates them | File |
|---|---|---|
| Dataset quality | `python -m src.dataset_audit` | `reports/improvement/dataset_audit.md` |
| Model comparison | `python -m src.train_model` | `reports/models/model_report.md`, `model_comparison.csv` |
| Final test evaluation, calibration | `python -m src.evaluate_model` | `reports/evaluation/evaluation_report.md` |
| Feature importance | `python -m src.explainability` | `reports/explainability/explainability_report.md` |
| Before/after, ablation, analyzer benchmark | `python -m src.improvement_report` | `reports/improvement/improvement_report.md` |

---

## 1. Architecture and workflow (as found)

```
Streamlit UI (app.py, src/ui)
   └─ predictor.predict(source)                 one entry point, never executes code
        ├─ code_parser          ast + tokenize, size/depth limits
        ├─ feature_extractor    39 structural features (file and per function)
        ├─ pattern_features     18 algorithm-pattern features
        ├─ function_analyzer    per-function metrics, structural band
        ├─ algorithm_analyzer   rule-based time / auxiliary-space Big-O (symbolic costs)
        ├─ complexity_score     Derived Complexity Score (0-100)
        ├─ suggestions          6 refactoring rules
        └─ saved model          SHA-256 verified, then joblib.load

Offline: dataset_loader → preprocessing (grouped split) → train_model → tune_model → evaluate_model → explainability
```

The architecture was sound and was kept. Changes were made inside existing modules, plus four small new modules
(`analysis_features`, `dataset_audit`, `improvement_report`, and `analyze_script` inside `algorithm_analyzer`).

## 2. Audit findings

| # | Area | Finding | Severity | Action |
|---|---|---|---|---|
| 1 | Dataset | 4,458 programs, 275 problems, 7 classes; imbalance only 1.67× (largest/smallest) | info | Kept class weighting (`balanced_subsample`); no resampling needed |
| 2 | Labels | 0 identical feature vectors with different labels; 0 of 104 near-duplicate clusters with conflicting labels | info | No labels changed or removed |
| 3 | Leakage | 0 problems and 0 near-duplicate clusters shared between train / validation / test | info | Grouped split kept unchanged (same file hash), so before/after use the same test set |
| 4 | Generalisation | 95.6% of programs share their problem's majority label: the label is mostly a property of the problem | limitation | Documented; selection only by grouped CV |
| 5 | Distribution shift | Validation has 21.1% O(1) vs 13.6% in train | minor | Selection uses repeated CV over train + validation, not the single validation part |
| 6 | Over-fitting | In-sample macro F1 0.947 vs grouped CV 0.606 | expected for forests | Training scores never used for selection; regularised variants tested (no gain) |
| 7 | Data-limited | Grouped-CV macro F1 rises from 0.444 (25% of problems) to 0.548 (100%) | limitation | Documented: more labelled *problems* would help |
| 8 | Static analyzer | Only functions were analysed; module-level scripts (most CodeComplex programs) got no estimate | gap | `analyze_script` added |
| 9 | Static analyzer | Insertion-sort loops (`while j >= 0 and a[j] > key`) and loops between `lo`, `mid`, `hi` in divide-and-conquer code were Unknown | gap | Two sound worst-case rules added |
| 10 | ML features | The rule-based analyzer's knowledge was not available to the model | improvement | 4 static-estimate features (adopted after ablation) |
| 11 | Schema | Training (dataset_loader) and inference (predictor) built feature dictionaries separately | risk | One shared function `program_features()`, tested for equality |
| 12 | Models | Extra Trees not compared; tuning code RF-only | gap | Extra Trees added to the comparison; tuning generalised to forest models |
| 13 | Confidence | Vote shares shown with a reliability table, but calibration was never quantified | gap | ECE, multiclass Brier and log loss added to the evaluation |
| 14 | Dashboard | Global importance visible only after analysing code; a hard-coded sentence about why RF was selected | bug risk | Feature-importance, calibration and improvement tabs; text now generated from the saved results |
| 15 | Dashboard | The five complexity measures were shown side by side without a definition of how they differ | clarity | "What is the difference between these measures?" table on every result |
| 16 | Docs/tests | "57 features" hard-coded in UI, docs and tests | maintenance | Read from model metadata |

## 3. Dataset quality (Phase 2)

* **Source:** CodeComplex, Python subset (commit `c54631d`, SHA-256 verified), expert labels of worst-case time
  complexity for whole programs. 4,900 records → 4,458 after removing exact and renamed duplicates.
* **No fabricated or relabelled data.** The checks above found no label noise, so the rows were kept unchanged.
* **Splits:** `StratifiedGroupKFold` with groups = problem merged with near-duplicate clusters.
  Train 2,914 / validation 730 / test 814 (51 problems). Model selection uses repeated grouped CV (2 × 5 folds)
  on train + validation; the test set is used once.

## 4. Feature engineering (Phase 3)

Most features the brief lists already existed (loop nesting, recursion, branching, comprehensions, exceptions,
function length, boolean operators, cyclomatic complexity). Thirteen candidates were added and measured:

* **Static-estimate features (4):** whether the rule-based analyzer produced a time estimate, and its polynomial
  degree, power of log n and exponential flag. Known for 45.6% of programs; where known it agrees with the expert
  label 66.3% of the time.
* **Extra AST counts (9):** assignments, augmented assignments, subscripts, slices, collection-mutating calls,
  membership tests, call nesting depth, largest parameter list, `sys.setrecursionlimit`.

Ablation (Random Forest, same hyperparameters, repeated grouped CV on train + validation):

| Feature set | Features | CV macro F1 | Gain |
|---|---:|---:|---:|
| patterns (previous) | 57 | 0.5358 | — |
| + static estimate | 61 | **0.5560** | +0.0202 |
| + extra AST counts | 66 | 0.5485 | +0.0127 |
| + both | 70 | 0.5569 | +0.0211 |

The static-estimate set was adopted. Adding the AST counts on top changed the score by only +0.0009, below the
0.005 margin, so they are computed and shown but **not** given to the model (no redundant features).

## 5. Machine learning (Phase 4)

27 candidates (dummy baselines, a loop-nesting rule, LR, SVM, Decision Tree, RF, **Extra Trees**, GB, HGB, soft
voting) × 3 feature sets. Selection rule, fixed in advance: highest repeated-CV macro F1; within 0.005, the simplest.

| Model (61 features) | CV macro F1 | Test macro F1 [95% CI] | Test accuracy |
|---|---:|---:|---:|
| Soft-voting ensemble | 0.5595 | 0.4682 [0.342–0.565] | 0.4853 |
| **Random Forest (selected)** | **0.5560** | **0.4604 [0.336–0.558]** | **0.4693** |
| Extra Trees | 0.5514 | 0.4943 [0.352–0.593] | 0.5037 |
| Hist Gradient Boosting | 0.5484 | 0.4391 [0.322–0.547] | 0.4619 |
| Gradient Boosting | 0.5411 | 0.4458 [0.330–0.563] | 0.4681 |
| SVM (RBF) | 0.5265 | 0.4518 [0.325–0.546] | 0.4631 |
| Logistic Regression | 0.5141 | 0.4550 [0.337–0.541] | 0.4681 |
| Decision Tree | 0.4832 | 0.4194 [0.290–0.514] | 0.4349 |

* The ensemble is within 0.005 of the RF, so the simpler RF was kept.
* Extra Trees has the best *test* score, but a lower CV score. Choosing it because of the test set would be
  selection on the test set, so it was not chosen; the result is reported for transparency.
* Tuning (RandomizedSearchCV, 30 configurations per feature set) gained +0.003 in CV, below the margin: not adopted.
* Class imbalance is mild and handled by `class_weight="balanced_subsample"`.

## 6. Final results and before/after (same 814 test programs)

| Metric | Before (57 features) | After (61 features) | Change |
|---|---:|---:|---:|
| Accuracy | 0.4533 | 0.4693 | +0.0160 |
| Balanced accuracy | 0.4424 | 0.4597 | +0.0173 |
| Macro precision | 0.4797 | 0.5016 | +0.0219 |
| Macro recall | 0.4424 | 0.4597 | +0.0173 |
| **Macro F1** | **0.4422** | **0.4604** | **+0.0182** |
| Weighted F1 | 0.4515 | 0.4681 | +0.0166 |
| Expected calibration error (lower is better) | 0.0819 | 0.0739 | −0.0080 |
| Multiclass Brier score (lower is better) | 0.6765 | 0.6607 | −0.0158 |
| Repeated grouped CV macro F1 | 0.5358 | 0.5560 | +0.0202 |
| Out-of-fold macro F1, grouped by problem (all 4,458) | 0.5592 | 0.5713 | +0.0121 |

Per-class F1 rose for O(log n) (+0.041), O(n²) (+0.052) and O(n³) (+0.028), and stayed level for O(n) and O(n log n).
The "before" model was re-fitted with the same seed, and it reproduces the recorded 0.4422 exactly.

**Honest reading:** the improvement is real but modest. The test confidence intervals overlap, so the test change
alone is not statistically conclusive; the cross-validated gain (+0.020) is the main evidence. Roughly half of
the predictions on unseen problems are still wrong.

## 7. Explainability (Phase 5)

* Permutation importance on the test set (top): `sort_call_count` 0.079, `halving_loop_count` 0.022,
  `power_of_two_expr_count` 0.012, `max_input_loop_depth` 0.011, `for_loop_count` 0.010, `static_time_degree` 0.008.
* Grouped: loops 0.192, library calls 0.085, exponential patterns 0.050, halving 0.025, static estimate 0.010.
  Size, nesting and operator features contribute nothing measurable on unseen problems.
* Per prediction: exact tree-path contributions. Optional TreeSHAP shares 3.9 of the top-5 features on average.
* **Confidence vs calibration:** the app shows the forest's vote share and calls it a confidence score. ECE 0.074
  and the reliability table (vote share 0.8–1.0 → 86.5% correct; below 0.4 → 31.6%) are shown next to it.

## 8. Static complexity analysis (Phase 6)

Reference benchmark (`python -m src.improvement_report`): **12 of 14 match the textbook worst case, 2 Unknown, 0 wrong.**

| Matches | Unknown (with the reason shown) |
|---|---|
| linear search, binary search (iterative and recursive), sequential loops, bubble, selection and insertion sort, merge sort, heap sort with `heapq`, maximum subarray (divide and conquer), fast power, naive Fibonacci | quick sort (partition sizes depend on the data: O(n log n) average, O(n²) worst), hand-written heap sift-down (`while True` loop) |

New rules (both are upper bounds, so they cannot under-estimate):

* **Guarded counter loop:** in `while j >= 0 and a[j] > key`, the counter condition alone bounds the loop; the other
  conditions can only stop it earlier.
* **Divide-and-conquer intervals:** a `range` between `lo`, `hi` and `mid = (lo + hi) // 2` has at most `hi − lo`
  items, so the work per call is linear in the interval and the master theorem gives O(n log n).
* **Scripts:** module-level code is analysed with `input()` values treated as the input size.

## 9. Remaining limitations

* Macro F1 ≈ 0.46 on unseen problems: half of the predictions are wrong; vote shares are not calibrated probabilities.
* The labels describe whole Codeforces programs; per-function ML predictions are experimental.
* The static analyzer is a set of rules, not a proof system: data-dependent recursion (quick sort), `while True`
  loops, amortised costs and most library calls stay Unknown.
* Space complexity has no labelled data, so it is estimated by rules only and never validated by ML.
* C support is structural only.
