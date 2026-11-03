"""Before/after report for the final improvement phase.

    python -m src.improvement_report

Run it after ``python -m src.evaluate_model``. It writes
``reports/improvement/improvement_results.json`` (shown on the dashboard's
*Model evaluation → Improvement history* tab) and ``improvement_report.md``.

What it measures (nothing is typed in by hand):

1. **Before vs after on the same test set.** The previous release's model
   configuration (Random Forest, ``patterns``, 57 features) is re-fitted with
   the same seed on the same train + validation rows. Its test macro F1 must
   reproduce the value recorded in ``baseline_before.json``, which is checked.
   It is compared with the saved final model. Both use the same 814 test
   programs.
2. **Feature ablation.** Repeated grouped cross-validation (2 × 5 folds) on
   train + validation for the previous feature set, the static-estimate
   features, the extra AST counts, and both. Features are adopted only if they
   improve the cross-validated macro F1 by more than ``ADOPT_MARGIN``, and the
   smallest set within that margin of the best is preferred.
3. **Static analyzer benchmark.** Reference algorithms (sorting, searching,
   divide and conquer) with their textbook worst-case time, compared with the
   analyzer's output. Unknown counts as "not estimated", never as correct.
"""

from __future__ import annotations

import json
import sys
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone

from src import config
from src import preprocessing as pp
from src import train_model as tm
from src import tune_model
from src.analysis_features import AST_EXTRA_FEATURE_NAMES, STATIC_FEATURE_NAMES

OUT_DIR = config.PROJECT_ROOT / "reports" / "improvement"
BASELINE = OUT_DIR / "baseline_before.json"
ADOPT_MARGIN = 0.005

REFERENCE_ALGORITHMS = {
    "Linear search": ("O(n)", """
        def linear_search(a, x):
            for i, v in enumerate(a):
                if v == x:
                    return i
            return -1
    """),
    "Binary search (iterative)": ("O(log n)", """
        def binary_search(a, x):
            lo, hi = 0, len(a) - 1
            while lo <= hi:
                mid = (lo + hi) // 2
                if a[mid] == x:
                    return mid
                if a[mid] < x:
                    lo = mid + 1
                else:
                    hi = mid - 1
            return -1
    """),
    "Binary search (recursive)": ("O(log n)", """
        def bsearch(a, x, lo, hi):
            if lo > hi:
                return -1
            mid = (lo + hi) // 2
            if a[mid] == x:
                return mid
            if a[mid] < x:
                return bsearch(a, x, mid + 1, hi)
            return bsearch(a, x, lo, mid - 1)
    """),
    "Sequential loops": ("O(n)", """
        def two_passes(a):
            s = 0
            for x in a:
                s += x
            m = 0
            for x in a:
                m = max(m, x)
            return s, m
    """),
    "Bubble sort": ("O(n^2)", """
        def bubble_sort(a):
            n = len(a)
            for i in range(n):
                for j in range(0, n - i - 1):
                    if a[j] > a[j + 1]:
                        a[j], a[j + 1] = a[j + 1], a[j]
            return a
    """),
    "Selection sort": ("O(n^2)", """
        def selection_sort(a):
            n = len(a)
            for i in range(n):
                m = i
                for j in range(i + 1, n):
                    if a[j] < a[m]:
                        m = j
                a[i], a[m] = a[m], a[i]
            return a
    """),
    "Insertion sort": ("O(n^2)", """
        def insertion_sort(a):
            for i in range(1, len(a)):
                key = a[i]
                j = i - 1
                while j >= 0 and a[j] > key:
                    a[j + 1] = a[j]
                    j -= 1
                a[j + 1] = key
            return a
    """),
    "Merge sort": ("O(n log n)", """
        def merge_sort(a):
            if len(a) <= 1:
                return a
            mid = len(a) // 2
            left = merge_sort(a[:mid])
            right = merge_sort(a[mid:])
            out = []
            i = j = 0
            while i < len(left) and j < len(right):
                if left[i] <= right[j]:
                    out.append(left[i])
                    i += 1
                else:
                    out.append(right[j])
                    j += 1
            out.extend(left[i:])
            out.extend(right[j:])
            return out
    """),
    "Quick sort (list partitions)": ("O(n^2)", """
        def quick_sort(a):
            if len(a) <= 1:
                return a
            pivot = a[len(a) // 2]
            left = [x for x in a if x < pivot]
            mid = [x for x in a if x == pivot]
            right = [x for x in a if x > pivot]
            return quick_sort(left) + mid + quick_sort(right)
    """),
    "Heap sort (heapq)": ("O(n log n)", """
        import heapq
        def heap_sort(a):
            h = list(a)
            heapq.heapify(h)
            return [heapq.heappop(h) for _ in range(len(h))]
    """),
    "Heap sort (hand-written sift-down)": ("O(n log n)", """
        def sift_down(a, n, i):
            while True:
                largest = i
                l, r = 2 * i + 1, 2 * i + 2
                if l < n and a[l] > a[largest]:
                    largest = l
                if r < n and a[r] > a[largest]:
                    largest = r
                if largest == i:
                    return
                a[i], a[largest] = a[largest], a[i]
                i = largest
        def heap_sort(a):
            n = len(a)
            for i in range(n // 2 - 1, -1, -1):
                sift_down(a, n, i)
            for end in range(n - 1, 0, -1):
                a[0], a[end] = a[end], a[0]
                sift_down(a, end, 0)
            return a
    """),
    "Maximum subarray (divide and conquer)": ("O(n log n)", """
        def max_sub(a, lo, hi):
            if lo == hi:
                return a[lo]
            mid = (lo + hi) // 2
            best_left = s = a[mid]
            for i in range(mid - 1, lo - 1, -1):
                s += a[i]
                best_left = max(best_left, s)
            best_right = s = a[mid + 1]
            for i in range(mid + 2, hi + 1):
                s += a[i]
                best_right = max(best_right, s)
            return max(max_sub(a, lo, mid), max_sub(a, mid + 1, hi), best_left + best_right)
    """),
    "Fast power (divide and conquer)": ("O(log n)", """
        def power(x, n):
            if n == 0:
                return 1
            half = power(x, n // 2)
            if n % 2 == 0:
                return half * half
            return half * half * x
    """),
    "Fibonacci (naive recursion)": ("O(2^n)", """
        def fib(n):
            if n < 2:
                return n
            return fib(n - 1) + fib(n - 2)
    """),
}


def analyzer_benchmark() -> list[dict]:
    """Run the static analyzer on REFERENCE_ALGORITHMS; the last function in each snippet is the entry point."""
    from src.algorithm_analyzer import analyze_algorithms
    from src.comparison import growth_key

    rows = []
    for name, (expected, code) in REFERENCE_ALGORITHMS.items():
        result = analyze_algorithms(textwrap.dedent(code).strip() + "\n")
        est = result.estimates[-1]
        got = est.time_class if est.time_status == "estimated" else "Unknown"
        if got == "Unknown":
            verdict = "not estimated (Unknown)"
        elif growth_key(got) == growth_key(expected):
            verdict = "matches"
        else:
            verdict = "different"
        reason = ""
        if got == "Unknown":
            reason = est.explanation.replace("Time could not be estimated: ", "")
            for cut in (" Space could not", " Auxiliary space"):
                reason = reason.split(cut)[0]
            reason = reason.rstrip(".")
        rows.append({"algorithm": name, "expected (worst case)": expected, "static estimate": got,
                     "verdict": verdict, "reason when Unknown": reason})
    return rows


def feature_ablation(df: pd.DataFrame, split: pp.SplitResult, seed: int = config.RANDOM_STATE) -> list[dict]:
    y = df[pp.TARGET]
    dev = split.indices("train").append(split.indices("validation"))
    cv = tm.repeated_cv(y.loc[dev], split.groups.loc[dev], seed)
    base = pp.feature_columns("patterns")
    sets = {
        "patterns (previous, 57)": base,
        "patterns + static estimate (61)": base + list(STATIC_FEATURE_NAMES),
        "patterns + extra AST counts (66)": base + list(AST_EXTRA_FEATURE_NAMES),
        "patterns + both (70)": base + list(STATIC_FEATURE_NAMES) + list(AST_EXTRA_FEATURE_NAMES),
    }
    params = tune_model.PHASE10_CONFIG["params"]
    rows = []
    for name, cols in sets.items():
        pipe = pp.make_pipeline("tree", clone(tune_model.rf_pipeline("patterns", params, seed).steps[-1][1]), cols)
        s = tm.cv_score(pipe, df.loc[dev, cols], y.loc[dev], cv, -1)
        rows.append({"feature_set": name, "n_features": len(cols), "rcv_macro_f1": round(float(s.mean()), 4),
                     "rcv_std": round(float(s.std()), 4)})
    best = max(r["rcv_macro_f1"] for r in rows)
    prev = rows[0]["rcv_macro_f1"]
    for r in rows:
        r["gain_vs_previous"] = round(r["rcv_macro_f1"] - prev, 4)
        r["within_margin_of_best"] = bool(best - r["rcv_macro_f1"] <= ADOPT_MARGIN)
    return rows


def before_after(df: pd.DataFrame, split: pp.SplitResult, baseline: dict, meta: dict) -> tuple[list[dict], dict]:
    from src.evaluate_model import calibration_scores

    y = df[pp.TARGET]
    dev = split.indices("train").append(split.indices("validation"))
    te = split.indices("test")
    before = tune_model.rf_pipeline(baseline["feature_set"], baseline["hyperparameters"]).fit(df.loc[dev], y.loc[dev])
    pred = before.predict(df.loc[te])
    m_before = {k: round(float(v), 4) for k, v in tm.metrics(y.loc[te], pred).items()}
    cal_before = calibration_scores(y.loc[te].values, before.predict_proba(df.loc[te]), list(before.classes_))
    reproduced = abs(m_before["macro_f1"] - baseline["test_metrics"]["macro_f1"]) < 1e-4
    m_after, cal_after = meta["test_metrics"], meta.get("calibration_test") or {}
    rows = []
    for key, label in (("accuracy", "Accuracy"), ("balanced_accuracy", "Balanced accuracy"),
                       ("macro_precision", "Macro precision"), ("macro_recall", "Macro recall"),
                       ("macro_f1", "Macro F1"), ("weighted_f1", "Weighted F1")):
        rows.append({"metric": label + " (test)", "before": m_before[key], "after": m_after[key],
                     "change": round(m_after[key] - m_before[key], 4)})
    for key, label in (("ece_top_label", "Expected calibration error (lower is better)"),
                       ("brier_multiclass", "Multiclass Brier score (lower is better)")):
        if key in cal_after:
            rows.append({"metric": label, "before": cal_before[key], "after": cal_after[key],
                         "change": round(cal_after[key] - cal_before[key], 4)})
    after_rcv = meta.get("rcv_macro_f1_selected")
    rows.append({"metric": "Repeated grouped CV macro F1 (train + validation)", "before": baseline["rcv_macro_f1"],
                 "after": after_rcv, "change": None if after_rcv is None else round(after_rcv - baseline["rcv_macro_f1"], 4)})
    per_class = []
    for cls, b in baseline["per_class_test"].items():
        a = (meta.get("per_class_test") or {}).get(cls, {})
        per_class.append({"class": cls, "F1 before": b["f1"], "F1 after": a.get("f1"),
                          "change": None if a.get("f1") is None else round(a["f1"] - b["f1"], 3),
                          "support": b["support"]})
    check = {"baseline_reproduced": reproduced, "recomputed_macro_f1": m_before["macro_f1"],
             "recorded_macro_f1": baseline["test_metrics"]["macro_f1"]}
    return rows, {"per_class": per_class, "check": check}


def run(verbose: bool = True) -> dict:
    df = pd.read_csv(config.BIGO_DATASET_CSV, keep_default_na=False)
    split = pp.load_split(df)
    baseline = json.loads(BASELINE.read_text(encoding="utf-8"))
    meta = json.loads(config.MODEL_METADATA_PATH.read_text(encoding="utf-8"))
    comparison = config.PROJECT_ROOT / "reports" / "models" / "model_comparison.csv"
    if comparison.exists():
        mc = pd.read_csv(comparison)
        row = mc[(mc["model"] == meta["selected_model"]) & (mc["features"] == meta["feature_set"])]
        if len(row):
            meta["rcv_macro_f1_selected"] = round(float(row["rcv_macro_f1"].iloc[0]), 4)

    ablation = feature_ablation(df, split)
    if verbose:
        print("feature ablation done", flush=True)
    rows, extra = before_after(df, split, baseline, meta)
    bench = analyzer_benchmark()
    n_match = sum(r["verdict"] == "matches" for r in bench)
    n_unknown = sum(r["verdict"].startswith("not estimated") for r in bench)
    n_wrong = sum(r["verdict"] == "different" for r in bench)
    result = {
        "before_after": rows,
        "per_class": extra["per_class"],
        "baseline_check": extra["check"],
        "feature_ablation": ablation,
        "analyzer_benchmark": bench,
        "analyzer_summary": {"cases": len(bench), "matches": n_match, "unknown": n_unknown, "different": n_wrong},
        "final_model": {"model": meta["selected_model"], "feature_set": meta["feature_set"],
                        "n_features": len(meta["features"]), "hyperparameters": meta["hyperparameters"]},
        "notes": [
            "Before = the previous release's configuration re-fitted with the same seed on the same rows; "
            + ("its test macro F1 reproduces the recorded value." if extra["check"]["baseline_reproduced"]
               else "WARNING: its test macro F1 does not reproduce the recorded value."),
            "The test set was used once for the final model; feature and model choices were made by cross-validation.",
            "The 95% confidence intervals of the before and after test macro F1 overlap (see the evaluation report), so "
            "the test-set change alone is not statistically conclusive; the cross-validated gain is the main evidence.",
            "Static analyzer: Unknown means the pattern is not supported; it is never counted as correct.",
        ],
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "improvement_results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (OUT_DIR / "improvement_report.md").write_text(render(result), encoding="utf-8")
    return result


def _table(rows: list[dict]) -> str:
    if not rows:
        return ""
    cols = list(rows[0])
    fmt = lambda v: "—" if v is None else (f"{v:.4f}" if isinstance(v, float) else str(v))  # noqa: E731
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(fmt(r[c]) for c in cols) + " |" for r in rows]
    return "\n".join(lines)


def render(r: dict) -> str:
    s = r["analyzer_summary"]
    f = r["final_model"]
    return "\n".join([
        "# Final improvement phase: measured results",
        "",
        "Generated by `python -m src.improvement_report`. Every number below is computed from the saved dataset, split "
        "and model files.",
        "",
        f"Final model: **{f['model']}**, feature set `{f['feature_set']}` ({f['n_features']} features), "
        f"`{json.dumps(f['hyperparameters'])}`.",
        "",
        "## 1. Before vs after (same 814 test programs)",
        "",
        _table(r["before_after"]),
        "",
        "Per-class F1:",
        "",
        _table(r["per_class"]),
        "",
        "## 2. Feature ablation (repeated grouped CV, train + validation only)",
        "",
        _table(r["feature_ablation"]),
        "",
        f"Adoption rule: a feature group is adopted only if it raises the cross-validated macro F1 by more than "
        f"{ADOPT_MARGIN}; among sets within {ADOPT_MARGIN} of the best, the smallest is chosen.",
        "",
        "## 3. Static analyzer on reference algorithms",
        "",
        _table(r["analyzer_benchmark"]),
        "",
        f"{s['matches']} of {s['cases']} match the textbook worst case, {s['unknown']} are reported as Unknown, "
        f"{s['different']} differ.",
        "",
        "## Notes",
        "",
        *[f"- {n}" for n in r["notes"]],
        "",
    ])


def main() -> int:
    if not config.MODEL_METADATA_PATH.exists() or not BASELINE.exists():
        print("Run python -m src.evaluate_model first (and keep reports/improvement/baseline_before.json).")
        return 1
    r = run()
    print(render(r))
    return 0


if __name__ == "__main__":
    sys.exit(main())
