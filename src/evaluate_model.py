"""Final evaluation, generalization tests and model saving (Phase 11).

    python -m src.evaluate_model

Uses the configuration chosen by ``python -m src.tune_model``. Everything
is seeded (``config.RANDOM_STATE``).

1. **Final model and test evaluation.** The chosen pipeline (preprocessing +
   Random Forest) is fitted on train + validation and evaluated **once** on the
   grouped test set: metrics with 95% problem-bootstrap intervals, confusion
   matrix, per-class precision/recall/F1. This fitted pipeline is saved as the
   final model; its preprocessing part is also saved on its own.

2. **Generalization experiments.** These are diagnostics, separate from the
   final model. Some retrain on parts of the *whole* dataset; that does not
   affect the final model or its test score.

   a. *Different implementations of familiar algorithms.* Test problems are
      unseen, but their algorithm families (tags) occur in training. The test
      results are broken down by family and per problem: do different solutions
      of the same problem get the same, correct label?
   b. *Excluded source groups.* All data comes from one source (Codeforces), so
      true repository/source hold-out is impossible inside CodeComplex. Two
      substitutes: holding out whole **contests** (rounds) instead of
      problems, and an **external dataset** (BigO(Bench) time-complexity test
      set) if it has been downloaded. It comes from a different collection
      (CodeContests) with labels from a different method (runtime
      measurement, not expert reading).
   c. *Excluded algorithm families.* For each common tag, a model is trained on
      every problem *without* that tag and tested on the problems *with* it.
      This is compared with out-of-fold predictions for the same rows when the
      family is present in training. Tags overlap and are tied to labels, so a
      held-out family also shifts the class balance; both effects are reported.
   d. *Different structural patterns.* Test-set slices: with/without
      functions, recursion, sorting, template code, by loop depth and by length.

3. **Failure cases.** The most confident errors (by id and features only; the
   CodeComplex licence is unclear, so no source code is copied into reports),
   the most frequent confusions, and how far errors are from the true class.
"""

from __future__ import annotations

import json
import platform
import re
import time
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support
from sklearn.model_selection import StratifiedGroupKFold

from src import config
from src import model_integrity
from src import preprocessing as pp
from src import train_model as tm
from src import tune_model
from src.code_parser import parse_source
from src.dataset_loader import BIGO_LABELS
from src.feature_extractor import extract_features
from src.pattern_features import extract_pattern_features

REPORT_DIR = config.PROJECT_ROOT / "reports" / "evaluation"
PREPROCESSING_PATH = config.MODELS_DIR / "preprocessing_pipeline.joblib"
FAMILY_TAGS = (
    "binary search", "bitmasks", "brute force", "combinatorics", "constructive algorithms", "dp",
    "graphs", "greedy", "implementation", "math", "number theory", "sortings",
)
BIGOBENCH_CANDIDATES = (
    config.RAW_DATA_DIR / "bigobench" / "data" / "time_complexity_test_set.jsonl",
    config.RAW_DATA_DIR / "bigobench" / "time_complexity_test_set.jsonl",
)
RANK = {c: i for i, c in enumerate(pp.CLASS_ORDER)}
BIGO_OF = {c: BIGO_LABELS[c][1] for c in pp.CLASS_ORDER}

# BigO(Bench) label (normalised) -> CodeComplex class. Only unambiguous single-variable classes.
BIGOBENCH_MAP = {
    "1": "constant",
    "logn": "logn", "log(n)": "logn",
    "n": "linear",
    "nlogn": "nlogn", "n*logn": "nlogn", "nlog(n)": "nlogn", "n*log(n)": "nlogn", "nlog n": "nlogn",
    "n^2": "quadratic", "n2": "quadratic", "n*n": "quadratic", "n**2": "quadratic",
    "n^3": "cubic", "n3": "cubic", "n**3": "cubic",
    "2^n": "np", "2**n": "np",
}


# --------------------------------------------------------------------------- #
# Model construction
# --------------------------------------------------------------------------- #
def chosen_config(path: Path = tune_model.TUNING_RESULTS) -> dict:
    return tune_model.load_chosen(path)["chosen"]


def build(chosen: dict, seed: int = config.RANDOM_STATE):
    return tune_model.rf_pipeline(chosen["feature_set"], chosen["params"], seed)


def oof_predictions(df: pd.DataFrame, pipe, groups: pd.Series, seed: int, folds: int = 5) -> np.ndarray:
    """Out-of-fold predictions for every row, with whole groups held out."""
    y = df[pp.TARGET]
    out = np.empty(len(df), dtype=object)
    splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    for tr, te in splitter.split(np.zeros(len(df)), y, groups):
        out[te] = clone(pipe).fit(df.iloc[tr], y.iloc[tr]).predict(df.iloc[te])
    return out


def _scores(y_true, y_pred) -> dict:
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # small slices may lack some classes
        m = tm.metrics(y_true, y_pred)
    return {"n": int(len(y_true)), "accuracy": round(float(m["accuracy"]), 4),
            "macro_f1": round(float(m["macro_f1"]), 4), "weighted_f1": round(float(m["weighted_f1"]), 4)}


# --------------------------------------------------------------------------- #
# 1. Test evaluation
# --------------------------------------------------------------------------- #
def test_evaluation(df, split, fitted) -> dict:
    te = split.indices("test")
    y = df.loc[te, pp.TARGET]
    pred = fitted.predict(df.loc[te])
    proba = fitted.predict_proba(df.loc[te])
    m = {k: round(float(v), 4) for k, v in tm.metrics(y, pred).items()}
    lo, hi = tm.group_bootstrap_ci(y, pred, split.groups.loc[te])
    p, r, f, s = precision_recall_fscore_support(y, pred, labels=pp.CLASS_ORDER, zero_division=0)
    per_class = pd.DataFrame({"precision": p, "recall": r, "f1": f, "support": s},
                             index=[BIGO_OF[c] for c in pp.CLASS_ORDER]).round(3)
    return {"pred": pred, "proba": proba, "metrics": {**m, "macro_f1_ci": [round(lo, 4), round(hi, 4)]},
            "per_class": per_class, "cm": confusion_matrix(y, pred, labels=pp.CLASS_ORDER),
            "reliability": confidence_reliability(y.values, pred, proba)}


CONFIDENCE_BINS = (0.0, 0.4, 0.6, 0.8, 1.0001)


def confidence_reliability(y_true, y_pred, proba) -> list[dict]:
    """How often test predictions were right, by the model's top probability.

    Random Forest probabilities are averaged tree votes, not calibrated
    probabilities; this table records what they meant on the held-out test set.
    """
    top = np.asarray(proba).max(axis=1)
    correct = np.asarray(y_true) == np.asarray(y_pred)
    rows = []
    for lo, hi in zip(CONFIDENCE_BINS[:-1], CONFIDENCE_BINS[1:]):
        mask = (top >= lo) & (top < hi)
        rows.append({"from": lo, "to": min(hi, 1.0), "n": int(mask.sum()),
                     "accuracy": round(float(correct[mask].mean()), 4) if mask.any() else None})
    return rows


# --------------------------------------------------------------------------- #
# 2a. Different implementations of familiar algorithms
# --------------------------------------------------------------------------- #
def implementation_analysis(df, split, pred) -> dict:
    te = split.indices("test")
    t = df.loc[te, ["problem", "tags", pp.TARGET]].assign(pred=pred, correct=pred == df.loc[te, pp.TARGET].values)
    per_problem = t.groupby("problem").agg(
        solutions=("pred", "size"), accuracy=("correct", "mean"),
        distinct_predictions=("pred", "nunique"),
        majority_share=("pred", lambda s: s.value_counts(normalize=True).iloc[0]),
    )
    multi = per_problem[per_problem["solutions"] >= 3]
    train_tags = set(df.loc[split.indices("train").append(split.indices("validation")), "tags"].str.split(",").explode())
    tagged = t.assign(tag=t["tags"].str.split(",")).explode("tag")
    tagged = tagged[tagged["tag"].isin(FAMILY_TAGS)]
    by_family = {}
    for tag, g in tagged.groupby("tag"):
        if len(g) >= 20:
            by_family[tag] = {**_scores(g[pp.TARGET], g["pred"]), "seen_in_training": tag in train_tags,
                              "test_problems": int(g["problem"].nunique())}
    return {
        "test_problems": int(len(per_problem)),
        "problems_with_3plus_solutions": int(len(multi)),
        "mean_problem_accuracy": round(float(multi["accuracy"].mean()), 4) if len(multi) else None,
        "problems_all_correct": int((multi["accuracy"] == 1).sum()),
        "problems_all_wrong": int((multi["accuracy"] == 0).sum()),
        "mean_majority_prediction_share": round(float(multi["majority_share"].mean()), 4) if len(multi) else None,
        "problems_with_one_prediction_for_all_solutions": int((multi["distinct_predictions"] == 1).sum()),
        "by_family": by_family,
        "per_problem": per_problem,
    }


# --------------------------------------------------------------------------- #
# 2b/2c. Group hold-outs
# --------------------------------------------------------------------------- #
def family_mask(df: pd.DataFrame, tag: str) -> pd.Series:
    """True for rows whose problem carries ``tag`` (tags are comma-separated)."""
    return df["tags"].str.split(",").apply(lambda ts: tag in ts)


def holdout_experiments(df, pipe, seed) -> dict:
    y = df[pp.TARGET]
    oof_problem = oof_predictions(df, pipe, pp.make_groups(df, "problem"), seed)
    oof_contest = oof_predictions(df, pipe, pp.make_groups(df, "contest"), seed)
    families = {}
    for tag in FAMILY_TAGS:
        has = family_mask(df, tag)
        if has.sum() < 50:
            continue
        train_rows = ~has
        fitted = clone(pipe).fit(df[train_rows], y[train_rows])
        held_pred = fitted.predict(df[has])
        held, seen = _scores(y[has], held_pred), _scores(y[has], oof_problem[has.values])
        families[tag] = {
            "rows": int(has.sum()), "problems": int(df.loc[has, "problem"].nunique()),
            "family_held_out": held, "family_seen_problem_held_out": seen,
            "macro_f1_change": round(held["macro_f1"] - seen["macro_f1"], 4),
            "class_share_in_family_%": (100 * y[has].value_counts(normalize=True)).round(1).reindex(pp.CLASS_ORDER, fill_value=0).to_dict(),
            "training_rows_without_family": int(train_rows.sum()),
        }
    return {
        "problem_grouped_oof": _scores(y, oof_problem),
        "contest_grouped_oof": _scores(y, oof_contest),
        "families": families,
        "oof_problem_pred": oof_problem,
    }


# --------------------------------------------------------------------------- #
# 2b. External dataset (BigO(Bench)), if downloaded
# --------------------------------------------------------------------------- #
def normalise_bigobench_label(label: str) -> Optional[str]:
    if not isinstance(label, str) or not label.strip():
        return None
    s = label.strip().lower().replace(" ", "")
    if s.startswith("o(") and s.endswith(")"):
        s = s[2:-1]
    while s.startswith("(") and s.endswith(")"):
        s = s[1:-1]
    return BIGOBENCH_MAP.get(s)


def bigobench_problem_id(record: dict) -> Optional[str]:
    for key in ("problem_name", "problem_id"):
        m = re.match(r"^\s*(\d+)_([A-Z][0-9]?)\b", str(record.get(key, "")))
        if m:
            return f"{int(m.group(1)):04d}_{m.group(2)}"
    return None


def load_bigobench(path: Path, known_problems: set) -> tuple[pd.DataFrame, dict]:
    """Features for the BigO(Bench) time test set; never executes the code."""
    stats = {"records": 0, "unmapped_label": 0, "unparsable": 0, "overlapping_problem": 0, "labels_unmapped": {}}
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            rec = json.loads(line)
            stats["records"] += 1
            raw_label = rec.get("time_complexity_inferred")
            label = normalise_bigobench_label(raw_label)
            if label is None:
                stats["unmapped_label"] += 1
                stats["labels_unmapped"][str(raw_label)] = stats["labels_unmapped"].get(str(raw_label), 0) + 1
                continue
            report = extract_features(rec.get("solution_code", ""))
            if not report.success:
                stats["unparsable"] += 1
                continue
            pid = bigobench_problem_id(rec)
            overlap = pid is not None and pid in known_problems
            stats["overlapping_problem"] += int(overlap)
            rows.append({"problem": pid or str(rec.get("problem_name", "")), "label": label, "overlap": overlap,
                         **report.file_features, **extract_pattern_features(report.parse_result.tree)})
    return pd.DataFrame(rows), stats


def external_evaluation(fitted, df, path: Optional[Path] = None) -> Optional[dict]:
    path = path or next((p for p in BIGOBENCH_CANDIDATES if p.exists()), None)
    if path is None or not path.exists():
        return None
    ext, stats = load_bigobench(path, set(df["problem"]))
    out = {"file": str(path.name), **{k: v for k, v in stats.items()}}
    if len(ext):
        fresh = ext[~ext["overlap"]]
        for name, part in (("unseen_problems", fresh), ("all_mapped", ext)):
            if len(part):
                pred = fitted.predict(part)
                out[name] = {**_scores(part["label"], pred),
                             "class_counts": part["label"].map(BIGO_OF).value_counts().to_dict()}
                if name == "unseen_problems":
                    out["cm"] = confusion_matrix(part["label"], pred, labels=pp.CLASS_ORDER).tolist()
    return out


# --------------------------------------------------------------------------- #
# 2d. Structural slices of the test set
# --------------------------------------------------------------------------- #
def structural_slices(df, split, pred, sources: Optional[pd.Series]) -> pd.DataFrame:
    te = split.indices("test")
    t = df.loc[te].assign(pred=pred)
    q = t["code_lines"].quantile([0.25, 0.5, 0.75]).values
    slices = {
        "all test programs": t.index,
        "no functions (module-level code only)": t.index[t["function_count"] == 0],
        "defines functions": t.index[t["function_count"] > 0],
        "uses recursion": t.index[t.get("recursive_function_count", pd.Series(0, index=t.index)) > 0],
        "uses sorting": t.index[t.get("sort_call_count", pd.Series(0, index=t.index)) > 0],
        "no input-dependent loop": t.index[t.get("max_input_loop_depth", t["maximum_loop_nesting_depth"]) == 0],
        "input-dependent loop depth 1": t.index[t.get("max_input_loop_depth", t["maximum_loop_nesting_depth"]) == 1],
        "input-dependent loop depth 2": t.index[t.get("max_input_loop_depth", t["maximum_loop_nesting_depth"]) == 2],
        "input-dependent loop depth 3+": t.index[t.get("max_input_loop_depth", t["maximum_loop_nesting_depth"]) >= 3],
        f"shortest 25% (<= {q[0]:.0f} code lines)": t.index[t["code_lines"] <= q[0]],
        f"longest 25% (> {q[2]:.0f} code lines)": t.index[t["code_lines"] > q[2]],
    }
    if sources is not None:
        tmpl = sources.loc[te].str.contains(r"sys\.stdin", regex=True)
        slices["template code (reads sys.stdin)"] = t.index[tmpl.values]
        slices["no template code"] = t.index[~tmpl.values]
    rows = []
    for name, idx in slices.items():
        if len(idx) >= 15:
            rows.append({"slice": name, **_scores(t.loc[idx, pp.TARGET], t.loc[idx, "pred"])})
    return pd.DataFrame(rows).set_index("slice")


# --------------------------------------------------------------------------- #
# 3. Failure cases
# --------------------------------------------------------------------------- #
# Manual review of the most confident errors in the Phase 11 run (source read locally, not copied).
# Keyed by problem id; shown only when that problem appears among the confident errors.
REVIEWED_FAILURES = {
    "0990_A": ("labelled O(n), predicted O(1). The solutions are a few arithmetic operations on the input numbers, "
               "which is constant time: the model's answer looks right and the label looks wrong (label noise)."),
    "0276_D": ("labelled O(log n), predicted O(1). A single expression using int.bit_length(); whether bit operations "
               "count as O(log n) (number of bits) or O(1) is a labelling convention."),
    "1080_B": ("labelled O(1), predicted O(log n). A loop over Q queries with O(1) work each; CodeComplex measures each "
               "test case separately, so the per-query cost is the label. The model sees a loop and cannot know this convention."),
    "1100_C": ("labelled O(1), predicted O(log n). A binary search on a real number down to a fixed precision: the "
               "number of halvings is bounded by a constant, so O(1) is defensible, and so is the model's O(log n) "
               "reading of the halving loop. A labelling convention, not a clear model error."),
    "0975_C": ("labelled O(n log n), predicted O(n). bisect is called inside a loop over the queries, which is "
               "O(q log n): a genuine model error (a binary-search call inside a loop is not weighted enough)."),
}

FAILURE_FEATURES = ("max_input_loop_depth", "sort_call_count", "halving_loop_count", "power_of_two_expr_count",
                    "recursive_function_count", "code_lines")


def failure_analysis(df, split, pred, proba, classes, top: int = 15) -> dict:
    te = split.indices("test")
    t = df.loc[te].assign(pred=pred, confidence=proba.max(axis=1))
    wrong = t[t["pred"] != t[pp.TARGET]]
    distance = (t[pp.TARGET].map(RANK) - t["pred"].map(RANK)).abs()
    pairs = (wrong.groupby([pp.TARGET, "pred"]).size().sort_values(ascending=False).head(8)
             .rename("count").reset_index())
    pairs["true"] = pairs[pp.TARGET].map(BIGO_OF)
    pairs["predicted"] = pairs["pred"].map(BIGO_OF)
    cols = ["sample_id", "problem", pp.TARGET, "pred", "confidence"] + [f for f in FAILURE_FEATURES if f in t.columns]
    confident = wrong.sort_values("confidence", ascending=False).head(top)[cols].copy()
    confident[pp.TARGET] = confident[pp.TARGET].map(BIGO_OF)
    confident["pred"] = confident["pred"].map(BIGO_OF)
    confident["confidence"] = confident["confidence"].round(2)
    return {
        "errors": int(len(wrong)),
        "distance_%": {"correct": round(100 * float((distance == 0).mean()), 1),
                       "adjacent class": round(100 * float((distance == 1).mean()), 1),
                       "2+ classes away": round(100 * float((distance >= 2).mean()), 1)},
        "mean_confidence_correct": round(float(t.loc[t["pred"] == t[pp.TARGET], "confidence"].mean()), 3),
        "mean_confidence_wrong": round(float(wrong["confidence"].mean()), 3) if len(wrong) else None,
        "top_confusions": pairs[["true", "predicted", "count"]],
        "confident_errors": confident.rename(columns={pp.TARGET: "true", "pred": "predicted"}),
    }


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "savefig.facecolor": "#fcfcfb",
        "axes.edgecolor": "#e4e3df", "axes.labelcolor": "#52514e", "xtick.color": "#52514e",
        "ytick.color": "#52514e", "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": "#e4e3df", "grid.linewidth": 0.6, "axes.axisbelow": True,
        "axes.titlelocation": "left", "axes.titlesize": 12, "axes.titleweight": "bold", "legend.frameon": False,
    })
    return plt


def plot_confusion(cm: np.ndarray, title: str, path: Path) -> None:
    plt = _plt()
    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    tm._plot_confusion(ax, cm, title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True label")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_per_class(per_class: pd.DataFrame, path: Path) -> None:
    plt = _plt()
    fig, ax = plt.subplots(figsize=(8.5, 3.6))
    x = np.arange(len(per_class))
    w = 0.38
    ax.bar(x - w / 2, per_class["precision"], w, color="#2a78d6", label="precision", edgecolor="#fcfcfb", linewidth=1.5)
    ax.bar(x + w / 2, per_class["recall"], w, color="#eb6834", label="recall", edgecolor="#fcfcfb", linewidth=1.5)
    for i, f in enumerate(per_class["f1"]):
        ax.text(i, max(per_class["precision"].iloc[i], per_class["recall"].iloc[i]) + 0.03, f"F1 {f:.2f}",
                ha="center", fontsize=8, color="#52514e")
    ax.set_xticks(x, per_class.index)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper right", ncol=2, fontsize=9)
    ax.set_title("Per-class precision and recall on the test set")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_families(families: dict, path: Path) -> None:
    plt = _plt()
    names = sorted(families, key=lambda k: families[k]["family_seen_problem_held_out"]["macro_f1"], reverse=True)
    seen = [families[n]["family_seen_problem_held_out"]["macro_f1"] for n in names]
    held = [families[n]["family_held_out"]["macro_f1"] for n in names]
    fig, ax = plt.subplots(figsize=(9, 0.42 * len(names) + 1.4))
    y = np.arange(len(names))
    for i in range(len(names)):
        ax.plot([held[i], seen[i]], [i, i], color="#c3c2b7", linewidth=2, zorder=1)
    ax.scatter(seen, y, s=70, color="#2a78d6", edgecolor="#fcfcfb", linewidth=2, zorder=2, label="family seen in training (problem held out)")
    ax.scatter(held, y, s=70, color="#eb6834", edgecolor="#fcfcfb", linewidth=2, zorder=3, label="whole family held out")
    ax.set_yticks(y, names)
    ax.invert_yaxis()
    ax.set_xlabel("Macro F1 on that family's programs")
    ax.set_xlim(0, max(seen + held) + 0.1)
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right", fontsize=8.5)
    ax.set_title("Generalization to algorithm families excluded from training")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_slices(slices: pd.DataFrame, overall: float, path: Path) -> None:
    plt = _plt()
    data = slices.drop(index="all test programs", errors="ignore")
    fig, ax = plt.subplots(figsize=(9, 0.4 * len(data) + 1.4))
    y = np.arange(len(data))
    ax.barh(y, data["accuracy"], color="#2a78d6", height=0.6, edgecolor="#fcfcfb")
    for i, (v, n) in enumerate(zip(data["accuracy"], data["n"])):
        ax.text(v + 0.01, i, f"{v:.2f}  (n={n})", va="center", fontsize=8, color="#52514e")
    ax.axvline(overall, color="#0b0b0b", linestyle="--", linewidth=1)
    ax.text(overall, len(data) - 0.2, f" all test programs {overall:.2f}", fontsize=8, color="#0b0b0b", va="top")
    ax.set_yticks(y, data.index)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("Accuracy (macro F1 is misleading inside a slice that contains only a few classes)")
    ax.grid(axis="y", visible=False)
    ax.set_title("Test accuracy by structural pattern")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #
def _md(df: pd.DataFrame, index: bool = True) -> str:
    from src.eda import _md_table

    return _md_table(df, index=index)


def _reviewed_notes(errors: pd.DataFrame) -> list[str]:
    found = [p for p in dict.fromkeys(errors["problem"]) if p in REVIEWED_FAILURES]
    if not found:
        return []
    lines = ["Manual review of these errors (source read locally):", ""]
    lines += [f"- **{p}:** {REVIEWED_FAILURES[p]}" for p in found]
    lines += ["", "Several of the most confident \"errors\" are therefore disagreements about labelling conventions or label",
              "noise rather than model mistakes, so the measured accuracy somewhat understates agreement with a careful",
              "reading of the code, but the labels are the reference and are not corrected here.", ""]
    return lines


def render_report(r: dict) -> str:
    tune = r["tuning"]
    test = r["test"]
    imp, hold, fail = r["implementations"], r["holdouts"], r["failures"]
    m = test["metrics"]
    fam = pd.DataFrame({
        k: {"rows": v["rows"], "problems": v["problems"],
            "macro F1 (family seen)": v["family_seen_problem_held_out"]["macro_f1"],
            "macro F1 (family held out)": v["family_held_out"]["macro_f1"],
            "change": v["macro_f1_change"],
            "accuracy (held out)": v["family_held_out"]["accuracy"]}
        for k, v in hold["families"].items()}).T.sort_values("change")
    worst = fam.index[0] if len(fam) else None
    mean_change = float(fam["change"].mean()) if len(fam) else float("nan")
    by_family = (pd.DataFrame(imp["by_family"]).T[["n", "test_problems", "accuracy", "macro_f1"]]
                 .astype({"n": int, "test_problems": int, "accuracy": float, "macro_f1": float})
                 if imp["by_family"] else None)

    lines = [
        "# Tuning and generalization",
        "",
        "Generated by `python -m src.tune_model` and `python -m src.evaluate_model` (seed "
        f"{config.RANDOM_STATE}). All numbers are measured; the test set was used once, after tuning.",
        "",
        "## 1. Tuning (Random Forest)",
        "",
        f"RandomizedSearchCV, {tune['n_iter_per_feature_set']} configurations per feature set, grouped 5-fold CV on",
        "train + validation (test untouched). Before/after compared on the Phase 10 repeated grouped CV folds.",
        "",
        "| | Feature set | Repeated-CV macro F1 | Test macro F1 | Test accuracy |",
        "| --- | --- | ---: | ---: | ---: |",
        f"| Before (Phase 10) | {tune['before']['feature_set']} | {tune['before']['rcv_macro_f1']:.3f} ± {tune['before']['rcv_std']:.3f} | "
        f"{r['before_test']['macro_f1']:.3f} | {r['before_test']['accuracy']:.3f} |",
        f"| After tuning | {tune['tuned']['feature_set']} | {tune['tuned']['rcv_macro_f1']:.3f} ± {tune['tuned']['rcv_std']:.3f} | "
        f"{r['tuned_test']['macro_f1']:.3f} | {r['tuned_test']['accuracy']:.3f} |",
        "",
        "Best search score per feature set: " + "; ".join(
            f"`{fs}` {s['search_cv_macro_f1']:.3f} ± {s['search_cv_std']:.3f}" for fs, s in tune["searches"].items()) + ".",
        f"Tuned parameters: `{json.dumps(tune['tuned']['params'])}`.",
        "",
        f"**Decision: {'the tuned configuration was adopted' if tune['adopted'] == 'tuned' else 'the Phase 10 configuration was kept'}.** "
        f"The rule, fixed before testing, adopts the tuned model only if it improves repeated-CV macro F1 by more than "
        f"{tune['adopt_margin']}; the change was {tune['rcv_gain']:+.3f}. The tuned score is slightly optimistic because",
        "tuning used the same rows. Both test scores are shown for transparency; they did not influence the decision.",
        "",
        "## 2. Final model on the test set",
        "",
        f"Configuration: Random Forest, `{r['chosen']['feature_set']}` features, `{json.dumps(r['chosen']['params'])}`;",
        f"trained on {r['n_dev']:,} train + validation rows; tested once on {r['n_test']:,} rows from unseen problems.",
        "",
        "| Accuracy | Balanced acc. | Macro precision | Macro recall | Macro F1 [95% CI] | Weighted F1 |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
        f"| {m['accuracy']:.3f} | {m['balanced_accuracy']:.3f} | {m['macro_precision']:.3f} | {m['macro_recall']:.3f} | "
        f"{m['macro_f1']:.3f} [{m['macro_f1_ci'][0]:.3f}, {m['macro_f1_ci'][1]:.3f}] | {m['weighted_f1']:.3f} |",
        "",
        "![Confusion matrix](confusion_matrix_test.png)",
        "",
        "![Per-class](per_class_test.png)",
        "",
        _md(test["per_class"].rename_axis("class")),
        "",
        "## 3. Generalization",
        "",
        "### a. Different implementations of familiar algorithm families",
        "",
        f"Every test problem is unseen, but its algorithm families also occur in training. Across "
        f"{imp['problems_with_3plus_solutions']} test problems with 3+ solutions, the mean per-problem accuracy is "
        f"{imp['mean_problem_accuracy']:.3f}; {imp['problems_all_correct']} problems had every solution right and "
        f"{imp['problems_all_wrong']} had every solution wrong. On average {100 * imp['mean_majority_prediction_share']:.0f}% of a",
        f"problem's solutions receive the same prediction ({imp['problems_with_one_prediction_for_all_solutions']} problems got a",
        "single prediction for all solutions). Different implementations of one algorithm are therefore usually, but not always,",
        "classified alike, and when the model is wrong about a problem it tends to be wrong for most of its solutions.",
        "",
    ]
    if by_family is not None:
        lines += ["Test results by algorithm family (tags; a program can have several). Families with few test problems",
                  "(see the column) give unstable numbers, and macro F1 within one family is low because each family",
                  "covers only a few classes:", "", _md(by_family.rename_axis("family")), ""]
    lines += [
        "### b. Excluded source groups",
        "",
        "All CodeComplex programs come from Codeforces, so a true source or repository hold-out is not possible inside",
        "this dataset. Out-of-fold predictions over the whole dataset (5 folds, Random Forest re-fitted per fold):",
        "",
        "| Held out together | Accuracy | Macro F1 |",
        "| --- | ---: | ---: |",
        f"| Problems | {hold['problem_grouped_oof']['accuracy']:.3f} | {hold['problem_grouped_oof']['macro_f1']:.3f} |",
        f"| Contests (whole rounds) | {hold['contest_grouped_oof']['accuracy']:.3f} | {hold['contest_grouped_oof']['macro_f1']:.3f} |",
        "",
        f"Holding out whole contests instead of single problems changes macro F1 by "
        f"{hold['contest_grouped_oof']['macro_f1'] - hold['problem_grouped_oof']['macro_f1']:+.3f}, so problems from the same round",
        "do not leak much information to each other.",
        "",
        f"Note that this cross-validated estimate over **all** problems ({hold['problem_grouped_oof']['macro_f1']:.3f}) differs from the",
        f"single held-out test set ({m['macro_f1']:.3f}). Both measure unseen problems; the difference shows how much the score",
        "depends on *which* problems are tested (a few problems contribute many solutions). The honest summary is a range,",
        f"roughly {min(m['macro_f1'], hold['problem_grouped_oof']['macro_f1']):.2f}–{max(m['macro_f1'], hold['problem_grouped_oof']['macro_f1']):.2f} macro F1, not a single number.",
        "",
    ]
    ext = r.get("external")
    if ext is None:
        lines += [
            "**External dataset:** BigO(Bench) was not found in `data/raw/bigobench/`, so no cross-source evaluation was run.",
            "Download it (see the commands in the project notes) and rerun `python -m src.evaluate_model` to add it.",
            "",
        ]
    else:
        u = ext.get("unseen_problems")
        lines += [
            f"**External dataset: BigO(Bench) time-complexity test set** (`{ext['file']}`): {ext['records']} records; "
            f"{ext['unmapped_label']} excluded because their label has no single-variable equivalent "
            f"({', '.join(f'{k} ×{v}' for k, v in sorted(ext['labels_unmapped'].items(), key=lambda kv: -kv[1])[:6])}); "
            f"{ext['unparsable']} not parsable by this Python; {ext['overlapping_problem']} from problems also in CodeComplex.",
        ]
        if u:
            lines += [
                f"On {u['n']} solutions from problems **not** in CodeComplex: accuracy {u['accuracy']:.3f}, macro F1 {u['macro_f1']:.3f}.",
                "Its labels come from runtime measurement rather than expert reading, so disagreement can reflect the",
                "labelling method as well as the model.",
            ]
        lines.append("")
    lines += [
        "### c. Algorithm families excluded from training",
        "",
        "For each family, the model is trained on every problem without that tag and tested on the problems with it, and",
        "compared with the same rows predicted when the family is present in training (problem held out).",
        "",
        "![Families](family_holdout.png)",
        "",
        _md(fam.rename_axis("family")) if len(fam) else "_Not enough data per family._",
        "",
        (f"Holding out a family changes macro F1 by {mean_change:+.3f} on average; the largest drop is for **{worst}** "
         f"({fam.loc[worst, 'change']:+.3f})." if worst else ""),
        "A drop shows that part of the performance relies on having seen similar algorithms. Because tags are tied to labels,",
        "holding out a family also removes many examples of its typical class (for example, bitmasks are mostly O(2^n)).",
        "",
        "### d. Different structural patterns (test set)",
        "",
        "![Slices](structural_slices.png)",
        "",
        _md(r["slices"].rename_axis("slice")),
        "",
        "Accuracy is the fairer comparison between slices: a slice such as \"uses sorting\" contains mostly one or two",
        "classes, which pulls its macro F1 down even when most of its programs are classified correctly.",
        "",
        "## 4. Failure cases",
        "",
        f"{fail['errors']} of {r['n_test']} test programs are misclassified. Distance from the true class: "
        f"{fail['distance_%']['correct']}% correct, {fail['distance_%']['adjacent class']}% one class away, "
        f"{fail['distance_%']['2+ classes away']}% two or more away. Mean confidence: {fail['mean_confidence_correct']:.2f}"
        f" when right, {fail['mean_confidence_wrong']:.2f} when wrong.",
        "",
        "Most frequent confusions:",
        "",
        _md(fail["top_confusions"], index=False),
        "",
        "Most confident errors (ids and features only; source code is not copied because the CodeComplex licence is unclear):",
        "",
        _md(fail["confident_errors"], index=False),
        "",
        *_reviewed_notes(fail["confident_errors"]),
        "## 5. Honest assessment",
        "",
        f"- **Within this domain** (unseen Codeforces problems, familiar algorithm families) the model reaches macro F1 "
        f"{m['macro_f1']:.3f} ({m['macro_f1_ci'][0]:.2f}–{m['macro_f1_ci'][1]:.2f}), about three times chance. That is a real but modest skill.",
        f"- **Unseen algorithm families:** performance drops (average change {mean_change:+.3f}). The model should not be",
        "  described as generalising to algorithms unlike those it was trained on.",
        "- **Other sources:** " + ("only one external dataset was available (results above); everyday code (classes, libraries, I/O) "
                                    "has not been tested." if ext else "not tested (BigO(Bench) not downloaded); everyday code has not been tested."),
        *([f"- **Code with functions** scores lower (accuracy {r['slices'].loc['defines functions', 'accuracy']:.2f}, macro F1 "
            f"{r['slices'].loc['defines functions', 'macro_f1']:.2f}) than module-level scripts (accuracy "
            f"{r['slices'].loc['no functions (module-level code only)', 'accuracy']:.2f}, macro F1 "
            f"{r['slices'].loc['no functions (module-level code only)', 'macro_f1']:.2f}). This matters because the app analyses functions."]
          if {"defines functions", "no functions (module-level code only)"} <= set(r["slices"].index) else []),
        "- **Limitations:** whole-program labels; one platform and style; label noise (identical code with different labels);",
        "  syntax-level features that cannot see loop bounds or input sizes; no space-complexity labels.",
        "",
        "### How the project could be improved",
        "",
        "1. Label a few hundred **function-level** examples from everyday code to test the real use case.",
        "2. Add sources beyond Codeforces (BigO(Bench) solutions, textbook implementations with documented complexity).",
        "3. Features that capture **loop bounds** (which variable a loop runs to, and whether it depends on input size).",
        "4. Combine the model with the rule-based analyser: use the rules when they are confident, the model otherwise.",
        "5. Report predictions with their probabilities, and abstain below a confidence threshold.",
        "",
        f"Software: Python {r['versions']['python']}, scikit-learn {r['versions']['scikit-learn']}.",
        "",
    ]
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def run(df: Optional[pd.DataFrame] = None, split: Optional[pp.SplitResult] = None, tuning: Optional[dict] = None,
        out_dir: Path = REPORT_DIR, models_dir: Path = config.MODELS_DIR, seed: int = config.RANDOM_STATE,
        sources: Optional[pd.Series] = None, external_path: Optional[Path] = None, save: bool = True,
        verbose: bool = True) -> dict:
    t0 = time.time()
    if df is None:
        df = pd.read_csv(config.BIGO_DATASET_CSV, keep_default_na=False)
        from src.eda import attach_sources

        sources = attach_sources(df, config.CODECOMPLEX_DIR / config.CODECOMPLEX["file"])
    if split is None:
        split = pp.load_split(df)
    if tuning is None:
        tuning = tune_model.load_chosen()
    chosen = tuning["chosen"]
    y = df[pp.TARGET]
    dev = split.indices("train").append(split.indices("validation"))
    te = split.indices("test")

    def log(msg):
        if verbose:
            print(f"[{time.time() - t0:5.0f}s] {msg}", flush=True)

    # Final model (fitted on train + validation) and the single test evaluation
    features = pp.feature_columns(chosen["feature_set"])
    final = build(chosen, seed).fit(df.loc[dev, features], y.loc[dev])  # fitted on the feature columns only
    test = test_evaluation(df, split, final)
    log(f"test macro F1 {test['metrics']['macro_f1']:.3f}, accuracy {test['metrics']['accuracy']:.3f}")

    # Before vs after on the test set (reporting only)
    def test_of(cfg):
        p = build(cfg, seed).fit(df.loc[dev], y.loc[dev]).predict(df.loc[te])
        return _scores(y.loc[te], p)

    before_test = test_of({"feature_set": tuning["before"]["feature_set"], "params": tuning["before"]["params"]})
    tuned_test = test_of({"feature_set": tuning["tuned"]["feature_set"], "params": tuning["tuned"]["params"]})

    implementations = implementation_analysis(df, split, test["pred"])
    log("implementations done")
    holdouts = holdout_experiments(df, build(chosen, seed), seed)
    log("hold-out experiments done")
    slices = structural_slices(df, split, test["pred"], sources)
    failures = failure_analysis(df, split, test["pred"], test["proba"], final.classes_)
    external = external_evaluation(final, df, external_path)
    log("external evaluation " + ("done" if external else "skipped (BigO(Bench) not downloaded)"))

    result = {
        "tuning": tuning, "chosen": chosen, "test": test, "before_test": before_test, "tuned_test": tuned_test,
        "implementations": implementations, "holdouts": holdouts, "slices": slices, "failures": failures,
        "external": external, "n_dev": int(len(dev)), "n_test": int(len(te)),
        "versions": {"python": platform.python_version(), "scikit-learn": sklearn.__version__,
                     "pandas": pd.__version__, "numpy": np.__version__},
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    plot_confusion(test["cm"], "Final model: test confusion matrix", out_dir / "confusion_matrix_test.png")
    plot_per_class(test["per_class"], out_dir / "per_class_test.png")
    if holdouts["families"]:
        plot_families(holdouts["families"], out_dir / "family_holdout.png")
    plot_slices(slices, test["metrics"]["accuracy"], out_dir / "structural_slices.png")
    if external and external.get("cm"):
        plot_confusion(np.array(external["cm"]), "BigO(Bench), unseen problems", out_dir / "confusion_matrix_bigobench.png")
    (out_dir / "evaluation_report.md").write_text(render_report(result), encoding="utf-8")
    (out_dir / "evaluation_results.json").write_text(json.dumps(_summary(result), indent=2, default=str), encoding="utf-8")
    failures["confident_errors"].to_csv(out_dir / "failure_cases.csv", index=False)

    if save:
        models_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(final, models_dir / config.MODEL_PATH.name)
        joblib.dump(final.named_steps["preprocess"], models_dir / PREPROCESSING_PATH.name)
        meta = {
            "phase11": True,
            "selected_model": "Random Forest",
            "feature_set": chosen["feature_set"],
            "features": pp.feature_columns(chosen["feature_set"]),
            "hyperparameters": chosen["params"],
            "classes": pp.CLASS_ORDER,
            "class_bigo": [BIGO_OF[c] for c in pp.CLASS_ORDER],
            "trained_on": f"{len(dev)} train + validation rows (grouped by problem)",
            "test_metrics": test["metrics"],
            "per_class_test": test["per_class"].to_dict(orient="index"),
            "confidence_reliability_test": test["reliability"],
            "generalization": _summary(result)["generalization"],
            "tuning": {k: tuning[k] for k in ("adopted", "rcv_gain", "adopt_margin")},
            "seed": seed,
            "files": {"model": config.MODEL_PATH.name, "preprocessing": PREPROCESSING_PATH.name},
            "versions": result["versions"],
            "dataset_sha256": tm._sha(config.BIGO_DATASET_CSV),
            "split_file_sha256": tm._sha(pp.SPLIT_FILE),
            "scope": ("Agreement with CodeComplex expert labels on unseen Codeforces problems (whole programs). "
                      "Not validated on everyday code, on single functions, or for space complexity; weaker on "
                      "algorithm families absent from training."),
        }
        meta = model_integrity.attach(meta, models_dir)  # SHA-256 of both files; checked before every load
        (models_dir / config.MODEL_METADATA_PATH.name).write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    log("saved report" + (" and model" if save else ""))
    return result


def _summary(r: dict) -> dict:
    fams = r["holdouts"]["families"]
    return {
        "chosen": r["chosen"],
        "test_metrics": r["test"]["metrics"],
        "before_test": r["before_test"], "tuned_test": r["tuned_test"],
        "generalization": {
            "problem_grouped_oof": r["holdouts"]["problem_grouped_oof"],
            "contest_grouped_oof": r["holdouts"]["contest_grouped_oof"],
            "family_holdout_mean_macro_f1_change": round(float(np.mean([v["macro_f1_change"] for v in fams.values()])), 4) if fams else None,
            "families": {k: {"held_out": v["family_held_out"]["macro_f1"], "seen": v["family_seen_problem_held_out"]["macro_f1"]}
                         for k, v in fams.items()},
            "implementations_mean_problem_accuracy": r["implementations"]["mean_problem_accuracy"],
            "external": {k: v for k, v in (r["external"] or {}).items() if k != "cm"} or None,
        },
        "failure_distance_%": r["failures"]["distance_%"],
    }


def main() -> int:
    if not config.BIGO_DATASET_CSV.exists() or not pp.SPLIT_FILE.exists():
        print("Run: python -m src.dataset_loader build  and  python -m src.preprocessing  first.")
        return 1
    if not tune_model.TUNING_RESULTS.exists():
        print("Run: python -m src.tune_model  first.")
        return 1
    run()
    print(f"Report: {REPORT_DIR / 'evaluation_report.md'}")
    print(f"Model:  {config.MODEL_PATH}\nPreprocessing: {PREPROCESSING_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
