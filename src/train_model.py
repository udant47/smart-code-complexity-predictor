"""Classical ML model comparison for Big-O classification (Phase 10).

    python -m src.train_model          # full comparison (about 15-40 minutes, depending on the CPU)
    python -m src.train_model --quick  # small grids, a few minutes

Protocol (the same partitions for every model)
----------------------------------------------
The grouped-by-problem split from Phase 9 (``data/processed/bigo_splits.csv``):
no problem and no near-duplicate cluster appears in more than one partition.
"Development" = train + validation; "test" is never used before step 4.

1. **Tuning:** each model's hyperparameters are chosen by grid search with
   5-fold *grouped, stratified* cross-validation on the **training** rows,
   scored by macro F1. Preprocessing (imputation, log, scaling) is inside each
   Pipeline, so it is re-fitted on every training fold.
2. **Comparison:** every tuned (model, feature set) pair is scored by
   **repeated grouped cross-validation on the development rows**
   (``RCV_REPEATS`` × 5 folds, different shuffles). Averaging over many
   folds of unseen problems is far less noisy than one validation split.
3. **Selection:** the pair with the highest mean macro F1 in step 2. If
   several are within ``TIE_TOLERANCE``, the simplest wins.
4. **Test:** every pair is refitted on the development rows and evaluated
   **once** on the test rows, with 95% confidence intervals from bootstrapping
   whole test *problems*. Test scores are reported for transparency; they do
   not influence the selection.

Candidates: majority and stratified dummies; a one-feature loop-nesting rule;
Logistic Regression; SVM (RBF kernel); Decision Tree; Random Forest;
Gradient Boosting; Histogram Gradient Boosting; a soft-voting ensemble of the
tuned Logistic Regression, Random Forest and Histogram Gradient Boosting; and
XGBoost only if it is installed (it is not a dependency: two
gradient-boosting implementations are already compared).

Feature sets: ``all`` (39 structural features, as in the first round) and
``patterns`` (those plus 18 algorithm-pattern features from
``src/pattern_features.py``).

Macro F1 is the main metric (each of the seven classes counts equally).
Accuracy, balanced accuracy, macro precision and recall, weighted F1, per-class
F1 and confusion matrices are reported too.

Extra checks on the selected model: its score on a conventional random split
(leakage size) and without size features (length-shortcut ablation).

What the result means
---------------------
The labels are expert annotations, not thresholds on the features, so the
scores are not circular. They measure **agreement with the CodeComplex
annotators on Codeforces problems the model has not seen**, from static
features of whole programs, not accuracy on everyday code, on single
functions, or for space complexity.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    VotingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from src import config
from src import model_integrity
from src import preprocessing as pp
from src.dataset_loader import BIGO_LABELS

REPORT_DIR = config.PROJECT_ROOT / "reports" / "models"
TIE_TOLERANCE = 0.005
RCV_REPEATS = 2
N_BOOTSTRAP = 1000
FEATURE_SETS = ("all", "patterns")
CLASS_NAMES = [BIGO_LABELS[c][1] for c in pp.CLASS_ORDER]
FIRST_ROUND = {"model": "Logistic Regression", "features": "all", "test_macro_f1": 0.404}  # Phase 10, round 1


# --------------------------------------------------------------------------- #
# A transparent one-feature baseline
# --------------------------------------------------------------------------- #
class NestingDepthRule(BaseEstimator, ClassifierMixin):
    """Predicts the most common training label for each maximum loop-nesting depth."""

    def __init__(self, feature: str = "maximum_loop_nesting_depth", cap: int = 4):
        self.feature = feature
        self.cap = cap

    def fit(self, X, y):
        depth = pd.Series(np.asarray(X[self.feature]).clip(max=self.cap), name="d")
        labels = pd.Series(np.asarray(y), name="y")
        self.table_ = labels.groupby(depth).agg(lambda s: s.value_counts().index[0]).to_dict()
        self.default_ = labels.value_counts().index[0]
        self.classes_ = np.array(sorted(labels.unique()))
        return self

    def predict(self, X):
        depth = np.asarray(X[self.feature]).clip(max=self.cap)
        return np.array([self.table_.get(d, self.default_) for d in depth])


# --------------------------------------------------------------------------- #
# Candidates
# --------------------------------------------------------------------------- #
@dataclass
class Candidate:
    name: str
    family: str  # "tree" / "linear" / "distance" / "none" (no preprocessing)
    estimator: object
    grid: dict
    simplicity: int  # lower = simpler; breaks near-ties only
    feature_sets: tuple = FEATURE_SETS
    members: tuple = field(default_factory=tuple)  # for the ensemble


def candidates(quick: bool = False, seed: int = config.RANDOM_STATE) -> list[Candidate]:
    m = "model__"
    q = quick
    out = [
        Candidate("Dummy (most frequent)", "tree", DummyClassifier(strategy="most_frequent"), {}, 0, ("all",)),
        Candidate("Dummy (stratified)", "tree", DummyClassifier(strategy="stratified", random_state=seed), {}, 0, ("all",)),
        Candidate("Loop-nesting rule", "none", NestingDepthRule(), {}, 1, ("all",)),
        Candidate("Logistic Regression", "linear", LogisticRegression(max_iter=5000, random_state=seed),
                  {m + "C": [0.01, 0.1] if q else [0.003, 0.01, 0.03, 0.1, 0.3, 1.0],
                   m + "class_weight": ["balanced"] if q else [None, "balanced"]}, 2),
        Candidate("SVM (RBF)", "distance", SVC(kernel="rbf", class_weight="balanced", random_state=seed),
                  {m + "C": [1.0] if q else [0.3, 1.0, 3.0, 10.0], m + "gamma": ["scale"] if q else ["scale", 0.01]}, 3),
        Candidate("Decision Tree", "tree", DecisionTreeClassifier(random_state=seed),
                  {m + "max_depth": [6] if q else [4, 6, 8, 12],
                   m + "min_samples_leaf": [5] if q else [1, 5, 10],
                   m + "class_weight": ["balanced"] if q else [None, "balanced"]}, 3),
        Candidate("Random Forest", "tree",
                  RandomForestClassifier(n_estimators=80 if q else 300, random_state=seed, n_jobs=1),
                  {m + "max_features": ["sqrt"] if q else ["sqrt", 0.5],
                   m + "min_samples_leaf": [2] if q else [1, 3],
                   m + "class_weight": ["balanced_subsample"] if q else [None, "balanced_subsample"]}, 4),
        Candidate("Gradient Boosting", "tree",
                  GradientBoostingClassifier(n_estimators=40 if q else 200, learning_rate=0.1, subsample=0.8, random_state=seed),
                  {m + "max_depth": [2] if q else [2, 3]}, 5),
        Candidate("Hist Gradient Boosting", "tree",
                  HistGradientBoostingClassifier(max_iter=60 if q else 300, class_weight="balanced", random_state=seed),
                  {m + "learning_rate": [0.1] if q else [0.03, 0.1],
                   m + "max_leaf_nodes": [15] if q else [15, 31],
                   m + "l2_regularization": [1.0] if q else [0.0, 1.0]}, 5),
    ]
    try:
        from xgboost import XGBClassifier  # optional; not a project dependency
    except ImportError:
        XGBClassifier = None
    if XGBClassifier is not None:
        out.append(Candidate("XGBoost", "tree",
                             XGBClassifier(n_estimators=60 if q else 300, subsample=0.8, random_state=seed, n_jobs=1),
                             {m + "learning_rate": [0.1] if q else [0.05, 0.1], m + "max_depth": [3] if q else [3, 5]}, 6))
    out.append(Candidate("Soft-voting ensemble (LR + RF + HGB)", "none", None, {}, 7,
                         members=("Logistic Regression", "Random Forest", "Hist Gradient Boosting")))
    return out


def build_pipeline(c: Candidate, features: list[str]) -> Pipeline:
    if c.family == "none":
        return Pipeline([("model", clone(c.estimator))])
    return pp.make_pipeline(c.family, clone(c.estimator), features)


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def metrics(y_true, y_pred) -> dict:
    kw = {"zero_division": 0}  # averages over classes present in y_true or y_pred
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "macro_precision": precision_score(y_true, y_pred, average="macro", **kw),
        "macro_recall": recall_score(y_true, y_pred, average="macro", **kw),
        "macro_f1": f1_score(y_true, y_pred, average="macro", **kw),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", **kw),
    }


def per_class_f1(y_true, y_pred) -> dict:
    scores = f1_score(y_true, y_pred, labels=pp.CLASS_ORDER, average=None, zero_division=0)
    return {BIGO_LABELS[c][1]: round(float(s), 3) for c, s in zip(pp.CLASS_ORDER, scores)}


def group_bootstrap_ci(y_true, y_pred, groups, n: int = N_BOOTSTRAP, seed: int = config.RANDOM_STATE) -> tuple[float, float]:
    """95% interval for macro F1, resampling whole groups (problems) with replacement."""
    rng = np.random.default_rng(seed)
    y_true, y_pred, groups = np.asarray(y_true), np.asarray(y_pred), np.asarray(groups)
    uniq = np.unique(groups)
    members = {g: np.where(groups == g)[0] for g in uniq}
    stats = []
    for _ in range(n):
        idx = np.concatenate([members[g] for g in rng.choice(uniq, size=len(uniq), replace=True)])
        stats.append(f1_score(y_true[idx], y_pred[idx], average="macro", zero_division=0))
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return float(lo), float(hi)


# --------------------------------------------------------------------------- #
# Cross-validation helpers
# --------------------------------------------------------------------------- #
def inner_cv(y: pd.Series, groups: pd.Series, seed: int, folds: int = 5) -> list:
    splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    return list(splitter.split(np.zeros(len(y)), y, groups))


def repeated_cv(y: pd.Series, groups: pd.Series, seed: int, repeats: int = RCV_REPEATS, folds: int = 5) -> list:
    out = []
    for r in range(repeats):
        out += inner_cv(y, groups, seed + 101 * (r + 1), folds)
    return out


def tune(c: Candidate, X: pd.DataFrame, y: pd.Series, cv: list, features: list[str], n_jobs: int):
    """Grouped grid search on the given (training) rows -> unfitted best pipeline, params, mean, std."""
    pipe = build_pipeline(c, features)
    if not c.grid:
        return pipe, {}, np.nan, np.nan
    search = GridSearchCV(pipe, c.grid, scoring="f1_macro", cv=cv, n_jobs=n_jobs, refit=False, error_score="raise")
    search.fit(X, y)
    i = search.best_index_
    params = search.cv_results_["params"][i]
    return pipe.set_params(**params), params, float(search.cv_results_["mean_test_score"][i]), \
        float(search.cv_results_["std_test_score"][i])


def cv_score(pipe, X: pd.DataFrame, y: pd.Series, cv: list, n_jobs: int) -> np.ndarray:
    from sklearn.model_selection import cross_val_score

    return cross_val_score(pipe, X, y, cv=cv, scoring="f1_macro", n_jobs=n_jobs, error_score="raise")


def select(results: pd.DataFrame, tolerance: float = TIE_TOLERANCE, score: str = "rcv_macro_f1") -> int:
    """Row index with the highest ``score``; within ``tolerance`` of it, the simplest."""
    top = results[score].max()
    near = results[results[score] >= top - tolerance]
    return near.sort_values(["simplicity", score], ascending=[True, False]).index[0]


# --------------------------------------------------------------------------- #
# Experiment
# --------------------------------------------------------------------------- #
def run(quick: bool = False, out_dir: Path = REPORT_DIR, models_dir: Path = config.MODELS_DIR,
        df: Optional[pd.DataFrame] = None, split: Optional[pp.SplitResult] = None,
        seed: int = config.RANDOM_STATE, save_model: bool = True, verbose: bool = True,
        feature_sets: Optional[tuple] = None, n_jobs: int = -1) -> dict:
    t0 = time.time()
    if df is None:
        df = pd.read_csv(config.BIGO_DATASET_CSV, keep_default_na=False)
    if split is None:
        split = pp.load_split(df) if pp.SPLIT_FILE.exists() else pp.make_split(df, "problem", seed)
    available = [fs for fs in (feature_sets or FEATURE_SETS) if set(pp.feature_columns(fs)) <= set(df.columns)]
    y = df[pp.TARGET]
    tr, va, te = split.indices("train"), split.indices("validation"), split.indices("test")
    dev = tr.append(va)
    tune_cv = inner_cv(y.loc[tr], split.groups.loc[tr], seed)
    dev_cv = repeated_cv(y.loc[dev], split.groups.loc[dev], seed)

    rows, fitted, preds, tuned = [], {}, {}, {}
    for c in candidates(quick, seed):
        for fs in [f for f in c.feature_sets if f in available]:
            t = time.time()
            features = pp.feature_columns(fs)
            X = df[features]
            if c.members:  # ensemble of already-tuned members, same feature set
                if not all((mname, fs) in tuned for mname in c.members):
                    continue
                vote = VotingClassifier([(f"m{i}", clone(tuned[(mname, fs)])) for i, mname in enumerate(c.members)], voting="soft")
                pipe, params, tmean, tstd = Pipeline([("model", vote)]), {"members": list(c.members)}, np.nan, np.nan
            else:
                pipe, params, tmean, tstd = tune(c, X.loc[tr], y.loc[tr], tune_cv, features, n_jobs)
                tuned[(c.name, fs)] = pipe
            scores = cv_score(pipe, X.loc[dev], y.loc[dev], dev_cv, n_jobs)
            final = clone(pipe).fit(X.loc[dev], y.loc[dev])
            test_pred = final.predict(X.loc[te])
            s = metrics(y.loc[te], test_pred)
            lo, hi = group_bootstrap_ci(y.loc[te], test_pred, split.groups.loc[te], seed=seed)
            key = f"{c.name} [{fs}]"
            rows.append({
                "model": c.name, "features": fs, "key": key, "simplicity": c.simplicity,
                "tune_cv_macro_f1": tmean, "tune_cv_std": tstd,
                "rcv_macro_f1": float(scores.mean()), "rcv_std": float(scores.std()),
                **{f"test_{k}": v for k, v in s.items()},
                "test_macro_f1_ci_low": lo, "test_macro_f1_ci_high": hi,
                "params": json.dumps({k.replace("model__", ""): (round(v, 4) if isinstance(v, float) else v)
                                      for k, v in params.items()}),
                "seconds": round(time.time() - t, 1),
            })
            fitted[key], preds[key] = final, test_pred
            if verbose:
                print(f"{key:48s} RCV {scores.mean():.3f}±{scores.std():.3f}  test {s['macro_f1']:.3f}  ({time.time() - t:.0f}s)", flush=True)

    results = pd.DataFrame(rows)
    best_i = select(results)
    best = results.loc[best_i]
    best_c = next(c for c in candidates(quick, seed) if c.name == best["model"])
    checks = _extra_checks(df, split, best_c, fitted[best["key"]], best["features"], seed)

    out_dir.mkdir(parents=True, exist_ok=True)
    results.round(4).to_csv(out_dir / "model_comparison.csv", index=False)
    te_y = y.loc[te]
    shown = _figure_selection(results, best["key"])
    _save_confusion_matrices(te_y, {k: preds[k] for k in shown}, best["key"], out_dir)
    per_class = {k: per_class_f1(te_y, preds[k]) for k in preds}

    features = pp.feature_columns(best["features"])
    metadata = {
        "selected_model": best["model"],
        "feature_set": best["features"],
        "selection_rule": (f"highest mean macro F1 in {RCV_REPEATS}x5 repeated grouped CV on train+validation; "
                           f"within {TIE_TOLERANCE}, the simplest model"),
        "hyperparameters": json.loads(best["params"]),
        "classes": pp.CLASS_ORDER,
        "class_bigo": CLASS_NAMES,
        "features": features,
        "trained_on": "train + validation partitions (grouped by problem)",
        "n_train_rows": int(len(dev)),
        "n_test_rows": int(len(te)),
        "rcv_macro_f1": round(float(best["rcv_macro_f1"]), 4),
        "rcv_std": round(float(best["rcv_std"]), 4),
        "test_metrics": {k.replace("test_", ""): round(float(v), 4) for k, v in best.items() if str(k).startswith("test_")},
        "per_class_test_f1": per_class[best["key"]],
        "extra_checks": checks,
        "dataset_sha256": _sha(config.BIGO_DATASET_CSV),
        "split_file_sha256": _sha(pp.SPLIT_FILE),
        "versions": {"python": platform.python_version(), "scikit-learn": sklearn.__version__,
                     "pandas": pd.__version__, "numpy": np.__version__},
        "quick_mode": quick,
        "scope": ("Agreement with CodeComplex expert labels on unseen Codeforces problems (whole programs). "
                  "Not validated on everyday code, on single functions, or for space complexity."),
    }
    if save_model:
        models_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(fitted[best["key"]], models_dir / config.MODEL_PATH.name)
        metadata = model_integrity.attach(metadata, models_dir, [config.MODEL_PATH.name])
        (models_dir / config.MODEL_METADATA_PATH.name).write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (out_dir / "model_report.md").write_text(render_report(results, best_i, metadata, per_class, split), encoding="utf-8")
    if verbose:
        print(f"Selected: {best['key']}  ({time.time() - t0:.0f}s total)")
    return {"results": results, "selected": best["model"], "selected_key": best["key"],
            "feature_set": best["features"], "metadata": metadata, "per_class": per_class}


def _figure_selection(results: pd.DataFrame, best_key: str) -> list[str]:
    """Confusion-matrix panels: baselines, then each model's better feature set (by RCV)."""
    keys = []
    for name, grp in results.groupby("model", sort=False):
        keys.append(grp.sort_values("rcv_macro_f1", ascending=False)["key"].iloc[0])
    if best_key not in keys:
        keys.append(best_key)
    return keys


def _extra_checks(df, split, best_c, best_fitted, fs, seed) -> dict:
    y = df[pp.TARGET]
    te = split.indices("test")
    dev = split.indices("train").append(split.indices("validation"))
    out = {}
    features = pp.feature_columns(fs)
    rnd = pp.make_split(df, "random", seed)
    rdev = rnd.indices("train").append(rnd.indices("validation"))
    rp = clone(best_fitted).fit(df.loc[rdev, features], y.loc[rdev]).predict(df.loc[rnd.indices("test"), features])
    out["random_split_test_macro_f1"] = round(float(f1_score(y.loc[rnd.indices("test")], rp, average="macro")), 4)
    small_fs = pp.NO_SIZE_VARIANT.get(fs)
    if best_c.family != "none" and small_fs:
        small = pp.feature_columns(small_fs)
        pipe = build_pipeline(best_c, small)
        pipe.set_params(**{k: v for k, v in best_fitted.get_params().items() if k.startswith("model__")})
        p2 = pipe.fit(df.loc[dev, small], y.loc[dev]).predict(df.loc[te, small])
        out["no_size_test_macro_f1"] = round(float(f1_score(y.loc[te], p2, average="macro")), 4)
        out["no_size_features_removed"] = list(pp.SIZE_FEATURES)
    return out


def _sha(path: Path) -> Optional[str]:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None


# --------------------------------------------------------------------------- #
# Confusion matrices
# --------------------------------------------------------------------------- #
def _plot_confusion(ax, cm: np.ndarray, title: str):
    from matplotlib.colors import LinearSegmentedColormap

    ramp = LinearSegmentedColormap.from_list("blue", ["#fcfcfb", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
    row = cm.sum(axis=1, keepdims=True)
    share = np.divide(cm, row, out=np.zeros_like(cm, dtype=float), where=row > 0)
    ax.imshow(share, cmap=ramp, vmin=0, vmax=1)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if cm[i, j]:
                ax.text(j, i, f"{cm[i, j]}", ha="center", va="center", fontsize=7.5,
                        color="#ffffff" if share[i, j] > 0.55 else "#0b0b0b")
    ax.set_xticks(range(len(CLASS_NAMES)), CLASS_NAMES, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(CLASS_NAMES)), CLASS_NAMES, fontsize=8)
    ax.set_title(title, fontsize=9.5, loc="left")
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)


def _save_confusion_matrices(y_test, preds: dict, best: str, out_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "text.color": "#0b0b0b",
                         "axes.labelcolor": "#52514e", "xtick.color": "#52514e", "ytick.color": "#52514e"})
    names = list(preds)
    cols = 4
    rows = int(np.ceil(len(names) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.4 * cols, 4.2 * rows), squeeze=False)
    for ax, name in zip(np.ravel(axes), names):
        cm = confusion_matrix(y_test, preds[name], labels=pp.CLASS_ORDER)
        f1 = f1_score(y_test, preds[name], average="macro")
        _plot_confusion(ax, cm, f"{name}{'  (selected)' if name == best else ''}\ntest macro F1 {f1:.3f}")
        ax.set_xlabel("Predicted", fontsize=8)
        ax.set_ylabel("Expert label", fontsize=8)
    for ax in np.ravel(axes)[len(names):]:
        ax.set_visible(False)
    fig.suptitle("Confusion matrices on the grouped test set (counts; colour = share of the true class)",
                 x=0.01, ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_dir / "confusion_matrices.png", dpi=110)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    cm = confusion_matrix(y_test, preds[best], labels=pp.CLASS_ORDER)
    _plot_confusion(ax, cm, f"{best}: test confusion matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Expert label")
    fig.tight_layout()
    fig.savefig(out_dir / "confusion_matrix_selected.png", dpi=130)
    plt.close(fig)
    pd.DataFrame(cm, index=CLASS_NAMES, columns=CLASS_NAMES).to_csv(out_dir / "confusion_matrix_selected.csv")


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #
def render_report(results: pd.DataFrame, best_i, meta: dict, per_class: dict, split) -> str:
    r = results.sort_values("rcv_macro_f1", ascending=False)
    b = results.loc[best_i]
    checks = meta["extra_checks"]

    def get(model, fs="all", col="test_macro_f1"):
        hit = results[(results["model"] == model) & (results["features"] == fs)]
        return float(hit[col].iloc[0]) if len(hit) else float("nan")

    others = r[r.index != best_i]
    runner = others.iloc[0] if len(others) else None
    lines = [
        "# Model comparison",
        "",
        "Generated by `python -m src.train_model`" + (" (quick mode: small grids)" if meta["quick_mode"] else "") + ".",
        f"Grouped-by-problem partitions: {len(split.indices('train')):,} train, {len(split.indices('validation')):,} validation, "
        f"{len(split.indices('test')):,} test rows; no problem appears in two partitions.",
        "",
        f"**Selection score (RCV):** mean macro F1 over {RCV_REPEATS}×5 repeated grouped cross-validation on train + validation.",
        "**Test:** one evaluation after refitting on train + validation; reported for transparency, not used for selection.",
        "CI = 95% bootstrap interval over test problems. Feature sets: `all` = 39 structural features; `patterns` = those",
        "plus 18 algorithm-pattern features.",
        "",
        "| Model | Features | RCV macro F1 | Test macro F1 [95% CI] | Test acc. | Test bal. acc. | Test macro P | Test macro R | Test weighted F1 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for i, row in r.iterrows():
        mark = " **(selected)**" if i == best_i else ""
        lines.append(
            f"| {row.model}{mark} | {row.features} | {row.rcv_macro_f1:.3f} ± {row.rcv_std:.3f} | "
            f"{row.test_macro_f1:.3f} [{row.test_macro_f1_ci_low:.3f}, {row.test_macro_f1_ci_high:.3f}] | "
            f"{row.test_accuracy:.3f} | {row.test_balanced_accuracy:.3f} | {row.test_macro_precision:.3f} | "
            f"{row.test_macro_recall:.3f} | {row.test_weighted_f1:.3f} |"
        )
    lines += ["", "Tuned hyperparameters (grouped 5-fold CV on train):", ""]
    lines += [f"- {row.model} [{row.features}]: `{row.params}`" for _, row in r.iterrows() if row.params not in ("{}",)]

    gain_rows = []
    for model in results["model"].unique():
        a, p_ = get(model, "all", "rcv_macro_f1"), get(model, "patterns", "rcv_macro_f1")
        if not np.isnan(a) and not np.isnan(p_):
            gain_rows.append((model, a, p_))
    lines += ["", "## Effect of the algorithm-pattern features (RCV macro F1)", "",
              "| Model | Structural only | + pattern features | Change |", "| --- | ---: | ---: | ---: |"]
    lines += [f"| {m} | {a:.3f} | {p_:.3f} | {p_ - a:+.3f} |" for m, a, p_ in gain_rows]

    lines += [
        "",
        "## Why this model was selected",
        "",
        f"**{b.model}** with the **{b.features}** feature set had the highest repeated-CV macro F1 "
        f"({b.rcv_macro_f1:.3f} ± {b.rcv_std:.3f})"
        + (f"; next was {runner.model} [{runner.features}] at {runner.rcv_macro_f1:.3f}" if runner is not None else "")
        + f". The rule (highest RCV macro F1; within {TIE_TOLERANCE}, the simpler model) was fixed before any test score was computed.",
        f"On the untouched test problems it reached macro F1 **{b.test_macro_f1:.3f}** "
        f"(95% CI {b.test_macro_f1_ci_low:.3f}–{b.test_macro_f1_ci_high:.3f}), accuracy {b.test_accuracy:.3f}, against "
        f"{get('Dummy (most frequent)'):.3f} for the majority baseline and {get('Loop-nesting rule'):.3f} for the one-feature rule.",
        f"The first Phase 10 round ({FIRST_ROUND['model']}, structural features, selected on a single validation split) scored "
        f"{FIRST_ROUND['test_macro_f1']:.3f} on the same test problems ({b.test_macro_f1 - FIRST_ROUND['test_macro_f1']:+.3f}).",
        "Top models' confidence intervals overlap, so the exact ranking among them is not decisive.",
        "",
        "![Selected model](confusion_matrix_selected.png)",
        "",
        "Per-class test F1 of the selected model:",
        "",
        "| " + " | ".join(per_class[b.key]) + " |",
        "| " + " | ".join("---:" for _ in per_class[b.key]) + " |",
        "| " + " | ".join(f"{v:.3f}" for v in per_class[b.key].values()) + " |",
        "",
        "![All confusion matrices](confusion_matrices.png)",
        "",
        "## Extra checks",
        "",
        f"- **Random split:** the selected model scores macro F1 {checks['random_split_test_macro_f1']:.3f} on a conventional",
        f"  stratified random split vs {b.test_macro_f1:.3f} on unseen problems; the gap is leakage through shared problems.",
    ]
    if "no_size_test_macro_f1" in checks:
        d = checks["no_size_test_macro_f1"] - b.test_macro_f1
        lines.append(f"- **Without the {len(checks['no_size_features_removed'])} size features:** test macro F1 "
                     f"{checks['no_size_test_macro_f1']:.3f} ({d:+.3f}); "
                     + ("program length is not the main signal." if abs(d) < 0.03 else "size features carry a noticeable part of the signal."))
    lines += [
        "- **Caveat on the pattern features:** they were designed after inspecting the first round's errors, which were",
        "  measured on this test set. Selection used only cross-validation, but the test score of the new features may be",
        "  slightly optimistic; the repeated-CV score is the fairer estimate.",
        "",
        "## What this experiment establishes, and what it does not",
        "",
        "- The labels are **expert annotations** (CodeComplex), not thresholds on the inputs, so the score is not circular.",
        "  The structural dataset, whose labels are thresholds, was deliberately not used for training.",
        "- It measures **agreement with those annotators on Codeforces problems the model has never seen**, from static",
        "  features of whole programs.",
        "- It does **not** show accuracy on everyday code, on single functions (labels are per program), or for space",
        "  complexity (no labels exist). Features describe syntax and patterns; they cannot see input sizes or library costs.",
        "- Labels contain noise (identical code with different labels was found in Phase 7), so 100% is not attainable.",
        "- Present the model as a **probabilistic second opinion** next to the rule-based analyser, not as proof.",
        "",
        f"Software: Python {meta['versions']['python']}, scikit-learn {meta['versions']['scikit-learn']}. "
        + ("XGBoost was installed and compared." if "XGBoost" in set(results["model"]) else "XGBoost was not installed, so it was not compared."),
        "",
    ]
    return "\n".join(lines)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m src.train_model")
    parser.add_argument("--quick", action="store_true", help="small grids for a fast check")
    parser.add_argument("--jobs", type=int, default=-1, help="parallel jobs (default: all CPU cores)")
    args = parser.parse_args(argv)
    if not config.BIGO_DATASET_CSV.exists():
        print("Processed data not found. Run: python -m src.dataset_loader download, build, then python -m src.preprocessing")
        return 1
    out = run(quick=args.quick, n_jobs=args.jobs)
    cols = ["model", "features", "rcv_macro_f1", "test_macro_f1", "test_accuracy"]
    print(out["results"].sort_values("rcv_macro_f1", ascending=False)[cols].round(3).to_string(index=False))
    print(f"\nReport: {REPORT_DIR / 'model_report.md'}\nModel:  {config.MODEL_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
