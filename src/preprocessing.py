"""ML preprocessing for the Big-O classifier (Phase 9).

Covers which columns are features, how the data is split, and how
features are transformed. No final model is trained here.

Scope
-----
Only the Big-O dataset (CodeComplex) is prepared for supervised learning.
Phase 8 showed that the structural dataset's labels are thresholds on an
input feature, so it is not a valid training target (it remains a
rule-based score).

Key ideas
---------
**Data leakage** is information in the training process that would not be
available when the model is used, or that lets test data influence training.
It makes scores look better than real performance. Three forms matter here:

* *Identifier and metadata columns.* ``sample_id``, ``raw_index``,
  ``code_sha256``, ``problem``, ``tags`` and the ``label_*`` columns either
  identify a sample or encode the answer. Tags are Codeforces metadata
  strongly tied to the label, and user code has no tags. All are excluded.
* *Shared problems across partitions.* 95.6% of samples carry their
  problem's majority label. If solutions of one problem are in both training
  and test data, a model can recognise the problem instead of judging the
  complexity. Splits are therefore grouped.
* *Fitting transformations on all data.* Scaling statistics (means,
  standard deviations) computed on the full dataset leak information about
  the test set. Every transformer lives inside a scikit-learn ``Pipeline``,
  so ``fit`` only ever sees training rows.

**Stratification** keeps the class proportions roughly equal in every
partition, so a rare class is not missing from validation or test and
scores are comparable. With groups, ``StratifiedGroupKFold`` balances
classes as well as it can while never splitting a group.

**Grouped evaluation** keeps every sample of a group on one side of a split:

* ``problem``: all solutions of one Codeforces problem (default). Every
  near-duplicate cluster lies inside one problem, so near-duplicates cannot
  cross partitions either; the groups are formed as the union of problem
  and near-duplicate cluster to guarantee this even if that changes.
* ``contest``: all problems of one Codeforces round, a stronger
  family-level test of generalisation to unseen sets of problems.
* ``none``: a conventional stratified random split, kept only to measure
  how much it overestimates performance.

The data has a single source (Codeforces), so grouping by repository or
source is not possible; the tags could define algorithm families, but they
overlap (a problem has several tags), so contests are used as the family-level group.

**Feature scaling** matters for models that compare magnitudes:
distance-based models (k-NN, SVM with RBF kernel) and gradient-trained
linear models (logistic regression). Without it, ``identifier_count``
(hundreds) would swamp ``maximum_loop_nesting_depth`` (0-6). The counts
are also heavily right-skewed, so they are log-transformed (``log1p``)
before standardisation. **Tree models** (decision trees, random forests,
gradient boosting) split one feature at a time on thresholds. A monotonic
transform such as scaling or log does not change which split is best, so
trees need neither; they only need numeric input without missing values.

Usage
-----
    python -m src.preprocessing      # make splits, save config, write the split report
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from src import config
from src.dataset_loader import BIGO_LABELS
from src.feature_extractor import FILE_FEATURE_NAMES

TARGET = "label"
CLASS_ORDER = [k for k, _ in sorted(BIGO_LABELS.items(), key=lambda kv: kv[1][2])]

# Columns that must never be features, with the reason
EXCLUDED_COLUMNS: dict[str, str] = {
    "sample_id": "identifier",
    "source_dataset": "constant metadata",
    "source_commit": "constant metadata",
    "raw_index": "identifier (row position in the raw file)",
    "problem": "identifier; used only to form groups",
    "tags": "Codeforces metadata strongly tied to the label and unavailable for user code",
    "label": "target",
    "label_name": "target (renamed)",
    "label_bigo": "target (notation)",
    "label_rank": "target (ordinal)",
    "label_method": "constant metadata",
    "near_duplicate_cluster": "dataset bookkeeping; used only to form groups",
    "code_sha256": "identifier",
}

# Size and style features: used for the length-shortcut ablation in Phase 10
SIZE_FEATURES = (
    "total_lines", "code_lines", "blank_lines", "comment_lines", "docstring_lines",
    "comment_ratio", "function_count", "class_count", "average_function_length",
    "maximum_function_length", "import_count", "identifier_count", "unique_identifier_count",
)

FEATURE_SETS: dict[str, tuple[str, ...]] = {
    "all": tuple(FILE_FEATURE_NAMES),
    "no_size": tuple(f for f in FILE_FEATURE_NAMES if f not in SIZE_FEATURES),
}

SPLIT_FILE = config.PROCESSED_DATA_DIR / "bigo_splits.csv"
PREPROCESSING_CONFIG = config.MODELS_DIR / "preprocessing_config.json"
REPORT_DIR = config.PROJECT_ROOT / "reports" / "preprocessing"

TEST_FOLDS = 5  # 1 of 5 group folds -> about 20% test
VAL_FOLDS = 5  # 1 of 5 of the remainder -> about 16% validation, 64% train
CV_FOLDS = 5  # cross-validation folds on the development set (train + validation)


# --------------------------------------------------------------------------- #
# Features and target
# --------------------------------------------------------------------------- #
def feature_columns(feature_set: str = "all") -> list[str]:
    if feature_set not in FEATURE_SETS:
        raise ValueError(f"unknown feature set {feature_set!r}; choose from {sorted(FEATURE_SETS)}")
    return list(FEATURE_SETS[feature_set])


def split_features_target(df: pd.DataFrame, feature_set: str = "all") -> tuple[pd.DataFrame, pd.Series]:
    """X with only code-derived features (never metadata), y with the label."""
    cols = feature_columns(feature_set)
    leaked = sorted(set(cols) & set(EXCLUDED_COLUMNS))
    if leaked:
        raise ValueError(f"excluded columns requested as features: {leaked}")
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"dataset is missing feature columns: {missing}")
    return df[cols].copy(), df[TARGET].copy()


# --------------------------------------------------------------------------- #
# Groups
# --------------------------------------------------------------------------- #
def make_groups(df: pd.DataFrame, by: str = "problem") -> pd.Series:
    """Group id per row; near-duplicate clusters are always merged into one group.

    ``by``: "problem", "contest" (Codeforces round, from "0123_B" -> "0123"), or
    "none" (each row its own group, apart from near-duplicate clusters).
    """
    if by == "problem":
        base = df["problem"].astype(str)
    elif by == "contest":
        base = df["problem"].astype(str).str.split("_").str[0]
    elif by == "none":
        base = df["sample_id"].astype(str)
    else:
        raise ValueError("by must be 'problem', 'contest' or 'none'")

    # Union-find over base groups and near-duplicate clusters
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    clusters = df.get("near_duplicate_cluster", pd.Series(-1, index=df.index))
    for g, c in zip(base, clusters):
        find("g:" + g)
        if c is not None and int(c) >= 0:
            union("g:" + g, f"c:{int(c)}")
    return base.map(lambda g: find("g:" + g)).rename(f"group_{by}")


# --------------------------------------------------------------------------- #
# Splits
# --------------------------------------------------------------------------- #
@dataclass
class SplitResult:
    strategy: str
    assignment: pd.Series  # index -> "train" | "validation" | "test"
    cv_fold: pd.Series  # index -> 0..CV_FOLDS-1 for train+validation rows, -1 for test
    groups: pd.Series
    notes: list[str] = field(default_factory=list)

    def indices(self, part: str) -> pd.Index:
        return self.assignment.index[self.assignment == part]


BALANCE_SEARCH = 50  # seeded shuffles examined when choosing a grouped hold-out fold


def _fold_imbalance(y: pd.Series, held: np.ndarray, n_splits: int) -> float:
    """Largest class-share difference (held-out vs all) plus deviation from the target size."""
    overall = y.value_counts(normalize=True)
    part = y.iloc[held].value_counts(normalize=True).reindex(overall.index, fill_value=0)
    return float((part - overall).abs().max()) + abs(len(held) / len(y) - 1 / n_splits)


def _first_fold(y: pd.Series, groups: Optional[pd.Series], n_splits: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Hold out about 1/n_splits of the rows.

    Without groups: the first StratifiedKFold fold. With groups: whole groups
    can be large and single-labelled (one problem has up to 135 solutions), so a
    single shuffle can be badly unbalanced. The fold with the best class balance
    among BALANCE_SEARCH seeded StratifiedGroupKFold shuffles is used. Only the
    labels and groups are looked at (never features or model scores), and the
    choice is deterministic for a given seed.
    """
    if groups is None:
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        return next(splitter.split(np.zeros(len(y)), y))
    best, best_score = None, np.inf
    for t in range(BALANCE_SEARCH):
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed + 1000 * t)
        for rest, held in splitter.split(np.zeros(len(y)), y, groups):
            score = _fold_imbalance(y, held, n_splits)
            if score < best_score:
                best, best_score = (rest, held), score
    return best


def make_split(df: pd.DataFrame, strategy: str = "problem", seed: int = config.RANDOM_STATE) -> SplitResult:
    """Train / validation / test partitions plus CV folds on train+validation.

    strategy "problem" or "contest": stratified *and* grouped.
    strategy "random": conventional stratified split (for comparison only).
    """
    y = df[TARGET].reset_index(drop=True)
    if strategy == "random":
        groups = make_groups(df, "none").reset_index(drop=True)
        use_groups = None
        notes = ["Conventional stratified random split: solutions of one problem can land in different partitions."]
    else:
        groups = make_groups(df, strategy).reset_index(drop=True)
        use_groups = groups
        notes = [f"Grouped by {strategy}: no group appears in more than one partition."]

    dev, test = _first_fold(y, use_groups, TEST_FOLDS, seed)
    y_dev = y.iloc[dev].reset_index(drop=True)
    g_dev = groups.iloc[dev].reset_index(drop=True) if use_groups is not None else None
    tr, va = _first_fold(y_dev, g_dev, VAL_FOLDS, seed + 1)

    assignment = np.empty(len(df), dtype=object)
    assignment[test] = "test"
    assignment[dev[tr]] = "train"
    assignment[dev[va]] = "validation"

    folds = np.full(len(df), -1)
    splitter = (StratifiedKFold(CV_FOLDS, shuffle=True, random_state=seed) if use_groups is None
                else StratifiedGroupKFold(CV_FOLDS, shuffle=True, random_state=seed))
    split_args = (np.zeros(len(dev)), y_dev) if use_groups is None else (np.zeros(len(dev)), y_dev, g_dev)
    for k, (_, held) in enumerate(splitter.split(*split_args)):
        folds[dev[held]] = k

    index = df.index
    return SplitResult(
        strategy=strategy,
        assignment=pd.Series(assignment, index=index, name="split"),
        cv_fold=pd.Series(folds, index=index, name="cv_fold"),
        groups=pd.Series(groups.values, index=index, name="group"),
        notes=notes,
    )


def cv_splits(split: SplitResult) -> list[tuple[np.ndarray, np.ndarray]]:
    """(train_idx, val_idx) positional pairs over the development rows, for GridSearchCV(cv=...)."""
    dev = split.cv_fold[split.cv_fold >= 0]
    folds = dev.values
    return [(np.where(folds != k)[0], np.where(folds == k)[0]) for k in sorted(set(folds))]


# --------------------------------------------------------------------------- #
# Preprocessing pipelines
# --------------------------------------------------------------------------- #
class FiniteValues(BaseEstimator, TransformerMixin):
    """Replace +/-inf with NaN so the imputer can handle them; cast to float."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        arr = np.asarray(X, dtype=float).copy()
        arr[~np.isfinite(arr)] = np.nan
        return arr

    def get_feature_names_out(self, input_features=None):
        return np.asarray(input_features, dtype=object)


def _log1p_nonnegative(X):
    return np.log1p(np.clip(X, 0, None))


def build_preprocessor(model_family: str, features: Iterable[str]) -> ColumnTransformer:
    """Unfitted preprocessing for one model family.

    ``tree``     : inf -> NaN, median imputation. No scaling (trees are invariant
                   to monotonic transforms).
    ``linear`` / ``distance``: inf -> NaN, median imputation, log1p (counts are
                   right-skewed and non-negative), then standardisation.

    Columns are selected by name, so identifier or label columns that happen to
    be present in the input DataFrame are dropped, never used.
    """
    features = list(features)
    steps = [("finite", FiniteValues()), ("impute", SimpleImputer(strategy="median"))]
    if model_family in ("linear", "distance"):
        steps += [
            ("log1p", FunctionTransformer(_log1p_nonnegative, feature_names_out="one-to-one")),
            ("scale", StandardScaler()),
        ]
    elif model_family != "tree":
        raise ValueError("model_family must be 'tree', 'linear' or 'distance'")
    return ColumnTransformer(
        [("features", Pipeline(steps), features)],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def make_pipeline(model_family: str, estimator, features: Iterable[str]) -> Pipeline:
    """Preprocessing + estimator in one Pipeline, so fit() only sees training rows."""
    return Pipeline([("preprocess", build_preprocessor(model_family, features)), ("model", estimator)])


# --------------------------------------------------------------------------- #
# Diagnostics and comparison
# --------------------------------------------------------------------------- #
def split_summary(df: pd.DataFrame, split: SplitResult) -> dict:
    parts = ["train", "validation", "test"]
    overall = df[TARGET].value_counts(normalize=True)
    out = {"strategy": split.strategy, "partitions": {}}
    for p in parts:
        idx = split.indices(p)
        dist = df.loc[idx, TARGET].value_counts(normalize=True).reindex(CLASS_ORDER, fill_value=0)
        out["partitions"][p] = {
            "samples": int(len(idx)),
            "share_%": round(100 * len(idx) / len(df), 1),
            "problems": int(df.loc[idx, "problem"].nunique()),
            "class_%": (100 * dist).round(1).to_dict(),
            "max_class_deviation_pp": round(float(100 * (dist - overall.reindex(CLASS_ORDER)).abs().max()), 2),
        }
    train_probs = set(df.loc[split.indices("train"), "problem"])
    test_idx = split.indices("test")
    seen = df.loc[test_idx, "problem"].isin(train_probs)
    out["test_samples_whose_problem_is_in_train_%"] = round(100 * float(seen.mean()), 1)
    contests_train = set(df.loc[split.indices("train"), "problem"].str.split("_").str[0])
    out["test_samples_whose_contest_is_in_train_%"] = round(
        100 * float(df.loc[test_idx, "problem"].str.split("_").str[0].isin(contests_train).mean()), 1)
    clusters = df.loc[df["near_duplicate_cluster"] >= 0].groupby("near_duplicate_cluster").apply(
        lambda g: split.assignment.loc[g.index].nunique() > 1, include_groups=False)
    out["near_duplicate_clusters_crossing_partitions"] = int(clusters.sum())
    groups_by_part = split.groups.groupby(split.assignment).apply(set)
    shared = set()
    for a in parts:
        for b in parts:
            if a < b:
                shared |= groups_by_part.get(a, set()) & groups_by_part.get(b, set())
    out["groups_in_more_than_one_partition"] = len(shared) if split.strategy != "random" else None
    return out


def leakage_probe(df: pd.DataFrame, strategy: str, seed: int = config.RANDOM_STATE) -> dict:
    """Diagnostic only (not the project's model): 1-nearest-neighbour accuracy
    under 5-fold cross-validation with each split strategy.

    1-NN predicts the label of the most similar training program, so it profits
    directly from near-copies of the same problem. A large gap between the random
    and grouped scores measures how much a random split overestimates performance.
    The scaler is fitted inside each training fold.
    """
    from sklearn.model_selection import cross_val_score
    from sklearn.neighbors import KNeighborsClassifier

    X, y = split_features_target(df)
    pipe = make_pipeline("distance", KNeighborsClassifier(n_neighbors=1), X.columns)
    if strategy == "random":
        cv = StratifiedKFold(CV_FOLDS, shuffle=True, random_state=seed)
        scores = cross_val_score(pipe, X, y, cv=cv, scoring="accuracy")
    else:
        cv = StratifiedGroupKFold(CV_FOLDS, shuffle=True, random_state=seed)
        scores = cross_val_score(pipe, X, y, cv=cv, groups=make_groups(df, strategy), scoring="accuracy")
    majority = y.value_counts(normalize=True).max()
    return {
        "strategy": strategy,
        "accuracy_mean": round(float(scores.mean()), 4),
        "accuracy_std": round(float(scores.std()), 4),
        "fold_scores": [round(float(s), 4) for s in scores],
        "majority_class_rate": round(float(majority), 4),
    }


# --------------------------------------------------------------------------- #
# Saving
# --------------------------------------------------------------------------- #
def save_split(df: pd.DataFrame, split: SplitResult, path: Path = SPLIT_FILE) -> Path:
    out = pd.DataFrame({
        "sample_id": df["sample_id"].values,
        "problem": df["problem"].values,
        "group": split.groups.values,
        "split": split.assignment.values,
        "cv_fold": split.cv_fold.values,
    })
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False, lineterminator="\n")
    return path


def load_split(df: pd.DataFrame, path: Path = SPLIT_FILE) -> SplitResult:
    s = pd.read_csv(path, keep_default_na=False).set_index("sample_id")
    if set(s.index) != set(df["sample_id"]):
        raise ValueError("split file does not match the dataset; rerun python -m src.preprocessing")
    s = s.loc[df["sample_id"]]
    return SplitResult(
        strategy="problem",
        assignment=pd.Series(s["split"].values, index=df.index, name="split"),
        cv_fold=pd.Series(s["cv_fold"].values, index=df.index, name="cv_fold"),
        groups=pd.Series(s["group"].values, index=df.index, name="group"),
    )


def save_config(df: pd.DataFrame, split: SplitResult, comparison: dict, path: Path = PREPROCESSING_CONFIG) -> Path:
    cfg = {
        "dataset": config.BIGO_DATASET_CSV.name,
        "dataset_sha256": hashlib.sha256(config.BIGO_DATASET_CSV.read_bytes()).hexdigest()
        if config.BIGO_DATASET_CSV.exists() else None,
        "target": TARGET,
        "classes": CLASS_ORDER,
        "feature_sets": {k: list(v) for k, v in FEATURE_SETS.items()},
        "excluded_columns": EXCLUDED_COLUMNS,
        "split": {
            "strategy": split.strategy,
            "groups": "problem, merged with near-duplicate clusters",
            "test": f"1 of {TEST_FOLDS} StratifiedGroupKFold folds",
            "validation": f"1 of {VAL_FOLDS} folds of the remainder",
            "cv_folds_on_train_plus_validation": CV_FOLDS,
            "random_state": config.RANDOM_STATE,
            "file": SPLIT_FILE.name,
        },
        "pipelines": {
            "tree": ["replace inf with NaN", "median imputation"],
            "linear": ["replace inf with NaN", "median imputation", "log1p", "StandardScaler"],
            "distance": ["replace inf with NaN", "median imputation", "log1p", "StandardScaler"],
            "fit_on": "training rows only (inside a scikit-learn Pipeline)",
        },
        "split_comparison": comparison,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    return path


def render_report(summaries: list[dict], probes: list[dict]) -> str:
    by = {s["strategy"]: s for s in summaries}
    pr = {p["strategy"]: p for p in probes}
    names = {"random": "Stratified random", "problem": "Grouped by problem", "contest": "Grouped by contest"}
    order = [k for k in ("random", "problem", "contest") if k in by]
    lines = [
        "# Split comparison",
        "",
        "Generated by `python -m src.preprocessing`. The project uses the **grouped-by-problem** split; the others are",
        "shown for comparison.",
        "",
        "| | " + " | ".join(names[k] for k in order) + " |",
        "| --- | " + " | ".join("---:" for _ in order) + " |",
    ]

    def row(label, fn):
        lines.append(f"| {label} | " + " | ".join(fn(k) for k in order) + " |")

    for part in ("train", "validation", "test"):
        row(f"{part.capitalize()} samples (%)",
            lambda k, p=part: f"{by[k]['partitions'][p]['samples']:,} ({by[k]['partitions'][p]['share_%']}%)")
    row("Largest class-share deviation in test (pp)", lambda k: f"{by[k]['partitions']['test']['max_class_deviation_pp']}")
    row("Test samples whose problem is also in train", lambda k: f"{by[k]['test_samples_whose_problem_is_in_train_%']}%")
    row("Test samples whose contest is also in train", lambda k: f"{by[k]['test_samples_whose_contest_is_in_train_%']}%")
    row("Near-duplicate clusters crossing partitions", lambda k: str(by[k]["near_duplicate_clusters_crossing_partitions"]))
    if pr:
        row("Leakage probe: 1-NN accuracy, 5-fold CV",
            lambda k: f"{100 * pr[k]['accuracy_mean']:.1f}% ± {100 * pr[k]['accuracy_std']:.1f}" if k in pr else "")
    lines += [
        "",
        "The leakage probe is a diagnostic, not the project's model: 1-nearest-neighbour copies the label of the most",
        "similar training program, so it benefits directly from seeing other solutions of the same problem. "
        + (f"The majority-class rate is {100 * probes[0]['majority_class_rate']:.1f}%." if probes else ""),
        "",
    ]
    if "random" in pr and "problem" in pr:
        gap = 100 * (pr["random"]["accuracy_mean"] - pr["problem"]["accuracy_mean"])
        lines += [
            f"The random split scores **{gap:.1f} percentage points higher** than the grouped split. That difference is",
            "not skill at judging complexity; it comes from test programs whose problem was already seen in training.",
            "Results in Phase 10 are therefore reported on grouped splits.",
            "",
        ]
    lines += ["## Class distribution per partition (grouped by problem)", "",
              "| Class | " + " | ".join(("train", "validation", "test")) + " |",
              "| --- | ---: | ---: | ---: |"]
    for c in CLASS_ORDER:
        lines.append(f"| {BIGO_LABELS[c][1]} | " + " | ".join(
            f"{by['problem']['partitions'][p]['class_%'][c]}%" for p in ("train", "validation", "test")) + " |")
    lines.append("")
    return "\n".join(lines)


def main(probe: bool = True) -> int:
    if not config.BIGO_DATASET_CSV.exists():
        print("Processed data not found. Run: python -m src.dataset_loader download  then  build")
        return 1
    df = pd.read_csv(config.BIGO_DATASET_CSV, keep_default_na=False)
    splits = {s: make_split(df, s) for s in ("random", "problem", "contest")}
    summaries = [split_summary(df, sp) for sp in splits.values()]
    probes = [leakage_probe(df, s) for s in splits] if probe else []
    comparison = {"summaries": summaries, "leakage_probe": probes}

    save_split(df, splits["problem"])
    save_config(df, splits["problem"], comparison)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report = render_report(summaries, probes)
    (REPORT_DIR / "split_report.md").write_text(report, encoding="utf-8")
    print(report)
    print(f"Saved {SPLIT_FILE}\nSaved {PREPROCESSING_CONFIG}\nSaved {REPORT_DIR / 'split_report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
