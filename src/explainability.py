"""Model explainability for the Random Forest time-complexity model (Phase 14).

What this module explains
-------------------------
The saved model (``models/complexity_model.joblib``) is a scikit-learn
Pipeline: median imputation, then a ``RandomForestClassifier`` (300 trees)
over the static features listed in ``models/model_metadata.json`` (structural, algorithm-pattern and,
since the final improvement phase, the static analyzer's own estimate). It
predicts one of seven time-complexity classes for a whole program. Everything
here explains **that model's behaviour**. It does not explain why code has a
given complexity, and it does not make the model more accurate (test macro F1
about 0.44).

Techniques and why they were chosen
-----------------------------------
1. **Native importance (mean decrease in impurity, MDI)** - read from the
   fitted forest, no data needed. Cheap and always available, but it is
   computed on the *training* data and is known to favour features with
   many distinct values; correlated features share (split) the credit.
2. **Permutation importance on the held-out test set** - the drop in macro F1
   when one feature's column is shuffled (10 repeats, fixed seed). It measures
   how much the model *relies* on a feature for unseen problems. Weakness:
   when features are strongly correlated (e.g. ``code_lines`` and
   ``operator_count``), shuffling one leaves the others to carry the same
   information, so each looks less important than the group really is.
3. **Grouped permutation importance** - the same, but all features of one
   documented group (``FEATURE_GROUPS``) are shuffled together with the same
   row order. This answers "how much does the model rely on loop features as
   a whole" and is less affected by correlation inside a group.
4. **Per-prediction explanations** (optional):
   * ``tree_path`` (default, no extra dependency): the decision-path
     decomposition of each tree (Saabas, "treeinterpreter"). For every split
     on the path from the root to the leaf, the change in the tree's class
     vote is credited to the feature used by that split; the forest averages
     over its trees. It is **exactly additive**:
     ``base value + sum of contributions = the model's vote share``.
     Limitation: the credit depends on the order of splits in each tree
     (features split near the root tend to receive more), and it lacks the
     consistency guarantees of SHAP values.
   * ``shap`` (optional): exact TreeSHAP (path-dependent) via the ``shap``
     package, only if it is installed. SHAP is suitable for this model
     (tree ensemble, about 60 features, one program at a time is milliseconds), but
     it pulls in ``numba``/``llvmlite`` (about 200 MB) and is therefore not a
     required dependency. Install with ``pip install shap`` to enable it.

How to read the results (and how not to)
----------------------------------------
* Importance is **association, not causation**. A feature being important
  means the model uses it to separate classes in CodeComplex; changing that
  feature in your code does not change the code's real complexity.
* **Global importance is not the reason for any individual prediction.** It
  is an average over the test set. Use a per-prediction explanation for one
  program, and remember that it explains the model's vote, not the truth.
* Contributions are in **vote-share points** (percentage points of the
  forest's vote). The vote shares are not calibrated probabilities.
* When the model is unsure (top vote share below 0.4), the explanation
  describes an uncertain vote and deserves even less weight.
* Per-function ML predictions are experimental (the model was trained on
  whole programs); their explanations inherit that limitation.

Files (written by ``python -m src.explainability``)
---------------------------------------------------
``models/explainability.json``                    global results for the app
``reports/explainability/explainability_report.md`` human-readable report
``reports/explainability/*.png``                  figures

The JSON records the SHA-256 of the model file it was computed for;
``load_global_explanation`` flags it as stale if the model has changed.

Public API (for the frontend)
-----------------------------
feature_catalogue() -> DataFrame              name, label, group, description
describe_feature(name) -> dict
compute_global_explanation(...) -> dict       (used by the CLI)
load_global_explanation(path) -> GlobalExplanationFile
global_importance_table(glob, kind, top) -> DataFrame
group_importance_table(glob) -> DataFrame
class_importance_table(glob, big_o, top) -> DataFrame
explain_features(values, model=None, method="tree_path", target=None, top=5, global_data=None) -> LocalExplanation
explain_source(source, ...) -> LocalExplanation
shap_available() -> bool
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from src import config
from src import preprocessing as pp
from src.dataset_loader import BIGO_LABELS
from src.feature_extractor import FILE_FEATURE_NAMES, METRIC_DEFINITIONS
from src.pattern_features import PATTERN_DEFINITIONS, PATTERN_FEATURE_NAMES
from src.analysis_features import ANALYSIS_FEATURE_NAMES, FEATURE_DESCRIPTIONS as ANALYSIS_DESCRIPTIONS
from src.analysis_features import STATIC_FEATURE_NAMES

GLOBAL_PATH = config.MODELS_DIR / "explainability.json"
REPORT_DIR = config.PROJECT_ROOT / "reports" / "explainability"
N_REPEATS = 10
LOW_CONFIDENCE = 0.4
BIGO_OF = {k: v[1] for k, v in BIGO_LABELS.items()}
LABEL_OF_BIGO = {v: k for k, v in BIGO_OF.items()}
METHODS = ("tree_path", "shap")

GLOBAL_CAVEATS = (
    "Feature importance shows which inputs this model relies on; it is association, not causation.",
    "Global importance is an average over the test set, not the reason for any single prediction.",
    "Correlated features share importance: a feature can look unimportant because another one carries the same information.",
    "Native (MDI) importance is computed on training data and favours features with many distinct values.",
)
LOCAL_CAVEATS = (
    "This explains how the model's trees voted for this input, not why the code has this complexity.",
    "Contributions are in vote-share points; the vote shares are not calibrated probabilities.",
    "Changing a feature in your code would not by itself change the code's real complexity.",
)

# --------------------------------------------------------------------------- #
# Feature catalogue
# --------------------------------------------------------------------------- #
FEATURE_GROUPS: dict[str, tuple[str, ...]] = {
    "Size": (
        "total_lines", "code_lines", "blank_lines", "comment_lines", "docstring_lines", "comment_ratio",
        "function_count", "class_count", "average_function_length", "maximum_function_length",
        "import_count", "identifier_count", "unique_identifier_count",
    ),
    "Branching": (
        "if_count", "elif_count", "else_count", "try_count", "except_count", "conditional_expression_count",
        "return_count", "cyclomatic_complexity", "average_function_cyclomatic_complexity",
        "maximum_function_cyclomatic_complexity", "comparison_operator_count", "logical_operator_count",
    ),
    "Nesting": ("maximum_nesting_depth", "average_nesting_depth"),
    "Loops": (
        "for_loop_count", "while_loop_count", "loop_count", "comprehension_count", "nested_loop_count",
        "maximum_loop_nesting_depth", "constant_bound_loop_count", "max_input_loop_depth",
        "dependent_inner_loop_count", "while_true_count",
    ),
    "Halving and binary search": ("halving_loop_count", "binary_search_mid_count", "bisect_call_count"),
    "Recursion": (
        "recursive_call_count", "recursive_function_count", "max_recursive_branching",
        "dividing_recursion_count", "memoized_function_count",
    ),
    "Exponential patterns": (
        "exponential_range_loop_count", "power_of_two_expr_count", "combinatoric_call_count",
        "bitwise_operator_count",
    ),
    "Library calls and data structures": (
        "sort_call_count", "heap_call_count", "hash_container_count", "math_call_count",
        "function_call_count", "collection_literal_count",
    ),
    "Operators": ("operator_count", "arithmetic_operator_count"),
    "Static analyzer estimate": (
        "static_time_known", "static_time_degree", "static_time_log_power", "static_time_exponential",
    ),
    "Data handling (extra AST counts)": (
        "assignment_count", "augmented_assignment_count", "subscript_count", "slice_count",
        "collection_mutation_call_count", "membership_test_count", "max_call_nesting_depth",
        "max_parameter_count", "recursion_limit_set",
    ),
}
GROUP_OF = {f: g for g, fs in FEATURE_GROUPS.items() for f in fs}

_LABEL_OVERRIDES = {
    "total_lines": "Total lines",
    "code_lines": "Code lines",
    "blank_lines": "Blank lines",
    "comment_lines": "Comment lines",
    "docstring_lines": "Docstring lines",
    "comment_ratio": "Comment ratio",
    "average_function_length": "Average function length",
    "maximum_function_length": "Longest function",
    "cyclomatic_complexity": "Cyclomatic complexity (whole file)",
    "average_function_cyclomatic_complexity": "Average function cyclomatic complexity",
    "maximum_function_cyclomatic_complexity": "Highest function cyclomatic complexity",
    "maximum_nesting_depth": "Deepest nesting",
    "average_nesting_depth": "Average nesting depth",
    "maximum_loop_nesting_depth": "Deepest loop nesting",
    "static_time_known": "Static estimate available",
    "static_time_degree": "Static estimate: polynomial degree",
    "static_time_log_power": "Static estimate: power of log n",
    "static_time_exponential": "Static estimate: exponential",
    "max_input_loop_depth": "Deepest input-dependent loop nesting",
    "max_recursive_branching": "Most self-calls in one function",
    "loop_count": "Number of loops",
    "for_loop_count": "Number of for loops",
    "while_loop_count": "Number of while loops",
    "while_true_count": "Number of while-True loops",
    "if_count": "Number of if statements",
    "elif_count": "Number of elif branches",
    "else_count": "Number of else branches",
    "try_count": "Number of try blocks",
    "except_count": "Number of except handlers",
    "hash_container_count": "Sets and dicts created",
    "binary_search_mid_count": "Binary-search midpoints",
    "power_of_two_expr_count": "Power-of-two expressions",
    "dependent_inner_loop_count": "Inner loops bounded by an outer loop variable",
    "constant_bound_loop_count": "Constant-size loops",
    "exponential_range_loop_count": "Loops over an exponential range",
    "dividing_recursion_count": "Recursive calls that divide the input",
}


def _label(name: str) -> str:
    if name in _LABEL_OVERRIDES:
        return _LABEL_OVERRIDES[name]
    if name.endswith("_count"):
        words = name[: -len("_count")].replace("_", " ")
        return "Number of " + words + ("es" if words.endswith("s") else "s")
    return name.replace("_", " ").capitalize()


def _description(name: str) -> str:
    if name in PATTERN_DEFINITIONS:
        return PATTERN_DEFINITIONS[name] + " (whole program, static count)"
    if name in ANALYSIS_DESCRIPTIONS:
        return ANALYSIS_DESCRIPTIONS[name]
    return METRIC_DEFINITIONS[name]


def _kind(name: str) -> str:
    if name in PATTERN_DEFINITIONS:
        return "algorithm pattern"
    if name in STATIC_FEATURE_NAMES:
        return "static analyzer estimate"
    return "structural"


# Every feature the dataset and the predictor compute; the model uses the subset in its metadata
ALL_FEATURES: tuple[str, ...] = tuple(FILE_FEATURE_NAMES) + tuple(PATTERN_FEATURE_NAMES) + tuple(ANALYSIS_FEATURE_NAMES)


def feature_catalogue(features: Optional[Iterable[str]] = None) -> pd.DataFrame:
    """One row per model feature: name, readable label, group, kind and definition."""
    rows = []
    for name in features or ALL_FEATURES:
        rows.append({
            "feature": name, "label": _label(name), "group": GROUP_OF.get(name, "Other"),
            "kind": _kind(name),
            "description": _description(name),
        })
    return pd.DataFrame(rows)


def describe_feature(name: str) -> dict:
    if name not in GROUP_OF:
        raise KeyError(f"unknown feature {name!r}")
    return feature_catalogue([name]).iloc[0].to_dict()


# --------------------------------------------------------------------------- #
# Global explanation
# --------------------------------------------------------------------------- #
def _sha256(path: Path) -> Optional[str]:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None


def _forest(pipeline):
    return pipeline.steps[-1][1] if hasattr(pipeline, "steps") else pipeline


def _transform(pipeline, X: pd.DataFrame) -> np.ndarray:
    pre = pipeline[:-1] if hasattr(pipeline, "steps") and len(pipeline.steps) > 1 else None
    return np.asarray(pre.transform(X) if pre is not None else X, dtype=float)


def native_importance(pipeline, features: list) -> pd.DataFrame:
    """Mean decrease in impurity, normalised to sum to 1, with the spread across trees."""
    rf = _forest(pipeline)
    per_tree = np.array([t.feature_importances_ for t in rf.estimators_])
    return pd.DataFrame({
        "feature": features,
        "importance": rf.feature_importances_,
        "std_across_trees": per_tree.std(axis=0),
    }).sort_values("importance", ascending=False, ignore_index=True)


def _f1s(y_true, y_pred, classes) -> tuple[float, np.ndarray]:
    from sklearn.metrics import f1_score

    macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    per = f1_score(y_true, y_pred, labels=classes, average=None, zero_division=0)
    return float(macro), per


def permutation_importance(pipeline, X: pd.DataFrame, y, columns: dict, n_repeats: int = N_REPEATS,
                           seed: int = config.RANDOM_STATE) -> pd.DataFrame:
    """Drop in macro F1 (and per-class F1) when each column set is shuffled together.

    ``columns`` maps a name to a list of feature columns; a single feature is a
    one-element list. All columns of a set get the same row permutation, so
    their joint distribution is kept and only their link to the target breaks.
    """
    rf = _forest(pipeline)
    classes = list(rf.classes_)
    y = np.asarray(y)
    base_macro, base_per = _f1s(y, pipeline.predict(X), classes)
    rng = np.random.default_rng(seed)
    rows = []
    for name, cols in columns.items():
        drops, per_drops = [], []
        for _ in range(n_repeats):
            perm = rng.permutation(len(X))
            Xp = X.copy()
            Xp[cols] = X[cols].to_numpy()[perm]
            macro, per = _f1s(y, pipeline.predict(Xp), classes)
            drops.append(base_macro - macro)
            per_drops.append(base_per - per)
        drops, per_drops = np.array(drops), np.array(per_drops)
        rows.append({
            "name": name, "columns": list(cols), "importance": float(drops.mean()),
            "std": float(drops.std(ddof=1)) if n_repeats > 1 else 0.0,
            "positive_repeats": int((drops > 0).sum()),
            **{f"class_drop::{BIGO_OF[c]}": float(v) for c, v in zip(classes, per_drops.mean(axis=0))},
        })
    out = pd.DataFrame(rows).sort_values("importance", ascending=False, ignore_index=True)
    out.attrs["baseline_macro_f1"] = base_macro
    out.attrs["baseline_per_class_f1"] = {BIGO_OF[c]: float(v) for c, v in zip(classes, base_per)}
    return out


def training_quantiles(X: pd.DataFrame) -> dict:
    """Per feature: 101 quantiles (0..100%) of the training rows, for 'how unusual is this value'."""
    qs = np.linspace(0, 1, 101)
    return {c: [round(float(v), 4) for v in np.quantile(X[c].to_numpy(dtype=float), qs)] for c in X.columns}


def share_below(value: float, quantiles: list) -> Optional[float]:
    """Approximate share of training programs with a strictly lower value (0-100)."""
    if quantiles is None or value is None or not np.isfinite(value):
        return None
    q = np.asarray(quantiles, dtype=float)
    return float(np.searchsorted(q, value, side="left") / (len(q) - 1) * 100.0) if value <= q[-1] else 100.0


def correlated_features(X: pd.DataFrame, threshold: float = 0.8) -> dict:
    """Features whose Spearman correlation on the training rows is at least ``threshold`` in absolute value."""
    corr = X.rank().corr().fillna(0.0)
    out = {}
    for f in X.columns:
        others = corr[f].drop(f)
        strong = others[others.abs() >= threshold].sort_values(key=np.abs, ascending=False)
        out[f] = [{"feature": g, "spearman": round(float(r), 3)} for g, r in strong.items()]
    return out


def mean_abs_contributions(model, X: pd.DataFrame) -> pd.DataFrame:
    """Mean |tree-path contribution| toward each row's predicted class (local explanations aggregated)."""
    decomp = _path_tables(model.pipeline)
    Xt = _transform(model.pipeline, X)
    contrib, _, proba = _tree_path(decomp, Xt)
    pred = proba.argmax(axis=1)
    picked = contrib[np.arange(len(X)), :, pred]  # rows x features
    return pd.DataFrame({"feature": list(X.columns), "mean_abs_contribution": np.abs(picked).mean(axis=0)}) \
        .sort_values("mean_abs_contribution", ascending=False, ignore_index=True)


def compute_global_explanation(df: pd.DataFrame, split: "pp.SplitResult", model=None, n_repeats: int = N_REPEATS,
                               seed: int = config.RANDOM_STATE, verbose: bool = True) -> dict:
    """All global results for the saved model on the saved split (test rows are used once more, read-only)."""
    from src.predictor import get_model

    t0 = time.time()
    model = model or get_model()
    features = model.features
    dev = split.indices("train").append(split.indices("validation"))
    te = split.indices("test")
    X_dev, X_te, y_te = df.loc[dev, features], df.loc[te, features], df.loc[te, pp.TARGET]

    def log(msg):
        if verbose:
            print(f"[{time.time() - t0:5.0f}s] {msg}", flush=True)

    rf = _forest(model.pipeline)
    old_jobs = rf.n_jobs
    rf.set_params(n_jobs=-1)  # speed only; predictions are identical
    try:
        native = native_importance(model.pipeline, features)
        log("native importance done")
        perm = permutation_importance(model.pipeline, X_te, y_te, {f: [f] for f in features}, n_repeats, seed)
        log("permutation importance done")
        groups = permutation_importance(model.pipeline, X_te, y_te,
                                        {g: [f for f in fs if f in features] for g, fs in FEATURE_GROUPS.items()
                                         if any(f in features for f in fs)},
                                        n_repeats, seed)
        log("grouped permutation importance done")
        local_mean = mean_abs_contributions(model, X_te)
        log("aggregated tree-path contributions done")
        shap_check = _shap_agreement(model, X_te.iloc[:200]) if shap_available() else None
    finally:
        rf.set_params(n_jobs=old_jobs)

    rank_native = native.set_index("feature")["importance"].rank(ascending=False)
    rank_perm = perm.set_index("name")["importance"].rank(ascending=False)
    rank_local = local_mean.set_index("feature")["mean_abs_contribution"].rank(ascending=False)
    agreement = {
        "native_vs_permutation_spearman": round(float(rank_native.corr(rank_perm.reindex(rank_native.index), method="spearman")), 3),
        "native_vs_tree_path_spearman": round(float(rank_native.corr(rank_local.reindex(rank_native.index), method="spearman")), 3),
        "permutation_vs_tree_path_spearman": round(float(rank_perm.corr(rank_local.reindex(rank_perm.index), method="spearman")), 3),
    }
    top10 = lambda r: set(r.sort_values().index[:10])  # noqa: E731 - rank 1 = most important
    agreement.update({
        "native_vs_permutation_top10_overlap": len(top10(rank_native) & top10(rank_perm)),
        "native_vs_tree_path_top10_overlap": len(top10(rank_native) & top10(rank_local)),
        "permutation_vs_tree_path_top10_overlap": len(top10(rank_perm) & top10(rank_local)),
    })
    class_cols = [c for c in perm.columns if c.startswith("class_drop::")]

    meta_f1 = (model.metadata.get("test_metrics") or {}).get("macro_f1")
    baseline = perm.attrs["baseline_macro_f1"]
    return {
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model_sha256": _sha256(config.MODEL_PATH),
        "model_name": (model.metadata or {}).get("selected_model", "Random Forest"),
        "dataset_sha256": _sha256(config.BIGO_DATASET_CSV),
        "split_file_sha256": _sha256(pp.SPLIT_FILE),
        "versions": _versions(),
        "settings": {"n_repeats": n_repeats, "seed": seed, "scoring": "macro F1", "n_test_rows": int(len(te)),
                     "n_training_rows": int(len(dev)), "correlation_threshold": 0.8},
        "baseline": {"test_macro_f1": round(baseline, 4), "metadata_test_macro_f1": meta_f1,
                     "matches_metadata": meta_f1 is None or abs(baseline - meta_f1) < 1e-3,
                     "per_class_f1": {k: round(v, 4) for k, v in perm.attrs["baseline_per_class_f1"].items()}},
        "native": [{"feature": r.feature, "importance": round(r.importance, 5), "std_across_trees": round(r.std_across_trees, 5)}
                   for r in native.itertuples()],
        "permutation": [{"feature": r["name"], "importance": round(r["importance"], 5), "std": round(r["std"], 5),
                         "positive_repeats": r["positive_repeats"],
                         "per_class": {c.split("::")[1]: round(r[c], 5) for c in class_cols}}
                        for _, r in perm.iterrows()],
        "groups": [{"group": r["name"], "features": r["columns"], "importance": round(r["importance"], 5),
                    "std": round(r["std"], 5), "positive_repeats": r["positive_repeats"],
                    "per_class": {c.split("::")[1]: round(r[c], 5) for c in class_cols}}
                   for _, r in groups.iterrows()],
        "tree_path_mean_abs": [{"feature": r.feature, "mean_abs_contribution": round(r.mean_abs_contribution, 5)}
                               for r in local_mean.itertuples()],
        "rank_agreement": agreement,
        "shap_check": shap_check,
        "correlated": correlated_features(X_dev),
        "training_quantiles": training_quantiles(X_dev),
        "caveats": list(GLOBAL_CAVEATS),
    }


def _versions() -> dict:
    import sklearn

    return {"python": platform.python_version(), "scikit-learn": sklearn.__version__,
            "numpy": np.__version__, "pandas": pd.__version__}


# --------------------------------------------------------------------------- #
# Loading global results (frontend)
# --------------------------------------------------------------------------- #
@dataclass
class GlobalExplanationFile:
    available: bool
    data: Optional[dict] = None
    stale: bool = False
    message: str = ""


def load_global_explanation(path: Path = GLOBAL_PATH, model_path: Path = config.MODEL_PATH) -> GlobalExplanationFile:
    """Read the saved global explanation; never recomputes or invents values."""
    if not path.exists():
        return GlobalExplanationFile(False, message="Global explanation not computed yet. Run: python -m src.explainability")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return GlobalExplanationFile(False, message=f"Global explanation could not be read: {exc}")
    current = _sha256(model_path)
    stale = current is not None and data.get("model_sha256") != current
    msg = ("The model has changed since these results were computed; rerun python -m src.explainability."
           if stale else "")
    return GlobalExplanationFile(True, data, stale, msg)


def _with_catalogue(df: pd.DataFrame, key: str = "feature") -> pd.DataFrame:
    cat = feature_catalogue().set_index("feature")[["label", "group", "description"]]
    return df.join(cat, on=key)


def global_importance_table(glob: dict, kind: str = "permutation", top: Optional[int] = None) -> pd.DataFrame:
    """Importance per feature, sorted, with label, group and description.

    kind: "permutation" (drop in test macro F1), "native" (MDI on training data)
    or "tree_path" (mean |contribution| of the per-prediction explanations).
    """
    if kind == "permutation":
        df = pd.DataFrame(glob["permutation"])[["feature", "importance", "std", "positive_repeats"]]
    elif kind == "native":
        df = pd.DataFrame(glob["native"]).rename(columns={"std_across_trees": "std"})
    elif kind == "tree_path":
        df = pd.DataFrame(glob["tree_path_mean_abs"]).rename(columns={"mean_abs_contribution": "importance"})
    else:
        raise ValueError("kind must be 'permutation', 'native' or 'tree_path'")
    df = df.sort_values("importance", ascending=False, ignore_index=True)
    df.insert(0, "rank", np.arange(1, len(df) + 1))
    corr = glob.get("correlated", {})
    df["strongly_correlated_with"] = [", ".join(c["feature"] for c in corr.get(f, [])[:3]) for f in df["feature"]]
    df = _with_catalogue(df)
    return df.head(top) if top else df


def group_importance_table(glob: dict) -> pd.DataFrame:
    df = pd.DataFrame(glob["groups"])[["group", "importance", "std", "positive_repeats", "features"]]
    df["n_features"] = df["features"].map(len)
    return df.sort_values("importance", ascending=False, ignore_index=True)


def class_importance_table(glob: dict, big_o: str, top: int = 10) -> pd.DataFrame:
    """Features whose shuffling lowers the F1 of one class the most (permutation, test set)."""
    rows = [{"feature": r["feature"], "f1_drop": r["per_class"].get(big_o)} for r in glob["permutation"]]
    if all(r["f1_drop"] is None for r in rows):
        raise KeyError(f"unknown class {big_o!r}")
    df = pd.DataFrame(rows).sort_values("f1_drop", ascending=False, ignore_index=True).head(top)
    return _with_catalogue(df)


# --------------------------------------------------------------------------- #
# Per-prediction explanations
# --------------------------------------------------------------------------- #
@dataclass
class _PathTables:
    rf: object
    classes: list
    n_trees: int
    root_value: np.ndarray  # classes, mean normalised root value over trees
    delta: np.ndarray  # total_nodes x classes: value[node] - value[parent] (0 for roots)
    split_feature: np.ndarray  # total_nodes: feature of the parent split (-1 for roots)


_PATH_CACHE: dict = {}


def _path_tables(pipeline) -> _PathTables:
    rf = _forest(pipeline)
    cached = _PATH_CACHE.get(id(rf))
    if cached is not None and cached.rf is rf:
        return cached
    deltas, feats, roots = [], [], []
    for est in rf.estimators_:
        t = est.tree_
        v = t.value[:, 0, :].astype(float)
        v = v / np.clip(v.sum(axis=1, keepdims=True), 1e-12, None)  # class shares per node
        parent = np.full(t.node_count, -1)
        for side in (t.children_left, t.children_right):
            kids = np.where(side >= 0)[0]
            parent[side[kids]] = kids
        d = np.zeros_like(v)
        f = np.full(t.node_count, -1)
        has = parent >= 0
        d[has] = v[has] - v[parent[has]]
        f[has] = t.feature[parent[has]]
        deltas.append(d)
        feats.append(f)
        roots.append(v[0])
    tables = _PathTables(rf, [str(c) for c in rf.classes_], len(rf.estimators_), np.mean(roots, axis=0),
                         np.vstack(deltas), np.concatenate(feats))
    _PATH_CACHE.clear()
    _PATH_CACHE[id(rf)] = tables
    return tables


def _tree_path(tables: _PathTables, Xt: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """contributions (rows x features x classes), base (classes), proba (rows x classes)."""
    indicator, _ = tables.rf.decision_path(Xt)
    indicator = indicator.tocsr()
    n_features = Xt.shape[1]
    out = np.zeros((Xt.shape[0], n_features, len(tables.classes)))
    for i in range(Xt.shape[0]):
        nodes = indicator.indices[indicator.indptr[i]:indicator.indptr[i + 1]]
        nodes = nodes[tables.split_feature[nodes] >= 0]
        np.add.at(out[i], tables.split_feature[nodes], tables.delta[nodes])
    out /= tables.n_trees
    proba = tables.root_value + out.sum(axis=1)
    return out, tables.root_value, proba


def shap_available() -> bool:
    try:
        import shap  # noqa: F401
    except Exception:  # ImportError, or numba/llvmlite failing to load
        return False
    return True


_SHAP_CACHE: dict = {}


def _shap_values(pipeline, Xt: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """TreeSHAP values (rows x features x classes) and expected values (classes)."""
    import shap

    rf = _forest(pipeline)
    explainer = _SHAP_CACHE.get(id(rf))
    if explainer is None or explainer[0] is not rf:
        _SHAP_CACHE.clear()
        explainer = (rf, shap.TreeExplainer(rf))
        _SHAP_CACHE[id(rf)] = explainer
    ex = explainer[1]
    vals = ex.shap_values(Xt, check_additivity=False)
    if isinstance(vals, list):  # older shap: one array per class
        vals = np.stack(vals, axis=-1)
    return np.asarray(vals, dtype=float), np.asarray(ex.expected_value, dtype=float).reshape(-1)


def _shap_agreement(model, X: pd.DataFrame) -> Optional[dict]:
    """How often SHAP and the tree-path method agree on the top features (test rows)."""
    try:
        Xt = _transform(model.pipeline, X)
        tp, _, proba = _tree_path(_path_tables(model.pipeline), Xt)
        sv, _ = _shap_values(model.pipeline, Xt)
    except Exception as exc:  # report, do not fail the whole run
        return {"error": f"{type(exc).__name__}: {exc}"}
    pred = proba.argmax(axis=1)
    overlaps, rhos = [], []
    for i, c in enumerate(pred):
        a, b = tp[i, :, c], sv[i, :, c]
        ta, tb = set(np.argsort(-np.abs(a))[:5]), set(np.argsort(-np.abs(b))[:5])
        overlaps.append(len(ta & tb) / 5)
        rhos.append(pd.Series(a).corr(pd.Series(b), method="spearman"))
    import shap

    return {"rows": int(len(X)), "mean_top5_overlap": round(float(np.mean(overlaps)), 3),
            "median_spearman": round(float(np.nanmedian(rhos)), 3), "shap_version": shap.__version__}


@dataclass
class FeatureContribution:
    feature: str
    label: str
    group: str
    value: float
    contribution: float  # vote-share points (x100) toward the explained class
    direction: str  # "toward" or "away from"
    share_of_training_below: Optional[float]  # % of training programs with a lower value (if known)
    description: str
    test_permutation_importance: Optional[float] = None  # global drop in test macro F1 (if known)


@dataclass
class LocalExplanation:
    available: bool
    method: Optional[str] = None
    method_description: str = ""
    explained_class: Optional[str] = None  # Big-O
    predicted_class: Optional[str] = None
    vote_share: Optional[float] = None  # model vote share for the explained class (0-1)
    base_value: Optional[float] = None  # average vote share before looking at this input (0-1)
    contributions: list = field(default_factory=list)  # all features, sorted by |contribution|
    top_toward: list = field(default_factory=list)
    top_away: list = field(default_factory=list)
    groups: dict = field(default_factory=dict)  # group -> summed contribution (points)
    additivity_error: Optional[float] = None
    summary: list = field(default_factory=list)  # plain-language sentences
    notes: list = field(default_factory=list)
    caveats: list = field(default_factory=lambda: list(LOCAL_CAVEATS))
    error: Optional[dict] = None

    def to_dict(self) -> dict:
        return json.loads(json.dumps(asdict(self), default=float))

    def table(self, top: Optional[int] = None) -> pd.DataFrame:
        rows = [{"Feature": c.label, "Group": c.group, "Value": c.value,
                 "Contribution (points)": round(c.contribution, 2), "Direction": c.direction,
                 "Training programs with a lower value (%)": None if c.share_of_training_below is None
                 else round(c.share_of_training_below)} for c in self.contributions]
        df = pd.DataFrame(rows)
        return df.head(top) if top else df


_METHOD_TEXT = {
    "tree_path": ("Decision-path decomposition of the Random Forest: each split on the path from root to leaf "
                  "credits the change in the tree's vote to the split feature; averaged over all trees. "
                  "Exactly additive; credit depends on split order."),
    "shap": "TreeSHAP (path-dependent) from the shap package: exact Shapley values for the tree ensemble's vote shares.",
}


def _unavailable(code: str, message: str, hint: Optional[str] = None) -> LocalExplanation:
    return LocalExplanation(False, error={"code": code, "message": message, "hint": hint})


def explain_features(values: dict, model=None, method: str = "tree_path", target: Optional[str] = None,
                     top: int = 5, global_data: Optional[dict] = None) -> LocalExplanation:
    """Explain the model's vote for one program, given its 57 feature values.

    target: a Big-O string (e.g. "O(n log n)") or dataset label; default the predicted class.
    global_data: the saved global explanation (default: loaded from models/explainability.json
    when it belongs to this model). Used only for context: how unusual each value is in the
    training data, and whether the feature helped on the test set.
    """
    from src.predictor import ModelUnavailable, feature_vector, get_model

    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}")
    if method == "shap" and not shap_available():
        return _unavailable("shap_not_installed", "SHAP explanations need the optional 'shap' package.",
                            "pip install shap, or use method='tree_path'.")
    if model is None:
        try:
            model = get_model()
        except ModelUnavailable as exc:
            return _unavailable(exc.issue.code, exc.issue.message, exc.issue.hint)
    try:
        X = feature_vector(model.features, values)
    except ValueError as exc:
        return _unavailable("feature_schema_error", f"Could not build the model input: {exc}")

    Xt = _transform(model.pipeline, X)
    model_proba = model.pipeline.predict_proba(X)[0]
    if method == "tree_path":
        contrib, base, proba = _tree_path(_path_tables(model.pipeline), Xt)
        contrib, proba = contrib[0], proba[0]
    else:
        contrib, base = _shap_values(model.pipeline, Xt)
        contrib = contrib[0]
        proba = base + contrib.sum(axis=0)

    classes = model.classes
    predicted = classes[int(np.argmax(model_proba))]
    if target is None:
        k = classes.index(predicted)
    else:
        label = LABEL_OF_BIGO.get(target, target)
        if label not in classes:
            raise ValueError(f"unknown class {target!r}")
        k = classes.index(label)
    additivity = float(abs(proba[k] - model_proba[k]))

    if global_data is None:
        global_data = _saved_global(model)
    quantiles = (global_data or {}).get("training_quantiles") or {}
    test_imp = {r["feature"]: r["importance"] for r in (global_data or {}).get("permutation", [])}
    items = []
    for j, f in enumerate(model.features):
        c = float(contrib[j, k]) * 100
        items.append(FeatureContribution(
            feature=f, label=_label(f), group=GROUP_OF.get(f, "Other"), value=float(X.iloc[0, j]),
            contribution=round(c, 4), direction="toward" if c >= 0 else "away from",
            share_of_training_below=share_below(float(X.iloc[0, j]), quantiles.get(f)),
            description=_description(f), test_permutation_importance=test_imp.get(f)))
    items.sort(key=lambda c: abs(c.contribution), reverse=True)
    toward = [c for c in items if c.contribution > 0][:top]
    away = [c for c in items if c.contribution < 0][:top]
    groups: dict = {}
    for c in items:
        groups[c.group] = round(groups.get(c.group, 0.0) + c.contribution, 3)
    groups = dict(sorted(groups.items(), key=lambda kv: -abs(kv[1])))

    big_o = BIGO_OF[classes[k]]
    exp = LocalExplanation(
        available=True, method=method, method_description=_METHOD_TEXT[method], explained_class=big_o,
        predicted_class=BIGO_OF[predicted], vote_share=round(float(model_proba[k]), 4),
        base_value=round(float(base[k]), 4), contributions=items, top_toward=toward, top_away=away,
        groups=groups, additivity_error=round(additivity, 8))
    top_p = float(model_proba.max())
    if top_p < LOW_CONFIDENCE:
        exp.notes.append(f"The model is unsure (top vote share {top_p:.2f}); this explains an uncertain vote.")
    if additivity > 1e-6:
        exp.notes.append(f"Contributions do not add up exactly to the vote share (difference {additivity:.2g}).")
    if not global_data:
        exp.notes.append("Global context (training distribution, test importance) unavailable; "
                         "run python -m src.explainability.")
    else:
        weak = [c.label for c in (toward + away)
                if c.test_permutation_importance is not None and c.test_permutation_importance <= 0]
        if weak:
            exp.notes.append("The model uses these features here, but shuffling them did not lower its test-set "
                             "score, so their effect may not carry over to unseen problems: " + ", ".join(weak) + ".")
    exp.summary = _summary_sentences(exp)
    return exp


def _summary_sentences(e: LocalExplanation) -> list:
    pts = lambda c: f"{abs(c.contribution):.1f}"  # noqa: E731
    out = [f"The model's vote share for {e.explained_class} is {e.vote_share:.2f}, starting from an average of "
           f"{e.base_value:.2f} across the training programs."]
    if e.explained_class != e.predicted_class:
        out[0] += f" (The predicted class is {e.predicted_class}.)"
    if e.top_toward:
        parts = ", ".join(f"{c.label} = {c.value:g} (+{pts(c)} points)" for c in e.top_toward[:3])
        out.append(f"For this input, the features that moved the vote most toward {e.explained_class} were: {parts}.")
    if e.top_away:
        parts = ", ".join(f"{c.label} = {c.value:g} (-{pts(c)} points)" for c in e.top_away[:3])
        out.append(f"The features that moved it most away from {e.explained_class} were: {parts}.")
    out.append("These describe how the model used the measured features for this input; "
               "they are not causes of the code's complexity.")
    return out


def _saved_global(model) -> Optional[dict]:
    """The saved global explanation, only if it was computed for this exact model file."""
    g = load_global_explanation()
    sha = getattr(getattr(model, "info", None), "model_sha256", None)
    if g.available and sha and g.data.get("model_sha256") == sha:
        return g.data
    return None


def explain_source(source, method: str = "tree_path", target: Optional[str] = None, top: int = 5,
                   model=None) -> LocalExplanation:
    """Parse source code (never executed) and explain the file-level ML prediction."""
    from src.code_parser import parse_bytes, parse_source
    from src.predictor import _all_features

    if isinstance(source, (bytes, bytearray)):
        parsed = parse_bytes(bytes(source))
    elif isinstance(source, str):
        parsed = parse_source(source)
    else:
        return _unavailable("invalid_input_type", f"Expected source code as text, got {type(source).__name__}.")
    if not parsed.success:
        return _unavailable(parsed.error.kind, parsed.error.message, parsed.error.hint)
    return explain_features(_all_features(parsed), model=model, method=method, target=target, top=top)


# --------------------------------------------------------------------------- #
# Report and figures
# --------------------------------------------------------------------------- #
def _plot_bars(names, values, errors, title, xlabel, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 0.32 * len(names) + 1.2))
    y = np.arange(len(names))[::-1]
    ax.barh(y, values, xerr=errors, color="#3266ad", height=0.6, error_kw={"ecolor": "#555", "capsize": 2, "lw": 1})
    ax.set_yticks(y, names, fontsize=9)
    ax.axvline(0, color="#999", lw=0.8)
    ax.set_xlabel(xlabel)
    ax.set_title(title, fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", color="#e5e5e5", lw=0.6)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def render_report(g: dict) -> str:
    perm = global_importance_table(g, "permutation")
    nat = global_importance_table(g, "native")
    tp = global_importance_table(g, "tree_path")
    grp = group_importance_table(g)
    s, b, a = g["settings"], g["baseline"], g["rank_agreement"]
    near_zero = int(((perm["importance"].abs() < 0.002)).sum())
    negative = perm[perm["importance"] < 0]

    def table(df, cols, fmt):
        head = "| " + " | ".join(cols) + " |\n|" + "---|" * len(cols)
        lines = [head] + ["| " + " | ".join(fmt(r)) + " |" for r in df.itertuples()]
        return "\n".join(lines)

    L = [
        "# Model explainability report (Phase 14)",
        "",
        f"Model: {g.get('model_name', 'Random Forest')}, {len(g['native'])} features. Test rows: {s['n_test_rows']} (grouped by problem, "
        f"unseen in training). Training rows: {s['n_training_rows']}. Permutation repeats: {s['n_repeats']}, seed {s['seed']}.",
        f"Baseline test macro F1 {b['test_macro_f1']:.3f}" + (" (matches the saved model metadata)." if b["matches_metadata"]
                                                               else f" (metadata says {b['metadata_test_macro_f1']}; check the files)."),
        "",
        "**Read this first.** Importance shows which inputs this model relies on. It is association, not causation, "
        "and the global ranking is not the reason for any single prediction.",
        "",
        "## 1. Permutation importance (held-out test set)",
        "",
        "Drop in test macro F1 when one feature is shuffled (mean ± std over repeats). "
        "`+ repeats` = how many repeats lowered the score.",
        "",
        "![Permutation importance](permutation_importance.png)",
        "",
        table(perm.head(15), ["rank", "feature", "group", "F1 drop", "std", "+ repeats", "strongly correlated with (ρ ≥ 0.8)"],
              lambda r: [str(r.rank), f"`{r.feature}`", r.group, f"{r.importance:.4f}", f"{r.std:.4f}",
                         f"{r.positive_repeats}/{s['n_repeats']}", r.strongly_correlated_with or "-"]),
        "",
        f"{near_zero} of {len(perm)} features change macro F1 by less than 0.002 when shuffled on their own. "
        "That does **not** prove they are useless: many have strongly correlated partners that carry the same information.",
    ]
    if len(negative):
        within = int((negative["importance"].abs() <= negative["std"]).sum())
        L += ["", f"{len(negative)} features have a negative mean drop (shuffling them helped slightly on this test set); "
              f"{within} of them are within one standard deviation of zero. The model gains nothing measurable from "
              "these features on unseen problems; for the rest, its use of them seems to hurt slightly."]
    L += [
        "",
        "## 2. Grouped permutation importance",
        "",
        "All features of a group shuffled together (same row order), which keeps their correlation inside the group.",
        "",
        "![Group importance](group_importance.png)",
        "",
        table(grp, ["group", "features", "F1 drop", "std", "+ repeats"],
              lambda r: [r.group, str(r.n_features), f"{r.importance:.4f}", f"{r.std:.4f}", f"{r.positive_repeats}/{s['n_repeats']}"]),
    ]
    useless = grp[grp["importance"] <= 0]["group"].tolist()
    if useless:
        L += ["", f"Groups with no measurable benefit on unseen problems (mean drop ≤ 0): {', '.join(useless)}. "
              "The forest still splits on these features (see native importance), so they can appear in per-prediction "
              "explanations, which flag them. This suggests those splits fit the training problems rather than general "
              "patterns. It is a finding about this model, not proof that the features are irrelevant to complexity."]
    L += [
        "",
        "## 3. Which features matter for which class",
        "",
        "Mean drop in that class's F1 on the test set when the feature is shuffled (top 3 per class).",
        "",
        "| class | test F1 | top features (F1 drop) |",
        "|---|---|---|",
    ]
    for big_o in [BIGO_OF[c] for c in pp.CLASS_ORDER]:
        ct = class_importance_table(g, big_o, top=3)
        parts = ", ".join(f"`{r.feature}` ({r.f1_drop:.3f})" for r in ct.itertuples())
        L.append(f"| {big_o} | {b['per_class_f1'].get(big_o, float('nan')):.3f} | {parts} |")
    L += [
        "",
        "## 4. Native importance (mean decrease in impurity)",
        "",
        "From the fitted forest, on the training data. Shown for comparison; it favours features with many distinct "
        "values (line and token counts) and is not measured on unseen problems.",
        "",
        table(nat.head(10), ["rank", "feature", "MDI", "std across trees"],
              lambda r: [str(r.rank), f"`{r.feature}`", f"{r.importance:.4f}", f"{r.std:.4f}"]),
        "",
        "## 5. Aggregated per-prediction contributions",
        "",
        "Mean absolute tree-path contribution toward each test program's predicted class, in vote-share points.",
        "",
        table(tp.head(10), ["rank", "feature", "mean abs. contribution (points)"],
              lambda r: [str(r.rank), f"`{r.feature}`", f"{100 * r.importance:.2f}"]),
        "",
        "## 6. Do the methods agree?",
        "",
        "Spearman correlation between the feature rankings:",
        "",
        f"| pair | Spearman (all {len(g['native'])}) | shared top-10 features |",
        "|---|---|---|",
        f"| native vs permutation | {a['native_vs_permutation_spearman']} | {a['native_vs_permutation_top10_overlap']}/10 |",
        f"| native vs aggregated tree-path | {a['native_vs_tree_path_spearman']} | {a['native_vs_tree_path_top10_overlap']}/10 |",
        f"| permutation vs aggregated tree-path | {a['permutation_vs_tree_path_spearman']} | {a['permutation_vs_tree_path_top10_overlap']}/10 |",
        "",
        f"The Spearman values over all {len(g['native'])} features are dominated by the many features whose permutation importance is "
        "near zero (their order is noise), so the top-10 overlap is the more useful comparison. Native importance and "
        "tree-path contributions both describe how the trees *use* features (largely on training data); permutation "
        "importance describes what *helps on unseen problems*. When they disagree, the second is the safer guide.",
        "",
    ]
    sc = g.get("shap_check")
    if sc and "error" not in sc:
        L += [f"SHAP check (shap {sc['shap_version']}, {sc['rows']} test programs): the tree-path method and TreeSHAP share "
              f"on average {sc['mean_top5_overlap'] * 5:.1f} of their top-5 features for the predicted class; median Spearman "
              f"correlation of the attributions {sc['median_spearman']}.", ""]
    elif sc:
        L += [f"SHAP check failed: {sc['error']}", ""]
    else:
        L += ["SHAP was not installed when this report was made, so the per-prediction method was not compared with TreeSHAP.", ""]
    L += [
        "## 7. Per-prediction explanations",
        "",
        "`explain_source(code)` / `explain_features(values)` return, for one program, the base vote share, each "
        "feature's contribution toward the explained class (default: the predicted class) and plain-language sentences. "
        "Contributions add up exactly to the model's vote share (checked in the tests).",
        "",
        "Limitations:",
        "",
        "- They explain the model's vote, which is often wrong (test macro F1 about 0.44) and not calibrated.",
        "- The tree-path method credits features by split order; features near the root of many trees tend to get more credit. "
        "TreeSHAP (optional) avoids this but needs the `shap` package.",
        "- Correlated features can trade credit between them; read groups (also returned) rather than single features.",
        "- A large contribution is not a cause: changing that feature in the code would not change the code's complexity.",
        "",
        "## Caveats",
        "",
        *[f"- {c}" for c in g["caveats"]],
        "",
        f"Software: Python {g['versions']['python']}, scikit-learn {g['versions']['scikit-learn']}.",
        "",
    ]
    return "\n".join(L)


def write_outputs(g: dict, out_dir: Path = REPORT_DIR, json_path: Path = GLOBAL_PATH) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(g, indent=2), encoding="utf-8")
    perm = global_importance_table(g, "permutation", top=15)
    _plot_bars(perm["feature"], perm["importance"], perm["std"],
               "Permutation importance on the test set (top 15)", "Drop in macro F1 when shuffled",
               out_dir / "permutation_importance.png")
    grp = group_importance_table(g)
    _plot_bars(grp["group"], grp["importance"], grp["std"], "Grouped permutation importance",
               "Drop in macro F1 when the group is shuffled", out_dir / "group_importance.png")
    (out_dir / "explainability_report.md").write_text(render_report(g), encoding="utf-8")


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m src.explainability")
    parser.add_argument("--repeats", type=int, default=N_REPEATS)
    args = parser.parse_args(argv)
    if not config.BIGO_DATASET_CSV.exists() or not pp.SPLIT_FILE.exists():
        print("Run: python -m src.dataset_loader build  and  python -m src.preprocessing  first.")
        return 1
    from src.predictor import ModelUnavailable, get_model

    try:
        model = get_model()
    except ModelUnavailable as exc:
        print(f"{exc.issue.message} {exc.issue.hint or ''}")
        return 1
    df = pd.read_csv(config.BIGO_DATASET_CSV, keep_default_na=False)
    g = compute_global_explanation(df, pp.load_split(df), model, n_repeats=args.repeats)
    write_outputs(g)
    print(f"Saved {GLOBAL_PATH}\nReport: {REPORT_DIR / 'explainability_report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
