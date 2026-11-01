"""Hyperparameter tuning of the selected model (Phase 11).

    python -m src.tune_model           # full search (about 10-25 minutes)
    python -m src.tune_model --quick   # small search for a fast check

The Phase 10 comparison selected a Random Forest with the ``patterns``
feature set. This module tunes it more thoroughly.

Protocol
--------
* Only the **development rows** (train + validation) are used. The test set is
  not touched here.
* ``RandomizedSearchCV`` samples ``N_ITER`` configurations from a documented
  search space and scores each by macro F1 with 5-fold *grouped, stratified*
  cross-validation (grouped by problem). The folds use a different shuffle
  seed from the comparison folds below.
* The feature set is part of the search: ``patterns`` (57 features) vs
  ``patterns_no_size`` (44 features, without line, identifier and function
  counts). Phase 10 hinted that dropping size features might help, but that
  hint came from the test set, so it is decided here by cross-validation.
* **Before vs after.** The Phase 10 configuration and the tuned one are both
  scored on the same repeated grouped CV folds as Phase 10 (2 × 5 folds).
  Because tuning used the same development rows, the tuned score is slightly
  optimistic. The tuned configuration is adopted only if it beats the Phase 10
  one by more than ``ADOPT_MARGIN``; otherwise the simpler, already-validated
  configuration is kept.

The chosen configuration is written to ``models/tuning_results.json`` and is
used by ``python -m src.evaluate_model``, which makes the single test
evaluation and saves the final model.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV

from src import config
from src import preprocessing as pp
from src import train_model as tm

TUNING_RESULTS = config.MODELS_DIR / "tuning_results.json"
N_ITER = 30
ADOPT_MARGIN = 0.005
SEARCH_SEED_OFFSET = 7  # tuning folds differ from the comparison folds

# Phase 10 selection (Random Forest [patterns]); used if no metadata file is found
PHASE10_CONFIG = {
    "feature_set": "patterns",
    "params": {"n_estimators": 300, "max_features": "sqrt", "min_samples_leaf": 3,
               "class_weight": "balanced_subsample"},
}

SEARCH_SPACE = {
    "model__n_estimators": [200, 400, 600],
    "model__max_features": ["sqrt", "log2", 0.2, 0.35, 0.5],
    "model__min_samples_leaf": [1, 2, 3, 5, 8],
    "model__min_samples_split": [2, 5, 10],
    "model__max_depth": [None, 12, 20],
    "model__class_weight": [None, "balanced", "balanced_subsample"],
    "model__max_samples": [None, 0.6, 0.8],
    "model__criterion": ["gini", "entropy"],
}


def rf_pipeline(feature_set: str, params: dict, seed: int = config.RANDOM_STATE):
    est = RandomForestClassifier(random_state=seed, n_jobs=1)
    est.set_params(**{k.replace("model__", ""): v for k, v in params.items()})
    return pp.make_pipeline("tree", est, pp.feature_columns(feature_set))


def phase10_config() -> dict:
    """The configuration selected in Phase 10 (from the saved metadata when available)."""
    meta_path = config.MODEL_METADATA_PATH
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("selected_model") == "Random Forest" and "phase11" not in meta:
            params = dict(meta.get("hyperparameters", {}))
            params.setdefault("n_estimators", PHASE10_CONFIG["params"]["n_estimators"])
            return {"feature_set": meta["feature_set"], "params": params}
    return json.loads(json.dumps(PHASE10_CONFIG))


def run(quick: bool = False, df: Optional[pd.DataFrame] = None, split: Optional[pp.SplitResult] = None,
        seed: int = config.RANDOM_STATE, out_path: Path = TUNING_RESULTS, n_jobs: int = -1,
        verbose: bool = True, feature_sets: tuple = ("patterns", "patterns_no_size")) -> dict:
    t0 = time.time()
    if df is None:
        df = pd.read_csv(config.BIGO_DATASET_CSV, keep_default_na=False)
    if split is None:
        split = pp.load_split(df)
    y = df[pp.TARGET]
    dev = split.indices("train").append(split.indices("validation"))
    search_cv = tm.inner_cv(y.loc[dev], split.groups.loc[dev], seed + SEARCH_SEED_OFFSET)
    compare_cv = tm.repeated_cv(y.loc[dev], split.groups.loc[dev], seed)  # same folds as Phase 10

    before = phase10_config()
    if quick:
        before["params"]["n_estimators"] = 60
    before_pipe = rf_pipeline(before["feature_set"], before["params"], seed)
    before_scores = tm.cv_score(before_pipe, df.loc[dev], y.loc[dev], compare_cv, n_jobs)
    if verbose:
        print(f"Before (Phase 10): RCV macro F1 {before_scores.mean():.3f} ± {before_scores.std():.3f}", flush=True)

    space = dict(SEARCH_SPACE)
    if quick:
        space["model__n_estimators"] = [60]
    searches = {}
    for fs in feature_sets:
        if not set(pp.feature_columns(fs)) <= set(df.columns):
            continue
        search = RandomizedSearchCV(
            rf_pipeline(fs, {}, seed), space, n_iter=4 if quick else N_ITER, scoring="f1_macro",
            cv=search_cv, random_state=seed, n_jobs=n_jobs, refit=False, error_score="raise",
        )
        search.fit(df.loc[dev], y.loc[dev])
        i = search.best_index_
        searches[fs] = {
            "best_params": {k.replace("model__", ""): v for k, v in search.cv_results_["params"][i].items()},
            "search_cv_macro_f1": float(search.cv_results_["mean_test_score"][i]),
            "search_cv_std": float(search.cv_results_["std_test_score"][i]),
            "n_configurations": int(len(search.cv_results_["params"])),
            "all_scores": sorted((float(s) for s in search.cv_results_["mean_test_score"]), reverse=True),
        }
        if verbose:
            print(f"Search [{fs}]: best CV {searches[fs]['search_cv_macro_f1']:.3f}  {searches[fs]['best_params']}", flush=True)

    best_fs = max(searches, key=lambda k: searches[k]["search_cv_macro_f1"])
    tuned = {"feature_set": best_fs, "params": searches[best_fs]["best_params"]}
    after_scores = tm.cv_score(rf_pipeline(best_fs, tuned["params"], seed), df.loc[dev], y.loc[dev], compare_cv, n_jobs)
    gain = float(after_scores.mean() - before_scores.mean())
    adopted = gain > ADOPT_MARGIN
    chosen = tuned if adopted else before
    if verbose:
        print(f"After (tuned):     RCV macro F1 {after_scores.mean():.3f} ± {after_scores.std():.3f}  "
              f"({gain:+.3f}) -> {'adopt tuned' if adopted else 'keep Phase 10 configuration'}", flush=True)

    result = {
        "model": "Random Forest",
        "protocol": ("RandomizedSearchCV on train+validation with grouped stratified 5-fold CV; before/after compared "
                     "on the Phase 10 repeated grouped CV folds; test set not used"),
        "seed": seed,
        "quick_mode": quick,
        "search_space": {k.replace("model__", ""): v for k, v in space.items()},
        "n_iter_per_feature_set": 4 if quick else N_ITER,
        "before": {**before, "rcv_macro_f1": float(before_scores.mean()), "rcv_std": float(before_scores.std())},
        "searches": searches,
        "tuned": {**tuned, "rcv_macro_f1": float(after_scores.mean()), "rcv_std": float(after_scores.std())},
        "rcv_gain": gain,
        "adopt_margin": ADOPT_MARGIN,
        "adopted": "tuned" if adopted else "phase10",
        "chosen": chosen,
        "note": "The tuned RCV score is slightly optimistic: tuning and comparison used the same development rows.",
        "seconds": round(time.time() - t0, 1),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, default=_json_default), encoding="utf-8")
    return result


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    return str(o)


def load_chosen(path: Path = TUNING_RESULTS) -> dict:
    if not path.exists():
        raise FileNotFoundError("Run `python -m src.tune_model` first.")
    return json.loads(path.read_text(encoding="utf-8"))


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m src.tune_model")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--jobs", type=int, default=-1)
    args = parser.parse_args(argv)
    if not config.BIGO_DATASET_CSV.exists() or not pp.SPLIT_FILE.exists():
        print("Run: python -m src.dataset_loader build  and  python -m src.preprocessing  first.")
        return 1
    r = run(quick=args.quick, n_jobs=args.jobs)
    print(f"\nChosen: {r['adopted']} configuration, features [{r['chosen']['feature_set']}]: {r['chosen']['params']}")
    print(f"Saved {TUNING_RESULTS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
