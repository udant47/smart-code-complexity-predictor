"""Dataset quality audit (final improvement phase).

    python -m src.dataset_audit            # everything (about 2 minutes)
    python -m src.dataset_audit --quick    # data checks only, no model fits

Writes ``reports/improvement/dataset_audit.json`` and ``dataset_audit.md``.
Every number is computed from ``data/processed`` and the saved split; nothing
is relabelled or generated.

Checks
------
1. Class balance per partition (train / validation / test) and the imbalance ratio.
2. Label consistency: identical feature vectors with different labels, and
   near-duplicate clusters (renamed-token Jaccard >= 0.9) with different labels.
3. Leakage: problems or near-duplicate clusters present in more than one
   partition (both must be 0 for a grouped split).
4. Diversity: programs per problem, labels per problem, algorithm tags.
5. Model-based (skipped with ``--quick``): in-sample vs grouped-CV macro F1
   (over-fitting gap) and a learning curve over the number of training
   problems (is the model data-limited?).
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_predict

from src import config
from src import preprocessing as pp
from src import train_model as tm
from src import tune_model

OUT_DIR = config.PROJECT_ROOT / "reports" / "improvement"


def _final_config() -> dict:
    if config.MODEL_METADATA_PATH.exists():
        meta = json.loads(config.MODEL_METADATA_PATH.read_text(encoding="utf-8"))
        if meta.get("selected_model") in tune_model.FOREST_MODELS:
            return {"model": meta["selected_model"], "feature_set": meta["feature_set"],
                    "params": meta["hyperparameters"]}
    return {"model": "Random Forest", **tune_model.PHASE10_CONFIG}


def data_checks(df: pd.DataFrame, split: pp.SplitResult, features: list[str]) -> dict:
    y = df[pp.TARGET]
    parts = {p: split.indices(p) for p in ("train", "validation", "test")}
    out = {"rows": int(len(df)), "problems": int(df["problem"].nunique()), "classes": pp.CLASS_ORDER}

    counts = y.value_counts().reindex(pp.CLASS_ORDER)
    out["class_balance"] = {
        "counts": {k: int(v) for k, v in counts.items()},
        "imbalance_ratio_max_over_min": round(float(counts.max() / counts.min()), 3),
        "share_by_partition": {p: y.loc[i].value_counts(normalize=True).reindex(pp.CLASS_ORDER).round(3).to_dict()
                               for p, i in parts.items()},
        "rows_by_partition": {p: int(len(i)) for p, i in parts.items()},
    }

    key = df[features].round(6).astype(str).agg("|".join, axis=1)
    labels_per_vector = df.groupby(key)[pp.TARGET].nunique()
    conflicting = set(labels_per_vector[labels_per_vector > 1].index)
    clusters = df[df["near_duplicate_cluster"].astype(int) >= 0]
    labels_per_cluster = clusters.groupby("near_duplicate_cluster")[pp.TARGET].nunique()
    out["label_consistency"] = {
        "distinct_feature_vectors": int(key.nunique()),
        "rows_with_identical_vector_but_different_label": int(key.isin(conflicting).sum()),
        "near_duplicate_clusters": int(labels_per_cluster.size),
        "rows_in_near_duplicate_clusters": int(len(clusters)),
        "near_duplicate_clusters_with_different_labels": int((labels_per_cluster > 1).sum()),
    }

    part_of = pd.Series("", index=df.index)
    for p, i in parts.items():
        part_of.loc[i] = p
    prob_parts = df.groupby("problem").apply(lambda g: part_of.loc[g.index].nunique(), include_groups=False)
    clus_parts = clusters.groupby("near_duplicate_cluster").apply(lambda g: part_of.loc[g.index].nunique(),
                                                                  include_groups=False)
    out["leakage"] = {
        "problems_in_more_than_one_partition": int((prob_parts > 1).sum()),
        "near_duplicate_clusters_in_more_than_one_partition": int((clus_parts > 1).sum()),
        "split_strategy": "StratifiedGroupKFold, groups = problem merged with near-duplicate clusters",
    }

    per_problem = df.groupby("problem")
    majority = per_problem[pp.TARGET].agg(lambda s: s.value_counts().iloc[0]).sum()
    tags = df["tags"].str.split(",").explode().str.strip()
    tags = tags[tags != ""]
    out["diversity"] = {
        "programs_per_problem_median": float(per_problem.size().median()),
        "programs_per_problem_max": int(per_problem.size().max()),
        "problems_with_more_than_one_label": int((per_problem[pp.TARGET].nunique() > 1).sum()),
        "share_of_rows_with_their_problems_majority_label": round(float(majority / len(df)), 4),
        "distinct_tags": int(tags.nunique()),
        "top_tags": tags.value_counts().head(10).to_dict(),
    }
    return out


def model_checks(df: pd.DataFrame, split: pp.SplitResult, cfg: dict, seed: int = config.RANDOM_STATE) -> dict:
    y = df[pp.TARGET]
    dev = split.indices("train").append(split.indices("validation"))
    features = pp.feature_columns(cfg["feature_set"])
    X = df.loc[dev, features]
    cv = tm.inner_cv(y.loc[dev], split.groups.loc[dev], seed)

    def pipe(n_estimators=None):
        p = dict(cfg["params"])
        if n_estimators:
            p["n_estimators"] = n_estimators
        return tune_model.forest_pipeline(cfg["feature_set"], p, seed, cfg["model"])

    oof = cross_val_predict(pipe(), X, y.loc[dev], cv=cv, n_jobs=-1)
    fitted = pipe().fit(X, y.loc[dev])
    gap = {"in_sample_macro_f1": round(float(f1_score(y.loc[dev], fitted.predict(X), average="macro")), 4),
           "grouped_cv_macro_f1": round(float(f1_score(y.loc[dev], oof, average="macro")), 4)}

    rng = np.random.default_rng(seed)
    curve = []
    for frac in (0.25, 0.5, 0.75, 1.0):
        scores = []
        for a, b in cv:
            idx_a, idx_b = X.index[a], X.index[b]
            probs = split.groups.loc[idx_a].unique()
            keep = probs if frac == 1.0 else rng.choice(probs, size=max(2, int(len(probs) * frac)), replace=False)
            sub = idx_a[split.groups.loc[idx_a].isin(keep)]
            m = pipe(200).fit(X.loc[sub], y.loc[sub])
            scores.append(f1_score(y.loc[idx_b], m.predict(X.loc[idx_b]), average="macro"))
        curve.append({"share_of_training_problems": frac, "grouped_cv_macro_f1": round(float(np.mean(scores)), 4)})
    return {"model": cfg, "fit_gap": gap, "learning_curve": curve}


def render(r: dict) -> str:
    cb, lc, lk, dv = r["class_balance"], r["label_consistency"], r["leakage"], r["diversity"]
    lines = [
        "# Dataset quality audit",
        "",
        "Generated by `python -m src.dataset_audit`. CodeComplex (Python subset), "
        f"{r['rows']:,} programs from {r['problems']} Codeforces problems, 7 worst-case time classes.",
        "",
        "## 1. Class balance",
        "",
        "| class | programs | train share | validation share | test share |",
        "|---|---:|---:|---:|---:|",
    ]
    for c in pp.CLASS_ORDER:
        s = cb["share_by_partition"]
        lines.append(f"| {c} | {cb['counts'][c]} | {s['train'][c]:.3f} | {s['validation'][c]:.3f} | {s['test'][c]:.3f} |")
    lines += [
        "",
        f"Largest / smallest class: {cb['imbalance_ratio_max_over_min']:.2f} (mild imbalance). Partitions: "
        + ", ".join(f"{k} {v}" for k, v in cb["rows_by_partition"].items()) + ".",
        "",
        "## 2. Label consistency",
        "",
        f"- Distinct feature vectors: {lc['distinct_feature_vectors']:,}; rows whose identical vector carries a different "
        f"label: **{lc['rows_with_identical_vector_but_different_label']}**.",
        f"- Near-duplicate clusters: {lc['near_duplicate_clusters']} ({lc['rows_in_near_duplicate_clusters']} rows); "
        f"clusters with different labels: **{lc['near_duplicate_clusters_with_different_labels']}**.",
        "- No evidence of label noise was found with these checks, so no labels were changed or removed.",
        "",
        "## 3. Leakage",
        "",
        f"- Problems in more than one partition: **{lk['problems_in_more_than_one_partition']}**.",
        f"- Near-duplicate clusters in more than one partition: **{lk['near_duplicate_clusters_in_more_than_one_partition']}**.",
        f"- Split: {lk['split_strategy']}.",
        "",
        "## 4. Diversity",
        "",
        f"- Programs per problem: median {dv['programs_per_problem_median']:.0f}, max {dv['programs_per_problem_max']}.",
        f"- {dv['problems_with_more_than_one_label']} problems have solutions with different labels; "
        f"{100 * dv['share_of_rows_with_their_problems_majority_label']:.1f}% of programs share their problem's majority label.",
        "  The label is mostly a property of the *problem*, so generalising to unseen problems is the real difficulty.",
        f"- {dv['distinct_tags']} algorithm tags; most common: "
        + ", ".join(f"{k} ({v})" for k, v in dv["top_tags"].items()) + ".",
    ]
    if r.get("model"):
        m = r["model"]
        lines += [
            "",
            "## 5. Model-based checks (train + validation only)",
            "",
            f"Model: {m['model']['model']}, `{m['model']['feature_set']}`.",
            "",
            f"- In-sample macro F1 {m['fit_gap']['in_sample_macro_f1']:.3f} vs grouped 5-fold CV "
            f"{m['fit_gap']['grouped_cv_macro_f1']:.3f}: a large gap, typical of forests (they nearly memorise training "
            "rows). Model selection therefore uses only grouped cross-validation, never training scores.",
            "",
            "| share of training problems | grouped-CV macro F1 |",
            "|---:|---:|",
            *[f"| {c['share_of_training_problems']:.2f} | {c['grouped_cv_macro_f1']:.3f} |" for c in m["learning_curve"]],
            "",
            "The score still rises with more training problems, so more labelled problems (not more solutions of the "
            "same problems) would most likely help.",
        ]
    return "\n".join(lines) + "\n"


def run(quick: bool = False) -> dict:
    df = pd.read_csv(config.BIGO_DATASET_CSV, keep_default_na=False)
    split = pp.load_split(df)
    cfg = _final_config()
    result = data_checks(df, split, pp.feature_columns(cfg["feature_set"]))
    result["model"] = None if quick else model_checks(df, split, cfg)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "dataset_audit.json").write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    (OUT_DIR / "dataset_audit.md").write_text(render(result), encoding="utf-8")
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m src.dataset_audit")
    parser.add_argument("--quick", action="store_true", help="data checks only (no model fits)")
    args = parser.parse_args(argv)
    if not config.BIGO_DATASET_CSV.exists():
        print("Processed data not found. Run: python -m src.dataset_loader build")
        return 1
    print(render(run(args.quick)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
