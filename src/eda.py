"""Exploratory data analysis of the processed datasets (Phase 8).

Reproducible: every number and figure is computed from
``data/processed/*.csv`` (built by ``python -m src.dataset_loader build``).
Nothing is simulated.

    python -m src.eda            # writes reports/eda/*.png and reports/eda/eda_report.md

The functions are also used by ``notebooks/01_exploratory_data_analysis.ipynb``.

Statistics here are associations. A feature that correlates with a label
does not cause it, and a strong association can come from how the data
was collected (for example, problem identity or coding style) rather than
from algorithmic complexity.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src import config
from src.dataset_loader import BIGO_LABELS, STRUCTURAL_BANDS, structural_band
from src.feature_extractor import FILE_FEATURE_NAMES, FUNCTION_FEATURE_NAMES

REPORT_DIR = config.PROJECT_ROOT / "reports" / "eda"
BIGO_ORDER = [v[1] for v in sorted(BIGO_LABELS.values(), key=lambda v: v[2])]  # O(1) .. O(2^n)
BAND_ORDER = [b[0] for b in STRUCTURAL_BANDS]

# Chart styling (reference palette: one series colour, one highlight, muted ink)
BLUE, ORANGE = "#2a78d6", "#eb6834"
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
DIVERGING = ("#e34948", "#f0efec", "#2a78d6")  # negative - neutral - positive

# Patterns of competitive-programming template code (counted, never executed)
TEMPLATE_PATTERNS = {
    "reads sys.stdin": r"sys\.stdin",
    "FastIO class": r"class FastIO|IOWrapper",
    "raises recursion limit": r"setrecursionlimit",
}


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_bigo(path: Path = config.BIGO_DATASET_CSV) -> pd.DataFrame:
    df = pd.read_csv(path, keep_default_na=False)
    df["label_bigo"] = pd.Categorical(df["label_bigo"], categories=BIGO_ORDER, ordered=True)
    return df


def load_structural(path: Path = config.STRUCTURAL_DATASET_CSV) -> pd.DataFrame:
    df = pd.read_csv(path, keep_default_na=False)
    df["reference_label"] = pd.Categorical(df["reference_label"], categories=BAND_ORDER, ordered=True)
    return df


def attach_sources(bigo: pd.DataFrame, raw_jsonl: Path) -> Optional[pd.Series]:
    """Source text per row from the raw file (None if it is not downloaded)."""
    if not raw_jsonl.exists():
        return None
    lines = raw_jsonl.read_text(encoding="utf-8").splitlines()
    return bigo["raw_index"].map(lambda i: json.loads(lines[i])["src"])


# --------------------------------------------------------------------------- #
# Analyses (each returns plain data; no plotting)
# --------------------------------------------------------------------------- #
def overview(df: pd.DataFrame, features: list[str]) -> dict:
    """1-3. Dimensions, data types, missing values."""
    dtypes = df.dtypes.astype(str)
    return {
        "rows": len(df),
        "columns": df.shape[1],
        "feature_columns": len(features),
        "metadata_columns": df.shape[1] - len(features),
        "feature_dtypes": dtypes[features].value_counts().to_dict(),
        "missing_values": int(df[features].isna().sum().sum()),
        "empty_metadata_strings": {c: int((df[c] == "").sum()) for c in df.columns if df[c].dtype == object and (df[c] == "").any()},
        "constant_features": [c for c in features if df[c].nunique() <= 1],
        "mostly_zero_features": [c for c in features if (df[c] == 0).mean() > 0.95],
    }


def duplicate_summary(df: pd.DataFrame, features: list[str], label: str) -> dict:
    """4. Remaining duplicates after Phase 7 cleaning."""
    key = df[features].round(6).astype(str).agg("|".join, axis=1)
    groups = df.groupby(key)[label]
    shared = key.duplicated(keep=False)
    conflicting = groups.nunique() > 1
    return {
        "duplicate_sample_ids": int(df["sample_id"].duplicated().sum()),
        "duplicate_code_hashes": int(df["code_sha256"].duplicated().sum()),
        "rows_sharing_a_feature_vector": int(shared.sum()),
        "feature_vectors_with_conflicting_labels": int(conflicting.sum()),
        "rows_in_conflicting_vectors": int(key.map(conflicting).sum()),
        "accuracy_ceiling_from_features": round(float(groups.apply(lambda x: x.value_counts().iloc[0]).sum() / len(df)), 4),
    }


def class_distribution(series: pd.Series, order: list[str]) -> pd.DataFrame:
    """5. Counts and percentages in a fixed class order."""
    counts = series.value_counts().reindex(order, fill_value=0)
    return pd.DataFrame({"count": counts, "percent": (100 * counts / counts.sum()).round(2)})


def feature_summary(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """6. Distribution summary per feature."""
    d = df[features]
    out = pd.DataFrame({
        "mean": d.mean(), "median": d.median(), "std": d.std(), "min": d.min(), "max": d.max(),
        "zero_%": (100 * (d == 0).mean()), "skew": d.skew(),
    })
    return out.round(2)


def redundant_pairs(df: pd.DataFrame, features: list[str], threshold: float = 0.95) -> pd.DataFrame:
    """7. Feature pairs with |Spearman rho| >= threshold."""
    usable = [c for c in features if df[c].nunique() > 1]
    corr = df[usable].corr(method="spearman")
    rows = []
    for i, a in enumerate(usable):
        for b in usable[i + 1:]:
            r = corr.loc[a, b]
            if abs(r) >= threshold:
                rows.append({"feature_a": a, "feature_b": b, "spearman": round(float(r), 3)})
    return pd.DataFrame(rows, columns=["feature_a", "feature_b", "spearman"]).sort_values("spearman", ascending=False, key=abs).reset_index(drop=True)


def outlier_summary(df: pd.DataFrame, features: list[str], k: float = 3.0) -> pd.DataFrame:
    """8. Extreme values: beyond Q3 + k·IQR (or below Q1 - k·IQR).

    Features whose IQR is 0 (mostly zeros) flag every non-zero value, so
    the count is reported together with the IQR for interpretation.
    """
    d = df[features]
    q1, q3 = d.quantile(0.25), d.quantile(0.75)
    iqr = q3 - q1
    mask = (d > q3 + k * iqr) | (d < q1 - k * iqr)
    out = pd.DataFrame({
        "outliers": mask.sum(), "outlier_%": (100 * mask.mean()).round(2),
        "q1": q1, "q3": q3, "iqr": iqr, "max": d.max(),
    })
    return out.sort_values("outliers", ascending=False)


def feature_label_association(df: pd.DataFrame, features: list[str], rank_col: str) -> pd.DataFrame:
    """9. Spearman correlation of each feature with the ordinal label rank."""
    usable = [c for c in features if df[c].nunique() > 1]
    rho = df[usable].corrwith(df[rank_col], method="spearman")
    return rho.rename("spearman_with_label_rank").to_frame().sort_values(
        "spearman_with_label_rank", ascending=False, key=abs
    ).round(3)


def median_by_class(df: pd.DataFrame, features: list[str], label: str) -> pd.DataFrame:
    return df.groupby(label, observed=False)[features].median().T


def problem_label_purity(df: pd.DataFrame) -> dict:
    """10. How strongly the label is tied to the problem (leakage risk)."""
    g = df.groupby("problem")["label"]
    n_labels = g.nunique()
    majority = g.apply(lambda x: x.value_counts().iloc[0])
    return {
        "problems": int(len(n_labels)),
        "single_label_problems": int((n_labels == 1).sum()),
        "single_label_problem_%": round(100 * float((n_labels == 1).mean()), 1),
        "labels_per_problem": {int(k): int(v) for k, v in n_labels.value_counts().sort_index().items()},
        "samples_with_problem_majority_label_%": round(100 * float(majority.sum() / len(df)), 1),
        "samples_per_problem": df["problem"].value_counts().describe().round(1).to_dict(),
    }


def tag_label_table(df: pd.DataFrame, top: int = 12) -> pd.DataFrame:
    """10. Problem tags vs labels: tags are metadata, unavailable for user code."""
    t = df.assign(tag=df["tags"].str.split(",")).explode("tag").reset_index(drop=True)
    t = t[t["tag"] != ""]
    common = t["tag"].value_counts().head(top).index
    sel = t[t["tag"].isin(common)]
    table = pd.crosstab(sel["tag"], sel["label_bigo"].astype(str), normalize="index")
    return (100 * table).round(1).reindex(columns=BIGO_ORDER)


def structural_circularity(df: pd.DataFrame) -> dict:
    """10. The structural reference label is a function of a feature."""
    rule = df["cyclomatic_complexity"].apply(structural_band)
    per_band = df.groupby("reference_label", observed=False)["cyclomatic_complexity"].agg(["min", "max", "count"])
    rho = df[list(FUNCTION_FEATURE_NAMES)].corr(method="spearman")["cyclomatic_complexity"].drop("cyclomatic_complexity")
    return {
        "rule_reproduces_label_%": round(100 * float((rule == df["reference_label"].astype(str)).mean()), 2),
        "cc_range_per_band": {str(k): {kk: int(vv) for kk, vv in row.items()} for k, row in per_band.iterrows()},
        "features_with_rho_over_0.7": rho[rho.abs() >= 0.7].round(3).sort_values(ascending=False).to_dict(),
    }


def near_duplicate_summary(df: pd.DataFrame) -> dict:
    """11. Near-duplicate clusters flagged in Phase 7."""
    flagged = df[df["near_duplicate_cluster"] >= 0]
    by_cluster = flagged.groupby("near_duplicate_cluster")
    return {
        "samples_in_clusters": int(len(flagged)),
        "clusters": int(by_cluster.ngroups),
        "largest_cluster": int(by_cluster.size().max()) if len(flagged) else 0,
        "clusters_with_mixed_labels": int((by_cluster["label"].nunique() > 1).sum()),
        "clusters_spanning_problems": int((by_cluster["problem"].nunique() > 1).sum()),
    }


def template_prevalence(df: pd.DataFrame, sources: Optional[pd.Series]) -> Optional[pd.DataFrame]:
    """12. Share of each class using competitive-programming template code."""
    if sources is None:
        return None
    rows = {}
    for name, pattern in TEMPLATE_PATTERNS.items():
        hit = sources.str.contains(pattern, regex=True)
        rows[name] = (100 * hit.groupby(df["label_bigo"], observed=False).mean()).round(1)
        rows[name]["all"] = round(100 * float(hit.mean()), 1)
    return pd.DataFrame(rows).T[BIGO_ORDER + ["all"]]


# --------------------------------------------------------------------------- #
# Plots (each returns a matplotlib Figure)
# --------------------------------------------------------------------------- #
def _style():
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "axes.titlecolor": INK,
        "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "axes.labelsize": 10, "xtick.color": INK_2, "ytick.color": INK_2,
        "xtick.labelsize": 9, "ytick.labelsize": 9, "axes.grid": True, "grid.color": GRID,
        "grid.linewidth": 0.6, "axes.axisbelow": True, "axes.spines.top": False,
        "axes.spines.right": False, "font.size": 10, "legend.frameon": False,
    })
    return plt


def plot_class_distribution(dist: pd.DataFrame, title: str, xlabel: str):
    plt = _style()
    fig, ax = plt.subplots(figsize=(8, 3.6))
    bars = ax.bar(dist.index.astype(str), dist["count"], color=BLUE, width=0.62, edgecolor=SURFACE, linewidth=2)
    for bar, (cnt, pct) in zip(bars, dist[["count", "percent"]].itertuples(index=False)):
        ax.annotate(f"{cnt:,}\n{pct:.1f}%", (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8, color=INK_2)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Samples")
    ax.grid(axis="x", visible=False)
    ax.set_ylim(0, dist["count"].max() * 1.22)
    fig.tight_layout()
    return fig


def plot_feature_histograms(df: pd.DataFrame, features: list[str], title: str):
    plt = _style()
    n = len(features)
    cols = 3
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(11, 2.6 * rows))
    for ax, f in zip(np.ravel(axes), features):
        raw = df[f].astype(float)
        cap = float(np.ceil(raw.quantile(0.99)))
        values = raw.clip(upper=cap)  # the top 1% is grouped into the last bar
        if cap <= 30:
            bins = np.arange(cap + 2) - 0.5
        else:
            bins = np.linspace(0, cap, 31)
        ax.hist(values, bins=bins, color=BLUE, edgecolor=SURFACE, linewidth=0.8)
        title_suffix = f"  (last bar: ≥ {cap:g}, max {raw.max():g})" if raw.max() > cap else ""
        ax.set_title(f + title_suffix, fontsize=9.5, fontweight="normal")
        ax.grid(axis="x", visible=False)
        ax.text(0.98, 0.92, f"median {values.median():g}", transform=ax.transAxes, ha="right", fontsize=8, color=INK_2)
    for ax in np.ravel(axes)[n:]:
        ax.set_visible(False)
    fig.suptitle(title, x=0.01, ha="left", fontsize=12, fontweight="bold", color=INK)
    fig.tight_layout()
    return fig


def plot_correlation_heatmap(df: pd.DataFrame, features: list[str], title: str):
    from matplotlib.colors import LinearSegmentedColormap

    plt = _style()
    usable = [c for c in features if df[c].nunique() > 1]
    corr = df[usable].corr(method="spearman")
    cmap = LinearSegmentedColormap.from_list("diverging", DIVERGING)
    size = max(7, 0.24 * len(usable) + 2)
    fig, ax = plt.subplots(figsize=(size + 1.2, size))
    im = ax.imshow(corr.values, cmap=cmap, vmin=-1, vmax=1)
    ax.set_xticks(range(len(usable)), usable, rotation=90, fontsize=7)
    ax.set_yticks(range(len(usable)), usable, fontsize=7)
    ax.grid(False)
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("Spearman ρ", color=INK_2)
    cbar.outline.set_visible(False)
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_features_by_class(df: pd.DataFrame, features: list[str], label: str, order: list[str], title: str):
    plt = _style()
    cols = 2
    rows = int(np.ceil(len(features) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(11, 3.0 * rows))
    for ax, f in zip(np.ravel(axes), features):
        data = [df.loc[df[label] == c, f].astype(float).values for c in order]
        bp = ax.boxplot(data, patch_artist=True, widths=0.55, showfliers=True,
                        medianprops={"color": INK, "linewidth": 1.6},
                        boxprops={"facecolor": "#b7d3f6", "edgecolor": BLUE, "linewidth": 1},
                        whiskerprops={"color": BLUE}, capprops={"color": BLUE},
                        flierprops={"marker": "o", "markersize": 2.5, "markerfacecolor": BLUE, "markeredgecolor": "none", "alpha": 0.35})
        ax.set_xticks(range(1, len(order) + 1), order, fontsize=8)
        ax.set_title(f, fontsize=10, fontweight="normal")
        ax.grid(axis="x", visible=False)
        if df[f].max() > 40:
            ax.set_yscale("symlog", linthresh=10)
        ax.set_ylim(bottom=-0.3 if df[f].max() <= 40 else 0)
    for ax in np.ravel(axes)[len(features):]:
        ax.set_visible(False)
    fig.suptitle(title, x=0.01, ha="left", fontsize=12, fontweight="bold", color=INK)
    fig.tight_layout()
    return fig


def plot_label_association(assoc: pd.DataFrame, title: str, top: int = 15):
    plt = _style()
    data = assoc.head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 0.32 * len(data) + 1.2))
    values = data.iloc[:, 0]
    ax.barh(data.index, values, color=[BLUE if v >= 0 else ORANGE for v in values], height=0.6, edgecolor=SURFACE)
    for y, v in enumerate(values):
        ax.annotate(f"{v:+.2f}", (v, y), xytext=(4 if v >= 0 else -4, 0), textcoords="offset points",
                    ha="left" if v >= 0 else "right", va="center", fontsize=8, color=INK_2)
    ax.axvline(0, color=INK_2, linewidth=0.8)
    ax.set_xlabel("Spearman ρ with label rank (association, not causation)")
    ax.set_xlim(min(0, values.min()) - 0.1, max(values.max(), 0) + 0.12)
    ax.grid(axis="y", visible=False)
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_problem_purity(df: pd.DataFrame):
    plt = _style()
    g = df.groupby("problem")["label"]
    purity = g.apply(lambda x: x.value_counts(normalize=True).iloc[0])
    size = g.size()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.6))
    a1.hist(100 * purity, bins=np.arange(30, 101, 5), color=BLUE, edgecolor=SURFACE, linewidth=1)
    a1.set_title("Label purity per problem")
    a1.set_xlabel("Majority-label share within the problem (%)")
    a1.set_ylabel("Problems")
    a1.grid(axis="x", visible=False)
    a2.scatter(size, 100 * purity, s=14, color=BLUE, alpha=0.6, edgecolors="none")
    a2.set_title("Solutions per problem vs purity")
    a2.set_xlabel("Solutions per problem")
    a2.set_ylabel("Majority-label share (%)")
    fig.tight_layout()
    return fig


def plot_structural_circularity(df: pd.DataFrame):
    plt = _style()
    fig, ax = plt.subplots(figsize=(9, 3.4))
    cap = 40
    clipped = df["cyclomatic_complexity"].max() > cap
    counts = df["cyclomatic_complexity"].clip(upper=cap).value_counts().sort_index()
    colors = [BLUE if structural_band(int(cc)) == "Low" else "#6da7ec" if structural_band(int(cc)) == "Medium" else ORANGE for cc in counts.index]
    ax.bar(counts.index, counts.values, color=colors, width=0.8, edgecolor=SURFACE, linewidth=1)
    for x in (5.5, 10.5):
        ax.axvline(x, color=INK, linewidth=1, linestyle="--")
    ymax = counts.max()
    for x, name in ((3, "Low (1-5)"), (8, "Medium (6-10)"), (14, "High (11+)")):
        ax.text(x, ymax * 0.92, name, ha="center", fontsize=9, color=INK)
    ax.set_yscale("log")
    ax.set_xlabel("Cyclomatic complexity (an input feature)" + (f"; values above {cap} shown at {cap}" if clipped else ""))
    ax.set_ylabel("Functions (log scale)")
    ax.set_title("Structural reference labels are thresholds on an input feature")
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    return fig


def plot_length_by_class(df: pd.DataFrame, sources: Optional[pd.Series]):
    plt = _style()
    fig, ax = plt.subplots(figsize=(8, 3.4))
    med = df.groupby("label_bigo", observed=False)["code_lines"].median().reindex(BIGO_ORDER)
    ax.bar(BIGO_ORDER, med.values, color=BLUE, width=0.6, edgecolor=SURFACE, linewidth=2, label="all programs")
    if sources is not None:
        plain = ~sources.str.contains("|".join(TEMPLATE_PATTERNS.values()), regex=True)
        med2 = df[plain].groupby("label_bigo", observed=False)["code_lines"].median().reindex(BIGO_ORDER)
        ax.plot(BIGO_ORDER, med2.values, color=ORANGE, marker="o", markersize=8, linewidth=2,
                markeredgecolor=SURFACE, markeredgewidth=2, label="without template code")
        ax.legend(loc="upper left", fontsize=9)
    ax.set_ylabel("Median code lines")
    ax.set_title("Program length differs by class (a possible shortcut)")
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #
LABEL_FEATURES = [
    "maximum_loop_nesting_depth", "nested_loop_count", "loop_count", "recursive_call_count",
    "function_count", "code_lines",
]
HIST_FEATURES = [
    "total_lines", "code_lines", "cyclomatic_complexity", "loop_count", "maximum_loop_nesting_depth",
    "maximum_nesting_depth", "function_count", "function_call_count", "identifier_count",
]


def _md_table(df: pd.DataFrame, index: bool = True) -> str:
    """Markdown table without extra dependencies."""
    def fmt(v) -> str:
        if isinstance(v, float):
            return f"{v:,.2f}".rstrip("0").rstrip(".") if v == v else ""
        if isinstance(v, (int, np.integer)):
            return f"{v:,}"
        return str(v)

    if index and df.index.name is None:
        df = df.rename_axis("feature")
    data = df.reset_index() if index else df
    header = [str(c) for c in data.columns]
    numeric = [pd.api.types.is_numeric_dtype(data[c]) for c in data.columns]
    rows = ["| " + " | ".join(header) + " |",
            "| " + " | ".join("---:" if n else "---" for n in numeric) + " |"]
    for rec in data.itertuples(index=False):
        rows.append("| " + " | ".join(fmt(v) for v in rec) + " |")
    return "\n".join(rows)


def run(
    out_dir: Path = REPORT_DIR,
    bigo_path: Path = config.BIGO_DATASET_CSV,
    structural_path: Path = config.STRUCTURAL_DATASET_CSV,
    raw_jsonl: Path = config.CODECOMPLEX_DIR / config.CODECOMPLEX["file"],
) -> dict:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    b, s = load_bigo(bigo_path), load_structural(structural_path)
    F, G = list(FILE_FEATURE_NAMES), list(FUNCTION_FEATURE_NAMES)
    sources = attach_sources(b, raw_jsonl)

    results = {
        "bigo_overview": overview(b, F),
        "structural_overview": overview(s, G),
        "bigo_duplicates": duplicate_summary(b, F, "label"),
        "structural_duplicates": duplicate_summary(s, G, "reference_label"),
        "problem_purity": problem_label_purity(b),
        "near_duplicates": near_duplicate_summary(b),
        "structural_circularity": structural_circularity(s),
    }
    bigo_dist = class_distribution(b["label_bigo"].astype(str), BIGO_ORDER)
    band_dist = class_distribution(s["reference_label"].astype(str), BAND_ORDER)
    assoc = feature_label_association(b, F, "label_rank")
    pairs = redundant_pairs(b, F)
    outliers = outlier_summary(b, F)
    tags = tag_label_table(b)
    templates = template_prevalence(b, sources)
    medians = median_by_class(b, LABEL_FEATURES, "label_bigo")

    figures = {
        "01_bigo_class_distribution.png": plot_class_distribution(bigo_dist, "Big-O dataset: samples per class", "Expert label"),
        "02_structural_band_distribution.png": plot_class_distribution(band_dist, "Structural dataset: samples per reference band", "Reference band (heuristic)"),
        "03_feature_histograms.png": plot_feature_histograms(b, HIST_FEATURES, "Big-O dataset: feature distributions"),
        "04_correlation_heatmap.png": plot_correlation_heatmap(b, F, "Big-O dataset: Spearman correlation between features"),
        "05_features_by_class.png": plot_features_by_class(b, LABEL_FEATURES, "label_bigo", BIGO_ORDER, "Big-O dataset: features by expert label"),
        "06_label_association.png": plot_label_association(assoc, "Features most associated with the Big-O label rank"),
        "07_problem_label_purity.png": plot_problem_purity(b),
        "08_structural_circularity.png": plot_structural_circularity(s),
        "09_length_by_class.png": plot_length_by_class(b, sources),
    }
    for name, fig in figures.items():
        fig.savefig(out_dir / name, dpi=130)
        plt.close(fig)

    results.update({
        "bigo_class_distribution": bigo_dist.to_dict(orient="index"),
        "structural_band_distribution": band_dist.to_dict(orient="index"),
        "top_label_associations": assoc.head(10)["spearman_with_label_rank"].to_dict(),
        "redundant_pairs": pairs.to_dict(orient="records"),
        "top_outlier_features": outliers.head(8)["outliers"].to_dict(),
        "template_prevalence": templates.to_dict(orient="index") if templates is not None else None,
    })
    (out_dir / "eda_results.json").write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    (out_dir / "eda_report.md").write_text(
        render_report(results, bigo_dist, band_dist, assoc, pairs, outliers, tags, templates, medians, b, s),
        encoding="utf-8",
    )
    return results


def render_report(r, bigo_dist, band_dist, assoc, pairs, outliers, tags, templates, medians, b, s) -> str:
    bo, so = r["bigo_overview"], r["structural_overview"]
    bd, sd = r["bigo_duplicates"], r["structural_duplicates"]
    pp, nd, sc = r["problem_purity"], r["near_duplicates"], r["structural_circularity"]
    top_feat, top_rho = assoc.index[0], assoc.iloc[0, 0]
    shares = bigo_dist["percent"]
    imbalance = round(float(shares.max() / shares.min()), 2)
    high = band_dist.loc["High", "percent"]
    tmpl = templates.loc["reads sys.stdin"] if templates is not None else None
    exp_recursive = round(100 * float((b.loc[b["label_bigo"] == "O(2^n)", "recursive_call_count"] > 0).mean()), 1)
    noise = None
    if config.DATASET_REPORT_JSON.exists():
        removed = json.loads(config.DATASET_REPORT_JSON.read_text(encoding="utf-8"))["bigo"]["removed"]
        noise = sum(v for k, v in removed.items() if k.endswith("conflicting_labels"))

    lines = [
        "# Exploratory data analysis",
        "",
        "Generated by `python -m src.eda` from `data/processed/`. Every number below is computed from the processed",
        "datasets. Correlations describe associations in this data; they do not show that a feature causes a label.",
        "",
        "## 1–3. Dimensions, types, missing values",
        "",
        "| Dataset | Rows | Columns | Features | Feature dtypes | Missing feature values |",
        "| --- | ---: | ---: | ---: | --- | ---: |",
        f"| Big-O (CodeComplex) | {bo['rows']:,} | {bo['columns']} | {bo['feature_columns']} | {bo['feature_dtypes']} | {bo['missing_values']} |",
        f"| Structural (TheAlgorithms) | {so['rows']:,} | {so['columns']} | {so['feature_columns']} | {so['feature_dtypes']} | {so['missing_values']} |",
        "",
        f"Empty metadata strings (not missing features): {bo['empty_metadata_strings'] or 'none'}. "
        f"Constant features: {bo['constant_features'] or 'none'}. "
        f"Features that are 0 in more than 95% of programs: {', '.join(bo['mostly_zero_features']) or 'none'}.",
        "",
        "## 4. Duplicates",
        "",
        "| Check | Big-O | Structural |",
        "| --- | ---: | ---: |",
        f"| Duplicate sample ids | {bd['duplicate_sample_ids']} | {sd['duplicate_sample_ids']} |",
        f"| Duplicate code hashes | {bd['duplicate_code_hashes']} | {sd['duplicate_code_hashes']} |",
        f"| Rows sharing an identical feature vector | {bd['rows_sharing_a_feature_vector']} | {sd['rows_sharing_a_feature_vector']} |",
        f"| Identical feature vectors with different labels | {bd['feature_vectors_with_conflicting_labels']} | {sd['feature_vectors_with_conflicting_labels']} |",
        f"| Highest accuracy achievable from these features | {100 * bd['accuracy_ceiling_from_features']:.1f}% | {100 * sd['accuracy_ceiling_from_features']:.1f}% |",
        "",
        "Different programs can share a feature vector (the features summarise structure, not the code). None of these",
        "collisions carry conflicting labels, so the features do not cap achievable accuracy below 100% on this data.",
        "",
        "## 5. Class distribution",
        "",
        "![Big-O classes](01_bigo_class_distribution.png)",
        "",
        _md_table(bigo_dist.rename_axis("class")),
        "",
        f"The Big-O classes are mildly imbalanced (largest/smallest share = {imbalance}).",
        "",
        "![Structural bands](02_structural_band_distribution.png)",
        "",
        _md_table(band_dist.rename_axis("band")),
        "",
        f"The structural bands are heavily imbalanced: only {high}% of functions fall in High.",
        "",
        "## 6. Feature distributions",
        "",
        "![Feature histograms](03_feature_histograms.png)",
        "",
        "Count features are right-skewed with long tails; each histogram shows values up to the 99th percentile, and the last bar collects the top 1%.",
        "",
        _md_table(feature_summary(b, HIST_FEATURES)),
        "",
        "## 7. Correlation between features",
        "",
        "![Correlation heatmap](04_correlation_heatmap.png)",
        "",
        f"{len(pairs)} feature pairs have |Spearman ρ| ≥ 0.95. Several are redundant by definition",
        "(`loop_count = for_loop_count + while_loop_count`, `operator_count` is a sum of categories,",
        "`comment_ratio` is derived from `comment_lines`). Linear models need one of each pair removed or regularisation.",
        "",
        _md_table(pairs, index=False) if len(pairs) else "_No pairs above the threshold._",
        "",
        "## 8. Outliers",
        "",
        "Values beyond Q3 + 3·IQR. Features with IQR = 0 are mostly zero, so any non-zero value is flagged there.",
        "",
        _md_table(outliers.head(10)),
        "",
        f"The longest programs ({int(b['total_lines'].max())} lines) consist mostly of template classes, not algorithmic code.",
        "Outliers are kept (they are real submissions); tree models are robust to them, and linear models should use",
        "log-transformed or scaled features.",
        "",
        "## 9. Relationship between features and labels",
        "",
        "![Features by class](05_features_by_class.png)",
        "",
        "![Label association](06_label_association.png)",
        "",
        f"The strongest association is `{top_feat}` (ρ = {top_rho:+.2f}). Loop-nesting features rise with the label rank as",
        "algorithm theory would suggest, but the overlap between classes is large: no single feature separates them.",
        "",
        f"`recursive_call_count` has median 0 in every class: only {exp_recursive}% of O(2^n) programs contain a direct",
        "recursive call; most enumerate subsets with loops over bitmasks (see the tag table in section 10).",
        "",
        "Median values per class:",
        "",
        _md_table(medians),
        "",
        "## 10. Potential target leakage",
        "",
        "![Problem purity](07_problem_label_purity.png)",
        "",
        f"- **Problem identity.** {pp['single_label_problems']} of {pp['problems']} problems "
        f"({pp['single_label_problem_%']}%) have the same label for every solution, and "
        f"{pp['samples_with_problem_majority_label_%']}% of samples carry their problem's majority label. "
        "If solutions of the same problem appear in both training and test data, a model can score well by",
        "  recognising the problem rather than the complexity. **Training and evaluation must split by `problem`.**",
        "- **Tags.** Problem tags are strongly associated with labels (table below), but they are Codeforces metadata",
        "  that user code does not have. They are excluded from the features.",
        "- **Metadata columns** (`sample_id`, `raw_index`, `code_sha256`, `near_duplicate_cluster`, `label_*`) must never be",
        "  used as features.",
        f"- **Structural labels are circular.** The reference band is reproduced by a threshold rule on",
        f"  `cyclomatic_complexity` for {sc['rule_reproduces_label_%']}% of functions, and that column is an input feature. A",
        "  model trained on these labels would only relearn the thresholds; removing the column does not fix it,",
        f"  because other features are strongly correlated with it: {', '.join(f'`{k}` ({v})' for k, v in list(sc['features_with_rho_over_0.7'].items())[:5])}.",
        "",
        "![Structural circularity](08_structural_circularity.png)",
        "",
        "Share of each tag's solutions per label (%), top tags:",
        "",
        _md_table(tags),
        "",
        "## 11. Near-duplicate samples",
        "",
        f"{nd['samples_in_clusters']} samples in {nd['clusters']} clusters (largest: {nd['largest_cluster']}); "
        f"{nd['clusters_with_mixed_labels']} clusters have mixed labels and {nd['clusters_spanning_problems']} span more than one problem.",
        "Because every cluster stays inside one problem, a problem-level split keeps near-duplicates on one side.",
        "",
        "## 12. Possible dataset bias",
        "",
        "- **One source and one style.** All programs are Codeforces solutions: short, single-file, reading from standard",
        "  input. Real-world code (classes, libraries, I/O, frameworks) is under-represented.",
        "- **Whole-program labels.** Labels describe complete programs, while the app analyses functions.",
        "- **Problem concentration.** "
        f"{pp['problems']} problems supply {bo['rows']:,} samples (median {pp['samples_per_problem']['50%']:.0f}, "
        f"max {pp['samples_per_problem']['max']:.0f} per problem), so the effective diversity is closer to {pp['problems']} tasks than to {bo['rows']:,}.",
    ]
    if tmpl is not None:
        lines += [
            f"- **Template code.** {tmpl['all']}% of programs read `sys.stdin` directly, from {tmpl['O(1)']}% of O(1) to "
            f"{tmpl['O(n^3)']}% of O(n^3) programs. Template code inflates size features (lines, classes, functions) "
            "independently of the algorithm, so size can act as a shortcut that reflects author style.",
            "",
            "![Length by class](09_length_by_class.png)",
            "",
            _md_table(templates.rename_axis("template pattern (% of class)")),
            "",
        ]
    lines += [
        (f"- **Label noise.** Phase 7 removed {noise} records whose identical code carried different expert labels; similar"
         if noise is not None else "- **Label noise.** Identical code with different expert labels was removed in Phase 7; similar"),
        "  disagreements likely remain in non-identical code.",
        "- **Structural corpus.** TheAlgorithms/Python is educational code: small, well-documented functions, so",
        f"  High-complexity functions are rare ({high}%).",
        "",
        "## Assessment: is the data suitable for training?",
        "",
        "**Big-O dataset: suitable, with conditions.** It has 4,000+ expert-labelled real programs, seven reasonably balanced",
        "classes, no missing values, no conflicting duplicates, and features whose association with the label is",
        "plausible (loop nesting rises from O(1) to O(n^3)). Conditions for honest results:",
        "",
        "1. Split by problem (grouped cross-validation and a held-out set of problems); report random-split scores only",
        "   to show how much they overestimate.",
        "2. Exclude tags and all metadata columns from the features.",
        "3. Compare against simple baselines: majority class, a single loop-nesting rule, and the Phase 6 static analyser.",
        "4. Check whether size features act as a shortcut, by training with and without them.",
        "5. Describe results as \"agreement with expert labels on Codeforces-style programs\", not general Big-O accuracy,",
        "   and remember that labels are per program while the app predicts per function.",
        "",
        "**Structural dataset: not suitable as a supervised target.** Its labels are thresholds on an input feature, so",
        "training would only reproduce the rule. It is suitable for describing distributions and for calibrating the",
        "transparent rule-based structural score.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    if not config.BIGO_DATASET_CSV.exists():
        print("Processed data not found. Run: python -m src.dataset_loader download  then  python -m src.dataset_loader build")
        return 1
    run()
    print(f"Wrote {REPORT_DIR / 'eda_report.md'} and {len(list(REPORT_DIR.glob('*.png')))} figures")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
