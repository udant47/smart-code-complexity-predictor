"""Frontend logic without Streamlit: input validation, result shaping, comparisons and
loading of saved evaluation artifacts. Kept separate so it can be unit-tested.

Nothing here re-implements analysis or model logic: results come from
``src.predictor.predict`` and artifacts come from files written by the
training and evaluation commands. Missing artifacts are reported as missing,
never filled in.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd

from src import config

# --------------------------------------------------------------------------- #
# Languages and input validation
# --------------------------------------------------------------------------- #
LANGUAGES = {
    "Python": {"supported": True, "note": "Fully supported (Python 3 syntax of the running interpreter)."},
    "JavaScript": {"supported": False, "note": "Not supported yet: the parser and model are Python-only."},
    "Java": {"supported": False, "note": "Not supported yet: the parser and model are Python-only."},
    "C++": {"supported": False, "note": "Not supported yet: the parser and model are Python-only."},
}
WARN_LINES = 1500  # analysis gets slower above this; still allowed up to the parser's hard limit


@dataclass
class InputCheck:
    ok: bool
    lines: int
    chars: int
    bytes: int
    errors: list = field(default_factory=list)  # blocking
    warnings: list = field(default_factory=list)


def count_lines(code: str) -> int:
    if not code:
        return 0
    return code.count("\n") + (0 if code.endswith("\n") else 1)


def check_input(code: str, language: str = "Python") -> InputCheck:
    """Checks the UI can do before calling the backend (the parser re-checks everything)."""
    code = code or ""
    n_bytes = len(code.encode("utf-8", errors="replace"))
    chk = InputCheck(True, count_lines(code), len(code), n_bytes)
    lang = LANGUAGES.get(language)
    if lang is None or not lang["supported"]:
        chk.errors.append({"code": "unsupported_language",
                           "message": f"{language} is not supported.",
                           "hint": "SmartCode currently analyses Python only. Switch the language to Python."})
    if not code.strip():
        chk.errors.append({"code": "empty", "message": "There is no code to analyse.",
                           "hint": "Paste Python code or choose a sample."})
    if n_bytes > config.MAX_SOURCE_BYTES:
        chk.errors.append({"code": "too_large", "message": f"The input is {n_bytes:,} bytes; the limit is "
                           f"{config.MAX_SOURCE_BYTES:,} bytes.", "hint": "Analyse one module or a few functions at a time."})
    if chk.lines > config.MAX_SOURCE_LINES:
        chk.errors.append({"code": "too_large", "message": f"The input has {chk.lines:,} lines; the limit is "
                           f"{config.MAX_SOURCE_LINES:,}.", "hint": "Analyse one module or a few functions at a time."})
    elif chk.lines > WARN_LINES:
        chk.warnings.append({"code": "large_input", "message": f"{chk.lines:,} lines: analysis may take a few seconds.",
                             "hint": None})
    chk.ok = not chk.errors
    return chk


# --------------------------------------------------------------------------- #
# Friendly errors
# --------------------------------------------------------------------------- #
ML_ERROR_TITLES = {
    "model_missing": "The machine-learning model has not been trained on this computer.",
    "model_metadata_invalid": "The model's metadata file could not be read (it may be corrupt).",
    "model_incompatible": "The saved model does not match this version of the feature extractor.",
    "model_load_failed": "The model file could not be loaded (it may be corrupt or from another library version).",
    "feature_schema_error": "The model input could not be built from the extracted features.",
}


def split_issues(errors: list) -> tuple[list, list]:
    """(model/ML issues, other issues)."""
    ml = [e for e in errors or [] if e.get("code") in ML_ERROR_TITLES]
    other = [e for e in errors or [] if e.get("code") not in ML_ERROR_TITLES]
    return ml, other


# --------------------------------------------------------------------------- #
# Result shaping (pure functions over predictor.to_dict())
# --------------------------------------------------------------------------- #
def overview(result: dict) -> dict:
    fs = result.get("file_summary") or {}
    fm = result.get("file_metrics") or {}
    ds = result.get("derived_score") or {}
    sc = result.get("structural_category") or {}
    return {
        "code_lines": fs.get("code_lines", fm.get("code_lines")),
        "total_lines": fs.get("total_lines", fm.get("total_lines")),
        "function_count": fs.get("function_count", fm.get("function_count")),
        "class_count": fs.get("class_count", fm.get("class_count")),
        "loop_count": fm.get("loop_count"),
        "file_cyclomatic_complexity": fs.get("file_cyclomatic_complexity", fm.get("cyclomatic_complexity")),
        "max_function_cyclomatic_complexity": fs.get("maximum_cyclomatic_complexity"),
        "most_complex_function": fs.get("most_complex_function"),
        "maximum_nesting_depth": fs.get("maximum_nesting_depth", fm.get("maximum_nesting_depth")),
        "structural_category": sc.get("highest") or sc.get("module_level"),
        "structural_scope": "most complex function" if sc.get("highest") else "module",
        "derived_score": ds.get("highest"),
        "derived_band": ds.get("highest_band"),
        "derived_scope": ds.get("scope"),
        "derived_function": ds.get("highest_function"),
    }


def functions_frame(result: dict) -> pd.DataFrame:
    rows = []
    for f in result.get("functions") or []:
        m, est = f.get("metrics") or {}, f.get("static_estimate") or {}
        ds = f.get("derived_score") or {}
        rows.append({
            "Function": f["function_id"],
            "Type": f.get("kind"),
            "Lines": f"{f['start_line']}–{f['end_line']}",
            "Cyclomatic": m.get("cyclomatic_complexity"),
            "Nesting": m.get("maximum_nesting_depth"),
            "Structural category": f.get("structural_category"),
            "Derived score": ds.get("score"),
            "Time (static)": est.get("time_complexity") or "Unknown",
            "Aux. space (static)": est.get("space_complexity") or "Unknown",
            "Status": est.get("status") or "unknown",
        })
    return pd.DataFrame(rows, columns=["Function", "Type", "Lines", "Cyclomatic", "Nesting", "Structural category",
                                       "Derived score", "Time (static)", "Aux. space (static)", "Status"])


# --------------------------------------------------------------------------- #
# Before / after comparison
# --------------------------------------------------------------------------- #
COMPARE_METRICS = [
    ("Code lines", "code_lines"),
    ("Functions", "function_count"),
    ("Classes", "class_count"),
    ("Loops", "loop_count"),
    ("File cyclomatic complexity", "file_cyclomatic_complexity"),
    ("Highest function cyclomatic complexity", "max_function_cyclomatic_complexity"),
    ("Maximum nesting depth", "maximum_nesting_depth"),
    ("Derived Complexity Score (highest)", "derived_score"),
]


def _delta(a, b):
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        d = b - a
        return round(d, 2) if isinstance(d, float) else d
    return None


def _show(v) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        return str(int(v)) if v.is_integer() else f"{v:.1f}"
    return str(v)


def _signed(d) -> str:
    if d is None:
        return "—"
    if d == 0:
        return "0"
    if isinstance(d, float) and d.is_integer():
        d = int(d)
    return f"{d:+.1f}" if isinstance(d, float) else f"{d:+d}"


def compare_numeric(before: dict, after: dict) -> pd.DataFrame:
    """Numeric metrics side by side (for charts and calculations)."""
    a, b = overview(before), overview(after)
    return pd.DataFrame([{"Metric": label, "Before": a.get(k), "After": b.get(k), "Change": _delta(a.get(k), b.get(k))}
                         for label, k in COMPARE_METRICS])


def compare_overview(before: dict, after: dict) -> pd.DataFrame:
    """Display table (all text, so mixed numeric and categorical rows render in one table)."""
    num = compare_numeric(before, after)
    rows = [{"Metric": r.Metric, "Before": _show(r.Before), "After": _show(r.After), "Change": _signed(r.Change)}
            for r in num.astype(object).where(num.notna(), None).itertuples()]
    a, b = overview(before), overview(after)
    pa, pb = before.get("prediction") or {}, after.get("prediction") or {}
    for label, x, y in (("Structural category", a["structural_category"], b["structural_category"]),
                        ("Derived score band", a["derived_band"], b["derived_band"]),
                        ("ML time-complexity class", pa.get("big_o"), pb.get("big_o"))):
        rows.append({"Metric": label, "Before": _show(x), "After": _show(y), "Change": "same" if x == y else "changed"})
    return pd.DataFrame(rows)


def compare_functions(before: dict, after: dict) -> pd.DataFrame:
    """Per-function metrics matched by function id (qualified name)."""
    def index(r):
        return {f["function_id"]: f for f in r.get("functions") or []}

    ia, ib = index(before), index(after)
    order = list(ia) + [k for k in ib if k not in ia]
    rows = []
    for fid in order:
        fa, fb = ia.get(fid), ib.get(fid)
        status = "unchanged id" if fa and fb else ("removed" if fa else "added")

        def g(f, key, sub="metrics"):
            if not f:
                return None
            if sub == "metrics":
                return (f.get("metrics") or {}).get(key)
            if sub == "score":
                return (f.get("derived_score") or {}).get("score")
            if sub == "time":
                return (f.get("static_estimate") or {}).get("time_complexity")
            return f.get(key)

        rows.append({
            "Function": fid, "Status": status,
            "Lines before": g(fa, "function_length"), "Lines after": g(fb, "function_length"),
            "Cyclomatic before": g(fa, "cyclomatic_complexity"), "Cyclomatic after": g(fb, "cyclomatic_complexity"),
            "Cyclomatic change": _delta(g(fa, "cyclomatic_complexity"), g(fb, "cyclomatic_complexity")),
            "Nesting before": g(fa, "maximum_nesting_depth"), "Nesting after": g(fb, "maximum_nesting_depth"),
            "Nesting change": _delta(g(fa, "maximum_nesting_depth"), g(fb, "maximum_nesting_depth")),
            "Score before": g(fa, None, "score"), "Score after": g(fb, None, "score"),
            "Score change": _delta(g(fa, None, "score"), g(fb, None, "score")),
            "Time before": g(fa, None, "time"), "Time after": g(fb, None, "time"),
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Saved evaluation artifacts
# --------------------------------------------------------------------------- #
REPORTS = config.PROJECT_ROOT / "reports"
ARTIFACTS = {
    "model_metadata": config.MODEL_METADATA_PATH,
    "evaluation_results": REPORTS / "evaluation" / "evaluation_results.json",
    "model_comparison": REPORTS / "models" / "model_comparison.csv",
    "confusion_matrix": REPORTS / "models" / "confusion_matrix_selected.csv",
    "dataset_report": config.DATASET_REPORT_JSON,
    "explainability": config.MODELS_DIR / "explainability.json",
}
COMMANDS = {
    "model_metadata": "python -m src.evaluate_model",
    "evaluation_results": "python -m src.evaluate_model",
    "model_comparison": "python -m src.train_model",
    "confusion_matrix": "python -m src.train_model",
    "dataset_report": "python -m src.dataset_loader build",
    "explainability": "python -m src.explainability",
}


@dataclass
class Artifacts:
    data: dict
    missing: dict  # name -> command that creates it
    errors: dict  # name -> message (file exists but could not be read)


def artifact_stamp() -> tuple:
    """Modification times, so cached loaders refresh when files change."""
    return tuple(p.stat().st_mtime_ns if p.exists() else 0 for p in ARTIFACTS.values())


def load_artifacts(paths: Optional[dict] = None) -> Artifacts:
    paths = paths or ARTIFACTS
    data, missing, errors = {}, {}, {}
    for name, path in paths.items():
        path = Path(path)
        if not path.exists():
            missing[name] = COMMANDS.get(name, "")
            continue
        try:
            if path.suffix == ".json":
                data[name] = json.loads(path.read_text(encoding="utf-8"))
            elif name == "confusion_matrix":
                data[name] = pd.read_csv(path, index_col=0)
            else:
                data[name] = pd.read_csv(path)
        except Exception as exc:  # corrupt or partial file
            errors[name] = f"{path.name} could not be read ({type(exc).__name__})."
    return Artifacts(data, missing, errors)


def comparison_table(df: pd.DataFrame) -> pd.DataFrame:
    """Model comparison rows sorted by test macro F1, with the columns the dashboard shows."""
    cols = {"model": "Model", "features": "Features", "rcv_macro_f1": "Repeated CV macro F1", "rcv_std": "CV std",
            "test_macro_f1": "Test macro F1", "test_macro_f1_ci_low": "CI low", "test_macro_f1_ci_high": "CI high",
            "test_accuracy": "Test accuracy"}
    out = df[[c for c in cols if c in df.columns]].rename(columns=cols)
    return out.sort_values("Test macro F1", ascending=False, ignore_index=True)


def class_distribution(dataset_report: dict) -> Optional[pd.DataFrame]:
    from src.dataset_loader import BIGO_LABELS

    dist = (dataset_report.get("bigo") or {}).get("class_distribution")
    if not dist:
        return None
    rows = [{"Class": BIGO_LABELS[k][1] if k in BIGO_LABELS else k, "Count": v["count"], "Percent": v["percent"]}
            for k, v in dist.items()]
    order = {v[1]: v[2] for v in BIGO_LABELS.values()}
    return pd.DataFrame(rows).sort_values("Class", key=lambda s: s.map(order), ignore_index=True)


def confusion_scores(cm: pd.DataFrame) -> dict:
    """Accuracy and macro F1 recomputed from a confusion matrix (rows true, columns predicted)."""
    import numpy as np

    z = cm.to_numpy(dtype=float)
    tp = np.diag(z)
    precision = np.divide(tp, z.sum(axis=0), out=np.zeros_like(tp), where=z.sum(axis=0) > 0)
    recall = np.divide(tp, z.sum(axis=1), out=np.zeros_like(tp), where=z.sum(axis=1) > 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros_like(tp), where=(precision + recall) > 0)
    return {"n": int(z.sum()), "accuracy": float(tp.sum() / z.sum()) if z.sum() else 0.0, "macro_f1": float(f1.mean())}
