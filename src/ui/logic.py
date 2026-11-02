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

from src import comparison, config

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
    "model_unverified": "The model files have no recorded integrity hash, so they were not loaded.",
    "model_integrity_failed": "The model files do not match the hashes recorded at training time, so they were not loaded.",
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
# Comparison calculations live in src.comparison; these helpers shape them for display.
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
    """Numeric metrics side by side (for calculations)."""
    comp = comparison.compare(before, after)
    return pd.DataFrame([{"Metric": m.label, "Before": m.before, "After": m.after, "Change": m.change}
                         for m in comp.metrics if m.key in dict(comparison.METRICS)])


_DIRECTION_TEXT = {"lower": "lower", "higher": "higher", "same": "same", "changed": "changed", "n/a": "—"}


def compare_overview(before: dict, after: dict) -> pd.DataFrame:
    """Display table (all text, so mixed numeric and categorical rows render in one Arrow table)."""
    comp = comparison.compare(before, after)
    rows = []
    for m in comp.metrics:
        numeric = m.key in dict(comparison.METRICS)
        rows.append({"Metric": m.label, "Before": _show(m.before), "After": _show(m.after),
                     "Change": _signed(m.change) if numeric else _DIRECTION_TEXT[m.direction],
                     "Direction": _DIRECTION_TEXT[m.direction]})
    return pd.DataFrame(rows)


def _pair(a, b) -> str:
    if a is None and b is None:
        return "—"
    return f"{_show(a)} → {_show(b)}"


def _ml(label, p) -> Optional[str]:
    if label is None:
        return None
    return f"{label} ({p:.0%})" if isinstance(p, (int, float)) else label


def compare_functions(before: dict, after: dict) -> pd.DataFrame:
    """Per-function comparison, matched by qualified name (display table)."""
    comp = comparison.compare(before, after)
    rows = []
    for f in comp.functions:
        rows.append({
            "Function": f.function_id, "Status": f.status,
            "Lines": _pair(f.lines_before, f.lines_after),
            "Cyclomatic": _pair(f.cyclomatic_before, f.cyclomatic_after),
            "Cyclomatic change": _signed(comparison._delta(f.cyclomatic_before, f.cyclomatic_after)),
            "Nesting": _pair(f.nesting_before, f.nesting_after),
            "Nesting change": _signed(comparison._delta(f.nesting_before, f.nesting_after)),
            "Score": _pair(f.score_before, f.score_after),
            "Time before": _show(f.time_before), "Time after": _show(f.time_after),
            "Time growth": {"lower": "lower", "higher": "higher", "same": "same"}.get(
                f.time_change, "not comparable" if f.status == "in both" else "—"),
            "Space": _pair(f.space_before, f.space_after),
            "ML before (exp.)": _show(_ml(f.ml_before, f.ml_probability_before)),
            "ML after (exp.)": _show(_ml(f.ml_after, f.ml_probability_after)),
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
