"""Results for languages with structural analysis only (C). Shows exactly what was computed, and says
plainly what is not available for the language."""

from __future__ import annotations

from typing import Optional

import pandas as pd
import streamlit as st

from src.languages.c_examples import CExample
from src.suggestions import THRESHOLDS
from src.ui import components as ui

HIGH_CC, DEEP_NESTING, LONG_FUNCTION = (THRESHOLDS["cyclomatic_medium"], THRESHOLDS["nesting_medium"],
                                        THRESHOLDS["length_medium"])

ERROR_TITLES = {"syntax": "The C code could not be parsed", "empty": "No code to analyse",
                "too_large": "Input too large", "encoding": "Unreadable text encoding",
                "invalid_characters": "Invalid characters in the code",
                "unsupported_language": "Unsupported language", "unavailable": "The C parser is not installed"}


def functions_frame(report: dict) -> pd.DataFrame:
    cols = ["Function", "Lines", "Parameters", "Cyclomatic", "Nesting", "Loop nesting", "Loops", "Conditionals",
            "Recursive calls", "Calls", "Structural category", "Flags"]
    rows = []
    for f in report.get("functions") or []:
        flags = [label for cond, label in ((f["cyclomatic_complexity"] >= HIGH_CC, f"cyclomatic ≥ {HIGH_CC}"),
                                           (f["maximum_nesting_depth"] >= DEEP_NESTING, f"nesting ≥ {DEEP_NESTING}"),
                                           (f["function_length"] >= LONG_FUNCTION, f"≥ {LONG_FUNCTION} lines")) if cond]
        rows.append({"Function": f["name"], "Lines": f"{f['start_line']}–{f['end_line']}",
                     "Parameters": f["parameter_count"], "Cyclomatic": f["cyclomatic_complexity"],
                     "Nesting": f["maximum_nesting_depth"], "Loop nesting": f["maximum_loop_nesting_depth"],
                     "Loops": f["loop_count"], "Conditionals": f["conditional_count"],
                     "Recursive calls": f["recursive_call_count"], "Calls": f["function_call_count"],
                     "Structural category": f["structural_category"], "Flags": ", ".join(flags) or "—"})
    return pd.DataFrame(rows, columns=cols)


def example_check(example: CExample, report: dict) -> pd.DataFrame:
    fm = report["file_metrics"]
    measured = {"function_count": fm["function_count"],
                "max_cyclomatic_complexity": fm["maximum_function_cyclomatic_complexity"],
                "max_nesting_depth": fm["maximum_nesting_depth"], "loop_count": fm["loop_count"],
                "recursive_function_count": sum(1 for f in report["functions"] if f["recursive_call_count"]),
                "structural_category": report["structural_category"]}
    labels = {"function_count": "Functions", "max_cyclomatic_complexity": "Highest cyclomatic complexity",
              "max_nesting_depth": "Maximum nesting depth", "loop_count": "Loops",
              "recursive_function_count": "Recursive functions", "structural_category": "Structural category"}
    return pd.DataFrame([{"Aspect": labels[k], "Documented": str(v), "Measured": str(measured[k]),
                          "Status": "✓ matches" if measured[k] == v else "✗ differs"}
                         for k, v in example.structure.items()])


def render(report: dict, example: Optional[CExample] = None) -> None:
    if not report["success"]:
        for e in report["errors"]:
            ui.callout("error", ERROR_TITLES.get(e["code"], "The code could not be analysed"),
                       f"{e['message']} {e.get('hint') or ''}".strip())
        return
    fm = report["file_metrics"]
    caps = report["capabilities"]
    ui.callout("info", "C analysis is experimental and structural only",
               " ".join(caps["notes"][1:]) or "Only structural metrics are available for this language.")
    for w in report["warnings"]:
        ui.callout("warning", w["message"])
    ui.cards([
        ui.card("Structural complexity", report["structural_category"], "from the highest cyclomatic complexity",
                accent=True),
        ui.card("Code lines", fm["code_lines"], f"{fm['total_lines']} total · {fm['comment_lines']} with comments"),
        ui.card("Functions", fm["function_count"]),
        ui.card("Loops", fm["loop_count"], f"max loop nesting {fm['maximum_loop_nesting_depth']}"),
        ui.card("Cyclomatic complexity", fm["maximum_function_cyclomatic_complexity"], "highest function"),
        ui.card("Max nesting depth", fm["maximum_nesting_depth"]),
    ])
    if example:
        with st.expander(f"Documented expectations vs this analysis ({example.title})", expanded=False):
            st.dataframe(example_check(example, report), hide_index=True, **ui.STRETCH)
            ui.muted(example.comparison_note)
    ui.section("Functions", "Structural metrics per function, in the common schema shared with Python.")
    df = functions_frame(report)
    if df.empty:
        ui.callout("info", "No function definitions found", "Prototypes and declarations are not functions.")
    else:
        st.dataframe(df, hide_index=True, **ui.STRETCH)
    ui.section("Not available for C")
    ui.muted("ML time-complexity class, rule-based Big-O estimate, Derived Complexity Score, refactoring "
             "recommendations and model explanations are Python-only. They are not shown because they were built "
             "and calibrated on Python and would not be reliable for C.")
