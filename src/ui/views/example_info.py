"""Documentation panels for the demonstration examples (src.examples)."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.examples import STRUCTURE_LABELS, Example, check_against_result
from src.ui import components as ui

ML_NOTE = ("No ML class is documented for any example: the model is a statistical estimate and may disagree with "
           "the documented complexity. Disagreements are shown, not hidden.")


def complexity_frame(example: Example) -> pd.DataFrame:
    return pd.DataFrame([{"Function": c.function, "Time": c.time, "Auxiliary space": c.space,
                          "Assumptions": " ".join(c.assumptions), "Note": c.note}
                         for c in example.complexity])


def structure_frame(example: Example) -> pd.DataFrame:
    return pd.DataFrame([{"Characteristic": label, "Expected": str(example.structure[k])}
                         for k, label in STRUCTURE_LABELS.items()])


def render_about(example: Example) -> None:
    with st.expander(f"About this example: {example.pattern}", expanded=False):
        st.markdown(ui.esc(example.summary))
        st.markdown("\n".join(f"- {ui.esc(c)}" for c in example.characteristics))
        st.markdown("**Expected structural characteristics**")
        st.dataframe(structure_frame(example), hide_index=True, **ui.STRETCH)
        st.markdown("**Expected asymptotic complexity (under the stated assumptions)**")
        st.dataframe(complexity_frame(example), hide_index=True, **ui.STRETCH)
        if example.expected_recommendations:
            ui.muted("Expected refactoring recommendations: " + ", ".join(example.expected_recommendations))
        ui.muted(ML_NOTE)


_STATUS = {"matches": "✓ matches", "differs": "✗ differs", "not estimated": "not estimated",
           "not documented": "not documented"}


def render_check(example: Example, result: dict) -> None:
    rows = check_against_result(example, result)
    differs = [r for r in rows if r.status == "differs"]
    with st.expander(f"Documented expectations vs this analysis ({example.title})", expanded=bool(differs)):
        if differs:
            ui.callout("warning", f"{len(differs)} documented value(s) differ from this analysis",
                       "The documentation or the analyzer has changed; the measured values below are authoritative.")
        df = pd.DataFrame([{"Aspect": r.aspect, "Documented": r.documented, "Measured": r.measured,
                            "Status": _STATUS[r.status]} for r in rows])
        st.dataframe(df, hide_index=True, **ui.STRETCH)
        ui.muted(ML_NOTE)
