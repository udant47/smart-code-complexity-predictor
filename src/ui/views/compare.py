"""Before/after comparison of two code versions. Differences are computed from real results;
the code itself is never rewritten."""

from __future__ import annotations

import streamlit as st

from src.ui import logic, services
from src.ui import components as ui
from src.ui.samples import COMPARE_AFTER, COMPARE_BEFORE

KEYS = {"before": "compare_before", "after": "compare_after"}


def _load_example():
    st.session_state[KEYS["before"]] = COMPARE_BEFORE
    st.session_state[KEYS["after"]] = COMPARE_AFTER
    st.session_state.pop("compare_result", None)


def _clear():
    st.session_state[KEYS["before"]] = ""
    st.session_state[KEYS["after"]] = ""
    st.session_state.pop("compare_result", None)


def render() -> None:
    ui.hero("Compare Versions", "Analyse two versions of the same code and see what changed in structure, "
            "categories, the Derived Complexity Score and per-function metrics.")
    st.session_state.setdefault(KEYS["before"], COMPARE_BEFORE)
    st.session_state.setdefault(KEYS["after"], COMPARE_AFTER)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Before**")
        before = st.text_area("Before", key=KEYS["before"], height=300, label_visibility="collapsed")
        chk_b = logic.check_input(before)
        ui.muted(f"{chk_b.lines:,} lines · {chk_b.chars:,} characters")
    with c2:
        st.markdown("**After**")
        after = st.text_area("After", key=KEYS["after"], height=300, label_visibility="collapsed")
        chk_a = logic.check_input(after)
        ui.muted(f"{chk_a.lines:,} lines · {chk_a.chars:,} characters")

    b1, b2, b3, _ = st.columns([1, 1, 1.4, 3])
    run = b1.button("Compare", type="primary", **ui.STRETCH)
    b2.button("Clear", on_click=_clear, **ui.STRETCH, key="compare_clear")
    b3.button("Load example", on_click=_load_example, **ui.STRETCH)

    if run:
        problems = [("Before", e) for e in chk_b.errors] + [("After", e) for e in chk_a.errors]
        if problems:
            for side, e in problems:
                ui.callout("error", f"{side}: {e['message']}", e.get("hint") or "")
            st.session_state.pop("compare_result", None)
            return
        with st.spinner("Analysing both versions…"):
            st.session_state["compare_result"] = {"before": services.analyze(before), "after": services.analyze(after),
                                                  "codes": (before, after)}

    out = st.session_state.get("compare_result")
    if not out:
        ui.callout("info", "Ready", "Paste two versions (or load the example) and press Compare.")
        return
    if out["codes"] != (before, after):
        ui.callout("info", "The code has changed since the last comparison", "Press Compare to update.")

    failed = False
    for side in ("before", "after"):
        o = out[side]
        if not o["ok"]:
            ui.callout("error", f"{side.title()}: {o['error']['message']}", o["error"].get("hint") or "")
            failed = True
        elif not o["result"]["success"]:
            for e in o["result"]["errors"]:
                ui.callout("error", f"{side.title()}: {e.get('message', '')}", e.get("hint") or "")
            failed = True
    if failed:
        return
    rb, ra = out["before"]["result"], out["after"]["result"]
    for side, r in (("Before", rb), ("After", ra)):
        ml, _ = logic.split_issues(r.get("errors"))
        if ml:
            ui.callout("warning", f"{side}: ML prediction unavailable", logic.ML_ERROR_TITLES.get(ml[0]["code"], ""))

    ob, oa = logic.overview(rb), logic.overview(ra)

    def delta(k, digits=0):
        a, b = ob.get(k), oa.get(k)
        if a is None or b is None:
            return "—"
        d = b - a
        return ("+" if d > 0 else "") + (f"{d:.{digits}f}" if digits else f"{d}")

    ui.section("Summary of changes", "After minus before. Lower structural values usually mean simpler code, "
               "but fewer lines or branches is not automatically better.")
    ui.cards([
        ui.card("Derived Complexity Score", f"{ui.fmt(ob['derived_score'], 1)} → {ui.fmt(oa['derived_score'], 1)}",
                f"change {delta('derived_score', 1)} · {ob['derived_band']} → {oa['derived_band']}", accent=True),
        ui.card("Structural category", f"{ob['structural_category']} → {oa['structural_category']}", "rule-based", accent=True),
        ui.card("ML time class", f"{(rb.get('prediction') or {}).get('big_o', '—')} → {(ra.get('prediction') or {}).get('big_o', '—')}",
                "model output; see caveats", accent=True),
        ui.card("Highest cyclomatic", f"{ob['max_function_cyclomatic_complexity']} → {oa['max_function_cyclomatic_complexity']}",
                f"change {delta('max_function_cyclomatic_complexity')}"),
        ui.card("Max nesting", f"{ob['maximum_nesting_depth']} → {oa['maximum_nesting_depth']}", f"change {delta('maximum_nesting_depth')}"),
        ui.card("Code lines", f"{ob['code_lines']} → {oa['code_lines']}", f"change {delta('code_lines')}"),
    ])

    tab1, tab2, tab3 = st.tabs(["Structural metrics", "Per-function", "Recommendations"])
    with tab1:
        df = logic.compare_overview(rb, ra)
        st.dataframe(df, hide_index=True, **ui.STRETCH)
    with tab2:
        fdf = logic.compare_functions(rb, ra)
        if fdf.empty:
            ui.callout("info", "No functions in either version", "Per-function comparison needs at least one def.")
        else:
            ui.muted("Functions are matched by qualified name. A renamed function shows as removed + added.")
            st.dataframe(fdf, hide_index=True, **ui.STRETCH)
    with tab3:
        nb = len((rb.get("recommendations") or {}).get("recommendations") or [])
        na = len((ra.get("recommendations") or {}).get("recommendations") or [])
        ui.muted(f"Recommendations: {nb} before → {na} after")
        c1, c2 = st.columns(2)
        for col, r, label in ((c1, rb, "Before"), (c2, ra, "After")):
            with col:
                st.markdown(f"**{label}**")
                recs = (r.get("recommendations") or {}).get("recommendations") or []
                if not recs:
                    ui.muted("No rule found evidence above its threshold.")
                for rec in recs:
                    ui.recommendation(rec)
    ui.muted("SmartCode compares the two versions you provide; it does not rewrite code.")
