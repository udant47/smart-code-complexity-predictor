"""Before/after comparison of two code versions.

Every difference shown is computed from the two real analysis results by
``src.comparison``. The code itself is never rewritten, and nothing is called
an improvement because it got shorter.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd
import streamlit as st

from src import comparison
from src.examples import ComparisonPair
from src.ui import components as ui
from src.ui import logic, services
from src.ui.samples import COMPARE_AFTER, COMPARE_BEFORE, PAIRS

KEYS = {"before": "compare_before", "after": "compare_after"}
PAIR_KEY = "compare_pair"
RESULT_KEY = "compare_result"
PAIR_PLACEHOLDER = "Choose an example pair…"


def _load_pair():
    pair = PAIRS.get(st.session_state.get(PAIR_KEY))
    if pair:
        st.session_state[KEYS["before"]] = pair.before
        st.session_state[KEYS["after"]] = pair.after
        st.session_state.pop(RESULT_KEY, None)


def _clear():
    st.session_state[KEYS["before"]] = ""
    st.session_state[KEYS["after"]] = ""
    st.session_state[PAIR_KEY] = PAIR_PLACEHOLDER
    st.session_state.pop(RESULT_KEY, None)


def _swap():
    b, a = st.session_state.get(KEYS["before"], ""), st.session_state.get(KEYS["after"], "")
    st.session_state[KEYS["before"]], st.session_state[KEYS["after"]] = a, b
    st.session_state.pop(RESULT_KEY, None)


def matching_pair(before: str, after: str) -> Optional[ComparisonPair]:
    return next((p for p in PAIRS.values()
                 if p.before.rstrip() == (before or "").rstrip() and p.after.rstrip() == (after or "").rstrip()), None)


def _about_pair(pair: ComparisonPair) -> None:
    with st.expander(f"About this pair: {pair.title}", expanded=False):
        st.markdown(ui.esc(pair.summary))
        st.markdown(f"**Behaviour.** {ui.esc(pair.behaviour)}")
        st.markdown("**Expected to change**\n" + "\n".join(f"- {ui.esc(c)}" for c in pair.changes))
        if pair.unchanged:
            st.markdown("**Expected to stay the same**\n" + "\n".join(f"- {ui.esc(c)}" for c in pair.unchanged))
        st.dataframe(pd.DataFrame([{"Function": e.function, "Time before": e.time_before, "Time after": e.time_after,
                                    "Space before": e.space_before, "Space after": e.space_after}
                                   for e in pair.expected]), hide_index=True, **ui.STRETCH)
        ui.callout("info", "Takeaway", pair.lesson)


def render() -> None:
    ui.hero("Compare Versions", "Analyse two versions of the same code and see what actually changed: structure, "
            "Derived Complexity Score, cyclomatic complexity, nesting, functions, ML output and Big-O estimates.")
    st.session_state.setdefault(KEYS["before"], COMPARE_BEFORE)
    st.session_state.setdefault(KEYS["after"], COMPARE_AFTER)

    st.selectbox("Example pair", [PAIR_PLACEHOLDER, *PAIRS], key=PAIR_KEY, on_change=_load_pair,
                 help="Documented before/after pairs, including ones where the shorter version is slower.")
    current = matching_pair(st.session_state[KEYS["before"]], st.session_state[KEYS["after"]])
    if current:
        _about_pair(current)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Before**")
        before = st.text_area("Before", key=KEYS["before"], height=300, label_visibility="collapsed",
                              placeholder="Paste the original version here.")
        chk_b = logic.check_input(before)
        ui.muted(f"{chk_b.lines:,} lines · {chk_b.chars:,} characters")
    with c2:
        st.markdown("**After**")
        after = st.text_area("After", key=KEYS["after"], height=300, label_visibility="collapsed",
                             placeholder="Paste the changed version here.")
        chk_a = logic.check_input(after)
        ui.muted(f"{chk_a.lines:,} lines · {chk_a.chars:,} characters")

    b1, b2, b3, _ = st.columns([1, 1, 1.4, 3])
    run = b1.button("Compare", type="primary", **ui.STRETCH)
    b2.button("Clear", on_click=_clear, **ui.STRETCH, key="compare_clear")
    b3.button("Swap before/after", on_click=_swap, **ui.STRETCH)

    if run:
        problems = [("Before", e) for e in chk_b.errors] + [("After", e) for e in chk_a.errors]
        if problems:
            for side, e in problems:
                ui.callout("error", f"{side}: {e['message']}", e.get("hint") or "")
            st.session_state.pop(RESULT_KEY, None)
            return
        with st.spinner("Analysing both versions…"):
            st.session_state[RESULT_KEY] = {"before": services.analyze(before), "after": services.analyze(after),
                                            "codes": (before, after)}

    out = st.session_state.get(RESULT_KEY)
    if not out:
        ui.callout("info", "Ready", "Choose an example pair or paste two versions, then press Compare.")
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
        ui.muted("Both versions must be valid Python before they can be compared.")
        return
    rb, ra = out["before"]["result"], out["after"]["result"]
    for side, r in (("Before", rb), ("After", ra)):
        ml, _ = logic.split_issues(r.get("errors"))
        if ml:
            ui.callout("warning", f"{side}: ML prediction unavailable",
                       logic.ML_ERROR_TITLES.get(ml[0]["code"], "") + " Structural and static results are still compared.")
    render_comparison(rb, ra)


def _arrow(m: comparison.MetricChange, digits: int = 0) -> str:
    return f"{ui.fmt(m.before, digits) if digits else ui.fmt(m.before)} → {ui.fmt(m.after, digits) if digits else ui.fmt(m.after)}"


def _sub(m: comparison.MetricChange, digits: int = 0) -> str:
    if m.change is None:
        return m.direction if m.direction != "n/a" else "—"
    if m.change == 0:
        return "no change"
    return f"change {m.change:+.{digits}f}" if digits else f"change {m.change:+g}"


def render_comparison(rb: dict, ra: dict) -> None:
    comp = comparison.compare(rb, ra)
    m = {x.key: x for x in comp.metrics}

    ui.section("Summary of changes", "After compared with before, as measured. Nothing here is a verdict: "
               "fewer lines or lower metrics are not automatically better.")
    ui.cards([
        ui.card("Structural category", _arrow(m["structural_category"]), "rule-based · " + m["structural_category"].direction,
                accent=True),
        ui.card("Derived Complexity Score", _arrow(m["derived_score"], 1),
                f"{_sub(m['derived_score'], 1)} · {m['derived_band'].before} → {m['derived_band'].after}", accent=True),
        ui.card("Highest cyclomatic", _arrow(m["max_cyclomatic_complexity"]), _sub(m["max_cyclomatic_complexity"])),
        ui.card("Max nesting", _arrow(m["max_nesting_depth"]), _sub(m["max_nesting_depth"])),
        ui.card("Functions", _arrow(m["function_count"]), _sub(m["function_count"])),
        ui.card("ML class (file)", _arrow(m["ml_class"]), "model output · not verified"),
        ui.card("Code lines", _arrow(m["code_lines"]), _sub(m["code_lines"]) + " · not a quality measure"),
    ])

    ui.section("What actually changed", "Observations computed from the two analyses. Time estimates are compared "
               "only when both are static estimates with status 'estimated'.")
    for o in comp.observations:
        ui.callout("warning" if o.kind == "caution" else "info", o.title, o.detail)

    tab1, tab2, tab3 = st.tabs(["Structural metrics", "Functions, predictions & Big-O", "Recommendations"])
    with tab1:
        st.dataframe(logic.compare_overview(rb, ra), hide_index=True, **ui.STRETCH)
    with tab2:
        fdf = logic.compare_functions(rb, ra)
        if fdf.empty:
            ui.callout("info", "No functions in either version", "Per-function comparison needs at least one def.")
        else:
            ui.muted("Functions are matched by qualified name; a renamed function shows as removed + added. "
                     "Time and space are the static analyzer's single-variable estimates. ML columns are "
                     "experimental per-function model outputs (blank for very short functions).")
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
    ui.muted("SmartCode compares the two versions you provide; it does not rewrite code and does not check that "
             "both versions behave the same.")
