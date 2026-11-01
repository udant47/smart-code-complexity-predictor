"""Main analyzer page: code input, validation and the results dashboard."""

from __future__ import annotations

import streamlit as st

from src.ui import logic, services
from src.ui import components as ui
from src.ui.samples import SAMPLES
from src.ui.views import results

CODE_KEY = "analyzer_code"
RESULT_KEY = "analyzer_result"
SAMPLE_PLACEHOLDER = "Choose a sample…"


def _load_sample():
    name = st.session_state.get("analyzer_sample")
    if name and name in SAMPLES:
        st.session_state[CODE_KEY] = SAMPLES[name]
        st.session_state.pop(RESULT_KEY, None)


def _clear():
    st.session_state[CODE_KEY] = ""
    st.session_state["analyzer_sample"] = SAMPLE_PLACEHOLDER
    st.session_state.pop(RESULT_KEY, None)


def render() -> None:
    ui.hero("Analyze Code. Understand Complexity.",
            "Explore structural complexity, function-level insights, and algorithmic analysis through static code inspection.")

    st.session_state.setdefault(CODE_KEY, SAMPLES["Merge sort (divide and conquer)"])
    c1, c2, c3 = st.columns([1, 2, 3])
    with c1:
        language = st.selectbox("Language", list(logic.LANGUAGES), key="analyzer_language",
                                help="Python is fully supported. Other languages are listed but not supported yet.")
    with c2:
        st.selectbox("Sample", [SAMPLE_PLACEHOLDER, *SAMPLES], key="analyzer_sample", on_change=_load_sample)
    with c3:
        ui.muted(logic.LANGUAGES[language]["note"])

    code = st.text_area("Python code", key=CODE_KEY, height=360, label_visibility="collapsed",
                        placeholder="Paste Python code here. It is parsed only, never executed.")
    check = logic.check_input(code, language)

    left, mid, right = st.columns([3, 1, 1])
    with left:
        limit = f"limit {logic.config.MAX_SOURCE_LINES:,} lines / {logic.config.MAX_SOURCE_BYTES // 1000:,} KB"
        st.markdown(ui.kv("Lines", f"{check.lines:,}") + " &nbsp;·&nbsp; " + ui.kv("Characters", f"{check.chars:,}")
                    + f' &nbsp;<span class="sc-muted">({limit})</span>', unsafe_allow_html=True)
    with mid:
        st.button("Clear", on_click=_clear, **ui.STRETCH)
    with right:
        run = st.button("Analyze", type="primary", **ui.STRETCH,
                        disabled=not logic.LANGUAGES[language]["supported"])

    if not logic.LANGUAGES[language]["supported"]:
        ui.callout("warning", f"{language} is not supported", "SmartCode analyses Python only. No results are produced "
                   "for other languages rather than showing results that would be wrong.")
    for w in check.warnings:
        ui.callout("info", w["message"])

    if run:
        if not check.ok:
            for e in check.errors:
                ui.callout("error", e["message"], e.get("hint") or "")
            st.session_state.pop(RESULT_KEY, None)
            return
        with st.spinner("Analysing…"):
            st.session_state[RESULT_KEY] = {"code": code, **services.analyze(code)}

    outcome = st.session_state.get(RESULT_KEY)
    if not outcome:
        ui.callout("info", "Ready", "Choose a sample or paste code, then press Analyze.")
        return
    if outcome["code"] != code:
        ui.callout("info", "The code has changed since the last analysis", "Press Analyze to update the results.")
    if not outcome["ok"]:
        e = outcome["error"]
        ui.callout("error", e["message"], e.get("hint") or "")
        return
    res = outcome["result"]
    if not res["success"]:
        for e in res.get("errors") or []:
            title = {"syntax": "Invalid Python syntax", "indentation": "Invalid indentation",
                     "invalid_characters": "Invalid characters in the code", "empty": "No code to analyse",
                     "too_large": "Input too large", "too_complex": "Code too deeply nested to analyse",
                     "encoding": "Unreadable text encoding"}.get(e.get("code"), "The code could not be analysed")
            ui.callout("error", title, f"{e.get('message', '')} {e.get('hint') or ''}".strip())
        return
    results.render(res, key="analyzer")
