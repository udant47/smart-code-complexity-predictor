"""SmartCode - Intelligent Static Code Analysis (Streamlit entry point).

Run with:  streamlit run app.py

This file only sets up the page, the theme and navigation. Analysis is done
by the existing backend (``src.predictor`` and friends) through
``src.ui.services``; nothing is trained at startup and submitted code is
never executed.
"""

import logging

import streamlit as st

st.set_page_config(page_title="SmartCode", page_icon="{ }", layout="wide", initial_sidebar_state="expanded")

from src import __version__  # noqa: E402
from src.ui import components as ui  # noqa: E402
from src.ui import services  # noqa: E402
from src.ui.views import analyzer, compare, docs, evaluation  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

PAGES = {
    "Analyzer": analyzer.render,
    "Compare versions": compare.render,
    "Model evaluation": evaluation.render,
    "Documentation": docs.render,
}


def sidebar() -> str:
    with st.sidebar:
        ui.brand()
        st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
        page = st.radio("Navigation", list(PAGES), key="nav", label_visibility="collapsed")
        st.divider()
        status = services.model_status()
        if status["available"]:
            info = status["info"]
            st.markdown(ui.badge("Model loaded", "#3fb950"), unsafe_allow_html=True)
            ui.muted(f"{info['model']} · {info['n_features']} features")
            if info.get("test_macro_f1") is not None:
                ui.muted(f"Test macro F1 {info['test_macro_f1']:.3f} · accuracy {info['test_accuracy']:.3f}")
            for w in status.get("warnings") or []:
                ui.muted("⚠ " + w["message"])
        else:
            st.markdown(ui.badge("Model unavailable", "#d29922"), unsafe_allow_html=True)
            ui.muted(status["issue"]["message"])
            if status["issue"].get("hint"):
                ui.muted(status["issue"]["hint"])
            ui.muted("Static analysis still works.")
        st.divider()
        ui.muted("Python: full analysis · C: structural only (experimental) · code is parsed, never executed")
        ui.muted(f"v{__version__}")
    return page


def main() -> None:
    ui.inject_theme()
    page = sidebar()
    try:
        PAGES[page]()
    except Exception:  # last line of defence: no tracebacks in the page
        logging.getLogger("smartcode").exception("Page %s failed", page)
        ui.callout("error", "Something went wrong while showing this page.",
                   "The error was logged. Try reloading the page or analysing different code.")
    ui.footer()


main()
