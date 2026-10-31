"""Streamlit entry point.

Run with:  streamlit run app.py

Phase 1: a placeholder page that confirms the environment works.
The dashboard is built in later phases on top of ``src.predictor``.
"""

import streamlit as st

from src import __version__
from src.config import MODEL_PATH

st.set_page_config(
    page_title="Smart Code Complexity Predictor",
    page_icon="🧮",
    layout="wide",
)

st.title("Smart Code Complexity Predictor")
st.caption(f"v{__version__} · Phase 1 scaffold · static analysis only - code is never executed")

st.success("Environment OK: Streamlit is running and the `src` package imports correctly.")

model_status = "found" if MODEL_PATH.exists() else "not trained yet"
st.info(f"Model file `{MODEL_PATH.name}`: {model_status}.")
