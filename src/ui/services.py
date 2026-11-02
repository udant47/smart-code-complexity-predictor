"""Cached calls from the Streamlit app into the existing backend.

The app never trains a model and never duplicates analysis logic: it calls
``src.predictor.predict`` (which parses, measures, estimates, scores, runs
the saved model, explains and recommends) and reads saved artifacts.

Caching
-------
* ``analyze`` is cached per (code, model-file timestamps), so re-rendering a
  page does not re-run the analysis, and retraining the model invalidates it.
* The model itself is cached by ``predictor.get_model`` (loaded once per
  process, reloaded if the files change).
* Saved reports are cached per file timestamps.

Errors
------
Unexpected exceptions are logged with their traceback for the developer and
turned into a short message with a reference id for the user; tracebacks
are never shown in the page.
"""

from __future__ import annotations

import logging
import uuid

import streamlit as st

from src import config
from src.ui import logic

log = logging.getLogger("smartcode")


def _model_stamp() -> tuple:
    from src.predictor import PREPROCESSING_PATH

    paths = (config.MODEL_PATH, config.MODEL_METADATA_PATH, PREPROCESSING_PATH, logic.ARTIFACTS["explainability"])
    return tuple(p.stat().st_mtime_ns if p.exists() else 0 for p in paths)


def _failure(where: str, exc: Exception) -> dict:
    ref = uuid.uuid4().hex[:8]
    log.exception("SmartCode %s failed (ref %s)", where, ref)
    return {"code": "analysis_failed", "message": f"The {where} could not be completed because of an internal error.",
            "hint": f"Try simplifying the input. If it keeps happening, report reference {ref} "
                    f"(error type: {type(exc).__name__})."}


def _analyze_uncached(code: str) -> dict:
    from src.predictor import predict

    try:
        return {"ok": True, "result": predict(code, explain=True).to_dict(), "error": None}
    except Exception as exc:  # never let a traceback reach the page
        return {"ok": False, "result": None, "error": _failure("analysis", exc)}


@st.cache_data(show_spinner=False, max_entries=64)
def _analyze_cached(code: str, stamp: tuple) -> dict:
    return _analyze_uncached(code)


def analyze(code: str) -> dict:
    """{"ok", "result" (predictor dict), "error"}; ok=False only on an internal failure."""
    try:
        code.encode("utf-8")
    except UnicodeEncodeError:
        # Text that is not valid Unicode (lone surrogates) cannot be hashed for the cache; the parser
        # rejects it with a friendly "encoding" error, so analyse it directly.
        return _analyze_uncached(code)
    except AttributeError:  # not a string
        return {"ok": False, "result": None, "error": _failure("analysis", TypeError("code must be text"))}
    return _analyze_cached(code, _model_stamp())


@st.cache_data(show_spinner=False)
def _model_status_cached(stamp: tuple) -> dict:
    from src.predictor import ModelUnavailable, get_model

    try:
        m = get_model()
    except ModelUnavailable as exc:
        return {"available": False, "issue": vars(exc.issue)}
    except Exception as exc:
        return {"available": False, "issue": _failure("model check", exc)}
    from dataclasses import asdict

    return {"available": True, "info": asdict(m.info), "warnings": [vars(w) for w in m.load_warnings]}


def model_status() -> dict:
    return _model_status_cached(_model_stamp())


@st.cache_data(show_spinner=False)
def _artifacts_cached(stamp: tuple) -> logic.Artifacts:
    return logic.load_artifacts()


def artifacts() -> logic.Artifacts:
    return _artifacts_cached(logic.artifact_stamp())


@st.cache_data(show_spinner=False)
def _global_explanation_cached(stamp: tuple) -> dict:
    from src.explainability import load_global_explanation

    try:
        g = load_global_explanation()
        return {"available": g.available, "stale": g.stale, "message": g.message, "data": g.data}
    except Exception as exc:
        return {"available": False, "stale": False, "message": _failure("explanation loading", exc)["message"], "data": None}


def global_explanation() -> dict:
    return _global_explanation_cached(_model_stamp())


def feature_catalogue():
    from src.explainability import feature_catalogue as fc

    return fc()
