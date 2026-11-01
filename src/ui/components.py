"""Reusable presentation widgets for the SmartCode dashboard.

All values shown come from already-computed backend results. Every piece of
user-derived text (function names, code, messages) is HTML-escaped before it
is placed in custom HTML.
"""

from __future__ import annotations

import html
from typing import Iterable, Optional

import streamlit as st

from src.ui.styles import BAND_COLORS, SEVERITY_COLORS, TOKENS, css

esc = html.escape


def _stretch_kwargs() -> dict:
    """Full-width keyword for st.dataframe / st.plotly_chart / st.button across Streamlit versions:
    ``width="stretch"`` from 1.50 (where use_container_width is deprecated), else use_container_width."""
    try:
        major, minor = (int(x) for x in st.__version__.split(".")[:2])
    except ValueError:
        return {"use_container_width": True}
    return {"width": "stretch"} if (major, minor) >= (1, 50) else {"use_container_width": True}


STRETCH = _stretch_kwargs()


def inject_theme() -> None:
    st.markdown(css(), unsafe_allow_html=True)


def brand() -> None:
    st.markdown(
        '<div class="sc-brand"><div class="sc-logo">{ }</div><div>'
        '<div class="sc-brand-name">SmartCode</div>'
        '<div class="sc-brand-sub">Intelligent Static Code Analysis</div></div></div>',
        unsafe_allow_html=True,
    )


def hero(title: str, subtitle: str) -> None:
    st.markdown(f'<div class="sc-hero"><h1>{esc(title)}</h1><p>{esc(subtitle)}</p></div>', unsafe_allow_html=True)


def section(title: str, description: str = "") -> None:
    d = f'<div class="d">{esc(description)}</div>' if description else ""
    st.markdown(f'<div class="sc-section"><div class="t">{esc(title)}</div>{d}</div>', unsafe_allow_html=True)


def badge(text: str, color: Optional[str] = None) -> str:
    color = color or TOKENS["text_muted"]
    return f'<span class="sc-badge" style="color:{color}">{esc(str(text))}</span>'


def band_badge(band: Optional[str]) -> str:
    if not band:
        return badge("n/a")
    return badge(band, BAND_COLORS.get(band, TOKENS["text_muted"]))


def severity_badge(sev: str) -> str:
    return badge(sev.upper(), SEVERITY_COLORS.get(sev, TOKENS["text_muted"]))


def status_badge(status: Optional[str]) -> str:
    colors = {"estimated": TOKENS["success"], "partial": TOKENS["warning"], "unknown": TOKENS["text_muted"]}
    label = {"estimated": "Estimated", "partial": "Partial", "unknown": "Unknown"}.get(status or "unknown", str(status))
    return badge(label, colors.get(status or "unknown", TOKENS["text_muted"]))


def card(label: str, value, sub: str = "", accent: bool = False, value_html: bool = False) -> str:
    v = value if value_html else esc("—" if value is None else str(value))
    return (f'<div class="sc-card{" accent" if accent else ""}"><div class="l">{esc(label)}</div>'
            f'<div class="v">{v}</div><div class="s">{esc(sub) if sub else "&nbsp;"}</div></div>')


def cards(items: Iterable[str]) -> None:
    st.markdown('<div class="sc-grid">' + "".join(items) + "</div>", unsafe_allow_html=True)


def callout(kind: str, title: str, body: str = "") -> None:
    """kind: info | warning | error | success. Title and body are escaped."""
    b = f'<div class="b">{esc(body)}</div>' if body else ""
    st.markdown(f'<div class="sc-callout {kind}"><div class="h">{esc(title)}</div>{b}</div>', unsafe_allow_html=True)


def issues(items: list, kind: str = "warning") -> None:
    """Render backend Issue dicts ({code, message, hint})."""
    for it in items or []:
        body = it.get("hint") or ""
        callout(kind, it.get("message", ""), body)


def muted(text: str) -> None:
    st.markdown(f'<div class="sc-muted">{esc(text)}</div>', unsafe_allow_html=True)


def kv(label: str, value) -> str:
    return f'<span class="sc-kv">{esc(label)}: <b>{esc("—" if value is None else str(value))}</b></span>'


def recommendation(rec: dict) -> None:
    ev = []
    for e in rec.get("evidence", []):
        val = e.get("value")
        if isinstance(val, dict):
            val = ", ".join(f"{k} {v}" for k, v in val.items())
        elif isinstance(val, list):
            val = "; ".join(str(v) for v in val)
        thr = f" (threshold {e['threshold']})" if e.get("threshold") is not None else ""
        ev.append(f"<li>{esc(str(e.get('metric')))}: <code>{esc(str(val))}</code>{esc(thr)}</li>")
    st.markdown(
        f'<div class="sc-rec"><div class="top">{severity_badge(rec["severity"])}'
        f'<span class="title">{esc(rec["title"])}</span>'
        f'<span class="meta">{esc(rec["rule_id"])} · {esc(rec["target"])} · lines {rec["start_line"]}-{rec["end_line"]}</span></div>'
        f'<div class="ev"><ul style="margin:0 0 0 18px;padding:0">{"".join(ev)}</ul></div>'
        f'<div class="sc-muted">{esc(rec["explanation"])}</div>'
        f'<div class="act" style="margin-top:6px"><b>Suggested review:</b> {esc(rec["suggested_action"])}</div></div>',
        unsafe_allow_html=True,
    )


def footer() -> None:
    st.markdown('<div class="sc-footer">SmartCode analyses code statically with Python\'s <code>ast</code> module. '
                "Submitted code is never executed, imported or evaluated.</div>", unsafe_allow_html=True)


def fmt(v, digits: int = 2) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.{digits}f}"
    return str(v)
