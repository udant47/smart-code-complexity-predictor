"""Theme tokens, injected CSS and the Plotly chart style for the SmartCode dashboard.

One place for colours, spacing and typography. Presentation only: nothing in
this module analyses code.

Palette notes
-------------
* Dark neutral surfaces with subtle borders; one restrained blue accent.
* Status colours (success / warning / danger) are reserved for state and are
  always shown with a text label, never colour alone.
* Charts: one hue for magnitudes (sequential), two hues plus grey for signed
  values (diverging), muted text tokens for labels.
"""

from __future__ import annotations

TOKENS = {
    "bg": "#0d1117",
    "surface": "#141a22",
    "surface_2": "#1a212b",
    "border": "#262d38",
    "border_strong": "#333c4a",
    "text": "#e6edf3",
    "text_muted": "#9aa5b1",
    "text_faint": "#6b7684",
    "accent": "#5b9cf5",
    "accent_soft": "rgba(91, 156, 245, 0.12)",
    "success": "#3fb950",
    "warning": "#d29922",
    "danger": "#f0605a",
    "info": "#5b9cf5",
    # charts
    "seq": ["#14263f", "#1d3a63", "#295489", "#3a70b0", "#5b9cf5", "#8dbbf8", "#c3dbfb"],
    "pos": "#5b9cf5",  # toward / increase
    "neg": "#e08a4f",  # away from / decrease
    "neutral": "#4a5361",
    "font": "Inter, 'Segoe UI', system-ui, -apple-system, sans-serif",
    "mono": "'JetBrains Mono', 'Cascadia Code', Consolas, 'Courier New', monospace",
}

SEVERITY_COLORS = {"high": TOKENS["danger"], "medium": TOKENS["warning"], "low": TOKENS["text_muted"]}
BAND_COLORS = {
    "Low": TOKENS["success"], "Medium": TOKENS["warning"], "High": TOKENS["danger"],
    "Moderate": TOKENS["warning"], "Very high": TOKENS["danger"],
}


def css() -> str:
    t = TOKENS
    return f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"], .stApp, .stMarkdown, button, input, textarea, select {{
    font-family: {t['font']};
}}
.stApp {{ background: {t['bg']}; color: {t['text']}; }}
.block-container {{ padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1280px; }}
section[data-testid="stSidebar"] {{ background: {t['surface']}; border-right: 1px solid {t['border']}; }}
header[data-testid="stHeader"] {{ background: transparent; }}
h1, h2, h3, h4 {{ color: {t['text']}; letter-spacing: -0.01em; }}
h2 {{ font-size: 1.25rem !important; font-weight: 600 !important; }}
h3 {{ font-size: 1.05rem !important; font-weight: 600 !important; }}
code, pre, .stCode, textarea {{ font-family: {t['mono']} !important; }}
p, li {{ color: {t['text']}; }}

/* code input */
.stTextArea textarea {{
    background: {t['surface']} !important; border: 1px solid {t['border']} !important;
    border-radius: 8px !important; font-size: 13px !important; line-height: 1.55 !important;
    color: {t['text']} !important;
}}
.stTextArea textarea:focus {{ border-color: {t['accent']} !important; box-shadow: none !important; }}

/* tabs */
.stTabs {{ margin-top: 18px; }}
.stTabs [data-baseweb="tab-list"] {{ gap: 4px; border-bottom: 1px solid {t['border']}; }}
.stTabs [data-baseweb="tab"] {{ padding: 8px 14px; color: {t['text_muted']}; }}
.stTabs [aria-selected="true"] {{ color: {t['text']} !important; }}

/* buttons */
.stButton button, .stDownloadButton button {{
    border-radius: 6px; border: 1px solid {t['border_strong']}; background: {t['surface_2']};
    color: {t['text']}; font-weight: 500;
}}
.stButton button[kind="primary"] {{ background: {t['accent']}; border-color: {t['accent']}; color: #0b1220; }}
.stButton button:hover {{ border-color: {t['accent']}; }}

/* dataframes and expanders */
[data-testid="stDataFrame"], [data-testid="stExpander"] {{
    border: 1px solid {t['border']}; border-radius: 8px;
}}
[data-testid="stExpander"] details {{ background: {t['surface']}; border-radius: 8px; }}

/* brand */
.sc-brand {{ display: flex; align-items: center; gap: 10px; margin-bottom: 2px; }}
.sc-logo {{
    width: 30px; height: 30px; border-radius: 7px; background: {t['accent_soft']};
    border: 1px solid {t['accent']}; color: {t['accent']}; display: flex; align-items: center;
    justify-content: center; font-family: {t['mono']}; font-weight: 600; font-size: 13px;
}}
.sc-brand-name {{ font-weight: 700; font-size: 1.05rem; color: {t['text']}; }}
.sc-brand-sub {{ color: {t['text_muted']}; font-size: 0.78rem; margin-top: -2px; }}

/* hero */
.sc-hero {{ padding: 6px 0 18px 0; border-bottom: 1px solid {t['border']}; margin-bottom: 20px; }}
.sc-hero h1 {{ font-size: 1.9rem; font-weight: 700; margin: 0 0 6px 0; padding: 0; }}
.sc-hero p {{ color: {t['text_muted']}; font-size: 0.98rem; margin: 0; max-width: 760px; }}

/* section header */
.sc-section {{ margin: 26px 0 10px 0; }}
.sc-section .t {{ font-size: 1.08rem; font-weight: 600; color: {t['text']}; }}
.sc-section .d {{ font-size: 0.86rem; color: {t['text_muted']}; margin-top: 2px; }}

/* metric cards */
.sc-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 10px; }}
.sc-card {{
    background: {t['surface']}; border: 1px solid {t['border']}; border-radius: 8px;
    padding: 12px 14px; min-height: 92px;
}}
.sc-card .l {{ font-size: 0.74rem; color: {t['text_muted']}; text-transform: uppercase; letter-spacing: 0.04em; }}
.sc-card .v {{ font-size: 1.45rem; font-weight: 600; color: {t['text']}; margin-top: 4px; font-variant-numeric: tabular-nums; }}
.sc-card .s {{ font-size: 0.78rem; color: {t['text_faint']}; margin-top: 3px; line-height: 1.35; }}
.sc-card.accent {{ border-color: {t['border_strong']}; box-shadow: inset 3px 0 0 {t['accent']}; }}

/* badges and pills */
.sc-badge {{
    display: inline-block; padding: 1px 8px; border-radius: 999px; font-size: 0.74rem; font-weight: 500;
    border: 1px solid currentColor; line-height: 1.6; white-space: nowrap;
}}
.sc-badge.big {{ font-size: 1.05rem; padding: 2px 14px; font-weight: 600; }}
.sc-kv {{ color: {t['text_muted']}; font-size: 0.85rem; }}
.sc-kv b {{ color: {t['text']}; font-weight: 500; }}

/* callouts */
.sc-callout {{
    border: 1px solid {t['border']}; border-left-width: 3px; border-radius: 6px; padding: 10px 14px;
    background: {t['surface']}; margin: 8px 0; font-size: 0.9rem; color: {t['text']};
}}
.sc-callout .h {{ font-weight: 600; margin-bottom: 2px; }}
.sc-callout .b {{ color: {t['text_muted']}; }}
.sc-callout.info {{ border-left-color: {t['info']}; }}
.sc-callout.warning {{ border-left-color: {t['warning']}; }}
.sc-callout.error {{ border-left-color: {t['danger']}; }}
.sc-callout.success {{ border-left-color: {t['success']}; }}

/* recommendation card */
.sc-rec {{ background: {t['surface']}; border: 1px solid {t['border']}; border-radius: 8px; padding: 12px 14px; margin-bottom: 10px; }}
.sc-rec .top {{ display: flex; gap: 8px; align-items: center; flex-wrap: wrap; }}
.sc-rec .title {{ font-weight: 600; }}
.sc-rec .meta {{ color: {t['text_faint']}; font-size: 0.78rem; font-family: {t['mono']}; }}
.sc-rec .ev {{ margin: 8px 0 4px 0; font-size: 0.84rem; color: {t['text_muted']}; }}
.sc-rec .ev code {{ font-size: 0.8rem; }}
.sc-rec .act {{ font-size: 0.88rem; }}

.sc-grid + div, .sc-grid {{ margin-bottom: 6px; }}
.stMarkdown p code, .sc-rec code, li code {{
    color: {t['text']} !important; background: {t['surface_2']} !important; border: 1px solid {t['border']};
    border-radius: 4px; padding: 0 4px;
}}
.sc-muted {{ color: {t['text_muted']}; font-size: 0.86rem; }}
.sc-footer {{ color: {t['text_faint']}; font-size: 0.78rem; margin-top: 40px; border-top: 1px solid {t['border']}; padding-top: 12px; }}
</style>
"""


def plotly_layout(fig, height: int = 340, legend: bool = True):
    """Apply the dashboard's chart style to a Plotly figure (returns the figure)."""
    t = TOKENS
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=t["font"], color=t["text_muted"], size=12),
        height=height,
        margin=dict(l=10, r=10, t=36, b=10),
        showlegend=legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, font=dict(color=t["text_muted"])),
        hoverlabel=dict(bgcolor=t["surface_2"], bordercolor=t["border_strong"], font=dict(color=t["text"], family=t["font"])),
        title=dict(font=dict(color=t["text"], size=13), x=0, xanchor="left"),
    )
    fig.update_xaxes(gridcolor=t["border"], zerolinecolor=t["border_strong"], linecolor=t["border"], tickfont=dict(color=t["text_muted"]))
    fig.update_yaxes(gridcolor=t["border"], zerolinecolor=t["border_strong"], linecolor=t["border"], tickfont=dict(color=t["text_muted"]))
    return fig
