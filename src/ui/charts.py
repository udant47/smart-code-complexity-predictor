"""Plotly figure builders. Pure functions of already-computed data (no Streamlit)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from src.ui.styles import TOKENS, plotly_layout

T = TOKENS


def probabilities_bar(probs: dict, predicted: str) -> go.Figure:
    labels = list(probs)
    values = [probs[k] for k in labels]
    colors = [T["accent"] if k == predicted else T["neutral"] for k in labels]
    fig = go.Figure(go.Bar(x=labels, y=values, marker_color=colors, marker_line_width=0,
                           text=[f"{v:.2f}" for v in values], textposition="outside",
                           textfont=dict(color=T["text_muted"], size=11),
                           hovertemplate="%{x}: vote share %{y:.3f}<extra></extra>"))
    fig.update_yaxes(range=[0, max(values + [0.01]) * 1.25], title="Vote share (not calibrated)")
    return plotly_layout(fig, height=260, legend=False)


def contributions_bar(contribs: list, top: int = 10, title: str = "") -> go.Figure:
    """Signed per-feature contributions (vote-share points)."""
    items = sorted(contribs, key=lambda c: abs(c["contribution"]), reverse=True)[:top][::-1]
    labels = [f"{c['label']} = {c['value']:g}" for c in items]
    values = [c["contribution"] for c in items]
    fig = go.Figure(go.Bar(
        x=values, y=labels, orientation="h", marker_line_width=0,
        marker_color=[T["pos"] if v > 0 else T["neg"] for v in values],
        customdata=[[c["group"], c["direction"]] for c in items],
        hovertemplate="%{y}<br>%{x:+.2f} points %{customdata[1]} the class<br>group: %{customdata[0]}<extra></extra>"))
    fig.add_vline(x=0, line_color=T["border_strong"], line_width=1)
    fig.update_xaxes(title="Contribution (vote-share points)")
    fig.update_layout(title=title)
    return plotly_layout(fig, height=40 + 30 * len(items), legend=False)


def importance_bar(df: pd.DataFrame, value: str, label: str, error: str = None, title: str = "",
                   xaxis: str = "", top: int = 15) -> go.Figure:
    d = df.head(top).iloc[::-1]
    fig = go.Figure(go.Bar(
        x=d[value], y=d[label], orientation="h", marker_color=T["accent"], marker_line_width=0,
        error_x=dict(type="data", array=d[error], color=T["text_faint"], thickness=1, width=3) if error else None,
        hovertemplate="%{y}: %{x:.4f}<extra></extra>"))
    fig.add_vline(x=0, line_color=T["border_strong"], line_width=1)
    fig.update_xaxes(title=xaxis)
    fig.update_layout(title=title)
    return plotly_layout(fig, height=60 + 26 * len(d), legend=False)


def score_components(components: list) -> go.Figure:
    """Derived Complexity Score: contribution of each component vs its maximum (weight x 100)."""
    comps = [c for c in components if c.get("normalized") is not None]
    names = [c["name"] for c in comps][::-1]
    got = [c["contribution"] for c in comps][::-1]
    cap = [c["effective_weight"] * 100 for c in comps][::-1]
    fig = go.Figure()
    fig.add_bar(x=cap, y=names, orientation="h", marker_color=T["surface_2"], marker_line_color=T["border_strong"],
                marker_line_width=1, name="Maximum possible", hovertemplate="%{y}: max %{x:.0f}<extra></extra>")
    fig.add_bar(x=got, y=names, orientation="h", marker_color=T["accent"], marker_line_width=0, name="This function",
                hovertemplate="%{y}: %{x:.1f} points<extra></extra>")
    fig.update_layout(barmode="overlay")
    fig.update_xaxes(title="Points", range=[0, max(cap + [1]) * 1.08])
    return plotly_layout(fig, height=60 + 30 * len(names))


def model_comparison(df: pd.DataFrame, selected: str = None, selected_features: str = None) -> go.Figure:
    """Test macro F1 per candidate; the saved model (name and feature set, from its metadata) is highlighted."""
    d = df.iloc[::-1]
    labels = d["Model"] + "  [" + d["Features"] + "]"
    colors = [T["accent"] if selected and m == selected and (selected_features is None or f == selected_features)
              else T["neutral"] for m, f in zip(d["Model"], d["Features"])]
    fig = go.Figure(go.Bar(
        x=d["Test macro F1"], y=labels, orientation="h", marker_color=colors, marker_line_width=0,
        error_x=dict(type="data", symmetric=False, array=d["CI high"] - d["Test macro F1"],
                     arrayminus=d["Test macro F1"] - d["CI low"], color=T["text_faint"], thickness=1, width=3),
        hovertemplate="%{y}<br>test macro F1 %{x:.3f}<extra></extra>"))
    fig.update_xaxes(title="Test macro F1 (95% problem-bootstrap interval)", range=[0, 0.7])
    return plotly_layout(fig, height=80 + 26 * len(d), legend=False)


def confusion_heatmap(cm: pd.DataFrame, normalize: bool = True) -> go.Figure:
    z = cm.to_numpy(dtype=float)
    shown = z / np.clip(z.sum(axis=1, keepdims=True), 1, None) if normalize else z
    text = [[f"{v:.0%}" if normalize else f"{int(v)}" for v in row] for row in shown]
    scale = [[i / (len(T["seq"]) - 1), c] for i, c in enumerate(T["seq"])]
    fig = go.Figure(go.Heatmap(
        z=shown, x=list(cm.columns), y=list(cm.index), colorscale=scale, showscale=False, xgap=2, ygap=2,
        customdata=z, hovertemplate="true %{y}, predicted %{x}<br>%{customdata:.0f} programs<extra></extra>"))
    top = shown.max() or 1
    for i, row_label in enumerate(cm.index):  # dark text on light cells, light text on dark cells
        for j, col_label in enumerate(cm.columns):
            light = shown[i, j] / top > 0.55
            fig.add_annotation(x=col_label, y=row_label, text=text[i][j], showarrow=False,
                               font=dict(size=11, color="#0b1220" if light else T["text"]))
    fig.update_yaxes(autorange="reversed", title="True class")
    fig.update_xaxes(title="Predicted class", side="bottom")
    return plotly_layout(fig, height=420, legend=False)


def simple_bar(x, y, title: str = "", yaxis: str = "", text_fmt: str = "{:.2f}", height: int = 300,
               color: str = None) -> go.Figure:
    fig = go.Figure(go.Bar(x=list(x), y=list(y), marker_color=color or T["accent"], marker_line_width=0,
                           text=[text_fmt.format(v) for v in y], textposition="outside",
                           textfont=dict(color=T["text_muted"], size=11)))
    fig.update_yaxes(title=yaxis, range=[0, max(list(y) + [0.01]) * 1.2])
    fig.update_layout(title=title)
    return plotly_layout(fig, height=height, legend=False)


def paired_bar(categories, a, b, name_a: str, name_b: str, yaxis: str = "", height: int = 320) -> go.Figure:
    fig = go.Figure()
    fig.add_bar(x=list(categories), y=list(a), name=name_a, marker_color=T["neutral"], marker_line_width=0)
    fig.add_bar(x=list(categories), y=list(b), name=name_b, marker_color=T["accent"], marker_line_width=0)
    fig.update_layout(barmode="group", bargap=0.25, bargroupgap=0.08)
    fig.update_yaxes(title=yaxis)
    return plotly_layout(fig, height=height)
