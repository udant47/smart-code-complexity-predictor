"""Results dashboard for one analysed program (used by the analyzer page)."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.ui import charts, logic, services
from src.ui import components as ui
from src.ui.styles import TOKENS


def render(result: dict, key: str = "main") -> None:
    ml_issues, other_errors = logic.split_issues(result.get("errors"))
    for e in other_errors:
        ui.callout("error", e.get("message", ""), e.get("hint") or "")
    if ml_issues:
        e = ml_issues[0]
        ui.callout("warning", "ML prediction unavailable: " + logic.ML_ERROR_TITLES.get(e["code"], e["message"]),
                   (e.get("message", "") + " " + (e.get("hint") or "")).strip()
                   + " The static analysis below does not need the model and is complete.")

    _overview(result)

    tabs = st.tabs(["Functions", "Algorithm analysis", "ML prediction & explainability", "Refactoring", "All metrics"])
    with tabs[0]:
        _functions(result, key)
    with tabs[1]:
        _algorithms(result)
    with tabs[2]:
        _ml(result, ml_issues)
    with tabs[3]:
        _refactoring(result)
    with tabs[4]:
        _all_metrics(result)

    notes = [w for w in result.get("warnings") or [] if w.get("code") not in ("low_confidence",)]
    if notes:
        with st.expander(f"Notes from the analysis ({len(notes)})"):
            for w in notes:
                ui.muted("• " + w.get("message", ""))


# --------------------------------------------------------------------------- #
def _overview(result: dict) -> None:
    o = logic.overview(result)
    pred = result.get("prediction")
    ui.section("Overview", "Structural measures (rule-based) and the model's time-complexity class are shown separately.")
    if pred:
        ml_value = pred["big_o"]
        acc = pred.get("measured_accuracy_in_range")
        ml_sub = f"vote share {pred['probability']:.2f}" + (
            f" · {acc['test_accuracy']:.0%} correct at this level on test" if acc else "")
    else:
        ml_value, ml_sub = "Unavailable", "model not loaded"
    struct_sub = ("rule: cyclomatic 1-5 Low, 6-10 Medium, 11+ High · "
                  + ("most complex function" if o["structural_scope"] == "most complex function" else "module level"))
    score_sub = (f"{o['derived_band']} · highest {'function' if o['derived_scope'] == 'functions' else 'module'}"
                 if o["derived_score"] is not None else "not enough metrics")
    ui.cards([
        ui.card("Structural complexity", ui.band_badge(o["structural_category"]).replace("sc-badge", "sc-badge big"),
                struct_sub, accent=True, value_html=True),
        ui.card("Derived Complexity Score", ui.fmt(o["derived_score"], 1), score_sub, accent=True),
        ui.card("ML time-complexity class", ml_value, ml_sub, accent=True),
    ])
    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
    ui.cards([
        ui.card("Code lines", o["code_lines"], f"{o['total_lines']} total lines"),
        ui.card("Functions", o["function_count"], "incl. methods and nested"),
        ui.card("Classes", o["class_count"]),
        ui.card("Loops", o["loop_count"], "for + while"),
        ui.card("Cyclomatic complexity", o["max_function_cyclomatic_complexity"] if o["max_function_cyclomatic_complexity"] is not None
                else o["file_cyclomatic_complexity"],
                f"highest function · file {o['file_cyclomatic_complexity']}" if o["max_function_cyclomatic_complexity"] is not None
                else "whole file"),
        ui.card("Max nesting depth", o["maximum_nesting_depth"], "deepest block nesting"),
    ])


# --------------------------------------------------------------------------- #
def _functions(result: dict, key: str) -> None:
    fns = result.get("functions") or []
    if not fns:
        ui.callout("info", "No functions found", "Function-level analysis needs at least one def. File-level results still apply.")
        return
    df = logic.functions_frame(result)
    st.dataframe(df, hide_index=True, **ui.STRETCH, column_config={
        "Derived score": st.column_config.ProgressColumn("Derived score", min_value=0, max_value=100, format="%.1f"),
        "Structural category": st.column_config.TextColumn("Structural category", help="Rule-based: cyclomatic 1-5 Low, 6-10 Medium, 11+ High"),
        "Time (static)": st.column_config.TextColumn("Time (static)", help="Rule-based Big-O estimate; Unknown when the analyser cannot tell"),
        "Aux. space (static)": st.column_config.TextColumn("Aux. space (static)", help="Extra memory beyond the input; Unknown when the analyser cannot tell"),
    })
    ids = [f["function_id"] for f in fns]
    chosen = st.selectbox("Inspect a function", ids, key=f"{key}_fn")
    f = next(x for x in fns if x["function_id"] == chosen)
    _function_detail(f)


def _function_detail(f: dict) -> None:
    m, ds, est = f.get("metrics") or {}, f.get("derived_score") or {}, f.get("static_estimate") or {}
    ui.cards([
        ui.card("Lines", f"{f['start_line']}–{f['end_line']}", f"{m.get('function_length')} lines · {f.get('kind')}"),
        ui.card("Cyclomatic", m.get("cyclomatic_complexity"), f"category {f.get('structural_category')}"),
        ui.card("Nesting depth", m.get("maximum_nesting_depth")),
        ui.card("Parameters", m.get("parameter_count")),
        ui.card("Loops / conditionals", f"{m.get('loop_count')} / {m.get('conditional_count')}"),
        ui.card("Recursive calls", m.get("recursive_call_count")),
    ])
    left, right = st.columns([1.1, 1])
    with left:
        ui.section("Derived Complexity Score",
                   f"{ui.fmt(ds.get('score'), 1)} · {ds.get('band') or 'n/a'} · "
                   + (f"higher than {ds['percentile']:.0f}% of reference functions" if ds.get("percentile") is not None else ""))
        if ds.get("score") is not None:
            st.plotly_chart(charts.score_components(ds.get("components") or []), **ui.STRETCH,
                            config={"displayModeBar": False})
        for w in ds.get("warnings") or []:
            ui.muted(w)
        ui.muted(ds.get("disclaimer", ""))
    with right:
        ui.section("Static Big-O estimate", "Rule-based analysis of this function's code.")
        _estimate_block(est)
        ml = f.get("ml_prediction")
        if ml:
            ui.section("ML class (experimental)", "The model was trained on whole programs, not single functions.")
            st.markdown(ui.kv("Class", ml["big_o"]) + " &nbsp; " + ui.kv("vote share", f"{ml['probability']:.2f}"),
                        unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
def _estimate_block(est: dict, nested: bool = False) -> None:
    """nested=True when already inside an expander (older Streamlit versions forbid nested expanders)."""
    if not est:
        ui.callout("info", "Unsupported", "No static estimate is available for this function.")
        return
    st.markdown(
        f"{ui.status_badge(est.get('status'))} &nbsp; "
        + ui.kv("Time", est.get("time_complexity") or "Unknown")
        + (f" ({est['time_confidence']} confidence)" if est.get("time_confidence") else "") + " &nbsp; "
        + ui.kv("Aux. space", est.get("space_complexity") or "Unknown")
        + (f" ({est['space_confidence']} confidence)" if est.get("space_confidence") else ""),
        unsafe_allow_html=True)
    if est.get("explanation"):
        st.markdown(ui.esc(est["explanation"]))
    if est.get("size_variables"):
        ui.muted("Size variables: " + ", ".join(f"{k} = {v}" for k, v in est["size_variables"].items()))
    if est.get("recurrence"):
        ui.muted("Recurrence: " + est["recurrence"])
    if est.get("assumptions"):
        st.markdown("**Assumptions**")
        for a in est["assumptions"]:
            st.markdown(f"- {ui.esc(a)}")
    if est.get("evidence"):
        if nested:
            st.markdown("**Evidence**")
            for e in est["evidence"]:
                st.markdown(f"- `{ui.esc(e)}`")
        else:
            with st.expander("Evidence"):
                for e in est["evidence"]:
                    st.markdown(f"- `{ui.esc(e)}`")


def _algorithms(result: dict) -> None:
    fns = result.get("functions") or []
    ui.section("Algorithm analysis (static Big-O)",
               "Separate from structural complexity: a short function can be O(2^n) and a long one O(1). "
               "Estimates come from rules over the code, worst case, and say Unknown when the rules cannot decide.")
    if not fns:
        ui.callout("info", "Unsupported for this input", "Big-O estimates are made per function; this code has no functions.")
        return
    counts = {s: sum(1 for f in fns if (f.get("static_estimate") or {}).get("status") == s) for s in ("estimated", "partial", "unknown")}
    ui.muted(f"{counts['estimated']} estimated · {counts['partial']} partial · {counts['unknown']} unknown")
    for f in fns:
        est = f.get("static_estimate") or {}
        title = f"{f['function_id']}  ·  time {est.get('time_complexity', 'Unknown')}  ·  aux. space {est.get('space_complexity', 'Unknown')}"
        with st.expander(title, expanded=len(fns) <= 2):
            _estimate_block(est, nested=True)


# --------------------------------------------------------------------------- #
def _ml(result: dict, ml_issues: list) -> None:
    pred, model = result.get("prediction"), result.get("model") or {}
    if not pred:
        ui.callout("warning", "No ML prediction for this input",
                   logic.ML_ERROR_TITLES.get(ml_issues[0]["code"], "") if ml_issues else "The model is unavailable.")
    else:
        ui.section("ML time-complexity class (whole file)",
                   "A Random Forest trained on CodeComplex expert labels. A second opinion, not proof.")
        c1, c2 = st.columns([1, 1.2])
        with c1:
            acc = pred.get("measured_accuracy_in_range")
            ui.cards([
                ui.card("Predicted class", pred["big_o"], f"runner-up {pred['runner_up']} (margin {pred['margin']:.2f})", accent=True),
                ui.card("Vote share", f"{pred['probability']:.2f}", "share of trees; not a calibrated probability"),
                ui.card("Measured accuracy at this level",
                        f"{acc['test_accuracy']:.0%}" if acc else "—",
                        f"{acc['test_predictions']} test programs with vote share {acc['probability_range'][0]:.1f}–{acc['probability_range'][1]:.1f}"
                        if acc else "not recorded"),
            ])
            if pred["probability"] < 0.4:
                ui.callout("warning", "Low confidence", "The model is unsure; treat this class as a weak hint and prefer the static estimate.")
            if model.get("test_macro_f1") is not None:
                ci = model.get("test_macro_f1_ci") or [None, None]
                ui.muted(f"Model: {model.get('model')} · test macro F1 {model['test_macro_f1']:.3f} "
                         f"(95% CI {ci[0]:.2f}–{ci[1]:.2f}) · accuracy {model.get('test_accuracy', 0):.3f}")
            ui.muted(model.get("scope") or "")
        with c2:
            st.plotly_chart(charts.probabilities_bar(pred["probabilities"], pred["big_o"]), **ui.STRETCH,
                            config={"displayModeBar": False})

    exp = result.get("explanation")
    ui.section("Why this prediction (per-prediction explanation)",
               "How the model's trees moved the vote for this input. It explains the model, not the code's real complexity.")
    if not exp or not exp.get("available"):
        msg = (exp or {}).get("error") or {}
        ui.callout("info", "Explanation unavailable", msg.get("message", "No model prediction to explain."))
    else:
        c1, c2 = st.columns([1.3, 1])
        with c1:
            st.plotly_chart(charts.contributions_bar(exp["contributions"], top=10,
                                                     title=f"Top features for {exp['explained_class']}"),
                            **ui.STRETCH, config={"displayModeBar": False})
            ui.muted("Blue bars moved the vote toward the class; orange bars moved it away. Values are in vote-share points.")
        with c2:
            for s in exp.get("summary") or []:
                st.markdown(ui.esc(s))
            for n in exp.get("notes") or []:
                ui.callout("info", "Note", n)
            ui.muted("Method: " + exp.get("method_description", ""))
        with st.expander("All feature contributions"):
            df = pd.DataFrame([{"Feature": c["label"], "Group": c["group"], "Value": c["value"],
                                "Contribution (points)": round(c["contribution"], 2), "Direction": c["direction"],
                                "Training programs with lower value (%)": None if c["share_of_training_below"] is None
                                else round(c["share_of_training_below"]),
                                "Test importance (F1 drop)": c.get("test_permutation_importance")}
                               for c in exp["contributions"]])
            st.dataframe(df, hide_index=True, **ui.STRETCH)
        with st.expander("Caveats"):
            for c in exp.get("caveats") or []:
                st.markdown(f"- {ui.esc(c)}")

    ui.section("Global feature importance", "Measured on the held-out test set; computed by python -m src.explainability.")
    g = services.global_explanation()
    if not g["available"]:
        ui.callout("info", "Global importance not computed yet", g["message"])
        return
    if g["stale"]:
        ui.callout("warning", "Out of date", g["message"])
    from src.explainability import global_importance_table, group_importance_table

    data = g["data"]
    c1, c2 = st.columns(2)
    with c1:
        perm = global_importance_table(data, "permutation", top=15)
        st.plotly_chart(charts.importance_bar(perm, "importance", "label", "std", "Permutation importance (top 15)",
                                              "Drop in test macro F1"),
                        **ui.STRETCH, config={"displayModeBar": False})
    with c2:
        grp = group_importance_table(data)
        st.plotly_chart(charts.importance_bar(grp, "importance", "group", "std", "Grouped permutation importance",
                                              "Drop in test macro F1"),
                        **ui.STRETCH, config={"displayModeBar": False})
    for c in data.get("caveats") or []:
        ui.muted("• " + c)
    with st.expander("Feature definitions"):
        st.dataframe(services.feature_catalogue()[["label", "group", "kind", "description"]],
                     hide_index=True, **ui.STRETCH)


# --------------------------------------------------------------------------- #
def _refactoring(result: dict) -> None:
    rec = result.get("recommendations") or {}
    ui.section("Refactoring review guidance",
               "Deterministic rules over measured metrics. Each item quotes its evidence. Not automatic fixes.")
    if not rec.get("success"):
        ui.callout("info", "No recommendations", (rec.get("error") or {}).get("message", "Analysis unavailable."))
        return
    items = rec.get("recommendations") or []
    c = rec.get("counts") or {}
    ui.muted(f"{len(items)} recommendation(s): {c.get('high', 0)} high · {c.get('medium', 0)} medium · {c.get('low', 0)} low · "
             f"{len(rec.get('rules_checked') or [])} rules checked")
    if not items:
        ui.callout("success", "No rule found evidence above its threshold",
                   "That does not prove the code is well designed; it means none of the measured signals stood out.")
    for r in items:
        ui.recommendation(r)
    ui.muted(rec.get("disclaimer", ""))


def _all_metrics(result: dict) -> None:
    fm = result.get("file_metrics") or {}
    if not fm:
        return
    cat = services.feature_catalogue().set_index("feature")
    df = pd.DataFrame([{"Feature": cat.loc[k, "label"] if k in cat.index else k, "Value": v,
                        "Group": cat.loc[k, "group"] if k in cat.index else "",
                        "Definition": cat.loc[k, "description"] if k in cat.index else ""} for k, v in fm.items()])
    ui.section("All file-level features", "The 57 static features the model receives, with their definitions.")
    st.dataframe(df, hide_index=True, **ui.STRETCH)
