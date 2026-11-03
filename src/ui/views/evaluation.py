"""Model evaluation dashboard: only results saved by the training and evaluation commands."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.ui import charts, logic, services
from src.ui import components as ui


def _missing(art: logic.Artifacts, name: str, what: str) -> bool:
    if name in art.errors:
        ui.callout("error", f"{what}: file could not be read", art.errors[name])
        return True
    if name not in art.data:
        ui.callout("info", f"{what}: not available", f"Create it with: {art.missing.get(name, '')}")
        return True
    return False


def render() -> None:
    ui.hero("Model Evaluation", "Saved results from the training and evaluation runs. Nothing on this page is computed "
            "live or estimated: if a result file is missing, the page says so.")
    art = services.artifacts()
    meta = art.data.get("model_metadata")
    ev = art.data.get("evaluation_results")

    # ------------------------------------------------------------------ headline
    ui.section("Final model on the held-out test set",
               "Test problems were never seen in training (grouped split by problem). The test set was used once.")
    if not _missing(art, "model_metadata", "Model metadata"):
        tm = meta.get("test_metrics") or {}
        ci = tm.get("macro_f1_ci") or [None, None]
        gen = (ev or {}).get("generalization") or meta.get("generalization") or {}
        oof = gen.get("problem_grouped_oof") or {}
        n_test = sum(v.get("support", 0) for v in (meta.get("per_class_test") or {}).values())
        ui.cards([
            ui.card("Macro F1 (test)", ui.fmt(tm.get("macro_f1"), 3),
                    f"95% CI {ci[0]:.2f}–{ci[1]:.2f} (problem bootstrap)" if ci[0] is not None else "", accent=True),
            ui.card("Accuracy (test)", ui.fmt(tm.get("accuracy"), 3), f"{n_test} test programs", accent=True),
            ui.card("Balanced accuracy", ui.fmt(tm.get("balanced_accuracy"), 3), "mean recall over 7 classes"),
            ui.card("Macro F1, all problems (CV)", ui.fmt(oof.get("macro_f1"), 3),
                    f"out-of-fold, grouped by problem, n = {oof.get('n', '—')}"),
            _baseline_card(art),
        ])
        ui.muted(f"{meta.get('selected_model')} · features: {meta.get('feature_set')} ({len(meta.get('features') or [])}) · "
                 f"{meta.get('trained_on')} · scikit-learn {(meta.get('versions') or {}).get('scikit-learn')}")
        ui.muted("Scope: " + (meta.get("scope") or ""))

    t1, t2, t3, t4, t5, t6, t7 = st.tabs(["Model comparison", "Confusion matrix & classes", "Feature importance",
                                          "Confidence & calibration", "Generalization", "Improvement history", "Dataset"])
    with t1:
        _comparison(art, meta)
    with t2:
        _confusion(art, meta)
    with t3:
        _importance()
    with t4:
        _confidence(meta, ev)
    with t5:
        _generalization(ev)
    with t6:
        _improvement(art)
    with t7:
        _dataset(art)


def _baseline_card(art) -> str:
    mc = art.data.get("model_comparison")
    if mc is not None and "model" in mc.columns:
        row = mc[mc["model"] == "Dummy (stratified)"]
        if len(row):
            return ui.card("Baseline macro F1", ui.fmt(float(row["test_macro_f1"].iloc[0]), 3),
                           "random guess in training proportions")
    return ui.card("Baseline macro F1", "—", "model comparison file missing")


def _comparison(art, meta) -> None:
    ui.section("Model comparison", "All candidates on the same grouped split. Selection used repeated cross-validation "
               "on the development set, not the test set; the test column is reported for transparency.")
    if _missing(art, "model_comparison", "Model comparison"):
        return
    df = logic.comparison_table(art.data["model_comparison"])
    st.plotly_chart(charts.model_comparison(df, (meta or {}).get("selected_model"), (meta or {}).get("feature_set")), **ui.STRETCH,
                    config={"displayModeBar": False})
    st.dataframe(df, hide_index=True, **ui.STRETCH, column_config={
        c: st.column_config.NumberColumn(c, format="%.3f") for c in df.columns if df[c].dtype.kind == "f"})
    raw = art.data["model_comparison"]
    if {"rcv_macro_f1", "key"} <= set(raw.columns) and len(raw):
        top = raw.loc[raw["rcv_macro_f1"].idxmax()]
        ui.muted(f"Highlighted: the selected model. Highest repeated-CV macro F1: {top['key']} ({top['rcv_macro_f1']:.3f}). "
                 "Selection rule: the highest repeated-CV score, and within 0.005 of it the simplest model. The test "
                 "intervals overlap heavily, so the top models are statistically indistinguishable on this test set.")


def _confusion(art, meta) -> None:
    ui.section("Confusion matrix (test set)", "Rows are expert labels, columns are predictions.")
    if not _missing(art, "confusion_matrix", "Confusion matrix"):
        cm = art.data["confusion_matrix"]
        normalize = st.toggle("Show row percentages", value=True, key="cm_norm")
        st.plotly_chart(charts.confusion_heatmap(cm, normalize), **ui.STRETCH, config={"displayModeBar": False})
        check = logic.confusion_scores(cm)
        tm = (meta or {}).get("test_metrics") or {}
        if tm.get("macro_f1") is not None:
            same = abs(check["macro_f1"] - tm["macro_f1"]) < 0.002 and abs(check["accuracy"] - tm["accuracy"]) < 0.002
            ui.muted(f"Recomputed from this matrix: accuracy {check['accuracy']:.3f}, macro F1 {check['macro_f1']:.3f} "
                     f"({check['n']} programs) — " + ("matches the saved model's test metrics." if same else
                                                     "does NOT match the saved model's metrics; the matrix file may be from an older run."))
    per = (meta or {}).get("per_class_test")
    if per:
        ui.section("Per-class results (test set)")
        df = pd.DataFrame(per).T.reset_index().rename(columns={"index": "Class", "precision": "Precision", "recall": "Recall",
                                                               "f1": "F1", "support": "Support"})
        st.plotly_chart(charts.simple_bar(df["Class"], df["F1"], yaxis="F1"), **ui.STRETCH,
                        config={"displayModeBar": False})
        st.dataframe(df, hide_index=True, **ui.STRETCH)


def _generalization(ev) -> None:
    if ev is None:
        ui.callout("info", "Generalization results not available", "Create them with: python -m src.evaluate_model")
        return
    gen = ev.get("generalization") or {}
    ui.section("Grouped cross-validation", "Every program predicted by a model that never saw its problem (or contest).")
    rows = []
    for key, label in (("problem_grouped_oof", "Grouped by problem"), ("contest_grouped_oof", "Grouped by contest")):
        if gen.get(key):
            rows.append({"Evaluation": label, **gen[key]})
    if rows:
        st.dataframe(pd.DataFrame(rows), hide_index=True, **ui.STRETCH)
    fams = gen.get("families") or {}
    if fams:
        ui.section("Algorithm families held out of training",
                   "Macro F1 on a family's problems when the family is in training (problem held out) vs absent entirely.")
        names = sorted(fams)
        st.plotly_chart(charts.paired_bar(names, [fams[n]["seen"] for n in names], [fams[n]["held_out"] for n in names],
                                          "Family seen in training", "Family held out", yaxis="Macro F1"),
                        **ui.STRETCH, config={"displayModeBar": False})
        if gen.get("family_holdout_mean_macro_f1_change") is not None:
            ui.muted(f"Mean change when a family is held out: {gen['family_holdout_mean_macro_f1_change']:+.3f} macro F1.")
    if gen.get("implementations_mean_problem_accuracy") is not None:
        ui.muted(f"Different implementations of the same problem: mean per-problem accuracy "
                 f"{gen['implementations_mean_problem_accuracy']:.3f}.")
    ext = gen.get("external")
    ui.section("External dataset (BigO(Bench))", "A different source, labelled by runtime measurement.")
    if not ext:
        ui.callout("info", "Not evaluated on this computer",
                   "BigO(Bench) was not found in data/raw/bigobench when the evaluation ran, so no external result exists.")
    else:
        unseen = ext.get("unseen_problems") or {}
        ui.cards([
            ui.card("Macro F1 (unseen problems)", ui.fmt(unseen.get("macro_f1"), 3), f"n = {unseen.get('n', '—')}", accent=True),
            ui.card("Accuracy (unseen problems)", ui.fmt(unseen.get("accuracy"), 3),
                    "higher than macro F1: a few common classes dominate"),
            ui.card("Records", ext.get("records", "—"), f"file {ext.get('file', '')}"),
        ])
        if unseen.get("class_counts"):
            cc = unseen["class_counts"]
            st.plotly_chart(charts.simple_bar(list(cc), list(cc.values()), yaxis="Programs", text_fmt="{:.0f}", height=260),
                            **ui.STRETCH, config={"displayModeBar": False})
    dist = ev.get("failure_distance_%")
    if dist:
        ui.section("How wrong are the errors?", "Share of test predictions by distance from the true class (classes ordered O(1) … O(2^n)).")
        st.plotly_chart(charts.simple_bar(list(dist), list(dist.values()), yaxis="% of test programs", text_fmt="{:.1f}%",
                                          height=260), **ui.STRETCH, config={"displayModeBar": False})


def _importance() -> None:
    from src.ui.views.results import render_global_importance

    render_global_importance()


def _improvement(art) -> None:
    ui.section("Final improvement phase: before vs after",
               "Same 814 test programs, same grouped split. Features were chosen by cross-validation on the development "
               "set; the test set was used once for the final model.")
    if _missing(art, "improvement", "Improvement report"):
        return
    imp = art.data["improvement"]
    ba = imp.get("before_after") or []
    if ba:
        st.dataframe(pd.DataFrame(ba), hide_index=True, **ui.STRETCH)
    abl = imp.get("feature_ablation") or []
    if abl:
        ui.section("Feature ablation (repeated grouped CV on train + validation)",
                   "Random Forest with the same hyperparameters; only the feature set changes.")
        df = pd.DataFrame(abl)
        st.plotly_chart(charts.simple_bar(df["feature_set"], df["rcv_macro_f1"], yaxis="Repeated-CV macro F1",
                                          height=280), **ui.STRETCH, config={"displayModeBar": False})
        st.dataframe(df, hide_index=True, **ui.STRETCH)
    bench = imp.get("analyzer_benchmark") or []
    if bench:
        ui.section("Static analyzer on reference algorithms",
                   "Expected values are standard worst-case results; Unknown is reported when a pattern is not supported.")
        st.dataframe(pd.DataFrame(bench), hide_index=True, **ui.STRETCH)
    for note in imp.get("notes") or []:
        ui.muted("• " + note)


def _confidence(meta, ev) -> None:
    cal = (meta or {}).get("calibration_test")
    if cal:
        ui.section("Confidence score vs calibrated probability",
                   "The app shows the forest's top vote share. These measurements, from the test set, show how far it is "
                   "from a calibrated probability.")
        ui.cards([
            ui.card("Expected calibration error", ui.fmt(cal.get("ece_top_label"), 3), "0 = perfectly calibrated", accent=True),
            ui.card("Mean top vote share", ui.fmt(cal.get("mean_top_vote_share"), 3), f"vs accuracy {ui.fmt(cal.get('accuracy'), 3)}"),
            ui.card("Multiclass Brier score", ui.fmt(cal.get("brier_multiclass"), 3), "lower is better"),
            ui.card("Log loss", ui.fmt(cal.get("log_loss"), 3), "lower is better"),
        ])
        ui.muted(cal.get("note", ""))
    ui.section("Is the vote share meaningful?", "Test accuracy for predictions grouped by the model's top vote share.")
    rel = (meta or {}).get("confidence_reliability_test")
    if not rel:
        ui.callout("info", "Not available", "Create it with: python -m src.evaluate_model")
        return
    rel = [r for r in rel if r.get("n")]
    labels = [f"{r['from']:.1f}–{r['to']:.1f}" for r in rel]
    st.plotly_chart(charts.simple_bar(labels, [r["accuracy"] for r in rel], yaxis="Test accuracy", text_fmt="{:.0%}",
                                      height=280), **ui.STRETCH, config={"displayModeBar": False})
    st.dataframe(pd.DataFrame([{"Vote share": lab, "Test programs": r["n"], "Accuracy": r["accuracy"]}
                               for lab, r in zip(labels, rel)]), hide_index=True, **ui.STRETCH)
    ui.muted("Vote shares are not calibrated probabilities, but higher shares were clearly more often right on the test set.")
    if ev:
        bt, tt = ev.get("before_test"), ev.get("tuned_test")
        if bt and tt:
            ui.section("Tuning", "Hyperparameter tuning result (adopted only if it beat the baseline by a margin in cross-validation).")
            st.dataframe(pd.DataFrame([{"Configuration": "Kept (Phase 10)", **bt}, {"Configuration": "Tuned", **tt}]),
                         hide_index=True, **ui.STRETCH)


def _dataset(art) -> None:
    ui.section("Training data: class distribution", "CodeComplex Python, after removing duplicates.")
    if _missing(art, "dataset_report", "Dataset report"):
        return
    rep = art.data["dataset_report"]
    df = logic.class_distribution(rep)
    if df is None:
        ui.callout("info", "Class distribution not recorded in the dataset report")
        return
    st.plotly_chart(charts.simple_bar(df["Class"], df["Count"], yaxis="Programs", text_fmt="{:.0f}", height=280),
                    **ui.STRETCH, config={"displayModeBar": False})
    b = rep.get("bigo") or {}
    ui.muted(f"{b.get('final_samples', '—')} programs from {b.get('problems', '—')} problems "
             f"({b.get('raw_records', '—')} raw records; {b.get('removed_total', '—')} duplicates removed).")
