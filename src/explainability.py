"""Why did the model predict this?

Responsibility:
    Global feature importance (model importances / permutation importance)
    and simple local explanations (which features pushed a function above
    the dataset median). No SHAP dependency unless justified later.

Planned public API (Phase 2):
    global_importance(model, feature_names) -> pd.DataFrame
    explain_function(features: dict[str, float]) -> list[str]
"""
