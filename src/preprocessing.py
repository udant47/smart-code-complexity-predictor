"""Feature cleaning and the scikit-learn preprocessing pipeline.

Responsibility:
    Handle missing/infinite values, clip outliers, and expose a
    ``ColumnTransformer``/``Pipeline`` (e.g. StandardScaler) that is saved
    together with the model so training and prediction stay identical.

Planned public API (Phase 2):
    clean_features(df: pd.DataFrame) -> pd.DataFrame
    build_preprocessor() -> sklearn.pipeline.Pipeline
"""
