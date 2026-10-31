"""Model training (classical ML only).

Responsibility:
    Train and compare candidate classifiers (Logistic Regression, Decision
    Tree, Random Forest, Gradient Boosting; XGBoost only if it measurably
    helps), using stratified cross-validation, then save the best pipeline
    to ``models/complexity_model.joblib`` with a metadata JSON
    (feature columns, label classes, metrics, library versions).

Run as:  python -m src.train_model
"""
