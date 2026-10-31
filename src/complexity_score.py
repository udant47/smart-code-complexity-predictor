"""Single 0-100 complexity score and label thresholds.

Responsibility:
    Combine selected features into an interpretable weighted score and
    map it to labels (e.g. Low / Medium / High). Also used to generate
    weak labels for training data when no human labels exist; this is
    documented so the evaluation is not circular.

Planned public API (Phase 2):
    compute_score(features: dict[str, float]) -> float
    score_to_label(score: float) -> str
"""
