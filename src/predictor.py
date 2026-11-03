"""Prediction pipeline: one call from source code to a structured result (Phase 12).

    from src.predictor import predict
    result = predict(source_code)          # PredictionResult
    result.to_dict()                       # JSON-friendly, for the frontend

Nothing in this module trains a model or executes the submitted code. The
code is parsed with ``ast`` and measured statically; the saved model and its
fitted preprocessing are loaded once per process and reused.

What the result contains
------------------------
* **File-level metrics:** the 39 structural features and 18 algorithm-pattern
  features of the whole submission.
* **Structural complexity category (rule-based):** Low / Medium / High from
  McCabe cyclomatic complexity using radon's published thresholds (1-5, 6-10,
  11+), per function and for the most complex function in the file. This is a
  transparent rule, not a model output. Phase 8 showed that a model trained to
  predict these bands would only relearn the thresholds.
* **Function-level metrics:** per-function structural metrics (Phase 5), the
  structural band, and the rule-based Big-O estimate of the static analyser
  (Phase 6) with its own status, confidence and explanation.
* **Machine-learning time-complexity prediction for the whole file:** the
  trained model (CodeComplex expert labels, whole programs) predicts one of
  seven classes, O(1) ... O(2^n), with its probabilities.
  - The probabilities are Random Forest vote shares, **not calibrated
    probabilities**. When the model metadata contains them, the measured test
    accuracy for predictions in the same probability range is attached, so the
    number can be interpreted honestly.
  - The model was evaluated on whole Codeforces programs (test macro F1 about
    0.44). It is a second opinion, not proof.
* **Per-function ML predictions (experimental):** the same model applied to
  each function's code on its own. It was never trained on single functions,
  so these are flagged ``experimental`` and carry a warning.
* **Derived Complexity Score (0-100):** a documented structural indicator per
  function, with a file summary (or a module score for scripts); see
  ``src/complexity_score.py``. It is separate from the ML prediction.
* **Refactoring recommendations:** deterministic, evidence-based review
  guidance from ``src/suggestions.py`` (not guaranteed fixes, not ML).
* **Optional explanation** (``explain=True``): which features moved the model's vote
  for this file, from ``src/explainability.py``. It explains the model, not the code.
* **Model metadata, errors and warnings.** If the model files are missing or
  incompatible, the ML parts are ``None`` and an error explains why. No
  prediction is ever invented.

Model files (written by ``python -m src.evaluate_model``)
---------------------------------------------------------
``models/complexity_model.joblib``       fitted Pipeline (preprocessing + Random Forest)
``models/preprocessing_pipeline.joblib`` the same fitted preprocessing on its own
``models/model_metadata.json``           feature list and order, classes, metrics, versions

Schema validation on load: the metadata feature list must equal the project's
feature set of that name, and the fitted preprocessing must expect exactly
those columns in that order. At prediction time, every feature is taken in
that order from the extractor output; a missing or non-finite value is an
error, not a silent default.

Security note: ``joblib`` files are pickles and can run code when loaded.
Before unpickling, ``src.model_integrity.check_file`` requires each file to be
a regular file below a size limit whose SHA-256 matches the hash recorded in
the metadata at training time. Model files are never accepted from users.
"""

from __future__ import annotations

import hashlib
import json
import textwrap
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd

from src import config
from src import model_integrity as integrity
from src import preprocessing as pp
from src.algorithm_analyzer import analyze_algorithms, analyze_script
from src import complexity_score as dcs
from src.code_parser import ParseResult, parse_bytes, parse_source
from src.dataset_loader import BIGO_LABELS, STRUCTURAL_BANDS, structural_band
from src.feature_extractor import FILE_FEATURE_NAMES, extract_features
from src.function_analyzer import analyze_functions
from src.pattern_features import PATTERN_FEATURE_NAMES
from src.analysis_features import ANALYSIS_FEATURE_NAMES, program_features
from src.suggestions import suggest

PREPROCESSING_PATH = config.MODELS_DIR / "preprocessing_pipeline.joblib"
LOW_CONFIDENCE = 0.4  # below this top probability, a warning is attached
MIN_FUNCTION_LINES_FOR_ML = 3  # experimental per-function ML only for functions this long or longer
BIGO_OF = {k: v[1] for k, v in BIGO_LABELS.items()}
STRUCTURAL_RULE = "McCabe cyclomatic complexity with radon thresholds: 1-5 Low, 6-10 Medium, 11+ High"


# --------------------------------------------------------------------------- #
# Result types
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Issue:
    code: str
    message: str
    hint: Optional[str] = None


@dataclass
class MLPrediction:
    label: str  # dataset class, e.g. "nlogn"
    big_o: str  # e.g. "O(n log n)"
    probability: float  # top vote share (uncalibrated)
    probabilities: dict  # Big-O -> vote share, in class order
    runner_up: str
    margin: float  # top minus second probability
    measured_accuracy_in_range: Optional[dict]  # test accuracy for this probability range, if recorded
    experimental: bool = False
    note: str = ""


@dataclass
class FunctionReport:
    function_id: str
    name: str
    kind: str
    class_name: Optional[str]
    start_line: int
    end_line: int
    metrics: dict
    structural_category: str
    static_estimate: dict
    derived_score: Optional[dict] = None  # Derived Complexity Score (0-100), see src/complexity_score.py
    ml_prediction: Optional[MLPrediction] = None


@dataclass
class ModelInfo:
    available: bool
    model: Optional[str] = None
    feature_set: Optional[str] = None
    n_features: Optional[int] = None
    classes: Optional[list] = None
    test_macro_f1: Optional[float] = None
    test_macro_f1_ci: Optional[list] = None
    test_accuracy: Optional[float] = None
    trained_on: Optional[str] = None
    scope: Optional[str] = None
    trained_with_sklearn: Optional[str] = None
    model_sha256: Optional[str] = None


@dataclass
class PredictionResult:
    success: bool  # the code was parsed and analysed (the ML part may still be unavailable)
    errors: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    file_metrics: Optional[dict] = None
    file_summary: Optional[dict] = None
    structural_category: Optional[dict] = None
    prediction: Optional[MLPrediction] = None
    functions: list = field(default_factory=list)
    model: Optional[ModelInfo] = None
    derived_score: Optional[dict] = None  # file summary of the Derived Complexity Score
    recommendations: Optional[dict] = None  # refactoring review guidance, see src/suggestions.py
    explanation: Optional[dict] = None  # per-prediction explanation (predict(..., explain=True)); see src/explainability.py
    program_estimate: Optional[dict] = None  # static estimate of module-level (script) code, see analyze_script

    def to_dict(self) -> dict:
        return json.loads(json.dumps(asdict(self), default=_json_default))

    def functions_table(self) -> pd.DataFrame:
        """One row per function, for display."""
        rows = []
        for f in self.functions:
            ml = f.ml_prediction
            rows.append({
                "Function": f.function_id, "Type": f.kind, "Lines": f"{f.start_line}-{f.end_line}",
                "Cyclomatic complexity": f.metrics.get("cyclomatic_complexity"),
                "Structural category": f.structural_category,
                "Derived Complexity Score": (f.derived_score or {}).get("score"),
                "Static time estimate": f.static_estimate.get("time_complexity"),
                "Static space estimate": f.static_estimate.get("space_complexity"),
                "ML time class (experimental)": ml.big_o if ml else None,
                "ML probability": round(ml.probability, 2) if ml else None,
            })
        return pd.DataFrame(rows)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


# --------------------------------------------------------------------------- #
# Model loading (once per process)
# --------------------------------------------------------------------------- #
class ModelUnavailable(Exception):
    def __init__(self, code: str, message: str, hint: Optional[str] = None):
        super().__init__(message)
        self.issue = Issue(code, message, hint)


@dataclass
class LoadedModel:
    pipeline: object
    preprocessing: object
    metadata: dict
    features: list
    classes: list
    info: ModelInfo
    load_warnings: list


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_model(model_path: Path = config.MODEL_PATH, metadata_path: Path = config.MODEL_METADATA_PATH,
               preprocessing_path: Path = PREPROCESSING_PATH) -> LoadedModel:
    """Load and validate the saved model. Raises ModelUnavailable with a clear reason."""
    import joblib
    import sklearn

    missing = [p.name for p in (model_path, metadata_path, preprocessing_path) if not p.exists()]
    if missing:
        raise ModelUnavailable(
            "model_missing", f"Trained model files not found: {', '.join(missing)}.",
            "Run: python -m src.train_model, python -m src.tune_model, then python -m src.evaluate_model.",
        )
    try:
        meta = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ModelUnavailable("model_metadata_invalid", f"Model metadata could not be read: {exc}") from exc

    features = list(meta.get("features") or [])
    feature_set = meta.get("feature_set")
    if feature_set not in pp.FEATURE_SETS or features != pp.feature_columns(feature_set):
        raise ModelUnavailable(
            "model_incompatible",
            "The saved model's feature list does not match this version of the feature extractor.",
            "Retrain: python -m src.dataset_loader build, then the training commands.",
        )
    for path in (model_path, preprocessing_path):
        try:
            integrity.check_file(path, meta)
        except integrity.IntegrityError as exc:
            raise ModelUnavailable(exc.code, exc.message, exc.hint) from exc
    try:
        pipeline = joblib.load(model_path)
        preprocessing = joblib.load(preprocessing_path)
    except Exception as exc:  # unpickling errors vary by library version
        raise ModelUnavailable("model_load_failed", f"The model file could not be loaded: {type(exc).__name__}: {exc}",
                               "Retrain the model with the installed library versions.") from exc

    for name, obj in (("model", pipeline), ("preprocessing", preprocessing)):
        expected = getattr(obj, "feature_names_in_", None)
        if expected is not None and [c for c in expected if c in features] != features:
            raise ModelUnavailable("model_incompatible",
                                   f"The fitted {name} expects a different feature order than the metadata.")
    if not hasattr(pipeline, "predict_proba"):
        raise ModelUnavailable("model_incompatible", "The saved model cannot produce probabilities.")
    classes = [str(c) for c in pipeline.classes_]
    if set(classes) != set(BIGO_OF):
        raise ModelUnavailable("model_incompatible", f"Unexpected model classes: {classes}")

    warnings = []
    trained = meta.get("versions", {}).get("scikit-learn")
    if trained and trained != sklearn.__version__:
        warnings.append(Issue("version_mismatch",
                              f"Model trained with scikit-learn {trained}; installed {sklearn.__version__}.",
                              "Predictions may differ slightly; retrain to be safe."))
    tm_ = meta.get("test_metrics", {})
    info = ModelInfo(
        available=True, model=meta.get("selected_model"), feature_set=feature_set, n_features=len(features),
        classes=[BIGO_OF[c] for c in classes], test_macro_f1=tm_.get("macro_f1"),
        test_macro_f1_ci=tm_.get("macro_f1_ci"), test_accuracy=tm_.get("accuracy"),
        trained_on=meta.get("trained_on"), scope=meta.get("scope"), trained_with_sklearn=trained,
        model_sha256=_sha256(model_path),
    )
    return LoadedModel(pipeline, preprocessing, meta, features, classes, info, warnings)


@lru_cache(maxsize=4)
def _cached_model(model_path: str, metadata_path: str, preprocessing_path: str, stamp: tuple) -> LoadedModel:
    return load_model(Path(model_path), Path(metadata_path), Path(preprocessing_path))


def get_model(model_path: Path = config.MODEL_PATH, metadata_path: Path = config.MODEL_METADATA_PATH,
              preprocessing_path: Path = PREPROCESSING_PATH) -> LoadedModel:
    """Cached loader: files are read once and reloaded only if they change on disk."""
    stamp = tuple(p.stat().st_mtime_ns if p.exists() else 0 for p in (model_path, metadata_path, preprocessing_path))
    return _cached_model(str(model_path), str(metadata_path), str(preprocessing_path), stamp)


# --------------------------------------------------------------------------- #
# Prediction
# --------------------------------------------------------------------------- #
def feature_vector(features: list, values: dict) -> pd.DataFrame:
    """One-row DataFrame in the exact training column order. Raises ValueError on bad values."""
    missing = [f for f in features if f not in values]
    if missing:
        raise ValueError(f"missing features: {missing[:5]}")
    row = [values[f] for f in features]
    arr = np.asarray(row, dtype=float)
    if not np.isfinite(arr).all():
        bad = [f for f, v in zip(features, arr) if not np.isfinite(v)]
        raise ValueError(f"non-finite feature values: {bad[:5]}")
    return pd.DataFrame([row], columns=features)


def _ml_from_proba(model: LoadedModel, proba, experimental: bool = False, note: str = "") -> MLPrediction:
    order = np.argsort(proba)[::-1]
    top, second = int(order[0]), int(order[1])
    label = model.classes[top]
    p = float(proba[top])
    measured = None
    for row in model.metadata.get("confidence_reliability_test") or []:
        if row["from"] <= p < row["to"] or (row["to"] >= 1.0 and p >= row["from"]):
            if row.get("n"):
                measured = {"probability_range": [row["from"], row["to"]], "test_predictions": row["n"],
                            "test_accuracy": row["accuracy"]}
            break
    probs = {BIGO_OF[c]: round(float(proba[model.classes.index(c)]), 4) for c in pp.CLASS_ORDER}
    return MLPrediction(label=label, big_o=BIGO_OF[label], probability=round(p, 4), probabilities=probs,
                        runner_up=BIGO_OF[model.classes[second]], margin=round(p - float(proba[second]), 4),
                        measured_accuracy_in_range=measured, experimental=experimental, note=note)


def _ml_prediction(model: LoadedModel, values: dict, experimental: bool = False, note: str = "") -> MLPrediction:
    X = feature_vector(model.features, values)
    return _ml_from_proba(model, model.pipeline.predict_proba(X)[0], experimental, note)


def _ml_predictions_batch(model: LoadedModel, rows: list, experimental: bool = True, note: str = "") -> list:
    """One predict_proba call for many feature dicts (same results as calling _ml_prediction on each).

    Rows whose features are invalid get None instead of failing the whole batch.
    """
    frames, keep = [], []
    for i, values in enumerate(rows):
        try:
            frames.append(feature_vector(model.features, values))
            keep.append(i)
        except ValueError:
            continue
    out: list = [None] * len(rows)
    if not frames:
        return out
    proba = model.pipeline.predict_proba(pd.concat(frames, ignore_index=True))
    for i, row in zip(keep, proba):
        out[i] = _ml_from_proba(model, row, experimental, note)
    return out


def _all_features(parsed: ParseResult) -> dict:
    return program_features(parsed)


def predict(source, *, model_path: Path = config.MODEL_PATH, metadata_path: Path = config.MODEL_METADATA_PATH,
            preprocessing_path: Path = PREPROCESSING_PATH, per_function_ml: bool = True,
            explain: bool = False) -> PredictionResult:
    """Analyse source code (str, or bytes from an upload) and predict its complexity.

    explain=True also attaches a per-prediction explanation of the file-level ML
    prediction (tree-path contributions, see src/explainability.py).
    """
    # 1-2. Validate and parse (never executes the code)
    if isinstance(source, (bytes, bytearray)):
        parsed = parse_bytes(bytes(source))
    elif isinstance(source, str):
        parsed = parse_source(source)
    else:
        return PredictionResult(False, errors=[Issue("invalid_input_type",
                                                     f"Expected source code as text, got {type(source).__name__}.")])
    if not parsed.success:
        e = parsed.error
        return PredictionResult(False, errors=[Issue(e.kind, e.message, e.hint)], warnings=list(parsed.warnings))

    result = PredictionResult(True, warnings=[Issue("parser_warning", w) for w in parsed.warnings
                                              if not w.startswith("No functions or classes")])  # reported below

    # 3. File-level metrics
    feats = extract_features(parsed)
    file_values = program_features(parsed, feats.file_features)  # same function the dataset builder uses
    result.file_metrics = {k: file_values[k] for k in (*FILE_FEATURE_NAMES, *PATTERN_FEATURE_NAMES, *ANALYSIS_FEATURE_NAMES)}

    # 4. Functions: structural metrics, rule-based category, static Big-O estimate
    functions = analyze_functions(parsed)
    estimates = {e.function_id: e for e in analyze_algorithms(parsed).estimates}
    feature_rows = {row["lineno"]: row for row in feats.function_features}
    result.file_summary = functions.summary.to_dict() if functions.summary else None
    scored = []
    for rec in functions.functions:
        est = estimates.get(rec.function_id)
        row = feature_rows.get(rec.start_line, {})
        score = dcs.score_function(row)
        scored.append((rec.function_id, score, int(row.get("code_lines", 1))))
        result.functions.append(FunctionReport(
            function_id=rec.function_id, name=rec.function_name, kind=rec.kind, class_name=rec.class_name,
            start_line=rec.start_line, end_line=rec.end_line, metrics=rec.to_dict(),
            structural_category=structural_band(rec.cyclomatic_complexity),
            static_estimate=est.to_dict() if est else {},
            derived_score=score.to_dict(),
        ))
    if scored:
        result.derived_score = {"scope": "functions", **dcs.summarize(scored)}
    else:  # a script with no functions: score the module as one unit
        module = dcs.score_module(feats.file_features)
        result.derived_score = {"scope": "module", "label": dcs.LABEL, "module": module.to_dict(),
                                "highest": module.score, "highest_band": module.band, "disclaimer": dcs.DISCLAIMER}
    bands = [f.structural_category for f in result.functions]
    worst = max(result.functions, key=lambda f: f.metrics["cyclomatic_complexity"], default=None)
    result.structural_category = {
        "method": STRUCTURAL_RULE,
        "kind": "rule-based (not a model prediction)",
        "highest": worst.structural_category if worst else None,
        "most_complex_function": worst.function_id if worst else None,
        "counts": {b[0]: bands.count(b[0]) for b in STRUCTURAL_BANDS},
        "module_level": structural_band(feats.file_features["cyclomatic_complexity"]),
    }
    script = analyze_script(parsed)  # module-level code (e.g. a competitive-programming script); None if none
    result.program_estimate = script.to_dict() if script is not None else None
    result.recommendations = suggest(parsed).to_dict()  # rule-based, independent of the ML model
    if not result.functions:
        result.warnings.append(Issue("no_functions", "No functions found: function-level results are empty; "
                                     "file-level metrics and the file prediction still apply."))

    # 5-7. Model (loaded once, validated), exact training feature order
    try:
        model = get_model(model_path, metadata_path, preprocessing_path)
    except ModelUnavailable as exc:
        result.model = ModelInfo(available=False)
        result.errors.append(exc.issue)
        return result
    result.model = model.info
    result.warnings.extend(model.load_warnings)
    try:
        result.prediction = _ml_prediction(model, file_values)
    except ValueError as exc:
        result.errors.append(Issue("feature_schema_error", f"Could not build the model input: {exc}"))
        return result
    if result.prediction.probability < LOW_CONFIDENCE:
        result.warnings.append(Issue(
            "low_confidence",
            f"The model is unsure (top probability {result.prediction.probability:.2f}, next "
            f"{result.prediction.runner_up}); treat the prediction as a weak hint."))
    if explain:
        from src.explainability import explain_features

        result.explanation = explain_features(file_values, model=model).to_dict()

    # 8. Per-function ML (experimental: the model was trained on whole programs).
    #    Features are collected first and predicted in one batch: one forest call instead of one per function.
    if per_function_ml:
        targets, rows = [], []
        source_by_line: dict = {}
        for fi in parsed.functions:
            source_by_line.setdefault(fi.lineno, fi.source)
        for f, info in zip(result.functions, functions.functions):
            if info.end_line - info.start_line + 1 < MIN_FUNCTION_LINES_FOR_ML:
                continue
            source_segment = source_by_line.get(f.start_line)
            if not source_segment:
                continue
            sub = parse_source(textwrap.dedent(source_segment))
            if not sub.success:
                continue
            targets.append(f)
            rows.append(_all_features(sub))
        n_done = 0
        note = "Model trained on whole programs; applied to this function alone."
        for f, ml in zip(targets, _ml_predictions_batch(model, rows, experimental=True, note=note)):
            if ml is not None:
                f.ml_prediction = ml
                n_done += 1
        if n_done:
            result.warnings.append(Issue(
                "experimental_function_predictions",
                "Per-function ML predictions are experimental: the model was trained and evaluated on whole "
                "programs, not single functions. Prefer the static estimate when it is available."))
    return result


def predict_json(source, **kwargs) -> str:
    return json.dumps(predict(source, **kwargs).to_dict(), indent=2)
