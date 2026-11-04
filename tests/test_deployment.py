"""Deployment readiness for Streamlit Community Cloud.

Community Cloud clones the repository, installs requirements.txt and runs
``streamlit run app.py`` from the repository root on Linux. These tests check
that everything it needs is present and portable, and that the model shipped in
the repository is the one recorded at training time.
"""

import json
import re
import tomllib
from pathlib import Path

import pytest

from src import config
from src import model_integrity

ROOT = config.PROJECT_ROOT
RUNTIME_MODEL_FILES = ("complexity_model.joblib", "preprocessing_pipeline.joblib", "model_metadata.json",
                       "explainability.json")


def test_entry_point_and_dependency_files_at_repository_root():
    assert (ROOT / "app.py").is_file()
    assert (ROOT / "requirements.txt").is_file()
    assert (ROOT / ".streamlit" / "config.toml").is_file()


def test_requirements_are_runtime_only_and_pin_the_model_sklearn_version():
    lines = [ln.strip() for ln in (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()
             if ln.strip() and not ln.strip().startswith("#")]
    names = {re.split(r"[<>=!~\[]", ln)[0].lower() for ln in lines}
    assert {"streamlit", "pandas", "numpy", "scikit-learn", "plotly", "joblib"} <= names
    assert not names & {"pytest", "ipykernel", "jupyter", "torch", "tensorflow", "openai", "anthropic"}
    sklearn = next(ln for ln in lines if ln.lower().startswith("scikit-learn"))
    assert "==" in sklearn
    if config.MODEL_METADATA_PATH.exists():
        trained = json.loads(config.MODEL_METADATA_PATH.read_text(encoding="utf-8"))["versions"]["scikit-learn"]
        assert sklearn.split("==")[1] == trained, "requirements.txt must pin the scikit-learn version the model was trained with"


def test_gitignore_keeps_runtime_model_files_and_excludes_private_data():
    text = (ROOT / ".gitignore").read_text(encoding="utf-8")
    for name in RUNTIME_MODEL_FILES:
        assert f"!models/{name}" in text
    for private in (".venv/", "__pycache__/", ".streamlit/secrets.toml", "data/raw/*", ".env"):
        assert private in text
    assert "!data/raw/" not in text.replace("!data/raw/.gitkeep", "")  # CodeComplex is never redistributed


def test_streamlit_config_is_cloud_compatible():
    cfg = tomllib.loads((ROOT / ".streamlit" / "config.toml").read_text(encoding="utf-8"))
    assert "address" not in cfg.get("server", {}) and "port" not in cfg.get("server", {})
    assert cfg["theme"]["base"] == "dark"


def test_local_run_instructions_bind_to_localhost():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "streamlit run app.py --server.address localhost" in readme


def test_no_machine_specific_paths_in_code():
    pattern = re.compile(r"""["'](?:[A-Za-z]:\\\\|/home/|/Users/|/root/)""")
    for path in list((ROOT / "src").rglob("*.py")) + [ROOT / "app.py"]:
        assert not pattern.search(path.read_text(encoding="utf-8")), path


@pytest.mark.skipif(not config.MODEL_PATH.exists(), reason="no trained model in models/")
def test_shipped_model_matches_its_recorded_hashes():
    meta = json.loads(config.MODEL_METADATA_PATH.read_text(encoding="utf-8"))
    for name in ("complexity_model.joblib", "preprocessing_pipeline.joblib"):
        model_integrity.check_file(config.MODELS_DIR / name, meta)  # raises IntegrityError on any mismatch
    assert config.MODEL_PATH.stat().st_size < 50 * 1024 * 1024  # below GitHub's large-file warning


@pytest.mark.skipif(not config.MODEL_PATH.exists(), reason="no trained model in models/")
def test_shipped_reports_describe_the_shipped_model():
    """The dashboard shows saved reports, so they must come from the same training run as the model."""
    meta = json.loads(config.MODEL_METADATA_PATH.read_text(encoding="utf-8"))
    ev = json.loads((ROOT / "reports" / "evaluation" / "evaluation_results.json").read_text(encoding="utf-8"))
    assert ev["chosen"]["feature_set"] == meta["feature_set"]
    assert ev["test_metrics"]["macro_f1"] == meta["test_metrics"]["macro_f1"]
    exp = json.loads((config.MODELS_DIR / "explainability.json").read_text(encoding="utf-8"))
    assert len(exp["permutation"]) == len(meta["features"])


def test_tuning_fallback_is_reported_not_silent(tmp_path, monkeypatch):
    """If train_model selects a model tune_model cannot tune, the fallback must carry a warning."""
    from src import tune_model

    meta = tmp_path / "model_metadata.json"
    meta.write_text(json.dumps({"selected_model": "Soft-voting ensemble (LR + RF + HGB)",
                                "feature_set": "patterns_static", "hyperparameters": {}}), encoding="utf-8")
    monkeypatch.setattr(config, "MODEL_METADATA_PATH", meta)
    cfg = tune_model.phase10_config()
    assert cfg["model"] == "Random Forest" and "Soft-voting ensemble" in cfg["warning"]
    meta.write_text(json.dumps({"selected_model": "Random Forest", "feature_set": "patterns_static",
                                "hyperparameters": {"min_samples_leaf": 3}}), encoding="utf-8")
    cfg = tune_model.phase10_config()
    assert "warning" not in cfg and cfg["feature_set"] == "patterns_static"
