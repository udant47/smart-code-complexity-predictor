"""Phase 1 smoke tests: the scaffold is importable and the folders exist."""

import importlib

import pytest

from src.config import MODELS_DIR, PROCESSED_DATA_DIR, PROJECT_ROOT, RAW_DATA_DIR

MODULES = [
    "src",
    "src.config",
    "src.code_parser",
    "src.function_analyzer",
    "src.feature_extractor",
    "src.algorithm_analyzer",
    "src.complexity_score",
    "src.dataset_loader",
    "src.preprocessing",
    "src.train_model",
    "src.evaluate_model",
    "src.predictor",
    "src.explainability",
    "src.suggestions",
    "src.ui",
    "src.ui.components",
    "src.ui.styles",
]


@pytest.mark.parametrize("name", MODULES)
def test_module_imports(name):
    importlib.import_module(name)


@pytest.mark.parametrize("folder", [RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR])
def test_data_folders_exist(folder):
    assert folder.is_dir()


def test_core_dependencies_installed():
    for lib in ["streamlit", "pandas", "numpy", "sklearn", "plotly", "matplotlib", "joblib"]:
        importlib.import_module(lib)


def test_project_root_has_app():
    assert (PROJECT_ROOT / "app.py").is_file()
