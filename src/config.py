"""Central paths and constants, so no module hard-codes a path.

Uses pathlib, so paths work the same on Windows and Linux.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"

MODEL_PATH = MODELS_DIR / "complexity_model.joblib"
MODEL_METADATA_PATH = MODELS_DIR / "model_metadata.json"

RANDOM_STATE = 42

# Input safety limits for code pasted/uploaded into the app.
MAX_SOURCE_BYTES = 200_000
MAX_SOURCE_LINES = 5_000

# Structural (Task A) reference bands - see dataset_loader for the caveats.
COMPLEXITY_LABELS = ("Low", "Medium", "High")

# --------------------------------------------------------------------------- #
# Datasets (selected in Phase 2), pinned to exact versions for reproducibility
# --------------------------------------------------------------------------- #
CODECOMPLEX = {
    "name": "CodeComplex (Python subset)",
    "repository": "https://github.com/sybaik1/CodeComplex-Data",
    "commit": "c54631de24c0c3b0e180e876a9f244fb0905cf86",
    "file": "python_data.jsonl",
    "sha256": "0cf4923ad68e57a42392b4f4787476fed9392d182c886f4aec73cd70175e1c3b",
    "paper": "Baik et al., CodeComplex: Dataset for Worst-Case Time Complexity Prediction, arXiv:2401.08719",
    "paper_url": "https://arxiv.org/abs/2401.08719",
    "license": "Not stated in the repository; used for non-commercial research and not redistributed.",
}
CODECOMPLEX_DIR = RAW_DATA_DIR / "codecomplex"

THEALGORITHMS = {
    "name": "TheAlgorithms/Python",
    "repository": "https://github.com/TheAlgorithms/Python",
    "commit": "6e03a48579ca212a08b36be00378f887dc41a003",
    "py_file_count": 1608,
    # sha256 over sorted "relative/path\0sha256(file)\n" lines of every .py file
    "content_fingerprint": "2f51e3cbed9ca169769d2ef0efbbd78049ad19552c30074285f520b0cc1c11a6",
    "license": "MIT (Copyright (c) 2016-2022 TheAlgorithms and contributors)",
}
THEALGORITHMS_DIR = RAW_DATA_DIR / "thealgorithms_python"

BIGO_DATASET_CSV = PROCESSED_DATA_DIR / "bigo_codecomplex_python.csv"
STRUCTURAL_DATASET_CSV = PROCESSED_DATA_DIR / "structural_functions.csv"
DATASET_REPORT_JSON = PROCESSED_DATA_DIR / "dataset_report.json"
DATASET_REPORT_MD = PROCESSED_DATA_DIR / "dataset_report.md"
