"""Central paths and constants, so no module hard-codes a path.

Uses pathlib, so paths work the same on Windows and Linux.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"

FEATURES_CSV = PROCESSED_DATA_DIR / "features.csv"
MODEL_PATH = MODELS_DIR / "complexity_model.joblib"
MODEL_METADATA_PATH = MODELS_DIR / "model_metadata.json"

RANDOM_STATE = 42

# Input safety limits for code pasted/uploaded into the app.
MAX_SOURCE_BYTES = 200_000
MAX_SOURCE_LINES = 5_000

COMPLEXITY_LABELS = ("Low", "Medium", "High")
