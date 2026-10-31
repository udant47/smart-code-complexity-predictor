"""Builds the training dataset from Python files on disk.

Responsibility:
    Read .py files from ``data/raw/`` (read as text only, never imported),
    run parser -> function analyzer -> feature extractor, attach labels,
    and write ``data/processed/features.csv``.

Planned public API (Phase 2):
    build_dataset(raw_dir: Path, out_path: Path) -> pd.DataFrame
    load_processed(path: Path) -> pd.DataFrame
"""
