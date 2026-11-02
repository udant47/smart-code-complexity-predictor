"""Integrity checks for the saved model files (Phase 18).

``joblib`` model files are Python pickles: loading one can run arbitrary code.
SmartCode therefore never unpickles a file unless all of these hold:

1. The file is a regular file (not a symlink, directory or device).
2. It is no larger than ``MAX_MODEL_BYTES``.
3. Its SHA-256 equals the hash recorded in ``model_metadata.json`` under
   ``integrity.files`` when this project trained or registered it.

Hashes are recorded automatically by ``python -m src.train_model`` and
``python -m src.evaluate_model``. Model files trained before Phase 18 can be
registered once with ``python -m src.model_integrity register``. Only do that
for files you trained yourself.

What this protects against: a model file that was replaced, swapped,
downloaded or corrupted after training is refused before it is unpickled.
What it does not protect against: someone who can already write to both the
model file and its metadata on your computer. That is not authentication.
Model files are never accepted from users through the app.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

from src import config

PREPROCESSING_FILE = "preprocessing_pipeline.joblib"
TRUSTED_FILES = (config.MODEL_PATH.name, PREPROCESSING_FILE)
MAX_MODEL_BYTES = 512 * 1024 * 1024  # the real model is about 10 MB
ALGORITHM = "sha256"
REGISTER_COMMAND = "python -m src.model_integrity register"


class IntegrityError(Exception):
    def __init__(self, code: str, message: str, hint: Optional[str] = None):
        super().__init__(message)
        self.code, self.message, self.hint = code, message, hint


def file_sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def integrity_block(paths: Iterable[Path]) -> dict:
    """The ``integrity`` metadata entry for the given files (missing files are skipped)."""
    files = {Path(p).name: file_sha256(Path(p)) for p in paths if Path(p).is_file()}
    return {"algorithm": ALGORITHM, "files": files}


def attach(meta: dict, models_dir: Path, names: Iterable[str] = TRUSTED_FILES) -> dict:
    """Return ``meta`` with an integrity entry for the model files present in ``models_dir``."""
    out = dict(meta)
    out["integrity"] = integrity_block(Path(models_dir) / n for n in names)
    return out


def check_file(path: Path, meta: dict) -> None:
    """Raise IntegrityError unless ``path`` is safe to unpickle under ``meta``."""
    path = Path(path)
    if path.is_symlink():
        raise IntegrityError("model_integrity_failed", f"{path.name} is a symbolic link; model files must be "
                             "regular files inside the models folder.")
    if not path.is_file():
        raise IntegrityError("model_missing", f"{path.name} is not a regular file.")
    size = path.stat().st_size
    if size > MAX_MODEL_BYTES:
        raise IntegrityError("model_integrity_failed",
                             f"{path.name} is {size / 1e6:,.0f} MB, above the {MAX_MODEL_BYTES / 1e6:,.0f} MB limit.")
    block = meta.get("integrity") if isinstance(meta, dict) else None
    recorded = (block or {}).get("files", {}).get(path.name) if isinstance(block, dict) else None
    if not block or (block.get("algorithm") != ALGORITHM) or not isinstance(recorded, str):
        raise IntegrityError(
            "model_unverified",
            f"No integrity hash is recorded for {path.name}, so it will not be loaded.",
            f"If you trained this model yourself, register it once with: {REGISTER_COMMAND}. "
            "Otherwise retrain with python -m src.evaluate_model.")
    actual = file_sha256(path)
    if actual != recorded:
        raise IntegrityError(
            "model_integrity_failed",
            f"{path.name} does not match the hash recorded when it was trained, so it will not be loaded.",
            "The file was changed or replaced after training. Retrain with python -m src.evaluate_model, "
            f"or, only if you replaced it yourself with a model you trained, run: {REGISTER_COMMAND}.")


@dataclass
class VerifyReport:
    ok: bool
    lines: list


def verify(models_dir: Path = config.MODELS_DIR) -> VerifyReport:
    meta_path = Path(models_dir) / config.MODEL_METADATA_PATH.name
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return VerifyReport(False, [f"metadata unreadable: {exc}"])
    lines, ok = [], True
    for name in TRUSTED_FILES:
        try:
            check_file(Path(models_dir) / name, meta)
            lines.append(f"OK        {name}")
        except IntegrityError as exc:
            ok = False
            lines.append(f"REFUSED   {name}: {exc.message}")
    return VerifyReport(ok, lines)


def register(models_dir: Path = config.MODELS_DIR) -> dict:
    """Record the current hashes of the local model files in the metadata (trust-on-first-use)."""
    meta_path = Path(models_dir) / config.MODEL_METADATA_PATH.name
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    present = [n for n in TRUSTED_FILES if (Path(models_dir) / n).is_file()]
    if not present:
        raise FileNotFoundError(f"No model files found in {models_dir}.")
    for n in present:
        p = Path(models_dir) / n
        if p.is_symlink() or p.stat().st_size > MAX_MODEL_BYTES:
            raise IntegrityError("model_integrity_failed", f"{n} cannot be registered (symlink or too large).")
    meta = attach(meta, models_dir, present)
    meta_path.write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    return meta["integrity"]


def main(argv: Optional[list] = None) -> int:
    ap = argparse.ArgumentParser(description="Record or verify SHA-256 hashes of the saved model files.")
    ap.add_argument("command", choices=["register", "verify"])
    ap.add_argument("--models-dir", type=Path, default=config.MODELS_DIR)
    args = ap.parse_args(argv)
    if args.command == "register":
        block = register(args.models_dir)
        print("Registered model files (only do this for models you trained yourself):")
        for name, h in block["files"].items():
            print(f"  {name}  sha256 {h}")
        return 0
    rep = verify(args.models_dir)
    print("\n".join(rep.lines))
    return 0 if rep.ok else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
