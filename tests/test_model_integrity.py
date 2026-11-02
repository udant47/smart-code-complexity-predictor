"""Model files are never unpickled unless they match the hashes recorded at training time (Phase 18)."""

import ast
import json
import pickle
import shutil
from pathlib import Path

import pytest

from src import config
from src import model_integrity as mi
from src import predictor as pr
from src.examples import get_example
from tests.test_predictor import model_dir, paths  # noqa: F401

CODE = get_example("merge_sort").code


class _MarkerOnUnpickle:
    """A pickle that creates a file when it is loaded: proof of whether unpickling happened."""

    def __init__(self, marker: Path):
        self.marker = marker

    def __reduce__(self):
        return (Path.touch, (self.marker,))


def _copy(src: Path, dst: Path) -> Path:
    d = dst / "models"
    shutil.copytree(src, d)
    return d


def _meta(d: Path) -> dict:
    return json.loads((d / "model_metadata.json").read_text())


def _write_meta(d: Path, meta: dict) -> None:
    (d / "model_metadata.json").write_text(json.dumps(meta))


def test_hash_helpers(tmp_path):
    f = tmp_path / "a.bin"
    f.write_bytes(b"abc")
    assert mi.file_sha256(f) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    block = mi.integrity_block([f, tmp_path / "missing.bin"])
    assert block == {"algorithm": "sha256", "files": {"a.bin": mi.file_sha256(f)}}
    assert mi.attach({"x": 1}, tmp_path, ["a.bin"])["integrity"] == block


def test_valid_model_loads(model_dir):  # noqa: F811
    assert pr.predict(CODE, **paths(model_dir)).prediction is not None


@pytest.mark.parametrize("break_meta", [
    lambda m: m.pop("integrity"),
    lambda m: m["integrity"].update(algorithm="md5"),
    lambda m: m["integrity"]["files"].pop("complexity_model.joblib"),
])
def test_unverified_model_is_refused(model_dir, tmp_path, break_meta):  # noqa: F811
    d = _copy(model_dir, tmp_path)
    meta = _meta(d)
    break_meta(meta)
    _write_meta(d, meta)
    r = pr.predict(CODE, **paths(d))
    assert r.prediction is None and r.errors[0].code == "model_unverified"
    assert mi.REGISTER_COMMAND in r.errors[0].hint
    assert r.functions and r.derived_score["highest"] is not None  # static analysis still works


@pytest.mark.parametrize("hash_recorded", [False, True], ids=["no-hash", "hash-of-original"])
def test_malicious_pickle_is_never_unpickled(model_dir, tmp_path, hash_recorded):  # noqa: F811
    d = _copy(model_dir, tmp_path)
    marker = tmp_path / "unpickled.txt"
    (d / "complexity_model.joblib").write_bytes(pickle.dumps(_MarkerOnUnpickle(marker)))
    if not hash_recorded:
        meta = _meta(d)
        meta.pop("integrity")
        _write_meta(d, meta)
    r = pr.predict(CODE, **paths(d))
    assert r.prediction is None
    assert r.errors[0].code == ("model_integrity_failed" if hash_recorded else "model_unverified")
    assert not marker.exists(), "the replaced model file was unpickled"


def test_preprocessing_file_is_checked_too(model_dir, tmp_path):  # noqa: F811
    d = _copy(model_dir, tmp_path)
    marker = tmp_path / "unpickled.txt"
    (d / "preprocessing_pipeline.joblib").write_bytes(pickle.dumps(_MarkerOnUnpickle(marker)))
    r = pr.predict(CODE, **paths(d))
    assert r.errors[0].code == "model_integrity_failed" and not marker.exists()


def test_joblib_load_is_not_called_when_integrity_fails(model_dir, tmp_path, monkeypatch):  # noqa: F811
    import joblib

    d = _copy(model_dir, tmp_path)
    (d / "complexity_model.joblib").write_bytes(b"tampered")
    monkeypatch.setattr(joblib, "load", lambda *a, **k: pytest.fail("joblib.load called on an unverified file"))
    assert pr.predict(CODE, **paths(d)).errors[0].code == "model_integrity_failed"


def test_symlinked_model_is_refused(model_dir, tmp_path):  # noqa: F811
    d = _copy(model_dir, tmp_path)
    real = tmp_path / "elsewhere.joblib"
    shutil.move(d / "complexity_model.joblib", real)
    try:
        (d / "complexity_model.joblib").symlink_to(real)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks are not available on this system")
    r = pr.predict(CODE, **paths(d))
    assert r.errors[0].code == "model_integrity_failed" and "symbolic link" in r.errors[0].message


def test_oversized_model_is_refused(model_dir, tmp_path, monkeypatch):  # noqa: F811
    d = _copy(model_dir, tmp_path)
    monkeypatch.setattr(mi, "MAX_MODEL_BYTES", 100)
    r = pr.predict(CODE, **paths(d))
    assert r.errors[0].code == "model_integrity_failed" and "limit" in r.errors[0].message


def test_register_and_verify_cli(model_dir, tmp_path, capsys):  # noqa: F811
    d = _copy(model_dir, tmp_path)
    assert mi.main(["verify", "--models-dir", str(d)]) == 0
    (d / "complexity_model.joblib").write_bytes(b"retrained elsewhere")
    assert mi.main(["verify", "--models-dir", str(d)]) == 1
    assert "REFUSED   complexity_model.joblib" in capsys.readouterr().out
    assert mi.main(["register", "--models-dir", str(d)]) == 0
    assert "only do this for models you trained yourself" in capsys.readouterr().out
    assert _meta(d)["integrity"]["files"]["complexity_model.joblib"] == mi.file_sha256(d / "complexity_model.joblib")
    assert mi.verify(d).ok
    # Registered junk passes the hash check but still fails safely as a load error.
    assert pr.predict(CODE, **paths(d)).errors[0].code == "model_load_failed"


def test_verify_reports_unreadable_metadata(tmp_path):
    (tmp_path / "model_metadata.json").write_text("{broken")
    rep = mi.verify(tmp_path)
    assert not rep.ok and "metadata unreadable" in rep.lines[0]


def test_training_commands_record_hashes():
    for rel in ("src/train_model.py", "src/evaluate_model.py"):
        src = (config.PROJECT_ROOT / rel).read_text(encoding="utf-8")
        tree = ast.parse(src)
        dumps = [n.lineno for n in ast.walk(tree) if isinstance(n, ast.Call) and getattr(n.func, "attr", "") == "dump"
                 and getattr(n.func.value, "id", "") == "joblib"]
        attaches = [n.lineno for n in ast.walk(tree) if isinstance(n, ast.Call)
                    and getattr(n.func, "attr", "") == "attach"]
        assert dumps and attaches and max(dumps) < min(attaches), rel


@pytest.mark.skipif(not config.MODEL_PATH.exists(), reason="no trained model on this computer")
def test_installed_model_is_registered():
    rep = mi.verify()
    assert rep.ok, "\n".join(rep.lines) + f"\nRun: {mi.REGISTER_COMMAND}"
