"""Validation tests for src.dataset_loader (Phase 7).

All tests use small hand-written fixtures and never touch the network.
The last section runs extra checks on the real data only if it has been
downloaded into data/raw (otherwise those tests are skipped).
"""

import io
import json
import zipfile
from pathlib import Path

import pandas as pd
import pytest

from src import config
from src import dataset_loader as dl
from src.feature_extractor import FILE_FEATURE_NAMES, FUNCTION_FEATURE_NAMES

# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
LOOP = "n = int(input())\ntotal = 0\nfor i in range(n):\n    total += i\nprint(total)\n"
LOOP_COMMENTED = "# read n\nn = int(input())\n\ntotal = 0  # sum\nfor i in range(n):\n    total += i\nprint(total)\n"
LOOP_RENAMED = "m = int(input())\nacc = 0\nfor j in range(m):\n    acc += j\nprint(acc)\n"
NESTED = "n = int(input())\nc = 0\nfor i in range(n):\n    for j in range(n):\n        c += 1\nprint(c)\n"
CONST = "a, b = map(int, input().split())\nprint(a + b)\n"
LONG_A = """a = int(input())
b = [0] * a
for i in range(a):
    b[i] = i * i
s = sum(b)
if s > 10:
    print(s)
else:
    print(-s)
t = sorted(b)
while a > 0:
    a //= 2
d = {}
for k, v in enumerate(t):
    d[k] = v + 1
m = max(d.values(), default=0)
r = [x for x in t if x % 2 == 0]
print(len(r), m)
q = min(b) if b else 0
print(q * 2, s - q)
print(t[0])
"""
LONG_B = LONG_A.replace("print(t[0])", "print(t[0], a)")  # small structural change: near duplicate


def record(i_src, label, problem="0001_A", **extra):
    return {"src": i_src, "complexity": label, "problem": problem, "from": "CODEFORCES", "tags": ["math"], **extra}


@pytest.fixture
def codecomplex_file(tmp_path: Path) -> Path:
    rows = [
        record(LOOP, "linear"),                         # 0 kept
        record(LOOP_COMMENTED, "linear"),               # 1 exact duplicate of 0 (comments ignored)
        record(LOOP_RENAMED, "linear"),                 # 2 renamed duplicate of 0
        record(NESTED, "quadratic", "0002_B"),          # 3 kept
        record(NESTED + "\n", "cubic", "0002_B"),       # 4 same code as 3, different label -> 3 and 4 removed
        record(CONST, "constant", "0003_C"),            # 5 kept
        record("def f(:\n    pass\n", "linear"),        # 6 syntax error
        record("   \n", "linear"),                      # 7 empty
        record(CONST, "superlinear"),                   # 8 unknown label
        {"src": CONST, "complexity": "constant"},       # 9 missing problem
        record(CONST, "constant", "not-an-id"),         # 10 bad problem id
        record(LONG_A, "constant", "0004_D"),           # 11 kept
        record(LONG_B, "constant", "0004_D"),           # 12 near duplicate of 11 (kept, flagged)
        record(LONG_A + "print(b[-1] // 3)\n", "constant", "0005_E", tags=[]),  # 13 kept (other problem)
    ]
    lines = [json.dumps(r) for r in rows]
    lines.insert(9, "{not json")  # raw line 9 is invalid JSON; later rows shift by one
    path = tmp_path / "python_data.jsonl"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


@pytest.fixture
def algorithms_dir(tmp_path: Path) -> Path:
    root = tmp_path / "algos"
    (root / "sorts").mkdir(parents=True)
    (root / "maths").mkdir()
    (root / ".github").mkdir()
    (root / "sorts" / "bubble.py").write_text(
        "def bubble(a):\n"
        "    for i in range(len(a)):\n"
        "        for j in range(len(a) - 1):\n"
        "            if a[j] > a[j + 1]:\n"
        "                a[j], a[j + 1] = a[j + 1], a[j]\n"
        "    return a\n",
        encoding="utf-8",
    )
    branches = "".join(f"    if x == {i}:\n        return {i}\n" for i in range(12))
    (root / "maths" / "lookup.py").write_text(
        f"def lookup(x):\n{branches}    return -1\n\n"
        "def stub():\n    pass\n\n"
        "def total(xs):\n    s = 0\n    for x in xs:\n        s += x\n    return s\n\n"
        "def total_again(values):\n    acc = 0\n    for v in values:\n        acc += v\n    return acc\n",
        encoding="utf-8",
    )
    (root / "maths" / "__init__.py").write_text("", encoding="utf-8")
    (root / "maths" / "broken.py").write_text("def f(:\n", encoding="utf-8")
    (root / ".github" / "hidden.py").write_text("def hidden():\n    return 1\n", encoding="utf-8")
    return root


# --------------------------------------------------------------------------- #
# Labels
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("cc, band", [(1, "Low"), (5, "Low"), (6, "Medium"), (10, "Medium"), (11, "High"), (60, "High")])
def test_structural_band_thresholds(cc, band):
    assert dl.structural_band(cc) == band


def test_structural_band_rejects_invalid():
    with pytest.raises(ValueError):
        dl.structural_band(0)


def test_bigo_label_map_is_complete_and_ordered():
    assert set(dl.BIGO_LABELS) == {"constant", "logn", "linear", "nlogn", "quadratic", "cubic", "np"}
    ranks = [v[2] for v in dl.BIGO_LABELS.values()]
    assert ranks == sorted(ranks) == list(range(7))
    assert dl.BIGO_LABELS["np"][1] == "O(2^n)"


# --------------------------------------------------------------------------- #
# Schema
# --------------------------------------------------------------------------- #
def test_schema_valid_record():
    assert dl.schema_problems(record(CONST, "constant")) == []


def test_schema_accepts_readme_alias_for_np():
    assert dl.schema_problems(record(CONST, "exponential")) == []


@pytest.mark.parametrize(
    "rec, problem",
    [
        ({"complexity": "linear", "problem": "0001_A"}, "missing_src"),
        ({"src": 3, "complexity": "linear", "problem": "0001_A"}, "wrong_type_src"),
        (record(CONST, "superlinear"), "unknown_label"),
        (record(CONST, "linear", "abc"), "bad_problem_id"),
        (record(CONST, "linear", tags=5), "wrong_type_tags"),
        ({"__invalid_json__": True}, "invalid_json"),
    ],
)
def test_schema_problems(rec, problem):
    assert problem in dl.schema_problems(rec)


def test_verify_schema_summary(codecomplex_file):
    records = dl.load_codecomplex(codecomplex_file)
    summary = dl.verify_schema(records)
    assert summary["records"] == 15
    assert summary["issues"] == {"invalid_json": 1, "unknown_label": 1, "missing_problem": 1, "bad_problem_id": 1}
    assert [r["raw_index"] for r in records] == list(range(15))


# --------------------------------------------------------------------------- #
# Duplicate detection
# --------------------------------------------------------------------------- #
def test_exact_tokens_ignore_comments_and_blank_lines():
    assert dl.code_tokens(LOOP, rename=False) == dl.code_tokens(LOOP_COMMENTED, rename=False)
    assert dl.code_tokens(LOOP, rename=False) != dl.code_tokens(LOOP_RENAMED, rename=False)


def test_renamed_tokens_match_renamed_code_but_keep_builtins():
    assert dl.code_tokens(LOOP, rename=True) == dl.code_tokens(LOOP_RENAMED, rename=True)
    a = dl.code_tokens("x = len(y)\n", rename=True)
    b = dl.code_tokens("x = sum(y)\n", rename=True)
    assert a != b  # different builtins are different programs


def test_tokenize_failure_returns_none():
    assert dl.code_tokens('s = """unterminated\n', rename=False) is None


def test_dedupe_keeps_first_and_drops_conflicting_groups():
    rows = [
        {"raw_index": 0, "k": "a", "label": "x"},
        {"raw_index": 1, "k": "a", "label": "x"},
        {"raw_index": 2, "k": "b", "label": "x"},
        {"raw_index": 3, "k": "b", "label": "y"},
        {"raw_index": 4, "k": "c", "label": "z"},
    ]
    removal = dl.Removal()
    kept = dl.dedupe(rows, "k", "label", removal, "dup")
    assert [r["raw_index"] for r in kept] == [0, 4]
    assert removal.counts == {"dup": 1, "dup_conflicting_labels": 2}


def test_near_duplicate_clusters_only_within_group():
    sh_a = dl.shingles(dl.code_tokens(LONG_A, rename=True))
    sh_b = dl.shingles(dl.code_tokens(LONG_B, rename=True))
    sh_c = dl.shingles(dl.code_tokens(NESTED, rename=True))
    assert dl.jaccard(sh_a, sh_b) >= 0.9 > dl.jaccard(sh_a, sh_c)
    rows = [
        {"raw_index": 0, "problem": "p1", "_shingles": sh_a},
        {"raw_index": 1, "problem": "p1", "_shingles": sh_b},
        {"raw_index": 2, "problem": "p1", "_shingles": sh_c},
        {"raw_index": 3, "problem": "p2", "_shingles": sh_a},  # same code, other problem
    ]
    clusters = dl.near_duplicate_clusters(rows, "problem")
    assert clusters == {0: 0, 1: 0}


# --------------------------------------------------------------------------- #
# Big-O dataset
# --------------------------------------------------------------------------- #
def test_build_bigo_dataset(codecomplex_file):
    df, stats = dl.build_bigo_dataset(dl.load_codecomplex(codecomplex_file))
    assert stats["raw_records"] == 15
    assert stats["final_samples"] == len(df) == 5
    assert stats["raw_records"] == stats["final_samples"] + stats["removed_total"]
    # raw lines: 0 LOOP, 5 CONST, 12-13 LONG_A/B near-duplicates, 14 LONG variant
    # (line 9 is the invalid JSON, so fixture rows 9-13 sit on lines 10-14)
    assert list(df["raw_index"]) == [0, 5, 12, 13, 14]


def test_bigo_removal_reasons(codecomplex_file):
    _, stats = dl.build_bigo_dataset(dl.load_codecomplex(codecomplex_file))
    removed = stats["removed"]
    assert removed["exact_duplicate"] == 1
    assert removed["exact_duplicate_conflicting_labels"] == 2
    assert removed["renamed_duplicate"] == 1
    assert removed["invalid_source:syntax"] == 1
    assert removed["empty_source"] == 1
    assert removed["schema:unknown_label"] == 1
    assert removed["schema:missing_problem"] == 1
    assert removed["schema:bad_problem_id"] == 1
    assert removed["schema:invalid_json"] == 1
    assert stats["raw_records"] == stats["final_samples"] + stats["removed_total"]


def test_bigo_columns_labels_and_origin(codecomplex_file):
    df, stats = dl.build_bigo_dataset(dl.load_codecomplex(codecomplex_file))
    assert list(df.columns) == list(dl.BIGO_META_COLUMNS) + list(dl.BIGO_FEATURE_COLUMNS)
    row = df.set_index("raw_index").loc[0]
    assert row["sample_id"] == "codecomplex-py-00000"
    assert (row["label"], row["label_name"], row["label_bigo"], row["label_rank"]) == ("linear", "linear", "O(n)", 2)
    assert row["source_commit"] == config.CODECOMPLEX["commit"]
    assert row["tags"] == "math"
    assert df.set_index("raw_index").loc[14, "tags"] == ""  # missing tags -> ""
    assert "expert annotation" in row["label_method"]
    assert df[list(dl.BIGO_FEATURE_COLUMNS)].notna().all().all()


def test_bigo_near_duplicates_flagged_not_removed(codecomplex_file):
    df, stats = dl.build_bigo_dataset(dl.load_codecomplex(codecomplex_file))
    flagged = df.set_index("raw_index")["near_duplicate_cluster"]
    assert flagged[12] == flagged[13] >= 0
    assert flagged[0] == -1
    assert stats["near_duplicates"]["samples_in_clusters"] == 2


def test_bigo_class_distribution_matches_rows(codecomplex_file):
    df, stats = dl.build_bigo_dataset(dl.load_codecomplex(codecomplex_file))
    dist = stats["class_distribution"]
    assert list(dist) == list(dl.BIGO_LABELS)
    assert sum(v["count"] for v in dist.values()) == len(df)
    assert dist["constant"]["count"] == 4 and dist["linear"]["count"] == 1
    assert dist["quadratic"]["count"] == 0  # both conflicting copies removed


def test_features_come_from_the_project_extractor(codecomplex_file):
    from src.feature_extractor import extract_features

    df, _ = dl.build_bigo_dataset(dl.load_codecomplex(codecomplex_file))
    row = df.set_index("raw_index").loc[0]
    expected = extract_features(LOOP).file_features
    assert {k: row[k] for k in FILE_FEATURE_NAMES} == pytest.approx(expected)


# --------------------------------------------------------------------------- #
# Structural dataset
# --------------------------------------------------------------------------- #
def test_build_structural_dataset(algorithms_dir):
    df, stats = dl.build_structural_dataset(algorithms_dir)
    assert stats["python_files"] == 4  # hidden .github folder ignored
    assert stats["files_removed"] == {"file_empty": 1, "file_invalid:syntax": 1}
    assert stats["removed"] == {"empty_function": 1, "renamed_duplicate_function": 1}
    assert list(df["function_id"]) == ["lookup", "total", "bubble"]
    assert list(df.columns) == list(dl.STRUCTURAL_META_COLUMNS) + list(FUNCTION_FEATURE_NAMES)
    assert set(df["source_license"]) == {"MIT"}
    assert df.set_index("function_id").loc["total", "source_path"] == "maths/lookup.py"


def test_structural_labels_are_derived_from_cyclomatic_complexity(algorithms_dir):
    """Documents the circularity: the reference label is a function of a feature."""
    df, stats = dl.build_structural_dataset(algorithms_dir)
    assert all(dl.structural_band(cc) == lab for cc, lab in zip(df["cyclomatic_complexity"], df["reference_label"]))
    labels = df.set_index("function_id")["reference_label"]
    assert labels["lookup"] == "High" and labels["total"] == "Low"
    assert "not ground truth" in stats["label_warning"]
    assert "not human-labelled" in stats["label_method"]


# --------------------------------------------------------------------------- #
# End-to-end build, report and reproducibility
# --------------------------------------------------------------------------- #
def test_build_writes_outputs_and_is_reproducible(codecomplex_file, algorithms_dir, tmp_path):
    out1, out2 = tmp_path / "out1", tmp_path / "out2"
    report = dl.build(codecomplex_file, algorithms_dir, out1)
    dl.build(codecomplex_file, algorithms_dir, out2)
    for name in (config.BIGO_DATASET_CSV.name, config.STRUCTURAL_DATASET_CSV.name):
        assert (out1 / name).read_bytes() == (out2 / name).read_bytes()

    saved = json.loads((out1 / config.DATASET_REPORT_JSON.name).read_text(encoding="utf-8"))
    assert saved["bigo"]["final_samples"] == len(pd.read_csv(out1 / config.BIGO_DATASET_CSV.name))
    assert saved["bigo"]["file_sha256_matches_pinned"] is False  # fixture, not the real file
    assert saved["environment"]["python"]
    md = (out1 / config.DATASET_REPORT_MD.name).read_text(encoding="utf-8")
    assert "Final samples" in md and "not ground truth" in md
    assert report["bigo"]["output_sha256"] == dl.sha256_bytes((out1 / config.BIGO_DATASET_CSV.name).read_bytes())


def test_build_requires_downloaded_data(tmp_path):
    with pytest.raises(FileNotFoundError, match="download"):
        dl.build(tmp_path / "missing.jsonl", tmp_path, tmp_path / "out")


# --------------------------------------------------------------------------- #
# Downloads (network replaced by fakes)
# --------------------------------------------------------------------------- #
def test_codecomplex_download_verifies_checksum(tmp_path, monkeypatch):
    monkeypatch.setattr(dl, "_fetch", lambda url: b"tampered")
    with pytest.raises(RuntimeError, match="checksum mismatch"):
        dl.download_codecomplex(tmp_path)
    assert not (tmp_path / config.CODECOMPLEX["file"]).exists()


def test_codecomplex_download_uses_pinned_commit(tmp_path, monkeypatch):
    data = b'{"src": "x", "complexity": "constant", "problem": "0001_A"}\n'
    seen = []
    monkeypatch.setattr(dl, "_fetch", lambda url: seen.append(url) or data)
    monkeypatch.setitem(config.CODECOMPLEX, "sha256", dl.sha256_bytes(data))
    path = dl.download_codecomplex(tmp_path)
    assert path.read_bytes() == data
    assert config.CODECOMPLEX["commit"] in seen[0] and seen[0].startswith("https://")


def test_archive_extraction_is_safe_and_verified(tmp_path, monkeypatch):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("Python-abc/sorts/ok.py", "def ok():\n    return 1\n")
        zf.writestr("Python-abc/../../evil.py", "x = 1\n")
        zf.writestr("Python-abc/.github/hidden.py", "x = 1\n")
        zf.writestr("Python-abc/README.md", "readme")
    monkeypatch.setattr(dl, "_fetch", lambda url: buf.getvalue())
    dest = tmp_path / "algos"
    with pytest.raises(RuntimeError, match="fingerprint mismatch"):
        dl.download_thealgorithms(dest)
    assert (dest / "sorts" / "ok.py").exists()
    assert not (tmp_path / "evil.py").exists() and not (tmp_path.parent / "evil.py").exists()
    assert not (dest / ".github").exists() and not (dest / "README.md").exists()


# --------------------------------------------------------------------------- #
# Real data (only when downloaded)
# --------------------------------------------------------------------------- #
REAL_CC = config.CODECOMPLEX_DIR / config.CODECOMPLEX["file"]


@pytest.mark.skipif(not REAL_CC.exists(), reason="CodeComplex not downloaded")
def test_real_codecomplex_file_matches_pinned_checksum():
    assert dl.sha256_bytes(REAL_CC.read_bytes()) == config.CODECOMPLEX["sha256"]
    records = dl.load_codecomplex(REAL_CC)
    assert len(records) == 4900
    assert dl.verify_schema(records)["issues"] == {}


@pytest.mark.skipif(not config.BIGO_DATASET_CSV.exists(), reason="dataset not built")
def test_real_processed_bigo_dataset():
    df = pd.read_csv(config.BIGO_DATASET_CSV, keep_default_na=False)
    assert len(df) > 4000
    assert set(df["label"]) == set(dl.BIGO_LABELS)
    assert df["sample_id"].is_unique
    assert df[list(FILE_FEATURE_NAMES)].notna().all().all()
