"""Reproducible dataset preparation (Phase 7).

Builds two processed datasets from the sources selected in Phase 2. No
synthetic or random feature values are used: every row comes from parsing
real source code with the project's own parser and feature extractor.

1. Big-O dataset: ``data/processed/bigo_codecomplex_python.csv``
   Source: CodeComplex, Python subset (4,900 Codeforces solutions).
   Labels: expert annotations of worst-case time complexity, one per
   *program* (not per function), in 7 classes. They are human judgements
   following the authors' written guidelines, not proofs, and no
   inter-annotator agreement is published. There are no space-complexity
   labels, so auxiliary space stays with the rule-based analyser.

2. Structural dataset: ``data/processed/structural_functions.csv``
   Source: functions from TheAlgorithms/Python (MIT licence).
   Labels: **heuristic reference bands, not ground truth.** No
   human-labelled dataset of structural complexity was found (Phase 2),
   so each function gets the band of its McCabe cyclomatic complexity under
   radon's published rank thresholds: Low = rank A (1-5), Medium = rank B
   (6-10), High = rank C or worse (11+).

   **Circularity risk.** These labels are a deterministic function of
   ``cyclomatic_complexity``, which is also one of the extracted
   features. A model trained on them with that feature would only
   relearn two thresholds, and its accuracy would measure nothing. Even
   without it, decision counts (if/elif/loops/and/or) nearly determine it.
   This dataset is therefore meant for distribution analysis and for
   checking the rule-based structural score, not for claiming that a
   model "predicts maintainability".

The two label sets are never mixed: structural bands are not Big-O labels.

Cleaning steps (each one is counted in the report)
--------------------------------------------------
* Schema check: required fields, types, allowed labels, problem-id format.
* Validation: empty code, oversized code and unparsable code are removed.
  Parsing uses the running Python version, so record it with the counts.
* Exact duplicates (type-1 clones): identical token streams, ignoring
  comments, blank lines and spacing.
* Renamed duplicates (type-2 clones): identical after replacing
  identifiers, numbers and strings with placeholders.
* Within each group of identical code, the first record (by raw index) is
  kept if all labels agree. If labels conflict, the whole group is removed
  as label noise.
* Likely near-duplicates (CodeComplex): token 5-gram Jaccard similarity
  >= 0.9 between solutions of the same problem. These are flagged with a
  cluster id but kept, because Phase 8 splits by problem, so they cannot
  leak between training and test data.
* Missing values: optional metadata (tags) is filled with "". Features
  come from the extractor and are complete when parsing succeeds; any
  NaN is an error.

Command line
------------
    python -m src.dataset_loader download   # fetch the pinned raw data
    python -m src.dataset_loader build      # produce the processed CSVs and report
    python -m src.dataset_loader all        # both

No submitted or downloaded code is ever executed: it is only read as text
and parsed with ``ast``.
"""

from __future__ import annotations

import argparse
import builtins
import hashlib
import io
import json
import keyword
import platform
import re
import sys
import tokenize
import urllib.request
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

from src import config
from src.code_parser import parse_source
from src.feature_extractor import FILE_FEATURE_NAMES, FUNCTION_FEATURE_NAMES, extract_features
from src.function_analyzer import analyze_functions

# --------------------------------------------------------------------------- #
# Labels
# --------------------------------------------------------------------------- #
# CodeComplex label -> (canonical name, Big-O notation, ordinal rank).
# The README calls the last class 'exponential'; the data uses 'np'; the paper
# describes it as NP-hard problems solved by exponential-time search.
BIGO_LABELS: dict[str, tuple[str, str, int]] = {
    "constant": ("constant", "O(1)", 0),
    "logn": ("logarithmic", "O(log n)", 1),
    "linear": ("linear", "O(n)", 2),
    "nlogn": ("linearithmic", "O(n log n)", 3),
    "quadratic": ("quadratic", "O(n^2)", 4),
    "cubic": ("cubic", "O(n^3)", 5),
    "np": ("exponential", "O(2^n)", 6),
}
BIGO_LABEL_ALIASES = {"exponential": "np"}

STRUCTURAL_BANDS = (("Low", 1, 5), ("Medium", 6, 10), ("High", 11, None))
STRUCTURAL_LABEL_METHOD = (
    "heuristic: band of McCabe cyclomatic complexity using radon rank thresholds "
    "(A 1-5 = Low, B 6-10 = Medium, C+ >= 11 = High); not human-labelled"
)

CODECOMPLEX_REQUIRED = {"src": str, "complexity": str, "problem": str}
CODECOMPLEX_OPTIONAL = {"from": str, "tags": (list, str)}
PROBLEM_ID_PATTERN = r"\d+_[A-Z][0-9]?"
NEAR_DUPLICATE_THRESHOLD = 0.9
SHINGLE_SIZE = 5


def structural_band(cyclomatic_complexity: int) -> str:
    for name, lo, hi in STRUCTURAL_BANDS:
        if cyclomatic_complexity >= lo and (hi is None or cyclomatic_complexity <= hi):
            return name
    raise ValueError(f"cyclomatic complexity must be >= 1, got {cyclomatic_complexity}")


# --------------------------------------------------------------------------- #
# Download (pinned versions, verified)
# --------------------------------------------------------------------------- #
def _fetch(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "smart-code-complexity-predictor"})
    with urllib.request.urlopen(req, timeout=120) as resp:  # noqa: S310 - fixed https URLs
        return resp.read()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def download_codecomplex(dest_dir: Path = config.CODECOMPLEX_DIR) -> Path:
    meta = config.CODECOMPLEX
    owner_repo = meta["repository"].split("github.com/")[1]
    url = f"https://raw.githubusercontent.com/{owner_repo}/{meta['commit']}/{meta['file']}"
    dest_dir.mkdir(parents=True, exist_ok=True)
    target = dest_dir / meta["file"]
    if target.exists() and sha256_bytes(target.read_bytes()) == meta["sha256"]:
        print(f"CodeComplex already present: {target}")
        return target
    print(f"Downloading {url}")
    data = _fetch(url)
    actual = sha256_bytes(data)
    if actual != meta["sha256"]:
        raise RuntimeError(f"CodeComplex checksum mismatch: expected {meta['sha256']}, got {actual}")
    target.write_bytes(data)
    print(f"Saved {target} ({len(data):,} bytes, sha256 verified)")
    return target


def algorithms_fingerprint(root: Path) -> tuple[int, str]:
    files = sorted(
        p for p in root.rglob("*.py")
        if not any(part.startswith(".") for part in p.relative_to(root).parts)
    )
    h = hashlib.sha256()
    for p in files:
        rel = p.relative_to(root).as_posix()
        h.update(rel.encode() + b"\0" + sha256_bytes(p.read_bytes()).encode() + b"\n")
    return len(files), h.hexdigest()


def download_thealgorithms(dest_dir: Path = config.THEALGORITHMS_DIR) -> Path:
    meta = config.THEALGORITHMS
    if dest_dir.exists():
        count, fp = algorithms_fingerprint(dest_dir)
        if fp == meta["content_fingerprint"]:
            print(f"TheAlgorithms/Python already present: {dest_dir}")
            return dest_dir
    owner_repo = meta["repository"].split("github.com/")[1]
    url = f"https://github.com/{owner_repo}/archive/{meta['commit']}.zip"
    print(f"Downloading {url} (about 15 MB)")
    data = _fetch(url)
    dest_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        for info in zf.infolist():
            parts = info.filename.split("/", 1)
            if len(parts) < 2 or not parts[1] or info.is_dir():
                continue
            rel = parts[1]
            if not (rel.endswith(".py") or rel == "LICENSE.md"):
                continue
            if any(seg.startswith(".") or seg == ".." for seg in rel.split("/")):
                continue  # skip hidden folders; never write outside dest_dir
            target = (dest_dir / rel).resolve()
            if dest_dir.resolve() not in target.parents:
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(zf.read(info))
    count, fp = algorithms_fingerprint(dest_dir)
    if fp != meta["content_fingerprint"]:
        raise RuntimeError(
            f"TheAlgorithms/Python content fingerprint mismatch ({count} files): expected "
            f"{meta['content_fingerprint']}, got {fp}"
        )
    print(f"Saved {count} .py files to {dest_dir} (fingerprint verified)")
    return dest_dir


def write_manifest(raw_dir: Path = config.RAW_DATA_DIR) -> Path:
    manifest = {
        "codecomplex": {**config.CODECOMPLEX},
        "thealgorithms_python": {**config.THEALGORITHMS},
    }
    path = raw_dir / "MANIFEST.json"
    raw_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return path


# --------------------------------------------------------------------------- #
# Loading and schema checks
# --------------------------------------------------------------------------- #
@dataclass
class Removal:
    """Counts of removed records by reason (for the report)."""

    counts: Counter = field(default_factory=Counter)

    def add(self, reason: str, n: int = 1) -> None:
        self.counts[reason] += n

    def total(self) -> int:
        return sum(self.counts.values())


def load_codecomplex(path: Path) -> list[dict]:
    """Read the JSONL file; each record keeps its 0-based line index as raw_index."""
    records = []
    with open(path, encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                obj = {"__invalid_json__": True}
            obj["raw_index"] = i
            records.append(obj)
    return records


def schema_problems(record: dict) -> list[str]:
    """Return a list of schema violations (empty when the record is valid)."""
    if record.get("__invalid_json__"):
        return ["invalid_json"]
    problems = []
    for key, typ in CODECOMPLEX_REQUIRED.items():
        if key not in record or record[key] is None:
            problems.append(f"missing_{key}")
        elif not isinstance(record[key], typ):
            problems.append(f"wrong_type_{key}")
    for key, typ in CODECOMPLEX_OPTIONAL.items():
        if key in record and record[key] is not None and not isinstance(record[key], typ):
            problems.append(f"wrong_type_{key}")
    label = record.get("complexity")
    if isinstance(label, str) and BIGO_LABEL_ALIASES.get(label, label) not in BIGO_LABELS:
        problems.append("unknown_label")
    problem = record.get("problem")
    if isinstance(problem, str) and not re.fullmatch(PROBLEM_ID_PATTERN, problem):
        problems.append("bad_problem_id")
    return problems


def verify_schema(records: list[dict]) -> dict:
    """Summarise schema conformance over all records."""
    issues = Counter()
    for r in records:
        for p in schema_problems(r):
            issues[p] += 1
    fields = Counter(k for r in records for k in r if k != "raw_index")
    return {"records": len(records), "field_presence": dict(sorted(fields.items())), "issues": dict(issues)}


# --------------------------------------------------------------------------- #
# Duplicate detection
# --------------------------------------------------------------------------- #
_BUILTINS = set(dir(builtins))
_SKIP_TOKENS = {tokenize.COMMENT, tokenize.NL, tokenize.ENCODING, tokenize.ENDMARKER}


def code_tokens(source: str, rename: bool) -> Optional[list[str]]:
    """Token stream without comments/blank lines; with ``rename``, identifiers,
    numbers and strings become placeholders (keywords and builtins are kept)."""
    out = []
    try:
        for tok in tokenize.generate_tokens(io.StringIO(source).readline):
            if tok.type in _SKIP_TOKENS:
                continue
            if tok.type == tokenize.NEWLINE:
                out.append("<NL>")
            elif tok.type == tokenize.INDENT:
                out.append("<IN>")
            elif tok.type == tokenize.DEDENT:
                out.append("<DE>")
            elif rename and tok.type == tokenize.NAME and not keyword.iskeyword(tok.string) and tok.string not in _BUILTINS:
                out.append("ID")
            elif rename and tok.type == tokenize.NUMBER:
                out.append("NUM")
            elif rename and tok.type == tokenize.STRING:
                out.append("STR")
            elif tok.type in (getattr(tokenize, "FSTRING_START", -1), getattr(tokenize, "FSTRING_MIDDLE", -1), getattr(tokenize, "FSTRING_END", -1)):
                out.append("STR" if rename else tok.string)
            else:
                out.append(tok.string)
    except (tokenize.TokenError, SyntaxError, IndentationError):
        return None
    if rename:  # collapse f-string pieces so 3.11 and 3.12+ agree
        collapsed = []
        for t in out:
            if not (t == "STR" and collapsed and collapsed[-1] == "STR"):
                collapsed.append(t)
        out = collapsed
    return out


def token_hash(tokens: list[str]) -> str:
    return sha256_bytes("\x1f".join(tokens).encode("utf-8"))


def dedupe(rows: list[dict], key: str, label_key: Optional[str], removal: Removal, reason: str) -> list[dict]:
    """Keep the first row of each group with equal ``key``; drop conflicting-label groups."""
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        groups[r[key]].append(r)
    kept = []
    for r in rows:
        group = groups[r[key]]
        if r is not group[0]:
            continue
        if label_key and len({g[label_key] for g in group}) > 1:
            removal.add(f"{reason}_conflicting_labels", len(group))
            continue
        removal.add(reason, len(group) - 1)
        kept.append(r)
    return kept


def shingles(tokens: list[str], k: int = SHINGLE_SIZE) -> frozenset:
    if len(tokens) < k:
        return frozenset({tuple(tokens)})
    return frozenset(tuple(tokens[i:i + k]) for i in range(len(tokens) - k + 1))


def jaccard(a: frozenset, b: frozenset) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def near_duplicate_clusters(rows: list[dict], group_key: str, threshold: float = NEAR_DUPLICATE_THRESHOLD) -> dict[int, int]:
    """Union-find over pairs with Jaccard >= threshold inside each group.

    Returns raw_index -> cluster id for rows in a cluster of size >= 2.
    """
    parent: dict[int, int] = {}

    def find(x: int) -> int:
        while parent.setdefault(x, x) != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    by_group: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_group[r[group_key]].append(r)
    for members in by_group.values():
        sh = [(m["raw_index"], m["_shingles"]) for m in members]
        for i in range(len(sh)):
            for j in range(i + 1, len(sh)):
                a, b = sh[i][1], sh[j][1]
                # cheap bound: Jaccard <= min/max size ratio
                if min(len(a), len(b)) < threshold * max(len(a), len(b)):
                    continue
                if jaccard(a, b) >= threshold:
                    ra, rb = find(sh[i][0]), find(sh[j][0])
                    if ra != rb:
                        parent[max(ra, rb)] = min(ra, rb)
    clusters: dict[int, list[int]] = defaultdict(list)
    for x in list(parent):
        clusters[find(x)].append(x)
    out = {}
    for root, members in clusters.items():
        if len(members) > 1:
            for m in members:
                out[m] = root
    return out


# --------------------------------------------------------------------------- #
# Big-O dataset (CodeComplex)
# --------------------------------------------------------------------------- #
BIGO_META_COLUMNS = (
    "sample_id", "source_dataset", "source_commit", "raw_index", "problem", "tags",
    "label", "label_name", "label_bigo", "label_rank", "label_method",
    "near_duplicate_cluster", "code_sha256",
)


def build_bigo_dataset(records: list[dict]) -> tuple[pd.DataFrame, dict]:
    removal = Removal()
    rows = []
    for r in records:
        problems = schema_problems(r)
        if problems:
            removal.add(f"schema:{problems[0]}")
            continue
        src = r["src"]
        if not src.strip():
            removal.add("empty_source")
            continue
        report = extract_features(src)
        if not report.success:
            removal.add(f"invalid_source:{report.error.kind}")
            continue
        tokens = code_tokens(report.parse_result.source, rename=False)
        renamed = code_tokens(report.parse_result.source, rename=True)
        if tokens is None or renamed is None:
            removal.add("invalid_source:tokenize")
            continue
        label = BIGO_LABEL_ALIASES.get(r["complexity"], r["complexity"])
        tags = r.get("tags")
        rows.append({
            "raw_index": r["raw_index"],
            "problem": r["problem"],
            "tags": ",".join(tags) if isinstance(tags, list) else (tags or ""),
            "label": label,
            "_exact": token_hash(tokens),
            "_renamed": token_hash(renamed),
            "_shingles": shingles(renamed),
            "code_sha256": sha256_bytes(src.encode("utf-8")),
            "features": report.file_features,
        })

    valid = len(rows)
    rows = dedupe(rows, "_exact", "label", removal, "exact_duplicate")
    rows = dedupe(rows, "_renamed", "label", removal, "renamed_duplicate")
    clusters = near_duplicate_clusters(rows, "problem")

    table = []
    for r in rows:
        name, bigo, rank = BIGO_LABELS[r["label"]]
        table.append({
            "sample_id": f"codecomplex-py-{r['raw_index']:05d}",
            "source_dataset": config.CODECOMPLEX["name"],
            "source_commit": config.CODECOMPLEX["commit"],
            "raw_index": r["raw_index"],
            "problem": r["problem"],
            "tags": r["tags"],
            "label": r["label"],
            "label_name": name,
            "label_bigo": bigo,
            "label_rank": rank,
            "label_method": "expert annotation (CodeComplex authors), whole program, worst-case time",
            "near_duplicate_cluster": clusters.get(r["raw_index"], -1),
            "code_sha256": r["code_sha256"],
            **{k: r["features"][k] for k in FILE_FEATURE_NAMES},
        })
    df = pd.DataFrame(table, columns=list(BIGO_META_COLUMNS) + list(FILE_FEATURE_NAMES))
    _check_no_missing(df, FILE_FEATURE_NAMES)

    in_clusters = int((df["near_duplicate_cluster"] >= 0).sum())
    stats = {
        "raw_records": len(records),
        "valid_records": valid,
        "removed": dict(sorted(removal.counts.items())),
        "removed_total": removal.total(),
        "final_samples": len(df),
        "problems": int(df["problem"].nunique()),
        "max_samples_per_problem": int(df["problem"].value_counts().max()) if len(df) else 0,
        "near_duplicates": {
            "threshold_jaccard": NEAR_DUPLICATE_THRESHOLD,
            "shingle_size": SHINGLE_SIZE,
            "samples_in_clusters": in_clusters,
            "clusters": int(df.loc[df["near_duplicate_cluster"] >= 0, "near_duplicate_cluster"].nunique()),
            "policy": "flagged, not removed; all near-duplicates share a problem, and Phase 8 splits by problem",
        },
        "class_distribution": _distribution(df["label"], [k for k in BIGO_LABELS]),
    }
    return df, stats


# --------------------------------------------------------------------------- #
# Structural dataset (TheAlgorithms/Python functions)
# --------------------------------------------------------------------------- #
STRUCTURAL_META_COLUMNS = (
    "sample_id", "source_dataset", "source_commit", "source_license", "source_path",
    "function_id", "kind", "start_line", "end_line",
    "reference_label", "label_method", "code_sha256",
)


def iter_python_files(root: Path) -> Iterable[Path]:
    for p in sorted(root.rglob("*.py")):
        if any(part.startswith(".") for part in p.relative_to(root).parts):
            continue
        yield p


def build_structural_dataset(root: Path) -> tuple[pd.DataFrame, dict]:
    removal = Removal()
    file_removal = Removal()
    rows = []
    files_seen = 0
    for path in iter_python_files(root):
        files_seen += 1
        rel = path.relative_to(root).as_posix()
        try:
            text = path.read_bytes().decode("utf-8")
        except UnicodeDecodeError:
            file_removal.add("file_not_utf8")
            continue
        if not text.strip():
            file_removal.add("file_empty")
            continue
        report = extract_features(text)
        if not report.success:
            file_removal.add(f"file_invalid:{report.error.kind}")
            continue
        records = {f.start_line: f for f in analyze_functions(report.parse_result).functions}
        for feat in report.function_features:
            rec = records[feat["lineno"]]
            if rec.is_empty:
                removal.add("empty_function")
                continue
            info = report.parse_result.get_function(feat["qualname"])
            segment = next(
                (f.source for f in report.parse_result.functions if f.lineno == feat["lineno"]),
                info.source if info else "",
            )
            renamed = code_tokens(_dedent(segment), rename=True)
            if renamed is None:
                removal.add("function_tokenize_error")
                continue
            rows.append({
                "path": rel,
                "record": rec,
                "features": feat,
                "_renamed": token_hash(renamed),
                "code_sha256": sha256_bytes(segment.encode("utf-8")),
            })
    candidates = len(rows) + removal.total()
    rows = dedupe(rows, "_renamed", None, removal, "renamed_duplicate_function")

    table = []
    for i, r in enumerate(rows):
        f, rec = r["features"], r["record"]
        table.append({
            "sample_id": f"thealgorithms-fn-{i:05d}",
            "source_dataset": config.THEALGORITHMS["name"],
            "source_commit": config.THEALGORITHMS["commit"],
            "source_license": "MIT",
            "source_path": r["path"],
            "function_id": rec.function_id,
            "kind": rec.kind,
            "start_line": rec.start_line,
            "end_line": rec.end_line,
            "reference_label": structural_band(f["cyclomatic_complexity"]),
            "label_method": STRUCTURAL_LABEL_METHOD,
            "code_sha256": r["code_sha256"],
            **{k: f[k] for k in FUNCTION_FEATURE_NAMES},
        })
    df = pd.DataFrame(table, columns=list(STRUCTURAL_META_COLUMNS) + list(FUNCTION_FEATURE_NAMES))
    _check_no_missing(df, FUNCTION_FEATURE_NAMES)
    stats = {
        "python_files": files_seen,
        "files_removed": dict(sorted(file_removal.counts.items())),
        "candidate_functions": candidates,
        "removed": dict(sorted(removal.counts.items())),
        "final_samples": len(df),
        "class_distribution": _distribution(df["reference_label"], [b[0] for b in STRUCTURAL_BANDS]),
        "label_method": STRUCTURAL_LABEL_METHOD,
        "label_warning": (
            "reference_label is a deterministic function of the cyclomatic_complexity feature. "
            "Training on it would relearn the thresholds; it is not ground truth."
        ),
    }
    return df, stats


def _dedent(segment: str) -> str:
    import textwrap

    return textwrap.dedent(segment)


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #
def _distribution(series: pd.Series, order: list[str]) -> dict:
    counts = series.value_counts()
    total = int(counts.sum())
    return {
        k: {"count": int(counts.get(k, 0)), "percent": round(100 * counts.get(k, 0) / total, 2) if total else 0.0}
        for k in order
    }


def _check_no_missing(df: pd.DataFrame, columns: Iterable[str]) -> None:
    missing = df[list(columns)].isna().sum()
    bad = missing[missing > 0]
    if len(bad):
        raise ValueError(f"missing feature values: {bad.to_dict()}")


def environment() -> dict:
    return {
        "python": platform.python_version(),
        "pandas": pd.__version__,
        "platform": platform.platform(),
        "note": "Parsing uses the running Python's grammar; counts can differ slightly between versions.",
    }


def render_markdown(report: dict) -> str:
    b, s = report["bigo"], report["structural"]
    lines = [
        "# Dataset report",
        "",
        f"Built with Python {report['environment']['python']}. Regenerate with `python -m src.dataset_loader build`.",
        "",
        "## Big-O dataset (CodeComplex, Python)",
        "",
        f"Source: {config.CODECOMPLEX['repository']} @ `{config.CODECOMPLEX['commit'][:12]}`. "
        f"Labels: expert annotations of worst-case time complexity per program ({config.CODECOMPLEX['paper']}).",
        "",
        f"- Raw records: {b['raw_records']:,}",
        f"- Valid after schema and parsing checks: {b['valid_records']:,}",
        f"- Removed: {b['removed_total']:,} ({', '.join(f'{k} {v}' for k, v in b['removed'].items()) or 'none'})",
        f"- **Final samples: {b['final_samples']:,}** across {b['problems']} problems "
        f"(max {b['max_samples_per_problem']} per problem)",
        f"- Near-duplicates flagged (Jaccard >= {b['near_duplicates']['threshold_jaccard']}): "
        f"{b['near_duplicates']['samples_in_clusters']:,} samples in {b['near_duplicates']['clusters']:,} clusters",
        "",
        "| Class | Big-O | Samples | % |",
        "| --- | --- | ---: | ---: |",
    ]
    for k, v in b["class_distribution"].items():
        lines.append(f"| {BIGO_LABELS[k][0]} | {BIGO_LABELS[k][1]} | {v['count']:,} | {v['percent']} |")
    lines += [
        "",
        "## Structural dataset (TheAlgorithms/Python functions)",
        "",
        f"Source: {config.THEALGORITHMS['repository']} @ `{config.THEALGORITHMS['commit'][:12]}` (MIT).",
        f"Label method: {STRUCTURAL_LABEL_METHOD}.",
        "",
        f"> **Warning:** {s['label_warning']}",
        "",
        f"- Python files: {s['python_files']:,} (removed: {', '.join(f'{k} {v}' for k, v in s['files_removed'].items()) or 'none'})",
        f"- Candidate functions: {s['candidate_functions']:,}",
        f"- Removed: {', '.join(f'{k} {v}' for k, v in s['removed'].items()) or 'none'}",
        f"- **Final samples: {s['final_samples']:,}**",
        "",
        "| Reference band | Samples | % |",
        "| --- | ---: | ---: |",
    ]
    for k, v in s["class_distribution"].items():
        lines.append(f"| {k} | {v['count']:,} | {v['percent']} |")
    lines += [
        "",
        "## What these labels are not",
        "",
        "- Structural bands are not Big-O labels, and Big-O labels are not maintainability scores.",
        "- CodeComplex has no space-complexity labels; auxiliary space remains rule-based.",
        "- CodeComplex labels describe whole programs; function-level Big-O is not directly validated by them.",
        "",
    ]
    return "\n".join(lines)


def build(
    codecomplex_file: Path = config.CODECOMPLEX_DIR / config.CODECOMPLEX["file"],
    algorithms_dir: Path = config.THEALGORITHMS_DIR,
    out_dir: Path = config.PROCESSED_DATA_DIR,
) -> dict:
    """Build both processed datasets and the report; returns the report dict."""
    if not codecomplex_file.exists():
        raise FileNotFoundError(f"{codecomplex_file} not found - run `python -m src.dataset_loader download` first")
    if not algorithms_dir.exists():
        raise FileNotFoundError(f"{algorithms_dir} not found - run `python -m src.dataset_loader download` first")

    cc_sha = sha256_bytes(codecomplex_file.read_bytes())
    records = load_codecomplex(codecomplex_file)
    schema = verify_schema(records)
    bigo_df, bigo_stats = build_bigo_dataset(records)
    bigo_stats["file_sha256"] = cc_sha
    bigo_stats["file_sha256_matches_pinned"] = cc_sha == config.CODECOMPLEX["sha256"]
    bigo_stats["schema"] = schema

    n_files, fingerprint = algorithms_fingerprint(algorithms_dir)
    struct_df, struct_stats = build_structural_dataset(algorithms_dir)
    struct_stats["content_fingerprint"] = fingerprint
    struct_stats["fingerprint_matches_pinned"] = fingerprint == config.THEALGORITHMS["content_fingerprint"]

    out_dir.mkdir(parents=True, exist_ok=True)
    bigo_path = out_dir / config.BIGO_DATASET_CSV.name
    struct_path = out_dir / config.STRUCTURAL_DATASET_CSV.name
    bigo_df.to_csv(bigo_path, index=False, lineterminator="\n")
    struct_df.to_csv(struct_path, index=False, lineterminator="\n")

    report = {
        "environment": environment(),
        "bigo": {**bigo_stats, "output": bigo_path.name, "output_sha256": sha256_bytes(bigo_path.read_bytes())},
        "structural": {**struct_stats, "output": struct_path.name, "output_sha256": sha256_bytes(struct_path.read_bytes())},
        "sources": {"codecomplex": config.CODECOMPLEX, "thealgorithms_python": config.THEALGORITHMS},
    }
    (out_dir / config.DATASET_REPORT_JSON.name).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / config.DATASET_REPORT_MD.name).write_text(render_markdown(report), encoding="utf-8")
    return report


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m src.dataset_loader", description=__doc__.split("\n")[0])
    parser.add_argument("command", choices=["download", "build", "all"])
    parser.add_argument("--codecomplex-file", type=Path, default=config.CODECOMPLEX_DIR / config.CODECOMPLEX["file"])
    parser.add_argument("--algorithms-dir", type=Path, default=config.THEALGORITHMS_DIR)
    parser.add_argument("--out-dir", type=Path, default=config.PROCESSED_DATA_DIR)
    args = parser.parse_args(argv)

    if args.command in ("download", "all"):
        download_codecomplex(args.codecomplex_file.parent)
        download_thealgorithms(args.algorithms_dir)
        print(f"Manifest: {write_manifest()}")
    if args.command in ("build", "all"):
        report = build(args.codecomplex_file, args.algorithms_dir, args.out_dir)
        print(render_markdown(report))
        print(f"Wrote {args.out_dir / config.BIGO_DATASET_CSV.name}")
        print(f"Wrote {args.out_dir / config.STRUCTURAL_DATASET_CSV.name}")
        print(f"Wrote {args.out_dir / config.DATASET_REPORT_MD.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
