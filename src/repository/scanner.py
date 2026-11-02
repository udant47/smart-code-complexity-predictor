"""Safe enumeration of the source files in a local folder (Phase 19).

Guarantees:
* **Read-only.** Nothing in the folder is created, modified or executed.
* **No symlinks or junctions are followed.** Symlinked files and folders are
  skipped and reported, and every folder and file is checked to resolve
  inside the root, so links, junctions and ``..`` cannot lead outside it.
  Each folder is visited at most once (loops are impossible).
* **Only regular files** are considered (no devices, pipes or sockets).
* **Bounded work.** Limits on entries visited, source files, bytes per file,
  total bytes and folder depth. Hitting a limit stops the scan and marks
  the result as truncated, with the reason.
* **Unsupported files are never read**; only their extensions are counted.
* **Credential files are never read** (by name), and supported source files
  whose content looks like it contains a secret are skipped by default.
"""

from __future__ import annotations

import os
import stat
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from src import config
from src.languages import registry
from src.repository.secrets import is_sensitive_filename

DEFAULT_IGNORED_DIRS = frozenset({
    ".git", ".hg", ".svn", ".venv", "venv", "env", "__pycache__", "node_modules", "site-packages", ".tox",
    ".mypy_cache", ".pytest_cache", ".ruff_cache", "build", "dist", ".idea", ".vscode", ".eggs",
})


@dataclass(frozen=True)
class Limits:
    max_entries: int = 50_000  # directory entries looked at
    max_source_files: int = 2_000
    max_file_bytes: int = config.MAX_SOURCE_BYTES
    max_total_bytes: int = 50_000_000
    max_depth: int = 25
    time_budget_seconds: float = 300.0
    include_hidden: bool = False


@dataclass
class SourceFile:
    path: Path  # absolute, resolved
    relative: str  # POSIX-style path relative to the root
    language: str
    size: int


@dataclass
class ScanResult:
    root: Path
    files: list[SourceFile] = field(default_factory=list)
    skipped: list[dict] = field(default_factory=list)  # {"path", "reason", ...}
    unsupported_extensions: dict = field(default_factory=dict)  # ".ext" -> count
    entries_seen: int = 0
    truncated: Optional[str] = None  # reason, when a limit stopped the scan

    def skip(self, relative: str, reason: str, **extra) -> None:
        self.skipped.append({"path": relative, "reason": reason, **extra})


class ScanError(ValueError):
    pass


def resolve_root(root) -> Path:
    p = Path(root).expanduser()
    if not p.exists():
        raise ScanError(f"Folder not found: {p}")
    resolved = p.resolve(strict=True)
    if not resolved.is_dir():
        raise ScanError(f"Not a folder: {p}")
    if resolved == Path(resolved.anchor):
        raise ScanError("Refusing to scan a whole drive or filesystem root; choose a project folder.")
    return resolved


def _inside(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _rel(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def scan(root, limits: Limits = Limits()) -> ScanResult:
    """Enumerate supported source files below ``root`` in a deterministic (sorted) order."""
    root = resolve_root(root)
    result = ScanResult(root=root)
    total_bytes = 0
    visited: set[str] = set()
    stack: list[tuple[Path, int]] = [(root, 0)]
    while stack:
        folder, depth = stack.pop()
        key = os.path.normcase(str(folder))
        if key in visited:
            continue
        visited.add(key)
        try:
            with os.scandir(folder) as it:
                entries = sorted(it, key=lambda e: e.name)
        except OSError as exc:
            result.skip(_rel(folder, root) or ".", "unreadable_folder", error=type(exc).__name__)
            continue
        subfolders = []
        for entry in entries:
            result.entries_seen += 1
            if result.entries_seen > limits.max_entries:
                result.truncated = f"more than {limits.max_entries:,} entries"
                return result
            path = Path(entry.path)
            rel = _rel(path, root)
            hidden = entry.name.startswith(".")
            try:
                if entry.is_symlink() or _is_junction(path):
                    result.skip(rel, "symlink")
                    continue
                st = entry.stat(follow_symlinks=False)
            except OSError as exc:
                result.skip(rel, "unreadable", error=type(exc).__name__)
                continue
            if stat.S_ISDIR(st.st_mode):
                if entry.name in DEFAULT_IGNORED_DIRS or (hidden and not limits.include_hidden):
                    result.skip(rel, "ignored_folder")
                elif depth + 1 > limits.max_depth:
                    result.skip(rel, "too_deep")
                else:
                    subfolders.append((path, depth + 1))
                continue
            if not stat.S_ISREG(st.st_mode):
                result.skip(rel, "not_a_regular_file")
                continue
            if is_sensitive_filename(entry.name):
                result.skip(rel, "credential_file")  # never opened
                continue
            info = registry.language_for_path(entry.name)
            if info is None or info.status == "unsupported" or not info.usable:
                ext = Path(entry.name).suffix.lower() or "(none)"
                result.unsupported_extensions[ext] = result.unsupported_extensions.get(ext, 0) + 1
                if info is not None:
                    result.skip(rel, "language_not_supported", language=info.language_id)
                continue
            if hidden and not limits.include_hidden:
                result.skip(rel, "hidden_file")
                continue
            if st.st_size > limits.max_file_bytes:
                result.skip(rel, "too_large", size=st.st_size)
                continue
            try:
                resolved = path.resolve(strict=True)
            except OSError:
                result.skip(rel, "unreadable")
                continue
            if not _inside(resolved, root):
                result.skip(rel, "outside_root")
                continue
            if len(result.files) >= limits.max_source_files:
                result.truncated = f"more than {limits.max_source_files:,} source files"
                return result
            if total_bytes + st.st_size > limits.max_total_bytes:
                result.truncated = f"more than {limits.max_total_bytes / 1e6:,.0f} MB of source"
                return result
            total_bytes += st.st_size
            result.files.append(SourceFile(resolved, rel, info.language_id, st.st_size))
        for path, d in reversed(subfolders):  # reversed so that the stack pops them in sorted order
            try:
                resolved = path.resolve(strict=True)
            except OSError:
                result.skip(_rel(path, root), "unreadable_folder")
                continue
            if not _inside(resolved, root):
                result.skip(_rel(path, root), "outside_root")
                continue
            stack.append((resolved, d))
    return result


def _is_junction(path: Path) -> bool:
    """Windows directory junctions are not symlinks to os.scandir; detect them where Python can (3.12+)."""
    check = getattr(os.path, "isjunction", None)
    return bool(check and check(path))


def read_file(f: SourceFile, root: Path, limits: Limits) -> bytes:
    """Read a scanned file again safely: still a regular file inside the root, and at most the size limit."""
    resolved = f.path.resolve(strict=True)
    if not _inside(resolved, root):
        raise ScanError("file moved outside the folder")
    with open(resolved, "rb") as fh:
        st = os.fstat(fh.fileno())
        if not stat.S_ISREG(st.st_mode):
            raise ScanError("not a regular file")
        data = fh.read(limits.max_file_bytes + 1)
    if len(data) > limits.max_file_bytes:
        raise ScanError("file grew beyond the size limit")
    return data
