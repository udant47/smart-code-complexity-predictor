"""Local repository analysis (Phase 19): safe enumeration, per-file structural analysis and a summary.

Kept separate from the core parser: it depends only on ``src.languages``. No
network access, no remote repositories, no authentication. Read-only on the
analysed folder.
"""

from src.repository.analyzer import RepositoryReport, analyze_repository, render_markdown, summarize
from src.repository.scanner import Limits, ScanError, scan

__all__ = ["Limits", "RepositoryReport", "ScanError", "analyze_repository", "render_markdown", "scan", "summarize"]
