"""Language-specific analyzers mapped onto a common structural schema (Phase 19).

See ``base.py`` for the schema and the extension interface, ``registry.py``
for what is supported, and ``docs/EXTENSIBILITY.md`` for the design.
"""

from src.languages.base import (COMMON_FILE_FIELDS, COMMON_FUNCTION_FIELDS, Capabilities, FunctionMetrics, Issue,
                                LanguageAnalyzer, LanguageReport)
from src.languages.registry import LANGUAGES, analyzer_for, get, language_for_path, supported_languages

__all__ = ["COMMON_FILE_FIELDS", "COMMON_FUNCTION_FIELDS", "Capabilities", "FunctionMetrics", "Issue",
           "LanguageAnalyzer", "LanguageReport", "LANGUAGES", "analyzer_for", "get", "language_for_path",
           "supported_languages"]
