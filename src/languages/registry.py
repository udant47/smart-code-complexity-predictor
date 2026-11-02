"""Which languages SmartCode supports, and how well (Phase 19).

Status values:
* ``full``          the complete Python pipeline (structural metrics, Big-O, ML, score, recommendations)
* ``structural``    structural metrics and category only, from a real parser for that language (C)
* ``unsupported``   listed so the UI can say so explicitly; nothing is analysed

C++, Java and JavaScript are deliberately ``unsupported``: no parser or
mapping exists for them, and pretending otherwise would produce wrong numbers.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import PurePath
from typing import Optional

from src.languages.base import LanguageAnalyzer
from src.languages.c_lang import CAnalyzer
from src.languages.python_lang import PythonAnalyzer


@dataclass(frozen=True)
class LanguageInfo:
    language_id: str
    display_name: str
    extensions: tuple[str, ...]
    status: str  # full | structural | unsupported
    analyzer: Optional[type] = None
    note: str = ""

    def available(self) -> tuple[bool, str]:
        if self.analyzer is None:
            return False, f"{self.display_name} is not supported."
        return self.analyzer.available()

    @property
    def usable(self) -> bool:
        return self.available()[0]


LANGUAGES: dict[str, LanguageInfo] = {
    "python": LanguageInfo("python", "Python", PythonAnalyzer.extensions, "full", PythonAnalyzer,
                           "Full analysis."),
    "c": LanguageInfo("c", "C", CAnalyzer.extensions, "structural", CAnalyzer,
                      "Experimental: structural metrics only. No ML class, Big-O or Derived Complexity Score."),
    "cpp": LanguageInfo("cpp", "C++", (".cpp", ".cc", ".cxx", ".hpp", ".hh", ".hxx"), "unsupported", None,
                        "Not supported: no C++ parser is integrated."),
    "java": LanguageInfo("java", "Java", (".java",), "unsupported", None,
                         "Not supported: no Java parser is integrated."),
    "javascript": LanguageInfo("javascript", "JavaScript", (".js", ".mjs", ".cjs"), "unsupported", None,
                               "Not supported: no JavaScript parser is integrated."),
}

_BY_EXTENSION = {ext: info for info in LANGUAGES.values() for ext in info.extensions}


def get(language_id: str) -> LanguageInfo:
    return LANGUAGES[language_id]


def language_for_path(path) -> Optional[LanguageInfo]:
    """Language by file extension (case-insensitive); None for unknown extensions. ``.h`` is treated as C."""
    return _BY_EXTENSION.get(PurePath(str(path)).suffix.lower())


def analyzer_for(language_id: str, **kwargs) -> LanguageAnalyzer:
    """An analyzer instance; raises ValueError for unsupported or unavailable languages."""
    info = LANGUAGES[language_id]
    if info.analyzer is None:
        raise ValueError(f"{info.display_name} is not supported.")
    ok, reason = info.available()
    if not ok:
        raise ValueError(reason)
    return info.analyzer(**kwargs)


def supported_languages() -> list[LanguageInfo]:
    """Languages that can be analysed on this computer right now."""
    return [i for i in LANGUAGES.values() if i.analyzer is not None and i.usable]
