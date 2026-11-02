"""Analyse every supported source file in a local folder and summarise it (Phase 19).

Separate from the core parser: this module only uses the language registry
(``src.languages``), the same analyzers the app uses for single files. It never
runs the ML model (that was validated on whole competitive-programming
programs, not on arbitrary repository files) and never executes anything.

Per file: structural metrics in the common schema (and, for Python, the
static Big-O class of each function). Summary: totals, per-language figures,
the cyclomatic category distribution, the most complex functions and files,
and every skipped file with its reason. Reports contain metrics, paths and
function names only, never source text.
"""

from __future__ import annotations

import statistics
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Optional

from src.languages import registry
from src.repository import scanner
from src.repository.secrets import scan_text
from src.suggestions import THRESHOLDS

HIGH_CC = THRESHOLDS["cyclomatic_medium"]  # 11: radon "High" band, same threshold as the refactoring rules
DEEP_NESTING = THRESHOLDS["nesting_medium"]  # 4
LONG_FUNCTION = THRESHOLDS["length_medium"]  # 50 lines


@dataclass
class FileResult:
    path: str
    language: str
    success: bool
    file_metrics: Optional[dict] = None
    functions: list = field(default_factory=list)
    error: Optional[dict] = None
    warnings: list = field(default_factory=list)


@dataclass
class RepositoryReport:
    root: str
    limits: dict
    files: list[FileResult]
    skipped: list[dict]
    unsupported_extensions: dict
    truncated: Optional[str]
    elapsed_seconds: float
    summary: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"root": self.root, "limits": self.limits, "truncated": self.truncated,
                "elapsed_seconds": round(self.elapsed_seconds, 2), "summary": self.summary,
                "files": [asdict(f) for f in self.files], "skipped": self.skipped,
                "unsupported_extensions": self.unsupported_extensions}


def _flags(fn: dict) -> list[str]:
    flags = []
    if fn["cyclomatic_complexity"] >= HIGH_CC:
        flags.append(f"cyclomatic ≥ {HIGH_CC}")
    if fn["maximum_nesting_depth"] >= DEEP_NESTING:
        flags.append(f"nesting ≥ {DEEP_NESTING}")
    if fn["function_length"] >= LONG_FUNCTION:
        flags.append(f"length ≥ {LONG_FUNCTION} lines")
    return flags


def analyze_repository(root, limits: scanner.Limits = scanner.Limits(), *, skip_files_with_secrets: bool = True,
                       top: int = 20, progress: Optional[Callable[[int, int, str], None]] = None,
                       clock: Callable[[], float] = time.monotonic) -> RepositoryReport:
    start = clock()
    scan = scanner.scan(root, limits)
    analyzers = {}
    results: list[FileResult] = []
    skipped = list(scan.skipped)
    truncated = scan.truncated
    for i, f in enumerate(scan.files):
        if clock() - start > limits.time_budget_seconds:
            truncated = f"time budget of {limits.time_budget_seconds:g} s reached"
            skipped.extend({"path": g.relative, "reason": "time_budget"} for g in scan.files[i:])
            break
        if progress:
            progress(i + 1, len(scan.files), f.relative)
        try:
            data = scanner.read_file(f, scan.root, limits)
        except (OSError, scanner.ScanError) as exc:
            skipped.append({"path": f.relative, "reason": "unreadable", "error": str(exc)[:80]})
            continue
        text = data.decode("utf-8", errors="replace")
        findings = scan_text(text)
        if findings and skip_files_with_secrets:
            skipped.append({"path": f.relative, "reason": "possible_secret",
                            "findings": [x.to_dict() for x in findings]})
            continue
        analyzer = analyzers.get(f.language) or analyzers.setdefault(f.language, registry.analyzer_for(f.language))
        try:
            rep = analyzer.analyze_bytes(data)
        except Exception as exc:  # an analyzer bug must not stop the whole repository
            results.append(FileResult(f.relative, f.language, False,
                                      error={"code": "analysis_failed", "message": type(exc).__name__}))
            continue
        d = rep.to_dict()
        if rep.success:
            fns = [{**fn, "flags": _flags(fn)} for fn in d["functions"]]
            for fn in fns:
                fn.pop("extra", None)
            results.append(FileResult(f.relative, f.language, True, d["file_metrics"], fns,
                                      warnings=[w["code"] for w in d["warnings"]]))
        else:
            e = d["errors"][0]
            results.append(FileResult(f.relative, f.language, False,
                                      error={"code": e["code"], "message": e["message"], "line": e.get("line")}))
    report = RepositoryReport(str(scan.root), asdict(limits), results, skipped, scan.unsupported_extensions,
                              truncated, clock() - start)
    report.summary = summarize(report, top)
    return report


def summarize(report: RepositoryReport, top: int = 20) -> dict:
    ok = [f for f in report.files if f.success]
    functions = [{**fn, "path": f.path, "language": f.language} for f in ok for fn in f.functions]
    by_lang: dict[str, dict] = {}
    for f in ok:
        lang = by_lang.setdefault(f.language, {"files": 0, "code_lines": 0, "functions": 0, "cc": []})
        lang["files"] += 1
        lang["code_lines"] += f.file_metrics["code_lines"]
        lang["functions"] += len(f.functions)
        lang["cc"].extend(fn["cyclomatic_complexity"] for fn in f.functions)
    languages = {}
    for name, v in sorted(by_lang.items()):
        cc = v.pop("cc")
        languages[name] = {**v, "mean_cyclomatic": round(statistics.fmean(cc), 2) if cc else None,
                           "median_cyclomatic": statistics.median(cc) if cc else None,
                           "high_complexity_functions": sum(c >= HIGH_CC for c in cc)}
    categories = {"Low": 0, "Medium": 0, "High": 0}
    for fn in functions:
        categories[fn["structural_category"]] += 1
    order = sorted(functions, key=lambda fn: (-fn["cyclomatic_complexity"], -fn["maximum_nesting_depth"],
                                              -fn["function_length"], fn["path"], fn["start_line"]))
    keep = ("path", "language", "qualname", "start_line", "end_line", "cyclomatic_complexity",
            "maximum_nesting_depth", "function_length", "structural_category", "static_time", "flags")
    flagged = [fn for fn in order if fn["flags"]]
    file_rows = sorted(({"path": f.path, "language": f.language, "functions": len(f.functions),
                         "max_cyclomatic": f.file_metrics["maximum_function_cyclomatic_complexity"],
                         "total_function_cyclomatic": sum(fn["cyclomatic_complexity"] for fn in f.functions),
                         "code_lines": f.file_metrics["code_lines"]} for f in ok),
                       key=lambda r: (-r["max_cyclomatic"], -r["total_function_cyclomatic"], r["path"]))
    skipped_by_reason: dict[str, int] = {}
    for s in report.skipped:
        skipped_by_reason[s["reason"]] = skipped_by_reason.get(s["reason"], 0) + 1
    return {
        "files_analyzed": len(ok),
        "files_failed": len(report.files) - len(ok),
        "files_skipped": len(report.skipped),
        "skipped_by_reason": dict(sorted(skipped_by_reason.items())),
        "code_lines": sum(f.file_metrics["code_lines"] for f in ok),
        "functions": len(functions),
        "languages": languages,
        "category_distribution": categories,
        "thresholds": {"high_cyclomatic": HIGH_CC, "deep_nesting": DEEP_NESTING, "long_function": LONG_FUNCTION},
        "high_complexity_function_count": len(flagged),
        "most_complex_functions": [{k: fn.get(k) for k in keep} for fn in order[:top]],
        "flagged_functions": [{k: fn.get(k) for k in keep} for fn in flagged[:top]],
        "most_complex_files": file_rows[:top],
        "failed_files": [{"path": f.path, **(f.error or {})} for f in report.files if not f.success][:top],
        "unsupported_extensions": dict(sorted(report.unsupported_extensions.items(), key=lambda kv: -kv[1])[:10]),
    }


def render_markdown(report: RepositoryReport) -> str:
    s = report.summary
    lines = [f"# Repository complexity report", "", f"Folder: `{report.root}`  ",
             f"Analysed in {report.elapsed_seconds:.1f} s. Structural metrics only (no ML predictions); "
             "nothing was executed.", ""]
    if report.truncated:
        lines += [f"> **Incomplete:** the scan stopped because of a limit ({report.truncated}).", ""]
    lines += ["## Summary", "", "| | |", "|---|---|",
              f"| Files analysed | {s['files_analyzed']} |", f"| Files that could not be parsed | {s['files_failed']} |",
              f"| Files skipped | {s['files_skipped']} |", f"| Code lines | {s['code_lines']:,} |",
              f"| Functions | {s['functions']:,} |",
              f"| Functions flagged (cyclomatic ≥ {HIGH_CC}, nesting ≥ {DEEP_NESTING} or ≥ {LONG_FUNCTION} lines) | "
              f"{s['high_complexity_function_count']} |", ""]
    if s["languages"]:
        lines += ["## Languages", "", "| Language | Files | Code lines | Functions | Mean CC | Median CC | CC ≥ 11 |",
                  "|---|---|---|---|---|---|---|"]
        for name, v in s["languages"].items():
            lines.append(f"| {registry.get(name).display_name} | {v['files']} | {v['code_lines']:,} | {v['functions']} | "
                         f"{v['mean_cyclomatic']} | {v['median_cyclomatic']} | {v['high_complexity_functions']} |")
        lines.append("")
    dist = s["category_distribution"]
    lines += ["## Cyclomatic categories (functions)", "",
              f"Low (1–5): {dist['Low']} · Medium (6–10): {dist['Medium']} · High (11+): {dist['High']}", ""]
    if s["most_complex_functions"]:
        lines += ["## Most complex functions", "", "| Function | File | Lines | CC | Nesting | Length | Static time | Flags |",
                  "|---|---|---|---|---|---|---|---|"]
        for fn in s["most_complex_functions"]:
            lines.append(f"| `{fn['qualname']}` | `{fn['path']}` | {fn['start_line']}–{fn['end_line']} | "
                         f"{fn['cyclomatic_complexity']} | {fn['maximum_nesting_depth']} | {fn['function_length']} | "
                         f"{fn['static_time'] or '—'} | {', '.join(fn['flags']) or '—'} |")
        lines.append("")
    if s["failed_files"]:
        lines += ["## Files that could not be parsed", ""]
        lines += [f"- `{f['path']}`: {f.get('message', f.get('code'))}" for f in s["failed_files"]]
        lines.append("")
    if s["skipped_by_reason"]:
        lines += ["## Skipped", ""] + [f"- {reason}: {n}" for reason, n in s["skipped_by_reason"].items()]
        lines += ["", "Files with possible secrets are listed with the rule and line only; the values are never "
                      "copied into this report.", ""]
    lines += ["## Notes", "",
              "- Cyclomatic complexity and nesting measure control flow, not quality or speed.",
              "- C support is experimental and structural only. C files that rely on syntax-changing macros "
              "cannot be parsed and are listed above instead of being measured approximately.",
              "- Static time is the rule-based Big-O estimate for Python functions; it is blank where unknown or "
              "not available for the language."]
    return "\n".join(lines) + "\n"
