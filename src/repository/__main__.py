"""Command line: ``python -m src.repository PATH [options]``.

Analyses a local folder (read-only) and writes ``repository_report.json`` and
``repository_report.md`` to the output folder, which defaults to
``reports/repository`` in this project. The output folder may not be inside
the analysed folder, so the analysed project is never modified.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from src import config
from src.repository.analyzer import analyze_repository, render_markdown
from src.repository.scanner import Limits, ScanError, resolve_root

DEFAULT_OUT = config.PROJECT_ROOT / "reports" / "repository"


def write_outputs(report, out_dir: Path) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path, md_path = out_dir / "repository_report.json", out_dir / "repository_report.md"
    json_path.write_text(json.dumps(report.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
    md_path.write_text(render_markdown(report), encoding="utf-8")
    return json_path, md_path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m src.repository",
                                 description="Structural complexity summary of a local folder (read-only).")
    ap.add_argument("path", type=Path, help="folder to analyse")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT, help="output folder (default: reports/repository)")
    ap.add_argument("--max-files", type=int, default=Limits.max_source_files)
    ap.add_argument("--max-total-mb", type=float, default=Limits.max_total_bytes / 1e6)
    ap.add_argument("--time-budget", type=float, default=Limits.time_budget_seconds, help="seconds")
    ap.add_argument("--max-depth", type=int, default=Limits.max_depth)
    ap.add_argument("--include-hidden", action="store_true", help="also scan hidden files and folders")
    ap.add_argument("--analyze-files-with-secrets", action="store_true",
                    help="analyse files that look like they contain secrets (values are never reported)")
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)

    try:
        root = resolve_root(args.path)
    except ScanError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    out = args.out.expanduser().resolve()
    if out == root or root in out.parents:
        print("Error: the output folder must be outside the analysed folder (it is never modified).",
              file=sys.stderr)
        return 2
    limits = Limits(max_source_files=args.max_files, max_total_bytes=int(args.max_total_mb * 1e6),
                    time_budget_seconds=args.time_budget, max_depth=args.max_depth,
                    include_hidden=args.include_hidden)

    def progress(i, n, rel):
        if not args.quiet and (i == n or i % 50 == 0):
            print(f"  {i}/{n} files", file=sys.stderr)

    report = analyze_repository(root, limits, skip_files_with_secrets=not args.analyze_files_with_secrets,
                                top=args.top, progress=progress)
    json_path, md_path = write_outputs(report, out)
    s = report.summary
    print(f"Analysed {s['files_analyzed']} files ({s['functions']} functions, {s['code_lines']:,} code lines) "
          f"in {report.elapsed_seconds:.1f} s; {s['files_failed']} could not be parsed; {s['files_skipped']} skipped.")
    if report.truncated:
        print(f"Incomplete: stopped at a limit ({report.truncated}).")
    for fn in s["flagged_functions"][:5]:
        print(f"  {fn['path']}:{fn['start_line']} {fn['qualname']}  CC {fn['cyclomatic_complexity']}, "
              f"nesting {fn['maximum_nesting_depth']}, {fn['function_length']} lines")
    print(f"Reports: {md_path}\n         {json_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
