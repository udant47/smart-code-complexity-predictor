"""Tests for local repository analysis (Phase 19)."""

import json
import os
from pathlib import Path

import pytest

from src.examples import get_example
from src.languages import registry
from src.repository import Limits, ScanError, analyze_repository, render_markdown, scan
from src.repository.__main__ import main
from src.repository.secrets import is_sensitive_filename, scan_text

C_AVAILABLE = registry.get("c").usable


def write(root: Path, rel: str, text: str = "x = 1\n") -> Path:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")
    return p


@pytest.fixture
def repo(tmp_path):
    root = tmp_path / "project"
    write(root, "pkg/search.py", get_example("binary_search").code)
    write(root, "pkg/branching.py", get_example("complex_branching").code)
    write(root, "pkg/nested.py", get_example("deep_nesting").code)
    write(root, "pkg/broken.py", "def f(:\n")
    write(root, "native/fib.c", "long fib(int n) { return n < 2 ? n : fib(n - 1) + fib(n - 2); }\n")
    write(root, "native/macro.c", "int f(int n) { FOR(i, n) { n++; } return n; }\n")
    write(root, "native/util.cpp", "int main() { return 0; }\n")
    write(root, "README.md", "# readme\n")
    write(root, "app.js", "console.log(1)\n")
    write(root, ".hidden/secret_tool.py", "x = 1\n")
    write(root, "node_modules/lib/index.py", "x = 1\n")
    write(root, ".env", "API_KEY=abc\n")
    write(root, "keys/server.pem", "-----BEGIN RSA PRIVATE KEY-----\n")
    write(root, "config/settings.py", 'DEBUG = True\nAPI_KEY = "sk_9f8a7b6c5d4e3f2a1b0c"\n')
    return root


# --------------------------------------------------------------------------- #
# Enumeration and safety
# --------------------------------------------------------------------------- #
def test_scan_finds_supported_files_and_reports_the_rest(repo):
    s = scan(repo)
    found = [f.relative for f in s.files]
    assert "README.md" not in found
    py = ["config/settings.py", "pkg/branching.py", "pkg/broken.py", "pkg/nested.py", "pkg/search.py"]
    c = ["native/fib.c", "native/macro.c"] if C_AVAILABLE else []
    assert sorted(found) == sorted(py + c)
    reasons = {x["path"]: x["reason"] for x in s.skipped}
    assert reasons[".hidden"] == "ignored_folder" and reasons["node_modules"] == "ignored_folder"
    assert reasons[".env"] == "credential_file" and reasons["keys/server.pem"] == "credential_file"
    assert reasons["native/util.cpp"] == "language_not_supported"
    assert s.unsupported_extensions[".md"] == 1 and s.unsupported_extensions[".js"] == 1
    assert s.truncated is None


def test_symlinks_are_never_followed(repo, tmp_path):
    outside = tmp_path / "outside"
    write(outside, "stolen.py", "def stolen():\n    return 1\n")
    try:
        (repo / "link_dir").symlink_to(outside, target_is_directory=True)
        (repo / "link_file.py").symlink_to(outside / "stolen.py")
        (repo / "pkg" / "loop").symlink_to(repo, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks are not available on this system")
    s = scan(repo)
    assert not any("stolen" in f.relative or f.relative.startswith(("link", "pkg/loop")) for f in s.files)
    reasons = {x["path"]: x["reason"] for x in s.skipped}
    assert reasons["link_dir"] == reasons["link_file.py"] == reasons["pkg/loop"] == "symlink"
    assert all(str(f.path).startswith(str(repo.resolve())) for f in s.files)


def test_root_validation(tmp_path):
    with pytest.raises(ScanError, match="not found"):
        scan(tmp_path / "nope")
    f = write(tmp_path, "file.py")
    with pytest.raises(ScanError, match="Not a folder"):
        scan(f)
    with pytest.raises(ScanError, match="whole drive"):
        scan(Path(tmp_path.anchor))


def test_root_given_through_a_symlink_or_dotdot_is_resolved(repo, tmp_path):
    s = scan(repo / "pkg" / ".." / "pkg")
    assert s.root == (repo / "pkg").resolve() and all(not f.relative.startswith("..") for f in s.files)


def test_limits_stop_the_scan_and_say_why(repo):
    assert "source files" in scan(repo, Limits(max_source_files=2)).truncated
    assert "entries" in scan(repo, Limits(max_entries=3)).truncated
    small = scan(repo, Limits(max_total_bytes=600))
    assert small.truncated and "MB" in small.truncated
    deep = scan(repo, Limits(max_depth=0))
    assert deep.files == [] and {"pkg", "native", "config", "keys"} <= {
        x["path"] for x in deep.skipped if x["reason"] == "too_deep"}


def test_large_files_are_skipped_not_read(repo):
    big = write(repo, "pkg/huge.py", "x = 1\n" * 40_000)
    s = scan(repo)
    assert "pkg/huge.py" not in [f.relative for f in s.files]
    assert {"path": "pkg/huge.py", "reason": "too_large", "size": big.stat().st_size} in s.skipped


def test_hidden_files_only_with_opt_in(repo):
    write(repo, ".tool.py", "x = 1\n")
    assert ".tool.py" not in [f.relative for f in scan(repo).files]
    included = [f.relative for f in scan(repo, Limits(include_hidden=True)).files]
    assert ".tool.py" in included and ".hidden/secret_tool.py" in included


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="named pipes need POSIX")
def test_special_files_are_not_opened(repo):
    os.mkfifo(repo / "pipe.py")  # opening a FIFO for reading would block forever
    s = scan(repo)
    assert {"path": "pipe.py", "reason": "not_a_regular_file"} in s.skipped


# --------------------------------------------------------------------------- #
# Secrets
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name, sensitive", [(".env", True), (".env.local", True), ("id_rsa", True),
                                             ("server.KEY", True), ("credentials.json", True), ("main.py", False),
                                             ("environment.py", False), ("keyboard.c", False)])
def test_sensitive_filenames(name, sensitive):
    assert is_sensitive_filename(name) is sensitive


@pytest.mark.parametrize("text, rule", [
    ("-----BEGIN OPENSSH PRIVATE KEY-----\nabc", "private_key"),
    ("key = 'AKIAABCDEFGHIJKLMNOP'", "aws_access_key_id"),
    ("t = 'ghp_" + "a" * 36 + "'", "github_token"),
    ('password = "hunter2hunter2"', "password_assignment"),
    ('API_KEY: "9f8a7b6c5d4e3f2a1b0c"', "password_assignment"),
])
def test_secret_rules(text, rule):
    findings = scan_text("x = 1\n" + text)
    assert [f.rule for f in findings] == [rule] and findings[0].line == 2
    assert set(findings[0].to_dict()) == {"rule", "line"}  # never the value


@pytest.mark.parametrize("text", ['password = os.environ["PW"]', 'password = "changeme123"',
                                  'api_key = "<your-key-here>"', 'secret = ""', "def password_ok(p):\n    return p"])
def test_placeholders_and_code_are_not_secrets(text):
    assert scan_text(text) == []


def test_files_with_secrets_are_skipped_and_values_never_reported(repo, tmp_path):
    report = analyze_repository(repo)
    skipped = {x["path"]: x for x in report.skipped}
    assert skipped["config/settings.py"]["reason"] == "possible_secret"
    assert skipped["config/settings.py"]["findings"] == [{"rule": "password_assignment", "line": 2}]
    dumped = json.dumps(report.to_dict()) + render_markdown(report)
    assert "sk_9f8a7b6c5d4e3f2a1b0c" not in dumped and "API_KEY=abc" not in dumped
    assert "PRIVATE KEY" not in dumped
    opted_in = analyze_repository(repo, skip_files_with_secrets=False)
    assert "config/settings.py" in [f.path for f in opted_in.files]
    assert "sk_9f8a7b6c5d4e3f2a1b0c" not in json.dumps(opted_in.to_dict())


# --------------------------------------------------------------------------- #
# Analysis and summary
# --------------------------------------------------------------------------- #
def test_repository_summary(repo):
    r = analyze_repository(repo)
    s = r.summary
    files = {f.path: f for f in r.files}
    assert files["pkg/broken.py"].success is False and files["pkg/broken.py"].error["code"] == "syntax"
    assert s["files_analyzed"] == 3 + (1 if C_AVAILABLE else 0)
    assert s["files_failed"] == 1 + (1 if C_AVAILABLE else 0)  # broken.py, and macro.c when C is available
    top = s["most_complex_functions"][0]
    assert (top["qualname"], top["path"], top["cyclomatic_complexity"], top["structural_category"]) == (
        "shipping_cost", "pkg/branching.py", 15, "High")
    flagged = {fn["qualname"]: fn["flags"] for fn in s["flagged_functions"]}
    assert flagged == {"shipping_cost": ["cyclomatic ≥ 11"], "approve_loan": ["nesting ≥ 4"]}
    assert s["high_complexity_function_count"] == 2
    search = next(fn for fn in s["most_complex_functions"] if fn["qualname"] == "binary_search")
    assert search["static_time"] == "O(log n)"
    assert s["category_distribution"]["High"] == 1
    assert s["languages"]["python"]["files"] == 3
    if C_AVAILABLE:
        assert s["languages"]["c"]["functions"] == 1
        assert files["native/macro.c"].error["code"] == "syntax"
        fib = next(fn for fn in s["most_complex_functions"] if fn["qualname"] == "fib")
        assert fib["static_time"] is None  # no Big-O for C
    assert s["skipped_by_reason"]["credential_file"] == 2
    assert s["most_complex_files"][0]["path"] == "pkg/branching.py"
    json.dumps(r.to_dict())


def test_report_contains_no_source_text(repo):
    r = analyze_repository(repo)
    text = json.dumps(r.to_dict()) + render_markdown(r)
    assert "weight must be positive" not in text and "credit_score" not in text


def test_time_budget_stops_cleanly(repo):
    ticks = iter(range(0, 1000, 10))  # every clock() call advances 10 s
    r = analyze_repository(repo, Limits(time_budget_seconds=25), clock=lambda: next(ticks))
    assert r.truncated and "time budget" in r.truncated
    assert any(x["reason"] == "time_budget" for x in r.skipped)
    assert len(r.files) < len(scan(repo).files)


def test_an_analyzer_crash_affects_only_that_file(repo, monkeypatch):
    from src.languages.python_lang import PythonAnalyzer

    real = PythonAnalyzer.analyze_bytes

    def flaky(self, data):
        if b"approve_loan" in data:
            raise RuntimeError("boom")
        return real(self, data)

    monkeypatch.setattr(PythonAnalyzer, "analyze_bytes", flaky)
    r = analyze_repository(repo)
    nested = next(f for f in r.files if f.path == "pkg/nested.py")
    assert nested.success is False and nested.error == {"code": "analysis_failed", "message": "RuntimeError"}
    assert r.summary["files_analyzed"] >= 2


def test_analysis_never_modifies_the_folder(repo):
    def snapshot():
        return sorted((str(p.relative_to(repo)), p.stat().st_mtime_ns, p.stat().st_size)
                      for p in repo.rglob("*") if not p.is_symlink())

    before = snapshot()
    analyze_repository(repo)
    assert snapshot() == before


# --------------------------------------------------------------------------- #
# Command line
# --------------------------------------------------------------------------- #
def test_cli_writes_reports_outside_the_repository(repo, tmp_path, capsys):
    out = tmp_path / "reports"
    assert main([str(repo), "--out", str(out), "--quiet"]) == 0
    data = json.loads((out / "repository_report.json").read_text(encoding="utf-8"))
    assert data["summary"]["most_complex_functions"][0]["qualname"] == "shipping_cost"
    md = (out / "repository_report.md").read_text(encoding="utf-8")
    assert md.startswith("# Repository complexity report") and "shipping_cost" in md
    assert "Analysed" in capsys.readouterr().out


def test_cli_refuses_output_inside_the_repository(repo, capsys):
    assert main([str(repo), "--out", str(repo / "reports")]) == 2
    assert "outside the analysed folder" in capsys.readouterr().err
    assert not (repo / "reports").exists()


def test_cli_reports_bad_paths(tmp_path, capsys):
    assert main([str(tmp_path / "missing"), "--out", str(tmp_path / "o")]) == 2
    assert "Folder not found" in capsys.readouterr().err
