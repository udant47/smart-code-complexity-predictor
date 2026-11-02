"""Security tests (Phase 18).

* Static: the application code contains no execution primitives, no shell or
  subprocess use, no dynamic imports, and no unpickling outside the guarded
  model loader. Runtime modules write no files.
* Runtime: while analysing hostile code, exec/eval/dynamic imports/shell calls
  are trapped and must never be called, compile() is only used in AST-only
  mode (that is what ast.parse does), and no file is opened for writing.
  The hostile samples would kill, hang or write files if they were ever run.
* Configuration: no upload widget, no error details shown to users.
"""

import ast
import builtins
import importlib
import os
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

from src import config
from src import predictor as pr

ROOT = config.PROJECT_ROOT
APP_FILES = sorted([ROOT / "app.py", *(ROOT / "src").rglob("*.py")])

FORBIDDEN_CALLS = {"exec", "eval", "compile", "__import__", "execfile", "breakpoint", "input"}
FORBIDDEN_ATTR_CALLS = {"system", "popen", "spawnl", "spawnv", "spawnlp", "spawnvp", "execv", "execve", "execl",
                        "execvp", "fork", "forkpty", "startfile", "import_module", "load_module", "exec_module"}
FORBIDDEN_MODULES = {"subprocess", "pty", "runpy", "importlib", "ctypes", "marshal", "pickle", "shelve", "dill",
                     "cloudpickle", "code", "codeop", "multiprocessing", "socket"}
UNPICKLE = {("joblib", "load"), ("pickle", "load"), ("pickle", "loads")}

# Modules used at analysis time (CLI-only training/report modules are excluded).
RUNTIME_MODULES = ["app.py", "src/predictor.py", "src/code_parser.py", "src/feature_extractor.py",
                   "src/function_analyzer.py", "src/algorithm_analyzer.py", "src/pattern_features.py",
                   "src/complexity_score.py", "src/suggestions.py", "src/explainability.py", "src/comparison.py",
                   "src/examples.py", "src/asymptotics.py", "src/model_integrity.py", "src/config.py",
                   *[str(p.relative_to(ROOT)) for p in (ROOT / "src" / "ui").rglob("*.py")],
                   *[str(p.relative_to(ROOT)) for p in (ROOT / "src" / "languages").rglob("*.py")],
                   *[str(p.relative_to(ROOT)) for p in (ROOT / "src" / "repository").rglob("*.py")]]
# File-system writes. (json.dump/pickle-style writes need a handle from open(), which is checked separately;
# DataFrame.rename and similar in-memory methods are deliberately not listed.)
WRITE_CALLS = {"write_text", "write_bytes", "unlink", "rmtree", "mkdir", "makedirs", "touch", "chmod", "savefig",
               "to_csv", "to_parquet"}
FS_MODULES = {"os", "shutil"}
FS_MODULE_CALLS = {"remove", "rename", "replace", "unlink", "rmdir", "removedirs", "makedirs", "mkdir", "chmod",
                   "symlink", "link", "truncate", "copy", "copy2", "copyfile", "copytree", "move", "rmtree"}
# Functions that legitimately write files. They are command-line report writers, never called by the app:
# write_outputs is `python -m src.explainability`, _plot_bars is only called from write_outputs, and
# register/main are `python -m src.model_integrity register`, and write_outputs is the report writer of
# `python -m src.repository` (it refuses an output folder inside the analysed folder).
WRITE_ALLOWED = {("src/explainability.py", "write_outputs"), ("src/explainability.py", "_plot_bars"),
                 ("src/model_integrity.py", "register"), ("src/model_integrity.py", "main"),
                 ("src/repository/__main__.py", "write_outputs")}


def _calls(path: Path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    parents = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parents[child] = node
    for node in ast.walk(tree):
        yield node, tree, parents


def _enclosing_function(node, parents):
    while node in parents:
        node = parents[node]
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return node.name
    return None


# --------------------------------------------------------------------------- #
# Static checks
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("path", APP_FILES, ids=lambda p: str(p.relative_to(ROOT)))
def test_no_execution_primitives_shell_or_dynamic_imports(path):
    offenders = []
    for node, _, _ in _calls(path):
        if isinstance(node, ast.Call):
            fn = node.func
            if isinstance(fn, ast.Name) and fn.id in FORBIDDEN_CALLS:
                offenders.append(f"{node.lineno}: {fn.id}()")
            if isinstance(fn, ast.Attribute) and fn.attr in FORBIDDEN_ATTR_CALLS | {"compile"} and not (
                    isinstance(fn.value, ast.Name) and fn.value.id == "re"):
                offenders.append(f"{node.lineno}: .{fn.attr}()")
            for kw in node.keywords:
                if kw.arg == "shell":
                    offenders.append(f"{node.lineno}: shell= keyword")
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            mods = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module or ""]
            offenders += [f"{node.lineno}: import {m}" for m in mods if m.split(".")[0] in FORBIDDEN_MODULES]
        elif isinstance(node, ast.Name) and node.id in ("__builtins__", "globals", "locals", "vars"):
            if isinstance(node.ctx, ast.Load) and node.id != "vars":
                offenders.append(f"{node.lineno}: {node.id}")
    assert offenders == [], f"{path.name}: {offenders}"


def test_unpickling_only_happens_in_the_guarded_model_loader():
    sites = []
    for path in APP_FILES:
        for node, _, parents in _calls(path):
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and isinstance(node.func.value, ast.Name) and (node.func.value.id, node.func.attr) in UNPICKLE):
                sites.append((path.name, _enclosing_function(node, parents), node.lineno))
    assert {(f, fn) for f, fn, _ in sites} == {("predictor.py", "load_model")}
    # In load_model, the integrity check comes before every unpickling call.
    tree = ast.parse((ROOT / "src" / "predictor.py").read_text(encoding="utf-8"))
    fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "load_model")
    check = min(n.lineno for n in ast.walk(fn) if isinstance(n, ast.Call)
                and isinstance(n.func, ast.Attribute) and n.func.attr == "check_file")
    loads = [n.lineno for n in ast.walk(fn) if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Attribute) and n.func.attr == "load"]
    assert loads and all(check < line for line in loads)


@pytest.mark.parametrize("rel", RUNTIME_MODULES)
def test_runtime_modules_do_not_write_files(rel):
    path = ROOT / rel
    offenders = []
    for node, _, parents in _calls(path):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        name = fn.attr if isinstance(fn, ast.Attribute) else getattr(fn, "id", None)
        where = _enclosing_function(node, parents)
        if (rel, where) in WRITE_ALLOWED:
            continue
        if name in WRITE_CALLS:
            offenders.append(f"{node.lineno}: {name}() in {where}")
        if (isinstance(fn, ast.Attribute) and isinstance(fn.value, ast.Name) and fn.value.id in FS_MODULES
                and fn.attr in FS_MODULE_CALLS):
            offenders.append(f"{node.lineno}: {fn.value.id}.{fn.attr}() in {where}")
        if name == "open":
            mode = node.args[1] if len(node.args) > 1 else next((k.value for k in node.keywords if k.arg == "mode"), None)
            if isinstance(fn, ast.Attribute) and node.args and isinstance(node.args[0], ast.Constant):
                mode = node.args[0]
            if mode is not None and not (isinstance(mode, ast.Constant) and set(str(mode.value)) <= set("rbt")):
                offenders.append(f"{node.lineno}: open() for writing in {where}")
    assert offenders == [], f"{rel}: {offenders}"


def test_report_plot_helper_is_only_used_by_the_cli_writer():
    tree = ast.parse((ROOT / "src" / "explainability.py").read_text(encoding="utf-8"))
    parents = {c: n for n in ast.walk(tree) for c in ast.iter_child_nodes(n)}
    callers = {_enclosing_function(n, parents) for n in ast.walk(tree)
               if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "_plot_bars"}
    assert callers == {"write_outputs"}


def test_no_upload_widget_and_safe_streamlit_config():
    for path in APP_FILES:
        assert "file_uploader" not in path.read_text(encoding="utf-8"), path.name
    cfg = tomllib.loads((ROOT / ".streamlit" / "config.toml").read_text(encoding="utf-8"))
    assert cfg["client"]["showErrorDetails"] is False
    assert cfg["server"]["maxUploadSize"] <= 1
    assert cfg["browser"]["gatherUsageStats"] is False
    assert cfg["server"]["address"] == "localhost"  # not reachable from other devices by default


# --------------------------------------------------------------------------- #
# Runtime checks: hostile code is analysed, never run
# --------------------------------------------------------------------------- #
HOSTILE = {
    "kill_process": "import os\nos._exit(3)\n",
    "kill_signal": "import os, signal\nos.kill(os.getpid(), signal.SIGKILL)\n",
    "infinite_loop": "while True:\n    pass\n",
    "system_exit": "raise SystemExit(1)\n",
    "shell": "import os, subprocess\nos.system('touch {marker}')\nsubprocess.run(['touch', '{marker}'])\n",
    "file_write": "open('{marker}', 'w').write('pwned')\n",
    "exec_eval": "exec(\"open('{marker}', 'w')\")\neval('__import__(\"os\").system(\"touch {marker}\")')\n",
    "dynamic_import": "__import__('smartcode_evil_module_xyz')\nimport smartcode_evil_module_abc\n",
    "decorator_side_effect": "def deco(f):\n    open('{marker}', 'w')\n    return f\n\n@deco\ndef g():\n    pass\n",
    "class_body": "class A:\n    open('{marker}', 'w')\n",
    "default_argument": "def f(x=open('{marker}', 'w')):\n    return x\n",
    "annotation": "def f(x: open('{marker}', 'w')) -> None:\n    pass\n",
    "dunder_main": "if __name__ == '__main__':\n    open('{marker}', 'w')\n",
    "recursion_bomb": "def f():\n    return f()\nf()\n",
    "memory_bomb": "x = [0] * (10 ** 12)\n",
}


LIBRARY_ROOTS = tuple({str(Path(p).resolve()) for p in (sys.prefix, sys.base_prefix, *sys.path) if p})


def _is_library_code(obj) -> bool:
    """Code objects the import system runs for installed modules (e.g. a lazily imported sklearn submodule)."""
    filename = getattr(obj, "co_filename", None)
    if not filename or not filename.endswith(".py") or not Path(filename).is_file():
        return False
    resolved = str(Path(filename).resolve())
    return resolved.startswith(LIBRARY_ROOTS) and not resolved.startswith(str((ROOT / "tests").resolve()))


@pytest.fixture(scope="module", autouse=True)
def warm_up():
    """Import everything the analysis uses before trapping. First imports legitimately run library code:
    the import system executes module code objects, and the standard library's @dataclass builds
    __init__ methods with exec. After warm-up, analysis performs no exec/eval at all."""
    pr.predict("def f(xs):\n    for x in xs:\n        pass\n", explain=True)


class Trap:
    """Records forbidden operations. exec/eval of *source text* is never allowed; the only exec permitted
    is the import system running an installed library module from its own .py file."""

    def __init__(self, monkeypatch):
        self.events = []
        real_compile, real_import, real_open = builtins.compile, builtins.__import__, builtins.open
        real_exec, real_eval = builtins.exec, builtins.eval

        def compile_guard(source, filename, mode, flags=0, *a, **k):
            if not flags & ast.PyCF_ONLY_AST:
                self.events.append(f"compile(mode={mode}, flags={flags})")
            return real_compile(source, filename, mode, flags, *a, **k)

        def import_guard(name, *a, **k):
            if "evil" in name:
                self.events.append(f"import {name}")
            return real_import(name, *a, **k)

        def open_guard(file, mode="r", *a, **k):
            if set(str(mode)) & set("wax+"):
                self.events.append(f"open({file!r}, {mode!r})")
            return real_open(file, mode, *a, **k)

        def forbid(label):
            def f(*a, **k):
                self.events.append(label)
                raise AssertionError(f"{label} called during analysis")
            return f

        def guarded(label, real):
            def f(obj, *a, **k):
                if not _is_library_code(obj):
                    self.events.append(f"{label}({str(obj)[:60]!r})")
                    raise AssertionError(f"{label} called during analysis")
                return real(obj, *a, **k)
            return f

        monkeypatch.setattr(builtins, "exec", guarded("exec", real_exec))
        monkeypatch.setattr(builtins, "eval", guarded("eval", real_eval))
        monkeypatch.setattr(builtins, "compile", compile_guard)
        monkeypatch.setattr(builtins, "__import__", import_guard)
        monkeypatch.setattr(builtins, "open", open_guard)
        monkeypatch.setattr(os, "system", forbid("os.system"))
        monkeypatch.setattr(os, "popen", forbid("os.popen"))
        monkeypatch.setattr(os, "_exit", forbid("os._exit"))
        monkeypatch.setattr(os, "kill", forbid("os.kill"))
        monkeypatch.setattr(subprocess, "Popen", forbid("subprocess.Popen"))
        monkeypatch.setattr(subprocess, "run", forbid("subprocess.run"))
        monkeypatch.setattr(importlib, "import_module", forbid("importlib.import_module"))


@pytest.mark.parametrize("name", list(HOSTILE))
def test_hostile_code_is_never_executed(name, monkeypatch, tmp_path):
    marker = tmp_path / "pwned"
    code = HOSTILE[name].format(marker=marker.as_posix())
    trap = Trap(monkeypatch)
    result = pr.predict(code)  # with the real saved model when present; static analysis otherwise
    assert result.success, [e.code for e in result.errors]
    assert trap.events == []
    assert not marker.exists() and list(tmp_path.iterdir()) == []
    assert not any("evil" in m for m in sys.modules)


def test_ui_service_analyses_hostile_code_without_running_it(monkeypatch, tmp_path):
    from src.ui import services

    marker = tmp_path / "pwned"
    trap = Trap(monkeypatch)
    for name, code in HOSTILE.items():
        out = services.analyze(code.format(marker=marker.as_posix()) + f"\n# {name}\n")
        assert out["ok"] and out["result"]["success"], name
    assert trap.events == [] and not marker.exists()


def test_parse_bytes_upload_path_is_static_too(monkeypatch, tmp_path):
    from src.code_parser import parse_bytes

    marker = tmp_path / "pwned"
    trap = Trap(monkeypatch)
    r = parse_bytes(f"# -*- coding: utf-8 -*-\nopen('{marker.as_posix()}', 'w')\n".encode())
    assert r.success and trap.events == [] and not marker.exists()
