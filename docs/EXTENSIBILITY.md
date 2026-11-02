# Extensibility: languages and repository analysis (Phase 19)

## 1. What is supported

| Language | Status | What you get |
|---|---|---|
| Python | **Full** | Structural metrics, structural category, rule-based Big-O, ML class, Derived Complexity Score, refactoring recommendations, explanations |
| C | **Experimental, structural only** | Structural metrics in the common schema and the structural category. Requires `tree-sitter==0.25.2` and `tree-sitter-c==0.24.1` |
| C++ | Unsupported | Nothing. C++ input given as C is detected and refused |
| Java | Unsupported | Nothing |
| JavaScript | Unsupported | Nothing |

The ML model, the static Big-O analyzer, the Derived Complexity Score and the refactoring rules were built and
calibrated on Python. They are **not** applied to C, and the app says so instead of showing numbers that would
not mean anything.

The Python pipeline (`code_parser` → `feature_extractor` → `function_analyzer` → `algorithm_analyzer` →
`predictor`) is unchanged. The new code sits beside it in `src/languages/` and `src/repository/`.

## 2. Choosing a parser for C

| Option | Verdict |
|---|---|
| Python `ast` | Not possible: it parses Python only. Never used for C. |
| Regular expressions | Rejected: C has nested expressions, declarators, comments and macros that regexes cannot parse reliably. |
| `pycparser` | Rejected: it needs fully preprocessed input. That means running an external C preprocessor (a subprocess, forbidden by the project's security policy) and having the system headers installed. |
| `libclang` | Rejected for now: it needs a native Clang installation, headers and compile flags, which is heavy and fragile on Windows. It is the right choice if exact semantics (types, macro expansion) are ever needed. |
| **tree-sitter + tree-sitter-c** | **Chosen.** It is a maintained grammar used by editors and GitHub, with prebuilt wheels for Windows, macOS and Linux. It parses without preprocessing, runs in linear time without recursion limits, and marks every part of the input it could not parse with `ERROR`/`MISSING` nodes. |

### Reliability policy

tree-sitter is *error-tolerant*: it always returns a tree. Metrics from a partly understood file would be wrong
without warning, so SmartCode is strict:

- **Any `ERROR` or `MISSING` node makes the whole file fail.** The error gives the line and column. No partial
  metrics are returned.
- C++ markers (`namespace`, `template<`, `std::`, `class X {`, access specifiers, C++ standard headers, `using
  namespace`), outside comments and strings, are refused as an unsupported language. Without this check some C++
  would parse "successfully" as C.
- The preprocessor is not run. Directives are syntax-tree nodes; function-like macro invocations look like calls.
  A macro that changes the syntax (`FOR(i, n) { ... }`) makes the parse fail, so the file is refused.
- Warnings: `macro_calls` when a function-like macro defined in the file is used inside a function (its internal
  control flow is not counted); `conditional_compilation` when `#if`/`#ifdef` is present (every branch is
  counted); `no_functions`.
- Same input limits as Python: 200 KB and 5,000 lines, no null bytes, valid Unicode.

## 3. How the languages differ, and how each construct is mapped

The common schema is defined in `src/languages/base.py` (`COMMON_FUNCTION_FIELDS`, `COMMON_FILE_FIELDS`). Each field
has one definition; the table shows how each language meets it.

| Concept | Python | C | Mapping and differences |
|---|---|---|---|
| Function | `def` / `async def`; methods inside `class`; nested `def` | `function_definition` with a body. Prototypes are declarations, not functions | C has no methods or nested functions (the GCC extension is not supported). `kind` is always `function` in C. A C function's lines include its return type line; Python's start at `def` (decorators excluded) |
| Name | `qualname` (`Class.method`, `outer.inner`) | The identifier inside the declarator, found through pointer, parenthesised and attributed declarators (`int (*pick(int))(int)` → `pick`) | C names are file-global, so `qualname = name` |
| Parameters | All kinds including `*args`, `**kwargs`, `self` | Declared parameters. `(void)` = 0. `...` counts 1 | `...` is treated like `*args` |
| Loops | `for`, `async for`, `while` | `for`, `while`, `do … while` | `do-while` is a loop (+1 decision, +1 nesting). Python comprehensions are not loops in either count, but their clauses add to Python CC |
| Conditionals | `if`, `elif`, `a if c else b` | `if`, `else if`, `c ? a : b` | `else if` = `elif` (same nesting level) |
| Multi-way branch | `match` / `case` | `switch` / `case` | Each non-default `case` is +1 CC in both. `switch` and `match` add one nesting level; `case` labels add none. C fall-through is not modelled |
| Boolean operators | `and`, `or` (+1 CC each), `not` | `&&`, `\|\|` (+1 CC each), `!` | Same rule |
| Exceptions | `try` (nesting), each `except` (+1 CC) | none | C has no exceptions |
| Jumps | `break`, `continue`, `return` | plus `goto` and labels | `goto` is counted separately (`extra.goto_count`), not as a decision point, as in common McCabe tools |
| Nesting | if, for, while, try, with, match | if, for, while, do, switch | Bare `{ }` blocks do not nest. A statement directly in the body is at depth 0 in both |
| Recursion | `f(...)` in `f`; `self.m` / `cls.m` in methods | `f(...)` in `f` | Lexical, direct only. Calls through function pointers are not detected in C |
| Calls | Every call expression | Every `call_expression` | **C macros look like calls** and are counted. They cannot be told apart without running the preprocessor |
| Operators | `and/or/not` in `logical_operator_count` | `&&`, `\|\|`, `!` | Other operator counts are Python-only for now |
| Lines | tokenize-based; docstrings are code | tree-sitter leaves; comments `//` and `/* */`; directives are code | C has no docstrings |
| Memory, types | Dynamic | Pointers, manual allocation, static types | Not measured: structural metrics describe control flow only |

The C examples (`src/languages/c_examples.py`) mirror the Python examples, and tests check that both languages
give the same cyclomatic complexity and nesting where the programs have the same structure. The one documented
difference is merge sort: C's `merge` needs two extra loops to copy leftover elements, which Python does with
slices, so its cyclomatic complexity is 6 against 4.

## 4. Adding a language

1. Pick a real parser for the language (tree-sitter has grammars for most). Never use Python's `ast` for another
   language, never call a compiler or interpreter, never execute the code.
2. Create `src/languages/<lang>_lang.py` with a `LanguageAnalyzer` subclass:
   - `language_id`, `display_name`, `extensions`, `capabilities` (structural only, unless more is genuinely
     validated);
   - `available()` returning `(False, install hint)` when the parser is missing;
   - `analyze(source)` calling `check_text_limits` first, refusing any parse error (no partial metrics), and
     returning a `LanguageReport` with `file_metrics` keyed exactly by `COMMON_FILE_FIELDS` and `FunctionMetrics`
     for each function.
3. Map every construct to the field definitions, and add a column for the language to the table above.
4. Register it in `src/languages/registry.py` with status `structural`, and add it to `LANGUAGES` in
   `src/ui/logic.py`.
5. Add tests in the style of `tests/test_languages.py`: hand-checked values for every construct, error cases,
   deep nesting, a performance bound, and examples that mirror the Python ones.
6. Do not enable ML, Big-O or the Derived Complexity Score for the language until they have been trained or
   validated on data in that language.

## 5. Repository analysis

`python -m src.repository PATH` analyses a local folder and writes `repository_report.md` and
`repository_report.json` (default output folder: `reports/repository`). It is a separate package that uses only
`src.languages`, not the core parser or the ML model.

### What it does
1. **Enumerate** supported source files (`.py`, and `.c`/`.h` when the C parser is installed), sorted and
   deterministic.
2. **Analyse** each file with the same language analyzers the app uses (plus the static Big-O class for Python
   functions).
3. **Aggregate** totals, per-language figures (files, code lines, functions, mean and median CC, CC ≥ 11), the
   Low/Medium/High distribution, and the most complex files.
4. **Identify high-complexity functions** with the same thresholds as the refactoring rules: cyclomatic ≥ 11,
   nesting ≥ 4, or ≥ 50 lines.
5. **Report** every file that was skipped or could not be parsed, with the reason.

### Protections

| Risk | Protection |
|---|---|
| Huge repositories | Limits on entries visited (50,000), source files (2,000), bytes per file (200 KB), total bytes (50 MB) and folder depth (25). Reaching a limit stops the scan and marks the report *incomplete* with the reason. Dependency and build folders (`.git`, `node_modules`, `venv`, `build`, `dist`, ...) are skipped |
| Excessive time | A time budget (300 s by default) is checked before each file. Remaining files are listed as skipped (`time_budget`). One file is bounded by the size limits (measured at about 5 s at worst) |
| Symlinks, junctions, path traversal | Symlinked files and folders are never followed. Every folder and file must resolve inside the root, and each folder is visited once, so links, junctions and `..` cannot lead outside or loop. Files are re-checked when opened. A whole drive or filesystem root is refused |
| Special files | Only regular files are opened (no FIFOs, devices or sockets, which could block or misbehave) |
| Unsupported file types | Never opened; only their extensions are counted. C++/Java/JS files are listed as `language_not_supported` |
| Files containing secrets | Credential files are never opened, by name (`.env`, `*.pem`, `*.key`, `id_rsa`, `credentials.json`, ...). Source files whose content matches secret patterns (private keys, AWS/GitHub/Slack/Google/Stripe tokens, `password = "..."` style assignments) are skipped by default. Only the rule name and line are reported, never the value. Reports contain metrics, paths and function names, never source text |
| Modifying the project | Read-only. The output folder may not be inside the analysed folder |
| Execution | Nothing is imported, compiled or executed (covered by the security tests) |
| Remote access | None: no network, no GitHub authentication, local folders only |

Secret detection is a best-effort safeguard, not a secret scanner: a file without findings may still contain
secrets.

### Possible next steps
- A *Repository* page in the app (a folder path field, a progress bar, and the report as tables).
- Incremental re-analysis using file hashes.
- More languages through the interface in section 4.
