# Testing SmartCode

## 1. Automated tests

```powershell
pip install -r requirements-dev.txt   # once: pytest is a development dependency
python -m pytest                      # everything (about 3 minutes)
python -m pytest tests/test_security.py tests/test_robustness.py tests/test_model_integrity.py   # Phase 18 only
python -m pytest -k "checklist"       # the Phase 18 requirement checklist
python -m src.model_integrity verify  # are the installed model files the ones recorded at training time?
```

The suite has 1,003 tests across 26 files. On Windows, with the datasets built, expect 999 passed and 4 skipped: the SHAP cross-check (SHAP
not installed), two symlink tests (Windows only allows symlinks with Developer Mode on) and one POSIX-only named-pipe test.
Tests that need a trained model build a small one in a temporary folder, so they do not depend on your `models/`
folder. The exception is two tests that check the installed model: they skip when no model has been trained.

| Area | Main test files |
|---|---|
| Parser (valid, invalid, empty, functions, classes, nested functions) | `test_code_parser.py`, `test_requirements_checklist.py` |
| Feature extraction (loops, conditionals, recursion, nesting, cyclomatic complexity, metric consistency) | `test_feature_extractor.py`, `test_function_analyzer.py`, `test_requirements_checklist.py` |
| Algorithm analysis (linear and binary search, nested and sequential loops, merge sort, Fibonacci, Unknown) | `test_algorithm_analyzer.py`, `test_examples.py`, `test_requirements_checklist.py` |
| ML (preprocessing, feature order, prediction, missing model, loading, output schema) | `test_preprocessing.py`, `test_predictor.py`, `test_requirements_checklist.py` |
| Model file integrity (hashes, malicious pickle, symlink, size) | `test_model_integrity.py` |
| Frontend (empty, invalid, missing model, unsupported language, formatting) | `test_app.py` (headless Streamlit), `test_requirements_checklist.py` |
| Security (no execution, no shell, no dynamic imports, no file writes, config) | `test_security.py` |
| Robustness (hostile and malformed input, exact size limits, timing, uploads) | `test_robustness.py` |
| Examples and before/after comparison | `test_examples.py`, `test_comparison.py` |
| Deployment readiness (entry point, pinned scikit-learn, cloud-compatible config, shipped model hashes, reports match the model) | `test_deployment.py` |
| Final improvement phase: script analysis, sorting algorithms, new features, training/inference feature equality, calibration, audit | `test_improvements.py` |
| Languages (common schema, C analyzer) and repository analysis | `test_languages.py`, `test_repository.py` |
| Training, evaluation, explainability, recommendations, dataset | `test_train_model.py`, `test_evaluate_model.py`, `test_explainability.py`, `test_suggestions.py`, `test_dataset_loader.py`, `test_eda.py` |

Line coverage measured with `pytest-cov` (optional, not a project requirement): 91% over `src/` and `app.py`.

## 2. Manual testing

Start the app:

```powershell
cd "$HOME\Downloads\smart-code-complexity-predictor"; .\.venv\Scripts\Activate.ps1; streamlit run app.py --server.address localhost
```

Open http://localhost:8501 and work through the checks below. Each one says what you should see. Anything else,
especially a red Python traceback, is a bug.

### A. Normal use
1. **Startup.** The sidebar shows *Model loaded* with *Random Forest · 61 features*. If it says
   *Model unavailable — No integrity hash is recorded*, run `python -m src.model_integrity register` once and
   restart the app.
2. **Examples.** In *Example*, pick each of the eight examples and press **Analyze**. Each one shows a
   *Documented expectations vs this analysis* table, and every row reads ✓ matches or *not documented* (ML rows).
3. **Results.** For *6. Merge sort*, the function table lists `merge` and `merge_sort`, and `merge_sort` has
   static time O(n log n). The ML class may differ: that is expected and shown honestly.
4. **Compare.** In the sidebar choose **Compare versions**, pick *Shorter but slower: sorting to find the minimum*,
   and press **Compare**. There is an orange warning: *Shorter, but with a higher static time estimate*.
   Press **Swap before/after**, then **Compare**: the directions reverse.
5. **Other pages.** *Model evaluation* shows the saved metrics (test macro F1 0.460) and the tabs *Feature importance*,
   *Confidence & calibration* (ECE 0.074) and *Improvement history* (before 0.442 → after 0.460). *Documentation* opens all
   its tabs, including *Examples & comparison* and *Security*.

6a. **Scripts.** Paste `n = int(input())`, then `a = list(map(int, input().split()))` and a double loop over `a`.
    In *Algorithm analysis* an expander *Module-level code (script) · time O(n^2)* appears. Above the cards, the
    expander *What is the difference between these measures?* explains the five measures.

### B. Bad input
6. **Empty input.** Press **Clear**, then **Analyze**. Expected: *There is no code to analyse*.
7. **Invalid code.** Paste `def broken(:` and press **Analyze**. Expected: *Invalid Python syntax* with line 1.
8. **Mixed indentation.** Paste code that indents one line with a tab and the next with spaces. Expected:
   *Invalid indentation*.
9. **Unsupported language.** Choose *JavaScript* as the language. **Analyze** is disabled, and the page says
   *JavaScript is not supported*.
10. **Too large.** In PowerShell, run `1..5001 | ForEach-Object { "x = 1" } | Set-Clipboard`, paste into the
    analyzer and press **Analyze**. Expected: an error *The input has 5,001 lines; the limit is
    5,000.* (5,002 if the clipboard adds a final line break).
11. **Near the limit.** Run
    `1..700 | ForEach-Object { "def f$_(xs):`n    t = 0`n    for x in xs:`n        t += x`n    return t`n" } | Set-Clipboard`,
    paste and analyze. The results appear within a few seconds.
12. **Too deeply nested.** Run `"x = " + ("-" * 3000) + "1" | Set-Clipboard`, paste and analyze. Expected:
    *Code too deeply nested to analyse*.

### C. Code is never executed
13. Paste this and press **Analyze**:
    ```python
    import os
    open(os.path.expanduser("~/smartcode_was_run.txt"), "w").write("executed!")
    os.system("notepad")
    ```
    The analysis completes normally, Notepad does not open, and
    `Test-Path "$HOME\smartcode_was_run.txt"` prints **False**.
14. Paste a real infinite loop (two lines: `while True:` and then `    pass`) and analyze. Results appear
    immediately: the loop is read, not run.

### D. Model problems (restore everything afterwards)
15. **Missing model.** Stop the app with Ctrl+C. Rename `models` to `models_off` and start the app again.
    The sidebar says *Model unavailable*. Analysis still shows all the static results, plus the message
    *ML prediction unavailable … has not been trained*. Stop the app and rename `models_off` back to `models`.
16. **Tampered model.** Make a backup with `Copy-Item models models_backup -Recurse`. Then run
    `Add-Content models\complexity_model.joblib "x"` and `python -m src.model_integrity verify`.
    Expected: `REFUSED complexity_model.joblib`. Start the app: the sidebar says *complexity_model.joblib does not match
    the hash recorded when it was trained*, and static analysis still works. Restore with
    `Remove-Item models -Recurse; Rename-Item models_backup models` and check that `verify` prints OK twice.

### E. Network
17. Started with `--server.address localhost`, the terminal shows a single line, `URL: http://localhost:8501`, and
    no *Network URL*. Other devices on your Wi-Fi cannot open it.
