# Smart Code Complexity Predictor

> Static analysis + classical machine learning to classify Python function
> complexity and estimate time / auxiliary-space Big-O — visualised in a
> Streamlit dashboard. Submitted code is **never executed**.

_Status: Phase 1 — project scaffold. Sections marked TODO are filled in later phases._

## 1. Overview
- Problem statement — TODO
- What the tool does / does not do — TODO

## 2. Features
- AST-based parsing (no code execution)
- Per-function metrics (LOC, cyclomatic complexity, nesting, loops, recursion, …)
- ML complexity classification (Low / Medium / High)
- Heuristic time & auxiliary space complexity estimation, with confidence
- Explainability and refactoring suggestions
- Interactive Streamlit + Plotly dashboard

## 3. Architecture
```
source text → code_parser → function_analyzer → feature_extractor
                                              ├→ algorithm_analyzer (rule-based Big-O)
                                              └→ preprocessing → predictor (ML)
                                                     → explainability / suggestions → UI
```
Folder-by-folder description — TODO

## 4. Tech stack
Python · ast · pandas · NumPy · scikit-learn · Plotly · Matplotlib · joblib · Streamlit · pytest

## 5. Setup (Windows)
```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 6. Usage
```powershell
python -m src.dataset_loader     # build data/processed/features.csv   (Phase 3)
python -m src.train_model        # train & save model                   (Phase 3)
python -m src.evaluate_model     # metrics & plots                      (Phase 3)
streamlit run app.py             # launch dashboard
```

## 7. Dataset
- Sources and licences — TODO
- Labelling strategy (and how circularity is avoided) — TODO

## 8. Model & evaluation
- Candidate models, cross-validation, metrics table, confusion matrix — TODO

## 9. Big-O estimation: method and limitations
- Supported patterns — TODO
- Known limitations (dynamic dispatch, input-dependent loops, library calls) — TODO

## 10. Testing
```powershell
python -m pytest
```

## 11. Screenshots
See `screenshots/` — TODO

## 12. Safety
Code is analysed with `ast.parse` only — never imported, `exec`'d or `eval`'d.

## 13. Future work
TODO

## 14. License
TODO
