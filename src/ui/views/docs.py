"""Documentation page: how SmartCode works, what it cannot do, and how it stays safe."""

from __future__ import annotations

import streamlit as st

from src import config
from src.examples import EXAMPLES
from src.ui import services
from src.ui import components as ui


def _model_numbers() -> str:
    meta = services.artifacts().data.get("model_metadata")
    if not meta:
        return "The model has not been evaluated on this computer yet (run `python -m src.evaluate_model`)."
    tm = meta.get("test_metrics") or {}
    ci = tm.get("macro_f1_ci") or [None, None]
    return (f"On held-out test problems the saved model reaches macro F1 **{tm.get('macro_f1', 0):.3f}** "
            f"(95% interval {ci[0]:.2f}–{ci[1]:.2f}) and accuracy **{tm.get('accuracy', 0):.3f}** over seven classes. "
            "That is far better than chance and far from reliable: roughly half of the predictions are wrong.")


def _examples_table() -> str:
    rows = ["| Example | Pattern | Highest cyclomatic | Max nesting | Expected time (static class) |",
            "|---|---|---|---|---|"]
    for e in EXAMPLES:
        times = "; ".join(f"`{c.function}` {c.static_time_class or 'Unknown'}" for c in e.complexity)
        rows.append(f"| {e.title} | {e.pattern} | {e.structure['max_cyclomatic_complexity']} | "
                    f"{e.structure['max_nesting_depth']} | {times} |")
    return "\n".join(rows)


SECTIONS = [
    ("Static analysis", """
SmartCode inspects source code **without running it**. Everything it reports is derived from the text of the code:
its syntax tree, its tokens and the counts and patterns found in them. This makes analysis safe and fast, but it also
means SmartCode cannot see runtime facts such as input sizes, data contents, which branches are actually taken,
or the cost of library functions it does not model.
"""),
    ("AST parsing", """
Python's own `ast` module turns the code into an abstract syntax tree; the `tokenize` module provides line, comment
and token information. The parser checks size limits ({lines:,} lines, {kb:,} KB), syntax, indentation and encoding
before anything else runs, and reports problems with the line number and a hint. Every function, method and nested
function gets a stable identifier from its qualified name (`Graph.bfs`, `outer.inner`).
"""),
    ("Structural metrics and feature engineering", """
For the whole file and for every function SmartCode counts lines (code, comments, docstrings), branches, loops,
`try`/`except`, comprehensions, operators, calls and identifiers, and measures **McCabe cyclomatic complexity**
(1 + decision points), nesting depth, loop nesting and recursion. Eighteen **algorithm-pattern features** add
signals that change complexity classes: sorting and heap calls, halving loops, binary-search midpoints,
constant-size loops, loops bounded by an outer loop variable, exponential ranges, combinatorics, branching or dividing
recursion and memoization. The **structural category** (Low / Medium / High) is a transparent rule on cyclomatic
complexity using radon's thresholds (1–5, 6–10, 11+), not a model output. The **Derived Complexity Score** (0–100)
combines seven log-scaled metrics with fixed weights (no metric can add more than 25 points). It is a
project-specific indicator for relative comparison, not an industry standard and not a probability of defects.
"""),
    ("Machine-learning methodology", """
The model predicts a program's **time-complexity class** (O(1), O(log n), O(n), O(n log n), O(n²), O(n³), O(2ⁿ)).
It was trained on **CodeComplex** (Python solutions to Codeforces problems labelled by experts), after removing
exact and renamed duplicates. Data were split **by problem**, so no test problem was seen in training, and
preprocessing lives inside scikit-learn pipelines so nothing from validation or test data leaks into training.
Ten kinds of model (from dummy baselines and a loop-nesting rule to logistic regression, SVM, trees, boosting
and a voting ensemble), each on two feature sets, were compared by repeated grouped cross-validation; a Random Forest with 300 trees was selected and the
test set was used once. Vote shares are the fraction of trees agreeing; they are not calibrated probabilities, so
SmartCode also shows how often predictions at that level were correct on the test set.

{model_numbers}
"""),
    ("Big-O analysis (rule-based)", """
Separately from the model, a rule-based analyser walks each function and builds a symbolic cost: loops over
`range` or collections, `while` loops with counters, halving or two pointers, recursion solved with recurrences or
the master theorem, built-in operations with documented costs, and calls to other functions in the same file.
It reports time and **auxiliary** space (extra memory beyond the input) in the function's own size variables
(`n = len(items)`), with assumptions, evidence and a confidence level. When the code is outside the supported
patterns (for example `while queue`, data-dependent splits or unknown library calls) it says **Unknown** instead of
guessing. Estimates are worst case and assume loops run to completion.
"""),
    ("Examples & comparison", """
The **Example** menu on the Analyzer loads eight documented programs. Each documents its expected structure and its
expected asymptotic complexity with the assumptions that make it true (input-size variable, O(1) primitive
operations, worst case). An automated test checks every documented structural value and static time class against
the analyzer. **No ML class is documented**: the model may disagree with the documented complexity (for example on
recursive Fibonacci), and the app shows that disagreement.

{examples_table}

The **Compare versions** page analyses two versions and reports measured differences only: structural category,
Derived Complexity Score, cyclomatic complexity, nesting, functions, per-function static Big-O and space estimates,
and ML outputs. Time estimates are compared by growth order only when both are static estimates with status
"estimated". Directions are described as lower or higher, never as improvements. Line count is always reported
with the caveat that shorter code is not necessarily simpler or faster; when code gets shorter while its time
estimate grows, the page says so. The example pairs include *Shorter but slower* (`sorted(items)[0]` instead of
a scan: fewer lines and lower structural metrics, but O(n log n) instead of O(n)).
"""),
    ("Evaluation", """
The **Model Evaluation** page shows only saved results: the model comparison, test macro F1 with a bootstrap
interval over problems, the confusion matrix, per-class scores, grouped cross-validation by problem and by contest,
experiments that remove whole algorithm families from training, and (when downloaded) an external test on
BigO(Bench). Model explanations (permutation importance on the test set and per-prediction tree-path
contributions) describe how the model uses features; they are associations, not causes.
"""),
    ("Limitations", """
* The model was trained on short competitive-programming programs. It has not been validated on everyday
  application code, on single functions, or for space complexity, and it is weaker on algorithm families absent from training.
* Expert labels are themselves imperfect, and static features cannot see loop bounds that depend on data.
* Per-function ML predictions are experimental; prefer the rule-based estimate when it is available.
* Structural metrics measure size and control flow. They do not measure naming, cohesion, coupling, tests or documentation quality.
* Refactoring recommendations are review prompts based on thresholds, not guaranteed improvements.
* Only Python is supported. Other languages are listed in the selector but are disabled rather than analysed incorrectly.
"""),
    ("Security", """
* Submitted code is **never executed, imported, compiled to bytecode or evaluated**: only `ast.parse` and `tokenize`
  read it. A test confirms that analysing code containing file writes does not perform them.
* Input size and nesting depth are limited to prevent resource exhaustion.
* Text from the code (function names, messages) is HTML-escaped before it is displayed.
* The saved model files are Python pickles (joblib), and loading a pickle can run code. Before loading, SmartCode
  checks that each file is a regular file under 512 MB whose SHA-256 matches the hash recorded in
  `model_metadata.json` when the model was trained. A replaced or tampered file is refused before it is unpickled.
  This guards against swapped or corrupted files, not against someone who can already edit both the model and its
  metadata on your computer. Model files are never accepted from users, and the app has no upload widget.
* Calls to code outside the file are not executed either: when such a call receives a whole input collection, the
  static estimate says Unknown rather than assuming it is cheap.
* Internal errors are logged for the developer and shown to users as a short message with a reference id, never as a traceback.
"""),
]


def render() -> None:
    ui.hero("Documentation", "How SmartCode analyses code, what its numbers mean, and where they stop being reliable.")
    fmt = {"lines": config.MAX_SOURCE_LINES, "kb": config.MAX_SOURCE_BYTES // 1000, "model_numbers": _model_numbers(),
           "examples_table": _examples_table()}
    names = [s[0] for s in SECTIONS]
    tabs = st.tabs(names)
    for tab, (title, body) in zip(tabs, SECTIONS):
        with tab:
            ui.section(title)
            st.markdown(body.format(**fmt))
