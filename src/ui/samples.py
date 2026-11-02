"""Sample programs for the UI, taken from the documented examples in ``src.examples``.

They are only parsed, never run.
"""

from src.examples import COMPARISON_PAIRS, EXAMPLES

SAMPLES: dict[str, str] = {e.title: e.code for e in EXAMPLES}
DEFAULT_SAMPLE = "6. Merge sort"
PAIRS = {p.title: p for p in COMPARISON_PAIRS}

# The default before/after pair shown on the Compare page.
COMPARE_BEFORE = COMPARISON_PAIRS[0].before
COMPARE_AFTER = COMPARISON_PAIRS[0].after
