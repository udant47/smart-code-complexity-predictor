"""Tests for src.eda (Phase 8).

Unit tests use tiny hand-made tables whose answers can be checked by
hand. The last test runs the full analysis only if the real processed
data exists.
"""

import json

import matplotlib

matplotlib.use("Agg")

import pandas as pd
import pytest

from src import config, eda


def small_bigo() -> pd.DataFrame:
    return pd.DataFrame({
        "sample_id": [f"s{i}" for i in range(6)],
        "code_sha256": [f"h{i}" for i in range(6)],
        "problem": ["p1", "p1", "p1", "p2", "p2", "p3"],
        "label": ["linear", "linear", "linear", "cubic", "quadratic", "constant"],
        "label_bigo": ["O(n)", "O(n)", "O(n)", "O(n^3)", "O(n^2)", "O(1)"],
        "label_rank": [2, 2, 2, 5, 4, 0],
        "tags": ["math", "math,dp", "", "dp", "dp", "math"],
        "near_duplicate_cluster": [0, 0, -1, -1, -1, -1],
        "loops": [1, 1, 1, 3, 2, 0],
        "loops_copy": [1, 1, 1, 3, 2, 0],
        "lines": [10, 12, 11, 40, 30, 5],
        "zeros": [0, 0, 0, 0, 0, 0],
    })


def test_overview_counts_and_flags():
    df = small_bigo()
    o = eda.overview(df, ["loops", "lines", "zeros"])
    assert (o["rows"], o["feature_columns"], o["missing_values"]) == (6, 3, 0)
    assert o["constant_features"] == ["zeros"]
    assert o["empty_metadata_strings"] == {"tags": 1}


def test_duplicate_summary_detects_shared_vectors_and_conflicts():
    df = small_bigo()
    d = eda.duplicate_summary(df, ["loops", "loops_copy"], "label")
    assert d["duplicate_sample_ids"] == 0
    assert d["rows_sharing_a_feature_vector"] == 3  # the three linear rows
    assert d["feature_vectors_with_conflicting_labels"] == 0
    conflict = df.assign(label=["linear", "cubic", "linear", "cubic", "quadratic", "constant"])
    d2 = eda.duplicate_summary(conflict, ["loops"], "label")
    assert d2["feature_vectors_with_conflicting_labels"] == 1
    assert d2["accuracy_ceiling_from_features"] == pytest.approx(5 / 6, abs=1e-4)


def test_class_distribution_keeps_order_and_zero_classes():
    dist = eda.class_distribution(small_bigo()["label_bigo"], eda.BIGO_ORDER)
    assert list(dist.index) == eda.BIGO_ORDER
    assert dist.loc["O(n)", "count"] == 3 and dist.loc["O(log n)", "count"] == 0
    assert dist["percent"].sum() == pytest.approx(100, abs=0.05)


def test_redundant_pairs_finds_identical_columns():
    pairs = eda.redundant_pairs(small_bigo(), ["loops", "loops_copy", "lines", "zeros"])
    first = pairs.iloc[0]
    assert {first["feature_a"], first["feature_b"]} == {"loops", "loops_copy"}
    assert first["spearman"] == 1.0
    assert "zeros" not in set(pairs["feature_a"]) | set(pairs["feature_b"])  # constant columns skipped


def test_outlier_summary():
    df = pd.DataFrame({"x": [1, 2, 2, 3, 3, 3, 4, 100]})
    out = eda.outlier_summary(df, ["x"], k=3)
    assert out.loc["x", "outliers"] == 1


def test_problem_label_purity():
    p = eda.problem_label_purity(small_bigo())
    assert p["problems"] == 3 and p["single_label_problems"] == 2
    assert p["labels_per_problem"] == {1: 2, 2: 1}
    assert p["samples_with_problem_majority_label_%"] == pytest.approx(round(100 * 5 / 6, 1))


def test_feature_label_association_sign():
    assoc = eda.feature_label_association(small_bigo(), ["loops", "lines"], "label_rank")
    assert (assoc["spearman_with_label_rank"] > 0).all()


def test_tag_table_rows_sum_to_100():
    table = eda.tag_label_table(small_bigo())
    assert set(table.index) == {"math", "dp"}
    assert table.fillna(0).sum(axis=1).round(0).eq(100).all()


def test_structural_circularity_is_detected():
    from src.feature_extractor import FUNCTION_FEATURE_NAMES

    rows = []
    for cc in [1, 3, 5, 6, 10, 11, 20]:
        row = {f: cc for f in FUNCTION_FEATURE_NAMES}
        row["reference_label"] = "Low" if cc <= 5 else "Medium" if cc <= 10 else "High"
        rows.append(row)
    df = pd.DataFrame(rows)
    df["reference_label"] = pd.Categorical(df["reference_label"], categories=eda.BAND_ORDER, ordered=True)
    c = eda.structural_circularity(df)
    assert c["rule_reproduces_label_%"] == 100.0
    assert c["cc_range_per_band"]["Medium"] == {"min": 6, "max": 10, "count": 2}


def test_near_duplicate_summary():
    n = eda.near_duplicate_summary(small_bigo())
    assert n == {"samples_in_clusters": 2, "clusters": 1, "largest_cluster": 2,
                 "clusters_with_mixed_labels": 0, "clusters_spanning_problems": 0}


def test_template_prevalence():
    df = small_bigo()
    df["label_bigo"] = pd.Categorical(df["label_bigo"], categories=eda.BIGO_ORDER)
    src = pd.Series(["import sys\nsys.stdin.readline()", "x", "x", "sys.stdin", "x", "x"])
    t = eda.template_prevalence(df, src)
    assert t.loc["reads sys.stdin", "all"] == pytest.approx(round(100 * 2 / 6, 1))
    assert t.loc["reads sys.stdin", "O(n^3)"] == 100.0
    assert eda.template_prevalence(df, None) is None


def test_markdown_table_has_header_and_rows():
    text = eda._md_table(pd.DataFrame({"a": [1, 2]}, index=["x", "y"]))
    lines = text.splitlines()
    assert lines[0] == "| feature | a |" and len(lines) == 4


@pytest.mark.skipif(not config.BIGO_DATASET_CSV.exists(), reason="processed data not built")
def test_full_report_on_real_data(tmp_path):
    results = eda.run(out_dir=tmp_path)
    assert len(list(tmp_path.glob("*.png"))) == 9
    report = (tmp_path / "eda_report.md").read_text(encoding="utf-8")
    assert "Assessment" in report and "not suitable as a supervised target" in report
    saved = json.loads((tmp_path / "eda_results.json").read_text(encoding="utf-8"))
    assert saved["bigo_overview"]["missing_values"] == 0
    assert results["structural_circularity"]["rule_reproduces_label_%"] == 100.0
