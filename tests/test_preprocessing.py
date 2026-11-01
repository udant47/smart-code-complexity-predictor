"""Tests for src.preprocessing (Phase 9).

Most tests use a small deterministic test table (not real data) built so
that the right answer is known: each "problem" has its own typical feature
values and one label. The last tests run on the real dataset if it has
been built.
"""

import json

import numpy as np
import pandas as pd
import pytest

from src import config
from src import preprocessing as pp
from src.feature_extractor import FILE_FEATURE_NAMES


def fixture_table(n_problems: int = 70, per_problem: int = 6, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for p in range(n_problems):
        label = pp.CLASS_ORDER[p % len(pp.CLASS_ORDER)]
        centre = rng.integers(0, 50, size=len(FILE_FEATURE_NAMES))
        for k in range(per_problem):
            feats = centre + rng.integers(0, 2, size=len(FILE_FEATURE_NAMES))
            rows.append({
                "sample_id": f"s-{p:03d}-{k:02d}",
                "source_dataset": "test", "source_commit": "x", "raw_index": len(rows),
                "problem": f"{p // 2:04d}_{'AB'[p % 2]}",
                "tags": "dp",
                "label": label, "label_name": label, "label_bigo": label, "label_rank": 0, "label_method": "x",
                "near_duplicate_cluster": -1, "code_sha256": f"h{len(rows)}",
                **dict(zip(FILE_FEATURE_NAMES, feats.tolist())),
            })
    df = pd.DataFrame(rows)
    # two near-duplicates in different problems: they must end up in one group
    df.loc[0, "near_duplicate_cluster"] = 7
    df.loc[per_problem * 5, "near_duplicate_cluster"] = 7  # problem 5
    return df


@pytest.fixture(scope="module")
def table():
    return fixture_table()


_SPLITS: dict = {}


def split_of(table, strategy, seed=config.RANDOM_STATE):
    key = (strategy, seed)
    if key not in _SPLITS:
        _SPLITS[key] = pp.make_split(table, strategy, seed=seed)
    return _SPLITS[key]


# --------------------------------------------------------------------------- #
# Features and target
# --------------------------------------------------------------------------- #
def test_features_exclude_identifiers_metadata_and_target(table):
    X, y = pp.split_features_target(table)
    assert list(X.columns) == list(FILE_FEATURE_NAMES)
    assert not set(X.columns) & set(pp.EXCLUDED_COLUMNS)
    assert y.name == "label" and len(y) == len(X)


def test_every_non_feature_column_has_an_exclusion_reason(table):
    non_features = set(table.columns) - set(FILE_FEATURE_NAMES)
    assert non_features <= set(pp.EXCLUDED_COLUMNS)


def test_no_size_feature_set(table):
    X, _ = pp.split_features_target(table, "no_size")
    assert not set(X.columns) & set(pp.SIZE_FEATURES)
    assert "maximum_loop_nesting_depth" in X.columns


def test_unknown_feature_set_rejected(table):
    with pytest.raises(ValueError):
        pp.split_features_target(table, "everything")


# --------------------------------------------------------------------------- #
# Groups
# --------------------------------------------------------------------------- #
def test_groups_by_problem_and_contest(table):
    g_problem = pp.make_groups(table, "problem")
    g_contest = pp.make_groups(table, "contest")
    assert g_problem.nunique() < table["problem"].nunique() + 1
    assert g_contest.nunique() < g_problem.nunique()


def test_near_duplicates_are_merged_into_one_group(table):
    for by in ("problem", "contest", "none"):
        g = pp.make_groups(table, by)
        assert g.iloc[0] == g.iloc[6 * 5]
    g = pp.make_groups(table, "problem")
    # the merge pulls both whole problems into the same group
    assert g[table["problem"] == table.loc[0, "problem"]].iloc[-1] == g.iloc[6 * 5]


# --------------------------------------------------------------------------- #
# Splits
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("strategy", ["problem", "contest", "random"])
def test_partitions_are_disjoint_and_complete(table, strategy):
    split = split_of(table, strategy)
    parts = split.assignment
    assert set(parts) == {"train", "validation", "test"}
    assert parts.notna().all()
    sizes = parts.value_counts(normalize=True)
    assert 0.1 <= sizes["test"] <= 0.3 and sizes["train"] > 0.5


@pytest.mark.parametrize("strategy", ["problem", "contest"])
def test_grouped_split_keeps_groups_and_near_duplicates_together(table, strategy):
    split = split_of(table, strategy)
    per_group = split.assignment.groupby(split.groups).nunique()
    assert (per_group == 1).all()
    assert split.assignment.iloc[0] == split.assignment.iloc[6 * 5]
    summary = pp.split_summary(table, split)
    assert summary["test_samples_whose_problem_is_in_train_%"] == 0.0
    assert summary["near_duplicate_clusters_crossing_partitions"] == 0
    assert summary["groups_in_more_than_one_partition"] == 0


def test_contest_split_keeps_contests_apart(table):
    summary = pp.split_summary(table, split_of(table, "contest"))
    assert summary["test_samples_whose_contest_is_in_train_%"] == 0.0


def test_random_split_mixes_problems(table):
    summary = pp.split_summary(table, split_of(table, "random"))
    assert summary["test_samples_whose_problem_is_in_train_%"] > 90


def test_stratification_keeps_every_class_in_every_partition(table):
    split = split_of(table, "problem")
    for part in ("train", "validation", "test"):
        assert set(table.loc[split.indices(part), "label"]) == set(pp.CLASS_ORDER)


def test_splits_are_reproducible(table):
    a, b = split_of(table, "problem"), pp.make_split(table, "problem")
    assert a.assignment.equals(b.assignment) and a.cv_fold.equals(b.cv_fold)
    c = split_of(table, "problem", seed=7)
    assert not a.assignment.equals(c.assignment)


def test_cv_folds_cover_development_rows_and_respect_groups(table):
    split = split_of(table, "problem")
    dev_mask = split.assignment != "test"
    assert (split.cv_fold[dev_mask] >= 0).all() and (split.cv_fold[~dev_mask] == -1).all()
    folds = pp.cv_splits(split)
    assert len(folds) == pp.CV_FOLDS
    dev_groups = split.groups[dev_mask].values
    seen = np.concatenate([val for _, val in folds])
    assert sorted(seen) == list(range(dev_mask.sum()))
    for tr, va in folds:
        assert not set(dev_groups[tr]) & set(dev_groups[va])


# --------------------------------------------------------------------------- #
# Pipelines: fitted on training rows only
# --------------------------------------------------------------------------- #
def test_scaler_statistics_come_from_training_rows_only(table):
    split = split_of(table, "problem")
    X, _ = pp.split_features_target(table)
    train = X.loc[split.indices("train")]
    pre = pp.build_preprocessor("linear", X.columns).fit(train)
    scaler = pre.named_transformers_["features"].named_steps["scale"]
    np.testing.assert_allclose(scaler.mean_, np.log1p(train.values.astype(float)).mean(axis=0))
    assert not np.allclose(scaler.mean_, np.log1p(X.values.astype(float)).mean(axis=0))


def test_linear_output_is_standardised_on_train(table):
    X, _ = pp.split_features_target(table)
    out = pp.build_preprocessor("distance", X.columns).fit_transform(X)
    np.testing.assert_allclose(out.mean(axis=0), 0, atol=1e-9)


def test_tree_preprocessing_does_not_rescale(table):
    X, _ = pp.split_features_target(table)
    out = pp.build_preprocessor("tree", X.columns).fit_transform(X)
    np.testing.assert_array_equal(out, X.values.astype(float))


def test_extra_columns_are_dropped_not_used(table):
    pre = pp.build_preprocessor("linear", FILE_FEATURE_NAMES).fit(table)  # full table incl. label/tags
    assert list(pre.get_feature_names_out()) == list(FILE_FEATURE_NAMES)
    assert pre.transform(table).shape[1] == len(FILE_FEATURE_NAMES)


def test_missing_and_infinite_values_are_imputed(table):
    X, _ = pp.split_features_target(table)
    X = X.astype(float)
    X.iloc[0, 0] = np.nan
    X.iloc[1, 1] = np.inf
    for family in ("tree", "linear"):
        out = pp.build_preprocessor(family, X.columns).fit_transform(X)
        assert np.isfinite(out).all()


def test_make_pipeline_fits_preprocessing_inside(table):
    from sklearn.dummy import DummyClassifier

    split = split_of(table, "problem")
    X, y = pp.split_features_target(table)
    tr = split.indices("train")
    pipe = pp.make_pipeline("linear", DummyClassifier(strategy="most_frequent"), X.columns).fit(X.loc[tr], y.loc[tr])
    assert pipe.predict(X.loc[split.indices("test")]).shape == (len(split.indices("test")),)


def test_unknown_model_family_rejected():
    with pytest.raises(ValueError):
        pp.build_preprocessor("neural", FILE_FEATURE_NAMES)


# --------------------------------------------------------------------------- #
# Leakage probe and saving
# --------------------------------------------------------------------------- #
def test_leakage_probe_shows_random_split_inflation(table):
    # Labels follow the problem, and features identify the problem: a random
    # split lets 1-NN copy labels from other solutions of the same problem.
    random = pp.leakage_probe(table, "random")
    grouped = pp.leakage_probe(table, "problem")
    assert random["accuracy_mean"] > 0.9
    assert grouped["accuracy_mean"] < random["accuracy_mean"] - 0.3


def test_save_and_load_split_roundtrip(table, tmp_path):
    split = split_of(table, "problem")
    path = pp.save_split(table, split, tmp_path / "splits.csv")
    loaded = pp.load_split(table, path)
    assert loaded.assignment.equals(split.assignment)
    assert loaded.cv_fold.tolist() == split.cv_fold.tolist()
    with pytest.raises(ValueError):
        pp.load_split(table.iloc[:-1], path)


def test_save_config_records_features_and_exclusions(table, tmp_path):
    split = split_of(table, "problem")
    path = pp.save_config(table, split, {"note": "test"}, tmp_path / "cfg.json")
    cfg = json.loads(path.read_text(encoding="utf-8"))
    assert cfg["feature_sets"]["all"] == list(FILE_FEATURE_NAMES)
    assert cfg["excluded_columns"]["tags"].startswith("Codeforces metadata")
    assert cfg["pipelines"]["fit_on"].startswith("training rows only")
    assert cfg["split"]["random_state"] == config.RANDOM_STATE


# --------------------------------------------------------------------------- #
# Real data (only when built)
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not config.BIGO_DATASET_CSV.exists(), reason="processed data not built")
def test_real_grouped_split_has_no_leakage_and_good_balance():
    df = pd.read_csv(config.BIGO_DATASET_CSV, keep_default_na=False)
    split = pp.make_split(df, "problem")
    s = pp.split_summary(df, split)
    assert s["test_samples_whose_problem_is_in_train_%"] == 0.0
    assert s["near_duplicate_clusters_crossing_partitions"] == 0
    assert s["partitions"]["test"]["max_class_deviation_pp"] < 5
    assert 15 <= s["partitions"]["test"]["share_%"] <= 25
