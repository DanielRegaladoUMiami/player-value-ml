"""End-to-end validation of the player-value-ml pipeline.

Hard asserts only — every test verifies a concrete property and fails
loudly if it's not met. Run with:

    PYTHONPATH=src pytest -v tests/

Coverage:
  - Feature parquets exist, schema correct, target sane
  - No leakage in temporal splits
  - Each trained model's metrics file is parseable and beats the floor
  - LightGBM model loads and produces sensible predictions
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

DATA = ROOT / "data" / "features"
MODELS = ROOT / "results" / "models"


# ====================== FEATURE PARQUETS ======================


@pytest.fixture(scope="module")
def splits() -> dict[str, pd.DataFrame]:
    return {
        "train": pd.read_parquet(DATA / "train.parquet"),
        "val":   pd.read_parquet(DATA / "val.parquet"),
        "test":  pd.read_parquet(DATA / "test.parquet"),
    }


def test_feature_files_exist():
    for name in ("train", "val", "test"):
        p = DATA / f"{name}.parquet"
        assert p.exists(), f"FAIL: {p} missing"
        assert p.stat().st_size > 1_000_000, f"FAIL: {p} suspiciously small"


def test_feature_count(splits):
    n_cols = splits["train"].shape[1]
    assert n_cols >= 80, f"FAIL: only {n_cols} features (expected >=80)"
    assert n_cols <= 120, f"FAIL: {n_cols} features (suspicious bloat)"


def test_required_columns_present(splits):
    must_have = {
        "player_id", "date", "market_value_in_eur",
        "log_value", "y_log_ratio", "y_horizon_days",
        "age", "position",
        "value_lag_1", "value_diff_1",
        "goals_6mo", "assists_6mo",
        "age_minus_position_peak",
    }
    cols = set(splits["train"].columns)
    missing = must_have - cols
    assert not missing, f"FAIL: missing columns {missing}"


def test_split_sizes(splits):
    assert len(splits["train"]) > 100_000
    assert len(splits["val"]) > 5_000
    assert len(splits["test"]) > 5_000


def test_no_future_leak_in_train(splits):
    assert (splits["train"]["date"] < pd.Timestamp("2023-01-01")).all(), \
        "FAIL: train contains dates >= 2023"


def test_no_future_leak_in_val(splits):
    d = splits["val"]["date"]
    assert d.min() >= pd.Timestamp("2023-01-01") and d.max() <= pd.Timestamp("2023-12-31"), \
        f"FAIL: val dates outside 2023, got {d.min()}..{d.max()}"


def test_test_split_in_future(splits):
    assert (splits["test"]["date"] >= pd.Timestamp("2024-01-01")).all(), \
        "FAIL: test contains dates < 2024"


def test_target_distribution_sane(splits):
    for name, df in splits.items():
        y = df["y_log_ratio"]
        assert y.notna().all(), f"FAIL: {name}: y has nulls"
        assert np.isfinite(y).all(), f"FAIL: {name}: y has inf"
        assert -3 < y.mean() < 3, f"FAIL: {name}: y mean off ({y.mean():.3f})"
        assert 0.1 < y.std() < 1.5, f"FAIL: {name}: y std off ({y.std():.3f})"


def test_lag_features_reproducible(splits):
    """value_lag_1 must equal a manual group-shift over (player_id, date)."""
    s = splits["train"].sort_values(["player_id", "date"]).copy()
    s["expected"] = s.groupby("player_id")["market_value_in_eur"].shift(1)
    mask = s["value_lag_1"].notna() & s["expected"].notna()
    diff = (s.loc[mask, "value_lag_1"] - s.loc[mask, "expected"]).abs().max()
    assert diff < 1e-6, f"FAIL: value_lag_1 mismatch (max abs diff {diff})"


def test_share_features_bounded(splits):
    for c in ["goal_share_team_6mo", "assist_share_team_6mo",
              "minutes_share_team_6mo", "skill_goal_share_6mo"]:
        col = splits["train"][c]
        col = col.dropna()
        if len(col) == 0:
            continue
        assert col.min() >= -1e-6, f"FAIL: {c} < 0 ({col.min()})"
        assert col.max() <= 1.0 + 1e-6, f"FAIL: {c} > 1 ({col.max()})"


def test_no_target_in_features(splits):
    """No feature should correlate >0.9 with y (would be leakage)."""
    train = splits["train"]
    y = train["y_log_ratio"]
    drop = {"y_log_ratio", "next_value", "y_horizon_days"}
    num = train.select_dtypes(include=[np.number]).drop(
        columns=drop & set(train.columns), errors="ignore"
    )
    corrs = num.corrwith(y).abs().dropna()
    leaks = corrs[corrs > 0.9]
    assert len(leaks) == 0, f"FAIL: features that look like target: {dict(leaks)}"


# ====================== TRAINED MODEL ARTIFACTS ======================


def _load_metrics(family: str) -> dict:
    path = MODELS / family / "metrics.json"
    assert path.exists(), f"FAIL: {path} missing — model not trained?"
    return json.loads(path.read_text())


def test_baselines_metrics_exist():
    m = _load_metrics("baselines")
    for key in ("B0_naive_zero", "B1_position_age_mean", "B2_ridge"):
        assert key in m, f"FAIL: baselines missing {key}"
        assert "test_mae_log" in m[key]
        assert "test_r2" in m[key]


def test_baselines_mae_in_expected_range():
    m = _load_metrics("baselines")
    # Sanity: every baseline should be within a known range
    for k, v in m.items():
        assert 0.15 < v["test_mae_log"] < 0.30, \
            f"FAIL: {k} test MAE {v['test_mae_log']:.3f} outside [0.15, 0.30]"


def test_lgbm_metrics_exist():
    m = _load_metrics("lgbm")
    assert "test_mae_log" in m
    assert "test_r2" in m
    assert "best_iteration" in m


def test_lgbm_beats_naive_baseline():
    """LightGBM MUST beat naive on R² (that's the whole point)."""
    bl = _load_metrics("baselines")
    lg = _load_metrics("lgbm")
    assert lg["test_r2"] > bl["B0_naive_zero"]["test_r2"], \
        f"FAIL: LGBM R²={lg['test_r2']:.3f} <= naive R²={bl['B0_naive_zero']['test_r2']:.3f}"


def test_lgbm_beats_ridge_on_r2():
    bl = _load_metrics("baselines")
    lg = _load_metrics("lgbm")
    assert lg["test_r2"] > bl["B2_ridge"]["test_r2"], \
        f"FAIL: LGBM R²={lg['test_r2']:.3f} <= Ridge R²={bl['B2_ridge']['test_r2']:.3f}"


def test_stats_metrics_exist():
    m = _load_metrics("stats")
    assert "Naive" in m
    assert "AutoETS" in m


def test_lgbm_model_loads_and_predicts(splits):
    """LightGBM .lgb file loads and predicts a finite vector."""
    import lightgbm as lgb
    path = MODELS / "lgbm" / "model.lgb"
    assert path.exists(), f"FAIL: {path} missing"
    booster = lgb.Booster(model_file=str(path))

    # Reconstruct the test feature matrix the way train_lgbm did
    EXCLUDE = {
        "player_id", "date", "next_date", "next_value", "date_of_birth",
        "y_log_ratio", "y_horizon_days", "market_value_in_eur",
    }
    CATEGORICAL = ["position", "sub_position", "foot", "career_stage",
                   "country_of_citizenship"]
    test = splits["test"].iloc[:200].copy()
    X = test[[c for c in test.columns if c not in EXCLUDE]].copy()
    for c in CATEGORICAL:
        if c in X.columns:
            X[c] = X[c].astype("category")
    preds = booster.predict(X)
    assert preds.shape == (200,), f"FAIL: pred shape {preds.shape}"
    assert np.isfinite(preds).all(), "FAIL: predictions contain non-finite"
    assert -2 < preds.mean() < 2, f"FAIL: predictions mean off ({preds.mean()})"


# ====================== FEATURE-IMPORTANCE SANITY ======================


def test_lgbm_top_features_make_sense():
    """The top-10 features should include at least one trajectory feature
    AND at least one age/career feature. If neither shows up, something
    is wrong with the model."""
    path = MODELS / "lgbm" / "feature_importance.csv"
    assert path.exists()
    imp = pd.read_csv(path).head(20)
    top_set = set(imp["feature"].head(10))

    trajectory_keywords = {"value_diff", "value_lag", "log_value",
                           "months_since_peak", "consec_"}
    age_keywords = {"age", "career_stage", "peak"}

    has_traj = any(any(k in f for k in trajectory_keywords) for f in top_set)
    has_age = any(any(k in f for k in age_keywords) for f in top_set)
    assert has_traj, f"FAIL: no trajectory feature in top 10: {top_set}"
    assert has_age, f"FAIL: no age/career feature in top 10: {top_set}"
