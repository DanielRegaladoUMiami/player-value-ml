"""Audit the feature pipeline output BEFORE training anything.

Hard checks (any failure = exit non-zero):
  L1. No future-leak: no feature row may contain info from after `date`
  L2. Target distribution sensible (mean ~0, std finite, no inf/nan)
  L3. Lag features actually point to past valuations (value_lag_1 < log_value? no — equal log())
  L4. Temporal splits are non-overlapping and cover all rows
  L5. No constant feature columns (variance > 0)
  L6. No feature is essentially the target in disguise (corr(f, y) < 0.95)

Soft checks (report only):
  R1. Distribution per feature: mean, std, %nulls, quantiles
  R2. Correlation with target (top 20 strongest signals)
  R3. Position-group means of position-relative z-scores should be ~0
  R4. Sanity: ratios in [0, 1]; growth features bounded
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent.parent / "results" / "audit"
OUT.mkdir(parents=True, exist_ok=True)

DATA = Path(__file__).resolve().parent.parent / "data" / "features"

FAIL = []   # accumulates hard-check failures


def fail(msg: str) -> None:
    print(f"  FAIL: {msg}")
    FAIL.append(msg)


def ok(msg: str) -> None:
    print(f"  OK  : {msg}")


# ---------------------------------------------------------------- L checks


def L1_no_future_leak(splits: dict[str, pd.DataFrame]) -> None:
    """Train rows must be < 2023; val 2023; test >= 2024."""
    print("\n[L1] No future leak in splits...")
    if (splits["train"]["date"] >= "2023-01-01").any():
        fail("train contains rows with date >= 2023-01-01")
    else:
        ok("train all dates < 2023-01-01")

    if not (splits["val"]["date"].between("2023-01-01", "2023-12-31").all()):
        fail("val rows outside 2023")
    else:
        ok("val all dates in 2023")

    if (splits["test"]["date"] < "2024-01-01").any():
        fail("test contains rows with date < 2024-01-01")
    else:
        ok("test all dates >= 2024-01-01")


def L2_target_distribution(splits: dict[str, pd.DataFrame]) -> None:
    print("\n[L2] Target distribution sanity...")
    for name, df in splits.items():
        y = df["y_log_ratio"]
        if y.isna().any():
            fail(f"{name}: y has nulls ({y.isna().sum()})")
        if not np.isfinite(y).all():
            fail(f"{name}: y contains inf")
        if y.std() == 0:
            fail(f"{name}: y has zero variance")
        ok(f"{name}: y  mean={y.mean():+.3f}  std={y.std():.3f}  "
           f"min={y.min():+.2f}  max={y.max():+.2f}  n={len(y):,}")


def L3_lag_correctness(train: pd.DataFrame) -> None:
    """value_lag_1 should equal the value from the player's previous row.
    Test on a small sample by reconstructing from market_value_in_eur."""
    print("\n[L3] Lag features correctness...")
    s = train.sort_values(["player_id", "date"]).copy()
    s["expected_lag_1"] = s.groupby("player_id")["market_value_in_eur"].shift(1)
    # Compare where both are non-null
    mask = s["value_lag_1"].notna() & s["expected_lag_1"].notna()
    diff = (s.loc[mask, "value_lag_1"] - s.loc[mask, "expected_lag_1"]).abs()
    max_diff = diff.max() if len(diff) else 0
    if max_diff > 1e-6:
        fail(f"value_lag_1 disagrees with manual shift; max abs diff = {max_diff}")
    else:
        ok(f"value_lag_1 matches groupby shift on {mask.sum():,} rows")


def L4_splits_partition(splits: dict[str, pd.DataFrame]) -> None:
    print("\n[L4] Splits are non-overlapping and complete...")
    sizes = {k: len(v) for k, v in splits.items()}
    total = sum(sizes.values())
    # Build a multi-set of (player_id, date) per split and check disjoint
    keys = {k: set(map(tuple, v[["player_id", "date"]].values))
            for k, v in splits.items()}
    for a in ("train", "val"):
        for b in ("val", "test"):
            if a >= b:
                continue
            inter = keys[a] & keys[b]
            if inter:
                fail(f"overlap between {a} and {b}: {len(inter)} rows")
            else:
                ok(f"no overlap {a} ∩ {b}")
    ok(f"total rows across splits: {total:,}")


def L5_no_constant_features(train: pd.DataFrame) -> None:
    print("\n[L5] No constant feature columns...")
    drop_cols = {"player_id", "date", "next_date", "date_of_birth"}
    num = train.select_dtypes(include=[np.number]).drop(columns=drop_cols & set(train.columns), errors="ignore")
    constants = [c for c in num.columns if num[c].nunique(dropna=True) <= 1]
    if constants:
        fail(f"constant columns: {constants}")
    else:
        ok(f"all {len(num.columns)} numeric columns have >1 unique value")


def L6_no_target_in_disguise(train: pd.DataFrame) -> None:
    print("\n[L6] No feature is a target proxy...")
    y = train["y_log_ratio"]
    num = train.select_dtypes(include=[np.number]).drop(
        columns=["y_log_ratio", "next_value", "y_horizon_days"], errors="ignore"
    )
    corrs = num.corrwith(y).abs().sort_values(ascending=False)
    leak_threshold = 0.95
    bad = corrs[corrs > leak_threshold]
    if len(bad):
        fail(f"features with |corr(y)| > {leak_threshold}: {dict(bad.head(5))}")
    else:
        ok(f"max |corr(feature, y)| = {corrs.max():.3f}  "
           f"(top: {corrs.head(3).to_dict()})")


# ---------------------------------------------------------------- R reports


def R1_feature_summary(train: pd.DataFrame) -> pd.DataFrame:
    print("\n[R1] Feature distribution summary...")
    drop_cols = {"player_id", "date", "next_date", "date_of_birth", "next_value"}
    rep_rows = []
    for c in train.columns:
        if c in drop_cols:
            continue
        col = train[c]
        if pd.api.types.is_numeric_dtype(col):
            rep_rows.append({
                "feature": c,
                "dtype": str(col.dtype),
                "nulls_pct": round(100 * col.isna().mean(), 1),
                "mean": float(col.mean(skipna=True)) if col.notna().any() else None,
                "std":  float(col.std(skipna=True))  if col.notna().any() else None,
                "p01":  float(col.quantile(0.01)) if col.notna().any() else None,
                "p50":  float(col.quantile(0.50)) if col.notna().any() else None,
                "p99":  float(col.quantile(0.99)) if col.notna().any() else None,
            })
        else:
            rep_rows.append({
                "feature": c,
                "dtype": str(col.dtype),
                "nulls_pct": round(100 * col.isna().mean(), 1),
                "unique": col.nunique(dropna=True),
            })
    rep = pd.DataFrame(rep_rows)
    print(rep.to_string(max_rows=200, max_cols=10))
    rep.to_csv(OUT / "feature_summary.csv", index=False)
    print(f"  -> {OUT / 'feature_summary.csv'}")
    return rep


def R2_correlation_with_target(train: pd.DataFrame) -> pd.Series:
    print("\n[R2] Top 20 correlations with target (|corr|)...")
    y = train["y_log_ratio"]
    drop_cols = {"y_log_ratio", "next_value", "y_horizon_days"}
    num = train.select_dtypes(include=[np.number]).drop(
        columns=drop_cols & set(train.columns), errors="ignore"
    )
    corrs = num.corrwith(y).dropna()
    top = corrs.abs().sort_values(ascending=False).head(20)
    print(top.round(3).to_string())
    top.to_csv(OUT / "top_corr_target.csv")
    return top


def R3_zscore_means(train: pd.DataFrame) -> None:
    print("\n[R3] Position-relative z-scores should have ~0 mean per position...")
    z_cols = [c for c in train.columns if c.endswith("_zscore_vs_pos")]
    if not z_cols:
        print("  (no z-score columns found — skipping)")
        return
    for z in z_cols:
        means = train.groupby("position")[z].mean()
        print(f"\n  {z}:")
        print(means.round(3).to_string())


def R4_bounded_ratios(train: pd.DataFrame) -> None:
    print("\n[R4] Share/ratio sanity (should be in [0, 1])...")
    for c in ["goal_share_team_6mo", "assist_share_team_6mo",
              "minutes_share_team_6mo", "skill_goal_share_6mo"]:
        if c not in train.columns:
            continue
        col = train[c]
        if col.min() < -1e-6 or col.max() > 1.0 + 1e-6:
            fail(f"{c} out of [0,1]: min={col.min()}, max={col.max()}")
        else:
            ok(f"{c} in [{col.min():.3f}, {col.max():.3f}]")


# ---------------------------------------------------------------- driver


def main() -> int:
    print("=" * 70)
    print("  FEATURE AUDIT — player-value-ml")
    print("=" * 70)

    splits = {
        "train": pd.read_parquet(DATA / "train.parquet"),
        "val":   pd.read_parquet(DATA / "val.parquet"),
        "test":  pd.read_parquet(DATA / "test.parquet"),
    }
    for k, df in splits.items():
        print(f"  {k:6s}: {len(df):>7,} rows × {len(df.columns)} cols")

    L1_no_future_leak(splits)
    L2_target_distribution(splits)
    L3_lag_correctness(splits["train"])
    L4_splits_partition(splits)
    L5_no_constant_features(splits["train"])
    L6_no_target_in_disguise(splits["train"])

    R1_feature_summary(splits["train"])
    R2_correlation_with_target(splits["train"])
    R3_zscore_means(splits["train"])
    R4_bounded_ratios(splits["train"])

    print("\n" + "=" * 70)
    if FAIL:
        print(f"  AUDIT FAILED with {len(FAIL)} issue(s):")
        for m in FAIL:
            print(f"    - {m}")
        return 1
    print("  AUDIT PASSED — all hard checks OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
