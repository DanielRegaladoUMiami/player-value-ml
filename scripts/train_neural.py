"""Neural forecasting via Nixtla NeuralForecast (NHITS, TFT, PatchTST).

Approach
--------
Each player's valuations form one irregular time series. We treat the
valuation index as the time axis (so step 1 -> step 2 represents one
update, regardless of calendar gap). NeuralForecast fits ONE global model
across all series; that scales well to 30k+ short series.

Format expected by neuralforecast:
    unique_id  ds  y
    p1         1  log(value_t1)
    p1         2  log(value_t2)
    p1         3  log(value_t3)
    ...
    p2         1  log(value_t1)
    ...

We predict 1 step ahead, then compute log_ratio = y_hat - y_last_observed.

For test evaluation: for every (player, T) in test split, we run a
1-step forecast given the player's history up to and including T, get
y_hat, and compare against the actual y_log_ratio.

To keep training time reasonable on CPU we use NHITS only (the fastest
of the three modern neural forecasters). PatchTST and TFT would be added
in a v2 — they have similar compute profiles but heavier hyperparameters.
"""
from __future__ import annotations

import json
import os
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch

warnings.filterwarnings("ignore")
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"     # MPS still has gaps

DATA = Path(__file__).resolve().parent.parent / "data" / "features"
OUT = Path(__file__).resolve().parent.parent / "results" / "models" / "neural"
OUT.mkdir(parents=True, exist_ok=True)
CACHE = Path.home() / ".cache/kagglehub/datasets/davidcariboo/player-scores/versions/655"


# ----------------------- panel construction -----------------------


def build_panel(min_history: int = 8) -> pd.DataFrame:
    """Build (unique_id, ds, y) panel from raw valuations.

    ds is the within-player index (1..N), making the series 'regular' in
    that sense. y = log(market_value_in_eur).
    """
    val = pd.read_csv(CACHE / "player_valuations.csv", parse_dates=["date"])
    val = val[val["market_value_in_eur"] > 0].copy()
    val = val.sort_values(["player_id", "date"]).reset_index(drop=True)
    val["log_value"] = np.log(val["market_value_in_eur"])

    counts = val.groupby("player_id").size()
    keep = counts[counts >= min_history].index
    val = val[val["player_id"].isin(keep)].copy()
    val["ds"] = val.groupby("player_id").cumcount() + 1
    val = val.rename(columns={"player_id": "unique_id", "log_value": "y"})
    return val[["unique_id", "ds", "y", "date"]]


def main() -> None:
    print("Building panel...")
    panel = build_panel(min_history=8)
    print(f"  {panel['unique_id'].nunique():,} players, {len(panel):,} rows")

    # Train uses everyone's history up to (but not including) the start of
    # the validation period. Targets for test rows come from the full set.
    test_feat = pd.read_parquet(DATA / "test.parquet")
    print(f"  Test feature rows: {len(test_feat):,}")

    test_targets = test_feat[
        test_feat["player_id"].isin(panel["unique_id"])
    ][["player_id", "date", "y_log_ratio"]].copy()
    print(f"  Eligible test targets: {len(test_targets):,}")

    # ---------------- Train ----------------
    # Use only data with date < 2024 for training to mirror the test split
    # (test is >= 2024). validation period is automatically picked.
    train_panel = panel[panel["date"] < pd.Timestamp("2024-01-01")].copy()
    train_panel = train_panel[["unique_id", "ds", "y"]]
    print(f"  Train panel rows: {len(train_panel):,}")

    from neuralforecast import NeuralForecast
    from neuralforecast.models import NHITS

    h = 1                # forecast horizon: 1 step ahead
    input_size = 8       # use last 8 observations as context

    model = NHITS(
        h=h,
        input_size=input_size,
        max_steps=300,           # short for CPU/MPS friendly time budget
        n_blocks=[1, 1, 1],
        mlp_units=[[128, 128], [128, 128], [128, 128]],
        n_pool_kernel_size=[2, 2, 1],
        n_freq_downsample=[4, 2, 1],
        batch_size=512,
        random_seed=42,
        scaler_type="standard",
        learning_rate=1e-3,
    )

    nf = NeuralForecast(models=[model], freq=1)
    print(f"\nTraining NHITS (input_size={input_size}, h=1, max_steps=300)...")
    t0 = time.time()
    nf.fit(df=train_panel)
    print(f"  Training done in {time.time() - t0:.1f}s")

    # ---------------- Predict in BATCHES, not per row ----------------
    # Group test rows by ds index. For each player we want to predict the
    # value at index current_ds + 1. We feed NHITS the panel sliced at the
    # MAX ds we need per player (per row, individually). To avoid 20k
    # separate predict() calls (each one re-runs lightning's init), we
    # use cross_validation: it runs forecasts from many cutoff points
    # over the panel in ONE pass.
    print("\nForecasting (batched cross-validation)...")

    panel_idx = panel.set_index(["unique_id", "date"])["ds"].to_dict()
    # Pair each test row with the cutoff ds.
    test_targets = test_targets.copy()
    test_targets["cutoff_ds"] = test_targets.apply(
        lambda r: panel_idx.get((r["player_id"], r["date"]), -1), axis=1
    )
    test_targets = test_targets[test_targets["cutoff_ds"] >= input_size].copy()
    print(f"  Predictions to issue: {len(test_targets):,}")

    # Group test rows by player to call predict once per player covering
    # all that player's cutoffs.
    pred_log_ratios = np.zeros(len(test_targets))
    panel_by_player = {pid: g for pid, g in panel.groupby("unique_id")}

    t0 = time.time()
    for k, (pid, group) in enumerate(test_targets.groupby("player_id")):
        if k % 200 == 0:
            print(f"  player {k}/{test_targets['player_id'].nunique()}  "
                  f"elapsed={time.time() - t0:.0f}s")
        full_panel = panel_by_player.get(pid)
        if full_panel is None:
            continue
        for _, row in group.iterrows():
            cutoff = int(row["cutoff_ds"])
            hist = full_panel[full_panel["ds"] <= cutoff][["unique_id", "ds", "y"]]
            if len(hist) < input_size:
                continue
            current_y = float(hist.iloc[-1]["y"])
            try:
                fc = nf.predict(df=hist)
                y_next = float(fc["NHITS"].iloc[0])
                # write into the row's index
                pred_log_ratios[row.name] = y_next - current_y
            except Exception:
                continue

    skipped = int((pred_log_ratios == 0.0).sum())

    print(f"\nDone. skipped={skipped}/{len(test_targets)}")

    # ---------------- Score ----------------
    from sklearn.metrics import mean_absolute_error, r2_score
    y_true = test_targets["y_log_ratio"].values
    mae = mean_absolute_error(y_true, pred_log_ratios)
    r2 = r2_score(y_true, pred_log_ratios)
    results = {
        "NHITS": {
            "test_mae_log": float(mae),
            "test_r2": float(r2),
            "n_eligible": int(len(test_targets) - skipped),
            "n_skipped": int(skipped),
            "input_size": input_size,
        },
    }
    print(f"\nNHITS  test MAE_log={mae:.4f}  R²={r2:+.4f}")

    (OUT / "metrics.json").write_text(json.dumps(results, indent=2))
    pd.DataFrame({
        "y_true": y_true,
        "y_pred_nhits": pred_log_ratios,
    }).to_parquet(OUT / "test_predictions.parquet", index=False)
    print(f"\nSaved to {OUT}")


if __name__ == "__main__":
    main()
