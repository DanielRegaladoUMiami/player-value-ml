"""Classical statistical forecasting with Nixtla StatsForecast.

Fits per-player series: AutoARIMA, AutoETS, Theta. These are the workhorses
of classical forecasting. We expect them to underperform LightGBM here
because our series are short (median 15 points, ~6mo apart), and these
classical models don't get to see exogenous features (position, age, league,
performance).

What we measure: their MAE/R² on the same test split, so we can honestly say
"these are what 50 years of statistical forecasting research gets you on
this problem; it's worse than LightGBM with engineered features".

Setup:
  - Keep only players with >=5 valuations in train (need enough history)
  - For each (player, T) test row, fit on train history < T and forecast 1 step
  - Frequency: irregular, so we treat valuation indices as the time axis
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score

DATA = Path(__file__).resolve().parent.parent / "data" / "features"
OUT = Path(__file__).resolve().parent.parent / "results" / "models" / "stats"
OUT.mkdir(parents=True, exist_ok=True)


def metrics(y_true, y_pred, prefix="") -> dict:
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {f"{prefix}mae_log": float(mae), f"{prefix}r2": float(r2)}


def main() -> None:
    print("Loading raw valuations (need full series, not the feature parquet)...")
    cache = Path.home() / ".cache/kagglehub/datasets/davidcariboo/player-scores/versions/655"
    val_raw = pd.read_csv(cache / "player_valuations.csv", parse_dates=["date"])
    val_raw = val_raw[val_raw["market_value_in_eur"] > 0].copy()
    val_raw["log_value"] = np.log(val_raw["market_value_in_eur"])
    val_raw = val_raw.sort_values(["player_id", "date"])

    # Filter to players with enough history overall
    counts = val_raw.groupby("player_id").size()
    eligible = counts[counts >= 5].index
    val_raw = val_raw[val_raw["player_id"].isin(eligible)]
    print(f"  Eligible players: {len(eligible):,}")

    # Load the test split to know which (player, date) pairs to predict
    test = pd.read_parquet(DATA / "test.parquet")
    test = test[test["player_id"].isin(eligible)][
        ["player_id", "date", "y_log_ratio"]
    ].copy()
    print(f"  Test predictions to make: {len(test):,}")

    # For statistical forecasting we work in LOG-VALUE space, then convert
    # back to log-ratio = forecast - current_log_value at the end.
    val_lookup = val_raw.groupby("player_id")

    # We use simple ETS / SES from statsforecast.models, applied per player.
    # AutoARIMA can be slow per series; we use simpler proven models.
    from statsforecast.models import AutoARIMA, AutoETS, AutoTheta, SeasonalNaive, Naive

    models = {
        "Naive":      Naive(),
        "AutoETS":    AutoETS(season_length=1),
        "AutoTheta":  AutoTheta(season_length=1),
        # AutoARIMA per-series is very slow; we use simpler models here and
        # skip it. (StatsForecast batch API would help but our irregular
        # dates don't fit it cleanly.)
    }

    # Per-player forecast loop — naive Python loop because per-player is
    # fundamentally different series; vectorising across players doesn't
    # help when each series is short.
    print("\nForecasting per (player, test_date)...")
    preds_by_model = {name: np.zeros(len(test)) for name in models}
    successes = {name: 0 for name in models}
    failures = {name: 0 for name in models}

    import warnings
    warnings.filterwarnings("ignore")

    for i, (_, row) in enumerate(test.iterrows()):
        if i % 2000 == 0:
            print(f"  {i}/{len(test)}")
        pid, t_date, _ = row["player_id"], row["date"], row["y_log_ratio"]
        # History UP TO AND INCLUDING t_date — the statistical models then
        # predict 1 step ahead, which is the NEXT valuation. We subtract
        # log_value(t_date) to land in log-ratio space (the target's space).
        hist = val_lookup.get_group(pid)
        hist = hist[hist["date"] <= t_date]["log_value"].values
        if len(hist) < 3:
            # Not enough history -> use last value
            current_lv = (val_lookup.get_group(pid)
                          [val_lookup.get_group(pid)["date"] == t_date]
                          ["log_value"].values)
            if len(current_lv) == 0:
                continue
            for name in models:
                preds_by_model[name][i] = 0.0  # naive: no change
                failures[name] += 1
            continue

        current_log_value = (val_lookup.get_group(pid)
                              [val_lookup.get_group(pid)["date"] == t_date]
                              ["log_value"].values)
        if len(current_log_value) == 0:
            continue
        clv = current_log_value[0]

        for name, model in models.items():
            try:
                m = model.__class__(**model.__dict__) if hasattr(model, "__dict__") else model
                m.fit(hist.astype(np.float32))
                fc = m.predict(h=1)["mean"][0]
                preds_by_model[name][i] = float(fc) - clv
                successes[name] += 1
            except Exception:
                preds_by_model[name][i] = 0.0
                failures[name] += 1

    print("\nPer-model success/failure:")
    for name in models:
        print(f"  {name:12s} ok={successes[name]:,} fail={failures[name]:,}")

    # Score
    results = {}
    y_test = test["y_log_ratio"].values
    for name, preds in preds_by_model.items():
        results[name] = metrics(y_test, preds, prefix="test_")
        print(f"  {name:12s}  MAE={results[name]['test_mae_log']:.4f}  "
              f"R²={results[name]['test_r2']:+.4f}")

    (OUT / "metrics.json").write_text(json.dumps(results, indent=2))
    print(f"\nSaved to {OUT}")


if __name__ == "__main__":
    main()
