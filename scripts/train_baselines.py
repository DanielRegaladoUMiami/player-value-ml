"""Train baselines that everything else has to beat.

Three baselines, in order of sophistication:

  B0. Naive: predict y_log_ratio = 0 (i.e. no change)
      Trivial but surprisingly hard to beat because the EDA target distribution
      has median = 0 and 31% of pairs are 'flat'.

  B1. Per-(position, age_band) mean of y_log_ratio
      Statistical baseline that captures the dominant signal: young players
      go up, old players go down.

  B2. Ridge regression on the full feature set, log-target
      Linear model with regularisation. Real baseline that uses ALL features.

Reports MAE, MAPE, R² on validation and test. Saves predictions and
model artifacts to results/models/.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

DATA = Path(__file__).resolve().parent.parent / "data" / "features"
OUT = Path(__file__).resolve().parent.parent / "results" / "models" / "baselines"
OUT.mkdir(parents=True, exist_ok=True)


# Features to use. We exclude leakage columns (next_*, y_*) and identifiers.
EXCLUDE = {
    "player_id", "date", "next_date", "next_value", "date_of_birth",
    "y_log_ratio", "y_horizon_days", "market_value_in_eur",
}

CATEGORICAL = ["position", "sub_position", "foot", "career_stage",
               "country_of_citizenship"]


def split_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    feature_cols = [c for c in df.columns if c not in EXCLUDE]
    return df[feature_cols], df["y_log_ratio"]


def metrics(y_true: np.ndarray, y_pred: np.ndarray, prefix: str = "") -> dict:
    """All metrics in log-ratio space."""
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    # MAPE in linear (€) space: convert back from log-ratio to multiplicative
    actual_mult = np.exp(y_true)
    pred_mult = np.exp(y_pred)
    mape = float(np.mean(np.abs(actual_mult - pred_mult) / np.maximum(actual_mult, 1e-3)))
    return {
        f"{prefix}mae_log": float(mae),
        f"{prefix}r2": float(r2),
        f"{prefix}mape_multiplicative": mape,
    }


# ---------- Baselines ----------


def baseline_zero(y: pd.Series) -> np.ndarray:
    return np.zeros(len(y))


def baseline_position_age(train: pd.DataFrame, holdout: pd.DataFrame) -> np.ndarray:
    """Mean of y_log_ratio per (position, age_band)."""
    t = train.copy()
    t["age_band"] = pd.cut(t["age"], bins=[0, 21, 25, 28, 32, 100],
                            labels=["U21", "22-25", "26-28", "29-32", "33+"])
    lookup = t.groupby(["position", "age_band"], observed=True)["y_log_ratio"].mean()
    global_mean = t["y_log_ratio"].mean()

    h = holdout.copy()
    h["age_band"] = pd.cut(h["age"], bins=[0, 21, 25, 28, 32, 100],
                            labels=["U21", "22-25", "26-28", "29-32", "33+"])
    preds = []
    for _, row in h.iterrows():
        key = (row["position"], row["age_band"])
        preds.append(lookup.get(key, global_mean))
    return np.asarray(preds)


def build_ridge_pipeline(X: pd.DataFrame) -> Pipeline:
    cat_cols = [c for c in CATEGORICAL if c in X.columns]
    num_cols = [c for c in X.columns
                if c not in cat_cols and pd.api.types.is_numeric_dtype(X[c])]

    num_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    cat_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value="missing")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", min_frequency=10,
                              sparse_output=False)),
    ])
    pre = ColumnTransformer([
        ("num", num_pipe, num_cols),
        ("cat", cat_pipe, cat_cols),
    ], remainder="drop")

    return Pipeline([
        ("pre", pre),
        ("ridge", Ridge(alpha=1.0)),
    ])


# ---------- driver ----------


def main() -> None:
    print("Loading splits...")
    train = pd.read_parquet(DATA / "train.parquet")
    val   = pd.read_parquet(DATA / "val.parquet")
    test  = pd.read_parquet(DATA / "test.parquet")

    X_train, y_train = split_features(train)
    X_val,   y_val   = split_features(val)
    X_test,  y_test  = split_features(test)
    print(f"  train: {len(X_train):>7,} rows, val: {len(X_val):>6,}, test: {len(X_test):>6,}")
    print(f"  features (all): {X_train.shape[1]}")

    results = {}

    # B0: Naive zero
    print("\nB0. Naive (y_hat = 0)...")
    p_val = baseline_zero(y_val)
    p_test = baseline_zero(y_test)
    results["B0_naive_zero"] = {
        **metrics(y_val.values, p_val, "val_"),
        **metrics(y_test.values, p_test, "test_"),
    }
    print(f"  val:  MAE_log={results['B0_naive_zero']['val_mae_log']:.4f}  "
          f"R2={results['B0_naive_zero']['val_r2']:+.4f}")
    print(f"  test: MAE_log={results['B0_naive_zero']['test_mae_log']:.4f}  "
          f"R2={results['B0_naive_zero']['test_r2']:+.4f}")

    # B1: Position-age mean
    print("\nB1. Position × age-band mean...")
    p_val = baseline_position_age(train, val)
    p_test = baseline_position_age(train, test)
    results["B1_position_age_mean"] = {
        **metrics(y_val.values, p_val, "val_"),
        **metrics(y_test.values, p_test, "test_"),
    }
    print(f"  val:  MAE_log={results['B1_position_age_mean']['val_mae_log']:.4f}  "
          f"R2={results['B1_position_age_mean']['val_r2']:+.4f}")
    print(f"  test: MAE_log={results['B1_position_age_mean']['test_mae_log']:.4f}  "
          f"R2={results['B1_position_age_mean']['test_r2']:+.4f}")

    # B2: Ridge on full feature set
    print("\nB2. Ridge regression on full feature set...")
    pipe = build_ridge_pipeline(X_train)
    # Fit on TRAIN only — no peeking at val/test
    pipe.fit(X_train, y_train)
    p_val = pipe.predict(X_val)
    p_test = pipe.predict(X_test)
    results["B2_ridge"] = {
        **metrics(y_val.values, p_val, "val_"),
        **metrics(y_test.values, p_test, "test_"),
    }
    print(f"  val:  MAE_log={results['B2_ridge']['val_mae_log']:.4f}  "
          f"R2={results['B2_ridge']['val_r2']:+.4f}")
    print(f"  test: MAE_log={results['B2_ridge']['test_mae_log']:.4f}  "
          f"R2={results['B2_ridge']['test_r2']:+.4f}")

    # Save predictions + metrics
    (OUT / "metrics.json").write_text(json.dumps(results, indent=2))
    pd.DataFrame({
        "y_true": y_test.values,
        "pred_B0": baseline_zero(y_test),
        "pred_B1": baseline_position_age(train, test),
        "pred_B2": pipe.predict(X_test),
    }).to_parquet(OUT / "test_predictions.parquet", index=False)
    print(f"\nSaved to {OUT}")

    # Print final comparison
    print("\n" + "=" * 60)
    print("  BASELINES — TEST SET")
    print("=" * 60)
    print(f"  {'model':<25s} {'MAE_log':>10s} {'R²':>10s}")
    for name, m in results.items():
        print(f"  {name:<25s} {m['test_mae_log']:>10.4f} {m['test_r2']:>+10.4f}")


if __name__ == "__main__":
    main()
