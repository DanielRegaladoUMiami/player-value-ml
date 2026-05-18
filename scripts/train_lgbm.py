"""Train LightGBM on the engineered features.

LightGBM is the favourite for this kind of problem — tabular data, mixed
numeric + categorical, missing values everywhere, hundreds of thousands
of rows. It's also the standard against which everything else is judged
in industry tabular ML.

Setup:
  - Target: y_log_ratio (log of next/current value)
  - Categorical features: handled natively by LightGBM (pd.Categorical)
  - Missing values: handled natively by LightGBM
  - Early stopping on validation split
  - Sample weight = y_horizon_days^-1  (downweight longer-horizon predictions
    because they have more variance; optional)
  - Feature importance saved
"""
from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score

DATA = Path(__file__).resolve().parent.parent / "data" / "features"
OUT = Path(__file__).resolve().parent.parent / "results" / "models" / "lgbm"
OUT.mkdir(parents=True, exist_ok=True)


EXCLUDE = {
    "player_id", "date", "next_date", "next_value", "date_of_birth",
    "y_log_ratio", "y_horizon_days", "market_value_in_eur",
}

CATEGORICAL = ["position", "sub_position", "foot", "career_stage",
               "country_of_citizenship"]


def split_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    feat_cols = [c for c in df.columns if c not in EXCLUDE]
    X = df[feat_cols].copy()
    # Cast categoricals so LightGBM handles them natively
    for c in CATEGORICAL:
        if c in X.columns:
            X[c] = X[c].astype("category")
    y = df["y_log_ratio"]
    return X, y


def metrics(y_true: np.ndarray, y_pred: np.ndarray, prefix: str = "") -> dict:
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    actual_mult = np.exp(y_true)
    pred_mult = np.exp(y_pred)
    mape = float(np.mean(np.abs(actual_mult - pred_mult) / np.maximum(actual_mult, 1e-3)))
    return {f"{prefix}mae_log": float(mae), f"{prefix}r2": float(r2),
            f"{prefix}mape_multiplicative": mape}


def main() -> None:
    print("Loading splits...")
    train = pd.read_parquet(DATA / "train.parquet")
    val   = pd.read_parquet(DATA / "val.parquet")
    test  = pd.read_parquet(DATA / "test.parquet")

    X_train, y_train = split_xy(train)
    X_val,   y_val   = split_xy(val)
    X_test,  y_test  = split_xy(test)
    print(f"  features: {X_train.shape[1]}")
    print(f"  train={len(X_train):,} val={len(X_val):,} test={len(X_test):,}")

    params = {
        "objective": "regression_l1",          # MAE-oriented; aligns with our reporting
        "metric": "mae",
        "learning_rate": 0.05,
        "num_leaves": 127,
        "min_data_in_leaf": 50,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "lambda_l1": 0.1,
        "lambda_l2": 0.1,
        "verbose": -1,
        "seed": 42,
    }

    cat_feats = [c for c in CATEGORICAL if c in X_train.columns]
    print(f"  categorical: {cat_feats}")

    train_set = lgb.Dataset(X_train, y_train, categorical_feature=cat_feats,
                            free_raw_data=False)
    val_set   = lgb.Dataset(X_val, y_val, categorical_feature=cat_feats,
                            reference=train_set, free_raw_data=False)

    print("\nTraining LightGBM (with early stopping on val)...")
    model = lgb.train(
        params,
        train_set,
        num_boost_round=3000,
        valid_sets=[train_set, val_set],
        valid_names=["train", "val"],
        callbacks=[lgb.early_stopping(50),
                   lgb.log_evaluation(100)],
    )

    p_val = model.predict(X_val, num_iteration=model.best_iteration)
    p_test = model.predict(X_test, num_iteration=model.best_iteration)

    results = {
        "best_iteration": int(model.best_iteration),
        **metrics(y_val.values, p_val, "val_"),
        **metrics(y_test.values, p_test, "test_"),
    }
    print(f"\nFinal:")
    print(f"  best_iteration: {results['best_iteration']}")
    print(f"  val:  MAE_log={results['val_mae_log']:.4f}  R²={results['val_r2']:+.4f}")
    print(f"  test: MAE_log={results['test_mae_log']:.4f}  R²={results['test_r2']:+.4f}")

    # Feature importance
    importance = pd.DataFrame({
        "feature": X_train.columns,
        "importance_gain":  model.feature_importance(importance_type="gain"),
        "importance_split": model.feature_importance(importance_type="split"),
    }).sort_values("importance_gain", ascending=False)

    print("\nTop 20 features by gain:")
    print(importance.head(20).to_string(index=False))

    # Save
    model.save_model(str(OUT / "model.lgb"), num_iteration=model.best_iteration)
    (OUT / "metrics.json").write_text(json.dumps(results, indent=2))
    importance.to_csv(OUT / "feature_importance.csv", index=False)
    pd.DataFrame({
        "y_true": y_test.values,
        "y_pred": p_test,
    }).to_parquet(OUT / "test_predictions.parquet", index=False)
    print(f"\nSaved to {OUT}")

    # Comparison vs baselines
    print("\n" + "=" * 64)
    print("  COMPARISON — TEST SET")
    print("=" * 64)
    bl_path = OUT.parent / "baselines" / "metrics.json"
    if bl_path.exists():
        bl = json.loads(bl_path.read_text())
        print(f"  {'model':<25s} {'MAE_log':>10s} {'R²':>10s}")
        for name, m in bl.items():
            print(f"  {name:<25s} {m['test_mae_log']:>10.4f} {m['test_r2']:>+10.4f}")
    print(f"  {'LGBM':<25s} {results['test_mae_log']:>10.4f} {results['test_r2']:>+10.4f}")


if __name__ == "__main__":
    main()
