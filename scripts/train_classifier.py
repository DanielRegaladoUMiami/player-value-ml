"""Reframe the problem as classification — what scouts/agents actually want.

Regression on log-ratio gave R² ~0.21 because absolute magnitude of value
change is mostly noise. But the DIRECTION and broad MAGNITUDE are
predictable: this script trains two LightGBM classifiers:

  T1. Binary: will the value go UP in 6 months?  (y_log_ratio > 0)
  T2. 3-class: DOWN (< -5%) / FLAT (±5%) / UP (> +5%)

Reports per-class metrics + calibration plots + the operating point that
maximises an F1-style "scout signal".

Same temporal splits as regression, same features.
"""
from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, brier_score_loss, classification_report,
    confusion_matrix, f1_score, log_loss, roc_auc_score,
)

DATA = Path(__file__).resolve().parent.parent / "data" / "features"
OUT = Path(__file__).resolve().parent.parent / "results" / "models" / "classifier"
PLOTS = Path(__file__).resolve().parent.parent / "results" / "plots"
OUT.mkdir(parents=True, exist_ok=True)
PLOTS.mkdir(parents=True, exist_ok=True)


EXCLUDE = {
    "player_id", "date", "next_date", "next_value", "date_of_birth",
    "y_log_ratio", "y_horizon_days", "market_value_in_eur",
}
CATEGORICAL = ["position", "sub_position", "foot", "career_stage",
               "country_of_citizenship"]


def split_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    feat_cols = [c for c in df.columns if c not in EXCLUDE]
    X = df[feat_cols].copy()
    for c in CATEGORICAL:
        if c in X.columns:
            X[c] = X[c].astype("category")
    return X, df["y_log_ratio"]


def make_3class(y: pd.Series, lo: float = -0.05, hi: float = 0.05) -> pd.Series:
    """0 = DOWN, 1 = FLAT, 2 = UP. Thresholds in log-ratio space."""
    out = pd.Series(1, index=y.index)
    out[y < lo] = 0
    out[y > hi] = 2
    return out


# ============== Binary classifier ==============


def train_binary(X_tr, y_tr, X_va, y_va, X_te, y_te) -> dict:
    print("\n[T1] Binary — y_log_ratio > 0 (will the value go UP)?")
    y_tr_b = (y_tr > 0).astype(int)
    y_va_b = (y_va > 0).astype(int)
    y_te_b = (y_te > 0).astype(int)
    print(f"  Class balance (train): {y_tr_b.mean():.3f} positive")

    cat = [c for c in CATEGORICAL if c in X_tr.columns]
    dtrain = lgb.Dataset(X_tr, y_tr_b, categorical_feature=cat, free_raw_data=False)
    dval   = lgb.Dataset(X_va, y_va_b, categorical_feature=cat, reference=dtrain)

    params = {
        "objective": "binary",
        "metric": "auc",
        "learning_rate": 0.05,
        "num_leaves": 127,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "min_data_in_leaf": 50,
        "lambda_l1": 0.1,
        "lambda_l2": 0.1,
        "verbose": -1,
        "seed": 42,
    }
    model = lgb.train(params, dtrain, num_boost_round=2000,
                      valid_sets=[dtrain, dval], valid_names=["train", "val"],
                      callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)])

    p_te = model.predict(X_te, num_iteration=model.best_iteration)
    auc = roc_auc_score(y_te_b, p_te)
    brier = brier_score_loss(y_te_b, p_te)
    # Pick the threshold that maximises balanced accuracy on val (avoids
    # the trivial threshold=0.5 trap if classes are imbalanced)
    p_va = model.predict(X_va, num_iteration=model.best_iteration)
    best_thr, best_bal = 0.5, 0
    for thr in np.linspace(0.3, 0.7, 41):
        bal = balanced_accuracy_score(y_va_b, (p_va > thr).astype(int))
        if bal > best_bal:
            best_bal, best_thr = bal, thr
    y_pred = (p_te > best_thr).astype(int)
    acc = accuracy_score(y_te_b, y_pred)
    f1 = f1_score(y_te_b, y_pred)
    bal_acc = balanced_accuracy_score(y_te_b, y_pred)

    print(f"\n  TEST  AUC={auc:.4f}  Brier={brier:.4f}  acc={acc:.3f}  "
          f"f1={f1:.3f}  bal_acc={bal_acc:.3f}  thr={best_thr:.2f}")
    print("\n  Confusion (rows=true, cols=pred):")
    print(confusion_matrix(y_te_b, y_pred))

    # Calibration plot
    frac_pos, mean_pred = calibration_curve(y_te_b, p_te, n_bins=15)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="perfectly calibrated")
    ax.plot(mean_pred, frac_pos, "o-", color="#2E86AB",
            label=f"LightGBM (AUC {auc:.3f})")
    ax.set_xlabel("mean predicted P(value goes up)")
    ax.set_ylabel("fraction actually went up")
    ax.set_title("Binary calibration on held-out test (≥2024)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(PLOTS / "05_calibration_binary.png", dpi=120)
    plt.close()

    model.save_model(str(OUT / "binary.lgb"), num_iteration=model.best_iteration)
    return {
        "auc": float(auc), "brier": float(brier),
        "accuracy": float(acc), "f1": float(f1),
        "balanced_accuracy": float(bal_acc),
        "best_threshold": float(best_thr),
        "positive_rate_train": float(y_tr_b.mean()),
        "positive_rate_test":  float(y_te_b.mean()),
    }


# ============== 3-class classifier ==============


def train_three_class(X_tr, y_tr, X_va, y_va, X_te, y_te) -> dict:
    print("\n[T2] 3-class — DOWN (<-5%) / FLAT (±5%) / UP (>+5%)")
    y_tr_c = make_3class(y_tr)
    y_va_c = make_3class(y_va)
    y_te_c = make_3class(y_te)
    print(f"  Class balance (train): "
          f"down={float((y_tr_c==0).mean()):.3f}  "
          f"flat={float((y_tr_c==1).mean()):.3f}  "
          f"up={float((y_tr_c==2).mean()):.3f}")

    cat = [c for c in CATEGORICAL if c in X_tr.columns]
    dtrain = lgb.Dataset(X_tr, y_tr_c, categorical_feature=cat, free_raw_data=False)
    dval   = lgb.Dataset(X_va, y_va_c, categorical_feature=cat, reference=dtrain)

    params = {
        "objective": "multiclass",
        "num_class": 3,
        "metric": "multi_logloss",
        "learning_rate": 0.05,
        "num_leaves": 127,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "min_data_in_leaf": 50,
        "verbose": -1,
        "seed": 42,
    }
    model = lgb.train(params, dtrain, num_boost_round=2000,
                      valid_sets=[dtrain, dval], valid_names=["train", "val"],
                      callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)])

    P_te = model.predict(X_te, num_iteration=model.best_iteration)  # (N, 3)
    y_pred = P_te.argmax(axis=1)

    acc = accuracy_score(y_te_c, y_pred)
    bal_acc = balanced_accuracy_score(y_te_c, y_pred)
    ll = log_loss(y_te_c, P_te, labels=[0, 1, 2])
    f1_macro = f1_score(y_te_c, y_pred, average="macro")
    print(f"\n  TEST  accuracy={acc:.3f}  balanced_acc={bal_acc:.3f}  "
          f"f1_macro={f1_macro:.3f}  log_loss={ll:.4f}")
    print("\n  Classification report:")
    print(classification_report(y_te_c, y_pred,
                                 target_names=["DOWN", "FLAT", "UP"], digits=3))
    print("  Confusion (rows=true, cols=pred):")
    cm = confusion_matrix(y_te_c, y_pred)
    print(cm)

    model.save_model(str(OUT / "three_class.lgb"), num_iteration=model.best_iteration)

    # Per-class AUC (one-vs-rest)
    aucs = {}
    for k, name in enumerate(["DOWN", "FLAT", "UP"]):
        y_bin = (y_te_c == k).astype(int)
        aucs[name] = float(roc_auc_score(y_bin, P_te[:, k]))
        print(f"  AUC vs rest, class {name}: {aucs[name]:.4f}")

    return {
        "accuracy": float(acc),
        "balanced_accuracy": float(bal_acc),
        "f1_macro": float(f1_macro),
        "log_loss": float(ll),
        "auc_one_vs_rest": aucs,
        "class_balance_train": {
            "down": float((y_tr_c == 0).mean()),
            "flat": float((y_tr_c == 1).mean()),
            "up":   float((y_tr_c == 2).mean()),
        },
        "confusion_matrix": cm.tolist(),
    }


def main() -> None:
    print("Loading splits...")
    train = pd.read_parquet(DATA / "train.parquet")
    val   = pd.read_parquet(DATA / "val.parquet")
    test  = pd.read_parquet(DATA / "test.parquet")
    X_tr, y_tr = split_xy(train)
    X_va, y_va = split_xy(val)
    X_te, y_te = split_xy(test)
    print(f"  train={len(X_tr):,}  val={len(X_va):,}  test={len(X_te):,}")

    results = {
        "binary":      train_binary(X_tr, y_tr, X_va, y_va, X_te, y_te),
        "three_class": train_three_class(X_tr, y_tr, X_va, y_va, X_te, y_te),
    }
    (OUT / "metrics.json").write_text(json.dumps(results, indent=2))
    print(f"\nSaved to {OUT}")
    print("\n" + "=" * 60)
    print("  RESULTS — TEST SET")
    print("=" * 60)
    print(f"  Binary  P(value up):  AUC = {results['binary']['auc']:.4f}")
    print(f"                        acc = {results['binary']['accuracy']:.3f}")
    print(f"                        f1  = {results['binary']['f1']:.3f}")
    print(f"  3-class               acc = {results['three_class']['accuracy']:.3f}")
    print(f"                       bal  = {results['three_class']['balanced_accuracy']:.3f}")
    print(f"                       AUC  = {results['three_class']['auc_one_vs_rest']}")


if __name__ == "__main__":
    main()
