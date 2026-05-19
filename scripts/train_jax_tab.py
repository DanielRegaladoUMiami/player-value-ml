"""Train the hand-rolled JAX TabTransformer on player_value features.

This is the 'craft' piece — same problem, same splits, same metric, but
written from first principles in pure JAX. We expect LightGBM to win on
absolute metrics. The point is the comparison: how close does a small
from-scratch Transformer get to a tuned gradient-boosted baseline on
tabular data with ~500k rows and ~70 features?
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score

from playerval.jax_tabtransformer import (
    TTConfig, init_params, forward, count_params,
)

DATA = Path(__file__).resolve().parent.parent / "data" / "features"
OUT = Path(__file__).resolve().parent.parent / "results" / "models" / "jax_tabtransformer"
OUT.mkdir(parents=True, exist_ok=True)


# Mirror the LightGBM script's column treatment so the comparison is fair.
EXCLUDE = {
    "player_id", "date", "next_date", "next_value", "date_of_birth",
    "y_log_ratio", "y_horizon_days", "market_value_in_eur",
}
CATEGORICAL = ["position", "sub_position", "foot", "career_stage",
               "country_of_citizenship"]


def build_feature_columns(train: pd.DataFrame) -> tuple[list[str], list[str], dict]:
    num_cols, cat_cols = [], []
    cat_vocab = {}
    for c in train.columns:
        if c in EXCLUDE:
            continue
        if c in CATEGORICAL:
            uniq = list(train[c].dropna().unique()) + ["__missing__"]
            cat_vocab[c] = {v: i for i, v in enumerate(uniq)}
            cat_cols.append(c)
        elif pd.api.types.is_numeric_dtype(train[c]):
            num_cols.append(c)
        else:
            # treat unknown-dtype columns as ignore
            pass
    return num_cols, cat_cols, cat_vocab


def encode(df: pd.DataFrame, num_cols, cat_cols, cat_vocab):
    X_num = df[num_cols].astype(np.float32).fillna(0.0).values
    # Robust normalization: per-column z-score from this split's stats
    # (we'll compute stats on train and apply to all splits)
    X_cat = np.zeros((len(df), len(cat_cols)), dtype=np.int32)
    for j, c in enumerate(cat_cols):
        idx = df[c].map(cat_vocab[c]).fillna(cat_vocab[c]["__missing__"]).astype(np.int32).values
        X_cat[:, j] = idx
    return X_num, X_cat


def main() -> None:
    print("Loading splits...")
    train = pd.read_parquet(DATA / "train.parquet")
    val   = pd.read_parquet(DATA / "val.parquet")
    test  = pd.read_parquet(DATA / "test.parquet")

    num_cols, cat_cols, cat_vocab = build_feature_columns(train)
    print(f"  numerical features: {len(num_cols)}")
    print(f"  categorical features: {len(cat_cols)}  cardinalities: "
          f"{[len(v) for v in cat_vocab.values()]}")

    Xn_train, Xc_train = encode(train, num_cols, cat_cols, cat_vocab)
    Xn_val,   Xc_val   = encode(val,   num_cols, cat_cols, cat_vocab)
    Xn_test,  Xc_test  = encode(test,  num_cols, cat_cols, cat_vocab)

    # Standardize numerical features using train stats only
    mu = Xn_train.mean(axis=0)
    sd = Xn_train.std(axis=0)
    sd[sd < 1e-6] = 1.0
    Xn_train = (Xn_train - mu) / sd
    Xn_val   = (Xn_val - mu) / sd
    Xn_test  = (Xn_test - mu) / sd
    # Clip outliers
    Xn_train = np.clip(Xn_train, -5, 5)
    Xn_val   = np.clip(Xn_val,   -5, 5)
    Xn_test  = np.clip(Xn_test,  -5, 5)

    y_train = train["y_log_ratio"].astype(np.float32).values
    y_val   = val["y_log_ratio"].astype(np.float32).values
    y_test  = test["y_log_ratio"].astype(np.float32).values

    print(f"  train shape: X_num={Xn_train.shape} X_cat={Xc_train.shape}  y={y_train.shape}")

    cfg = TTConfig(
        n_num=len(num_cols),
        n_cat=len(cat_cols),
        cat_cardinalities=tuple(len(cat_vocab[c]) for c in cat_cols),
        d_model=64,
        n_heads=4,
        n_layers=2,
        d_ff=128,
    )
    key = jax.random.PRNGKey(42)
    init_key, _ = jax.random.split(key)
    params = init_params(cfg, init_key)
    print(f"  TabTransformer params: {count_params(params):,}")

    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(learning_rate=3e-4, b1=0.9, b2=0.95, weight_decay=0.01),
    )
    opt_state = optimizer.init(params)

    def loss_fn(params, xn, xc, y):
        pred = forward(params, xn, xc, cfg)
        return jnp.mean((pred - y) ** 2)

    @jax.jit
    def step(params, opt_state, xn, xc, y):
        loss, grads = jax.value_and_grad(loss_fn)(params, xn, xc, y)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    # Train
    n_train = len(y_train)
    batch_size = 512
    n_epochs = 3
    rng = np.random.default_rng(42)
    print(f"\nTraining ({n_epochs} epochs, batch={batch_size})...")
    t0 = time.time()
    best_val_mae = float("inf")
    for ep in range(1, n_epochs + 1):
        order = rng.permutation(n_train)
        running = 0.0
        n_steps = 0
        for s in range(0, n_train, batch_size):
            idx = order[s:s + batch_size]
            xn = jnp.asarray(Xn_train[idx])
            xc = jnp.asarray(Xc_train[idx])
            y  = jnp.asarray(y_train[idx])
            params, opt_state, loss = step(params, opt_state, xn, xc, y)
            running += float(loss)
            n_steps += 1
        # Val score
        val_pred = forward(params, jnp.asarray(Xn_val), jnp.asarray(Xc_val), cfg)
        val_mae = float(jnp.mean(jnp.abs(val_pred - jnp.asarray(y_val))))
        val_r2 = r2_score(y_val, np.asarray(val_pred))
        print(f"  epoch {ep}  train_loss={running/n_steps:.4f}  "
              f"val_mae={val_mae:.4f}  val_r2={val_r2:+.4f}")
        if val_mae < best_val_mae:
            best_val_mae = val_mae
    print(f"  total: {time.time() - t0:.1f}s")

    # Test
    test_pred = np.asarray(forward(params, jnp.asarray(Xn_test), jnp.asarray(Xc_test), cfg))
    test_mae = mean_absolute_error(y_test, test_pred)
    test_r2 = r2_score(y_test, test_pred)
    print(f"\nTest: MAE={test_mae:.4f}  R²={test_r2:+.4f}")

    results = {
        "TabTransformer": {
            "test_mae_log": float(test_mae),
            "test_r2": float(test_r2),
            "val_mae_log": float(best_val_mae),
            "params": int(count_params(params)),
            "config": {
                "d_model": cfg.d_model, "n_layers": cfg.n_layers,
                "n_heads": cfg.n_heads, "d_ff": cfg.d_ff,
            },
        },
    }
    (OUT / "metrics.json").write_text(json.dumps(results, indent=2))
    pd.DataFrame({"y_true": y_test, "y_pred": test_pred}).to_parquet(
        OUT / "test_predictions.parquet", index=False
    )
    print(f"\nSaved to {OUT}")


if __name__ == "__main__":
    main()
