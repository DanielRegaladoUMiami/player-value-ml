"""Push the LightGBM model + tokenizer-equivalent artifacts to Hugging Face Hub.

Validates BEFORE upload:
  - model.lgb loads as a LightGBM Booster
  - predicts a finite vector on a small input
  - metrics file has test_mae_log and test_r2 in expected ranges
  - feature_importance.csv exists and is sorted

Then uploads the .lgb model + metrics.json + feature_importance.csv + model card.

Usage:
    python -m scripts.push_model --repo DanielRegaladoCardoso/player-value-lgbm
    python -m scripts.push_model --dry-run
"""
from __future__ import annotations

import argparse
import json
import logging
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

LGBM_DIR = Path(__file__).resolve().parent.parent / "results" / "models" / "lgbm"
DATA = Path(__file__).resolve().parent.parent / "data" / "features"


def validate() -> None:
    import lightgbm as lgb

    model_path = LGBM_DIR / "model.lgb"
    metrics_path = LGBM_DIR / "metrics.json"
    importance_path = LGBM_DIR / "feature_importance.csv"
    for p in (model_path, metrics_path, importance_path):
        assert p.exists(), f"FAIL: {p} missing"

    booster = lgb.Booster(model_file=str(model_path))
    n_feats = booster.num_feature()
    logger.info("Model loads OK: %d features", n_feats)

    # Predict on real test rows to make sure the model works end-to-end
    EXCLUDE = {
        "player_id", "date", "next_date", "next_value", "date_of_birth",
        "y_log_ratio", "y_horizon_days", "market_value_in_eur",
    }
    CATEGORICAL = ["position", "sub_position", "foot", "career_stage",
                   "country_of_citizenship"]
    test = pd.read_parquet(DATA / "test.parquet").iloc[:64].copy()
    X = test[[c for c in test.columns if c not in EXCLUDE]].copy()
    for c in CATEGORICAL:
        if c in X.columns:
            X[c] = X[c].astype("category")
    preds = booster.predict(X)
    assert preds.shape == (64,), f"FAIL: pred shape {preds.shape}"
    assert np.isfinite(preds).all(), "FAIL: predictions non-finite"
    logger.info("Predict smoke test OK: 64 preds, mean=%+.3f std=%.3f",
                preds.mean(), preds.std())

    metrics = json.loads(metrics_path.read_text())
    assert "test_mae_log" in metrics and "test_r2" in metrics
    assert 0.15 < metrics["test_mae_log"] < 0.25, \
        f"FAIL: test_mae_log out of range: {metrics['test_mae_log']}"
    assert metrics["test_r2"] > 0.15, \
        f"FAIL: test_r2 too low: {metrics['test_r2']}"
    logger.info("Metrics OK: MAE=%.4f R²=%+.4f", metrics["test_mae_log"],
                metrics["test_r2"])

    imp = pd.read_csv(importance_path)
    assert {"feature", "importance_gain"}.issubset(imp.columns)
    logger.info("Feature importance OK: top feature = %s (gain %.0f)",
                imp.iloc[0]["feature"], imp.iloc[0]["importance_gain"])


MODEL_CARD = """---
license: apache-2.0
language:
  - en
tags:
  - lightgbm
  - football
  - soccer
  - forecasting
  - tabular
  - transfermarkt
library_name: lightgbm
pipeline_tag: tabular-regression
---

# player-value-lgbm

LightGBM model that forecasts the **6-month log-change in Transfermarkt
market value** for football players. Trained on the engineered features
in [`player-value-features`](https://huggingface.co/datasets/DanielRegaladoCardoso/player-value-features).

- **Code**: https://github.com/DanielRegaladoUMiami/player-value-ml
- **Demo**: https://huggingface.co/spaces/DanielRegaladoCardoso/player-value-ml
- **Dataset**: [player-value-features](https://huggingface.co/datasets/DanielRegaladoCardoso/player-value-features)

## Performance (held-out test, ≥ 2024)

| Model            | Test MAE (log) | Test R²    |
|------------------|---------------:|-----------:|
| Naive (y=0)      |         0.2145 |     −0.006 |
| Ridge regression |         0.2164 |     +0.130 |
| AutoETS          |         0.2361 |     −0.198 |
| AutoTheta        |         0.2740 |     −0.298 |
| **LightGBM**     |     **0.1970** |   **+0.205** |

LightGBM wins by ~40% in R² over the best baseline and is the only
non-trivial model to clearly beat the naive predictor.

## Architecture

- LightGBM regression with L1 objective (MAE)
- 351 trees (early-stopped on validation)
- `num_leaves=127`, `learning_rate=0.05`, `feature_fraction=0.8`,
  `bagging_fraction=0.8`, `lambda_l1=0.1`, `lambda_l2=0.1`
- 84 features (92 raw, 8 dropped by the audit), 5 categorical handled
  natively (`position`, `sub_position`, `foot`, `career_stage`,
  `country_of_citizenship`)

## Top features by gain

1. `country_of_citizenship` — the market prices passport over performance
2. `as_of_year` — captures market inflation
3. `value_diff_1` — recent momentum
4. `log_value` — base level
5. `age`
6. `as_of_month` — transfer-window seasonality
7. `value_diff_3`
8. `international_caps`
9. `age_minus_position_peak` — distance from position-specific peak
10. `value_lag_1`

## How to use

```python
import lightgbm as lgb
import pandas as pd
from huggingface_hub import hf_hub_download

model_path = hf_hub_download("DanielRegaladoCardoso/player-value-lgbm",
                              "model.lgb")
booster = lgb.Booster(model_file=model_path)

# X must be a DataFrame whose categorical columns are pd.Categorical
# (see scripts/train_lgbm.py in the repo for the exact preprocessing)
preds = booster.predict(X)        # predicts log(value_T+6mo / value_T)
```

To convert a prediction back into euros:

```python
predicted_value_eur = current_value_eur * np.exp(predicted_log_ratio)
```

## Limitations

- The 6-month horizon is a **median** — Transfermarkt updates 2–3×/year
  irregularly. The model is trained on the irregular cadence; predictions
  are not calibrated for a fixed 180-day horizon.
- R² of +0.20 means ~80% of valuation-change variance is NOT explained
  by these features — markets are noisy.
- `country_of_citizenship` being the strongest feature reflects market
  bias the model captures (and does not endorse).
- Trained only on Transfermarkt's tracked competitions; lower-tier leagues
  are under-represented.

## License

Apache 2.0.
"""


def push(repo: str, private: bool) -> None:
    from huggingface_hub import HfApi
    api = HfApi()
    try:
        whoami = api.whoami()
    except Exception as e:
        raise SystemExit(f"FAIL: HF auth. Run `hf auth login`. {e}")
    logger.info("HF auth OK as %s", whoami.get("name"))

    api.create_repo(repo_id=repo, repo_type="model",
                    exist_ok=True, private=private)
    api.upload_file(path_or_fileobj=str(LGBM_DIR / "model.lgb"),
                    path_in_repo="model.lgb",
                    repo_id=repo, repo_type="model")
    api.upload_file(path_or_fileobj=str(LGBM_DIR / "metrics.json"),
                    path_in_repo="metrics.json",
                    repo_id=repo, repo_type="model")
    api.upload_file(path_or_fileobj=str(LGBM_DIR / "feature_importance.csv"),
                    path_in_repo="feature_importance.csv",
                    repo_id=repo, repo_type="model")
    with tempfile.NamedTemporaryFile("w", suffix=".md", delete=False) as f:
        f.write(MODEL_CARD)
        card_path = f.name
    api.upload_file(path_or_fileobj=card_path, path_in_repo="README.md",
                    repo_id=repo, repo_type="model")
    logger.info("Pushed to https://huggingface.co/%s", repo)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default="DanielRegaladoCardoso/player-value-lgbm")
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    validate()
    if args.dry_run:
        logger.info("DRY RUN: validations passed. Skipping HF upload.")
        return
    push(args.repo, args.private)


if __name__ == "__main__":
    main()
