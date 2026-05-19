"""Push the engineered feature parquets to Hugging Face Hub.

Validates BEFORE upload:
  - parquets exist and are non-empty
  - required columns present (target, key identifiers)
  - no future-leak in splits (train < 2023, val = 2023, test ≥ 2024)
  - target distribution sane (mean/std/no inf)
  - HF Hub auth functional

Then uploads train/val/test.parquet + a comprehensive data card.

Usage:
    python -m scripts.push_dataset --repo DanielRegaladoCardoso/player-value-features
    python -m scripts.push_dataset --dry-run  # validate without uploading
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

DATA = Path(__file__).resolve().parent.parent / "data" / "features"

REQUIRED_COLS = {
    "player_id", "date", "market_value_in_eur", "y_log_ratio",
    "age", "position", "value_lag_1", "value_diff_1",
}


def validate_split(name: str, df: pd.DataFrame, date_lo: str | None,
                   date_hi: str | None) -> None:
    assert len(df) > 1000, f"FAIL: {name} too small ({len(df)} rows)"
    missing = REQUIRED_COLS - set(df.columns)
    assert not missing, f"FAIL: {name} missing required cols {missing}"

    if date_lo:
        assert (df["date"] >= pd.Timestamp(date_lo)).all(), \
            f"FAIL: {name} contains dates < {date_lo}"
    if date_hi:
        assert (df["date"] <= pd.Timestamp(date_hi)).all(), \
            f"FAIL: {name} contains dates > {date_hi}"

    y = df["y_log_ratio"]
    assert y.notna().all(), f"FAIL: {name} target has nulls"
    assert np.isfinite(y).all(), f"FAIL: {name} target has inf"
    assert -2 < y.mean() < 2, f"FAIL: {name} target mean off ({y.mean():.3f})"

    logger.info("  %s OK: %d rows, %d cols, y mean=%+.3f std=%.3f",
                name, len(df), len(df.columns), y.mean(), y.std())


DATA_CARD = """---
license: apache-2.0
language:
  - en
size_categories:
  - 100K<n<1M
task_categories:
  - tabular-regression
tags:
  - football
  - soccer
  - forecasting
  - tabular
  - transfermarkt
  - time-series
---

# player-value-features

Engineered features for forecasting Transfermarkt player market values
6 months ahead. Companion dataset to
[player-value-ml](https://github.com/DanielRegaladoUMiami/player-value-ml).

## Provenance

- **Source**: `davidcariboo/player-scores` on Kaggle (CC0). 507k historical
  Transfermarkt valuations of ~31k players, 2000–2026.
- **Auxiliary**: 1.88M player-game appearances, 88k matches, 40k transfers,
  1.26M goal/card/sub events, 47k player profiles.

## Schema

| split | rows    | date range          |
|-------|---------|--------------------|
| train | 422,838 | 2000 – 2022-12-31  |
| val   |  28,491 | 2023               |
| test  |  23,876 | ≥ 2024             |

Each row is a `(player_id, as_of_date)` observation with **92 engineered
features** in 10 thematic groups (A–J). The target is

```
y_log_ratio = log(value_{T+next} / value_T)
```

where `T+next` is the player's next observed valuation (~6 months later
on average).

## Feature groups (high level)

| Group | Theme                          | Examples                                        |
|-------|--------------------------------|-------------------------------------------------|
| A     | Lag / trajectory               | `value_lag_1..5`, `value_diff_1..3`             |
| B     | Performance (absolute)         | `goals_6mo`, `minutes_played_12mo`              |
| C     | Performance vs team            | `goal_share_team_6mo`                           |
| D     | Performance vs position        | `goals_6mo_zscore_vs_pos`                       |
| E     | Competition-tier weighted      | `weighted_goals_6mo` (PL > MLS)                 |
| F     | Player static                  | `age`, `position`, `foot`, `height_in_cm`       |
| G     | Career / transfers             | `transfers_to_date`, `last_transfer_fee_eur`    |
| H     | Temporal                       | `as_of_year`, `is_summer_window`                |
| I     | Age × position interaction     | `age_minus_position_peak`                       |
| J     | xG / shot-quality proxies      | `chance_quality_index_6mo`                      |

Every feature is computed using only information available **strictly
before** `as_of_date` — no future leakage.

## Reproducibility

```python
from datasets import load_dataset
ds = load_dataset("DanielRegaladoCardoso/player-value-features")
print(ds)
```

## Splits methodology

Splits are **temporal**, not random. A given player can appear in multiple
splits (their career crosses the time boundary). What is guaranteed: no
row in a later split is used to inform features of an earlier split.

## License

Apache 2.0. Source Transfermarkt data via Kaggle is CC0.

## Citation

```bibtex
@misc{regalado2026playervalue,
  author = {Daniel Regalado},
  title  = {player-value-features: engineered tabular features for
            forecasting football player market values},
  year   = {2026},
  url    = {https://huggingface.co/datasets/DanielRegaladoCardoso/player-value-features}
}
```
"""


def push(out_dir: Path, repo: str, private: bool) -> None:
    from huggingface_hub import HfApi
    api = HfApi()
    try:
        whoami = api.whoami()
    except Exception as e:
        raise SystemExit(f"FAIL: HF auth not configured. Run `hf auth login`. {e}")
    logger.info("HF auth OK as %s", whoami.get("name"))

    api.create_repo(repo_id=repo, repo_type="dataset",
                    exist_ok=True, private=private)
    for name in ("train", "val", "test"):
        api.upload_file(
            path_or_fileobj=str(DATA / f"{name}.parquet"),
            path_in_repo=f"data/{name}.parquet",
            repo_id=repo, repo_type="dataset",
        )
    with tempfile.NamedTemporaryFile("w", suffix=".md", delete=False) as f:
        f.write(DATA_CARD)
        card_path = f.name
    api.upload_file(path_or_fileobj=card_path, path_in_repo="README.md",
                    repo_id=repo, repo_type="dataset")
    logger.info("Pushed to https://huggingface.co/datasets/%s", repo)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default="DanielRegaladoCardoso/player-value-features")
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    splits = {
        "train": pd.read_parquet(DATA / "train.parquet"),
        "val":   pd.read_parquet(DATA / "val.parquet"),
        "test":  pd.read_parquet(DATA / "test.parquet"),
    }
    validate_split("train", splits["train"], date_lo=None, date_hi="2022-12-31")
    validate_split("val",   splits["val"],   date_lo="2023-01-01", date_hi="2023-12-31")
    validate_split("test",  splits["test"],  date_lo="2024-01-01", date_hi=None)

    if args.dry_run:
        logger.info("DRY RUN: validations passed. Skipping HF upload.")
        return

    push(DATA, args.repo, args.private)


if __name__ == "__main__":
    main()
