# player-value-ml

> **Forecasting football player market values: an honest comparison of 7 model
> families on the same temporal split.**
>
> The interesting question isn't "which model wins" — it's "when does each
> family win, and what does that tell us about the data?"

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

## What this project is

Given a football player and a date `T`, predict their Transfermarkt market
value at `T + 6 months`. Same data, same features, same temporal split — but
seven different model families:

| # | family               | concrete implementation                          |
|---|----------------------|--------------------------------------------------|
| 0 | Naive baselines      | last-value, seasonal-naive                        |
| 1 | Linear regression    | Ridge / Lasso with engineered features            |
| 2 | Statistical          | `StatsForecast` (AutoARIMA, AutoETS, Theta)       |
| 3 | ML tabular           | `MLForecast` + LightGBM with lag features         |
| 4 | Neural forecasting   | `NeuralForecast` (NHITS, TFT, PatchTST)           |
| 5 | From-scratch JAX     | Tabular Transformer + LSTM hand-rolled            |
| 6 | Ensemble             | Stacking of the best from each family             |

This is **not** a "neural networks win" story. The point is to show what
honest model comparison looks like — which family wins where, and why.

## Data

- **Source**: `davidcariboo/player-scores` on Kaggle (Transfermarkt scrape, CC0)
- **Scale**: ~507k historical valuations, 31k unique players, 2003–2025
- **Auxiliary**: 1.88M appearances, 40k transfers, 88k matches, 47k player profiles

## Pipeline

```
Kaggle (CC0)
   │
   ▼
[1] EDA & data audit          scripts/eda.py            (results/eda/)
   │
   ▼
[2] Feature engineering       scripts/features.py       (~60 features)
   │
   ▼
[3] Temporal splits           scripts/splits.py         (train<2023, val=2023, test≥2024)
   │
   ▼
[4-9] Train each model family scripts/train_*.py        (one script per family)
   │
   ▼
[10] Honest comparison        scripts/evaluate.py       (MAE, MAPE, R², calibration)
   │
   ▼
[11] Plots + tables           results/plots/, results/metrics.json
   │
   ▼
[12] HF Space demo            space/app.py              (typeas un jugador → prediccion)
```

## Status

Work in progress. See `docs/decisions/` for architectural choices and
`results/eda/` for early data exploration.

## License

Apache 2.0. See [LICENSE](LICENSE).

## Author

[Daniel Regalado](https://github.com/DanielRegaladoUMiami) — MSBA,
University of Miami.
