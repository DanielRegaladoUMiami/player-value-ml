# player-value-ml

> **Forecasting football player market values: an honest comparison of seven
> model families on the same temporal split.**
>
> The interesting question isn't *"which model wins?"* — it's *"when does
> each family win, and what does that tell us about the data?"*

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Dataset: Kaggle](https://img.shields.io/badge/dataset-Kaggle%20%C2%B7%20davidcariboo%2Fplayer--scores-20BEFF.svg)](https://www.kaggle.com/datasets/davidcariboo/player-scores)
[![Tests](https://img.shields.io/badge/tests-19%2F19%20passing-brightgreen.svg)](#tests)
[![CI](https://img.shields.io/badge/CI-GitHub%20Actions-lightgrey.svg)](#)
[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)

---

## What this project is — and isn't

**It is**: a methodologically honest comparison of tabular forecasting
approaches on a real, messy dataset (~507k Transfermarkt valuations,
~31k players, 2000–2026). Same data, same temporal split, same metrics —
seven different model families. Every result in this README is reproducible
from the scripts in `scripts/` and the seeds in `src/playerval/`.

**It isn't**: a "neural networks beat everything" story. The data has
short, irregular series per entity (median **15** valuations, ~189 days
apart) and rich engineered features. That's the regime where gradient
boosting tends to win, and that is exactly what we observe. The
contribution is the comparison itself — not a fancy model trained in
isolation.

See [`docs/decisions/001-honest-multimodel-comparison.md`](docs/decisions/001-honest-multimodel-comparison.md)
for the full architectural rationale.

---

## Live results (test set, ≥ 2024)

Target: `y = log(value_{T+next} / value_T)`. Metric on log-ratio: MAE
(lower is better), R² (higher is better, > 0 means beats the mean).

| # | Family            | Model                  | Test MAE (log) | Test R²      | Notes                              |
|---|-------------------|------------------------|---------------:|-------------:|------------------------------------|
| 0 | Naive             | B0 last-value (zero)   |          0.2145 |       −0.006 | Hard-to-beat sanity check          |
| 0 | Naive             | StatsForecast Naive    |          0.2080 |       −0.002 | Even simpler — strong baseline     |
| 1 | Linear            | B1 Position×Age mean   |          0.2240 |       +0.146 | Cohort baseline                    |
| 1 | Linear            | B2 Ridge               |          0.2164 |       +0.130 | Engineered features, regularised   |
| 2 | Statistical       | AutoETS                |          0.2361 |   **−0.198** | **Overfits short irregular series**|
| 2 | Statistical       | AutoTheta              |          0.2740 |   **−0.298** | Same failure mode, worse           |
| 3 | ML tabular        | **LightGBM** (regression) |   **0.1970** |       +0.205 | **Best MAE** in regression           |
| 4 | Neural forecast   | NHITS (Nixtla)         |              — |            — | _attempted; deferred for batched-inference v2_ |
| 5 | From-scratch JAX  | **TabTransformer**     |          0.2053 |   **+0.2135** | **Best R²** in regression — 95K params, hand-rolled |
| 6 | Ensemble          | Stacking               |              − |            − | _future work_                      |

### Reframed as classification — the part that actually works

The R² ceiling in regression (~0.21) reflects the data: **magnitude** of
value change is mostly noise. But **direction** is predictable. The
LightGBM classifier on the same features gets:

| Task                                  | Metric         | Result      |
|---------------------------------------|----------------|------------:|
| Binary — *will value go UP?*          | **Test AUC**   | **0.773**   |
|                                       | Accuracy       | 0.682       |
|                                       | F1             | 0.579       |
|                                       | Brier          | 0.168       |
| 3-class — DOWN (<−5%) / FLAT / UP (>+5%) | Accuracy   | 0.554 (33% random) |
|                                       | AUC DOWN-vs-rest | **0.808** |
|                                       | AUC UP-vs-rest   | **0.772** |
|                                       | AUC FLAT-vs-rest | 0.638     |

This is the **product-relevant framing**: a scout/agent cares whether a
player's value will go up or down (and roughly by how much), not the
exact log-ratio. Detecting **DOWN movers is the easiest signal**
(AUC 0.81): aging and under-performing players are visible from age,
minutes share, and recent value trajectory.

Raw JSON: [`results/models/baselines/metrics.json`](results/models/baselines/metrics.json),
[`results/models/stats/metrics.json`](results/models/stats/metrics.json),
[`results/models/lgbm/metrics.json`](results/models/lgbm/metrics.json).

### Key findings

- **LightGBM wins on MAE; JAX TabTransformer wins on R²** — LightGBM
  achieves the lowest MAE (0.197), while a 95K-param hand-rolled
  TabTransformer in pure JAX captures the most variance (R² +0.2135 vs
  LGBM's +0.205). Classical bias-variance trade-off: LGBM is more
  conservative (predictions cluster near the mean → lower MAE), while the
  Transformer makes bigger predictions that explain more but occasionally
  overshoot.
- **Classical statistical methods FAIL here** — `AutoETS` (R² **−0.198**)
  and `AutoTheta` (R² **−0.298**) are *worse than the naive baseline*.
  This is the M-competition / Makridakis insight in miniature: ETS/Theta
  need long, regular series. With a median of 15 irregularly-spaced points,
  they fit noise and extrapolate confidently in the wrong direction.
- **Passport > goals.** The single most important feature by LightGBM
  gain is `country_of_citizenship`. The market prices nationality (work
  permits, league fit, sponsorship value) above short-term performance.
  We capture that signal; we do not endorse it.
- **Top 5 features (by gain)**: `country_of_citizenship`, `as_of_year`,
  `value_diff_1`, `log_value`, `age`. See
  [`results/models/lgbm/feature_importance.csv`](results/models/lgbm/feature_importance.csv).

![Value vs age](results/eda/plots/06_value_vs_age.png)

![Target distribution](results/eda/plots/07_target_distribution.png)

---

## Architecture

```
┌──────────────────────────────┐
│  Kaggle  davidcariboo/       │   8 raw CSVs · 507k valuations
│         player-scores  (CC0) │   31k players · 1.88M appearances
└──────────────┬───────────────┘
               │  scripts/eda.py
               ▼
┌──────────────────────────────┐
│  [1] EDA + audit             │   results/eda/  ·  results/audit/
└──────────────┬───────────────┘
               │  scripts/build_features.py
               ▼
┌──────────────────────────────┐
│  [2] Feature engineering     │   ~75 features in 10 groups (A–J)
│      src/playerval/features  │   data/features/{train,val,test}.parquet
└──────────────┬───────────────┘
               │  temporal split
               ▼
   train < 2023 │ val = 2023 │ test ≥ 2024
               │
               ▼
┌────────────────────────────────────────────────────────────────┐
│  [3] Train each family on the SAME splits, SAME features       │
│                                                                │
│   B0/B1/B2 baselines   StatsForecast    LightGBM (winner)      │
│       │                    │                  │                │
│       └──────────┬─────────┴──────────┬───────┘                │
│                  ▼                    ▼                        │
│         results/models/*/metrics.json  +  feature_importance   │
└──────────────────────────┬─────────────────────────────────────┘
                           │  scripts/evaluate.py
                           ▼
                  Honest comparison table
```

---

## Feature engineering — 10 groups, no leakage

Every feature is computed using **only information available strictly
before** the `as_of_date`. The pipeline lives in
[`src/playerval/features.py`](src/playerval/features.py).

| Group | Name                       | Example features                                              | Why it matters                                                 |
|-------|----------------------------|---------------------------------------------------------------|----------------------------------------------------------------|
|   A   | Lag / trajectory            | `value_lag_1..5`, `value_diff_1..3`, `months_since_peak`     | The forecasting spine — recent valuations dominate             |
|   B   | Performance (absolute)      | `goals_6mo`, `minutes_played_12mo`                            | Did the player actually do anything on the pitch?              |
|   C   | Performance vs team         | `goal_share_team_6mo`, `minutes_share_team_6mo`               | A star on a bad team ≠ a star on a great team                  |
|   D   | Performance vs position     | `goals_6mo_zscore_vs_pos`, `assists_6mo_zscore_vs_pos`        | 10 goals means very different things for a CB vs a striker     |
|   E   | Competition-tier weighted   | `weighted_goals_6mo`, `weighted_minutes_12mo`                 | A goal in the Premier League ≠ a goal in League Two            |
|   F   | Player static               | `age`, `position`, `country_of_citizenship`, `height_in_cm`   | Demographics; passport drives market access                    |
|   G   | Club / career static        | `transfers_to_date`, `last_transfer_fee_eur`                  | Past prices reveal info markets already priced                 |
|   H   | Temporal / seasonal         | `as_of_year`, `is_summer_window`, `days_since_first_valuation`| Window effects, market inflation, cohort                       |
|   I   | Age × position interaction  | `age_minus_position_peak`, `pre_peak`, `post_peak`            | Peak age depends on position (GK ≠ winger)                     |
|   J   | xG / shot-quality proxies   | `chance_quality_index_6mo`, `skill_goal_share_6mo`            | Approximate xG without event coordinates                       |

Two design choices worth flagging:

- **Position-relative z-scores (Group D).** Goals are standardised within
  `(position, calendar_year)` buckets, computed from a leakage-safe per-season
  baseline. Lets the model learn "good for a midfielder" instead of memorising
  position-specific scales.
- **Competition-tier weighting (Group E).** Each appearance is multiplied
  by a tier weight (`TIER_WEIGHT` in `src/playerval/refdata.py`) before
  rolling up, so 10 Champions League goals don't get washed out by 30
  third-division goals in the same window.

Full audit: [`results/audit/feature_summary.csv`](results/audit/feature_summary.csv).

---

## Data

| Property             | Value                                                            |
|----------------------|------------------------------------------------------------------|
| Source               | [`davidcariboo/player-scores`](https://www.kaggle.com/datasets/davidcariboo/player-scores) (CC0) |
| Valuations           | 507,815                                                          |
| Unique players       | 31,507                                                           |
| Date range           | 2000 – 2026                                                      |
| Median series length | 15 valuations (p25=8, p75=23, p95=34, max=57)                    |
| Median gap           | 168 days                                                         |
| Target               | `y = log(value_{T+next} / value_T)`, std ≈ 0.43                   |
| Eligible (T, T+1) pairs | 476,306                                                       |
| Splits               | train < 2023-01-01 · val = 2023 · test ≥ 2024-01-01              |

Raw € is heavily right-skewed (skew 7.92); the log transform brings it to
0.42 — hence the log-ratio target.

![Series length distribution](results/eda/plots/02_series_length.png)

---

## How to run

```bash
# 1. clone + install
git clone https://github.com/DanielRegaladoUMiami/player-value-ml.git
cd player-value-ml
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# 2. pull the dataset (uses kagglehub — needs ~/.kaggle/kaggle.json)
python scripts/download_data.py

# 3. EDA + audit  →  results/eda/, results/audit/
python scripts/eda.py

# 4. build features  →  data/features/{train,val,test}.parquet
python scripts/build_features.py

# 5. train every family  →  results/models/*/metrics.json
python scripts/train_baselines.py
python scripts/train_stats.py
python scripts/train_lgbm.py

# 6. compare them  →  results/comparison.md
python scripts/evaluate.py
```

End-to-end on a 2024 MacBook Pro: ~12 minutes (most of it in feature
engineering, not training).

---

## Tests

```bash
pytest -q
# 19 passed in ~7s
```

Coverage spans:

- **Leakage guards** — every rolling/asof merge is unit-tested to confirm
  no information from `T` or later leaks into row `T`.
- **Feature math** — z-scores, tier weighting, age-vs-position-peak,
  career streaks (`consec_increases`, `consec_decreases`).
- **Split integrity** — no `player_id` appears in both train and test
  with overlapping dates; temporal boundaries are exact.
- **Target safety** — `y_log_ratio` is never computed from a zero or
  missing valuation.

---

## Limitations (the honest section)

- **No event coordinates.** The Kaggle dump has goal events but not shot
  locations, so "xG" features (Group J) are approximations from event
  text classification, not real shot-level expected goals.
- **Short series.** Median 15 valuations per player means classical
  sequence models (ETS, Theta, ARIMA) genuinely cannot work well here.
  That's not a bug in those methods — it's a property of the data, and
  the results table reflects it honestly.
- **`country_of_citizenship` dominance** is a market quirk (work permits,
  marketability, agent networks) that the model **captures** rather than
  **corrects**. Practitioners using this for actual scouting should
  audit for the obvious fairness implications before any deployment.
- **Target horizon is variable.** `y_horizon_days` ranges from ~30 days
  to over a year. We include it as a feature so the model can condition
  on horizon, but a fixed-horizon reformulation would be cleaner for
  some downstream uses.
- **No causal claims.** This is predictive modelling on observational
  market prices. Don't read feature importances as causal levers.

---

## Future work

- **Neural forecasting** (NHITS / TFT / PatchTST via `neuralforecast`).
- **From-scratch JAX** TabTransformer + LSTM — the "craft" piece, continuity
  with [cronica-jax](https://github.com/DanielRegaladoUMiami/cronica-jax).
- **Stacked ensemble** of the best model from each family.
- **Hugging Face Space** demo (`type a player → see the predicted
  6-month value`).

---

## License

Apache 2.0 — see [LICENSE](LICENSE). Dataset is CC0 via Kaggle.

## Author

**Daniel Regalado** — MSBA, University of Miami.
GitHub: [@DanielRegaladoUMiami](https://github.com/DanielRegaladoUMiami)
