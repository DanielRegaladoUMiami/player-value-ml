# ADR-001: Compare 7 model families honestly on the same temporal split

**Date:** 2026-05-17
**Status:** Accepted

## Context

The default temptation when building a forecasting project is to pick one
model (usually a fancy neural net), tune it well, and report the result.
This is misleading: for tabular time-series data with rich features,
gradient boosting and even regularised linear regression are usually
competitive with — and often beat — neural networks.

The Transfermarkt player-value problem looks fancy ("forecasting") but is
fundamentally:

- Short series per player (avg ~16 valuations, max ~50)
- Irregular sampling (~3 updates/year)
- Heavy cold-start (new players arrive constantly)
- Rich tabular features (position, age, club, league, performance)

These conditions are NOT where neural sequence models shine. They are where
tree-based ML with engineered lag features tends to win.

## Decision

We compare seven model families on the **same temporal split**, the **same
feature set** where applicable, and report all results honestly:

| # | family               | role in the experiment                            |
|---|----------------------|---------------------------------------------------|
| 0 | Naive                | sanity baseline — surprisingly hard to beat        |
| 1 | Linear (Ridge/Lasso) | interpretable baseline                             |
| 2 | Statistical (Nixtla) | AutoARIMA / ETS / Theta — classical workhorses     |
| 3 | ML tabular           | LightGBM with lag features — the favourite         |
| 4 | Neural forecasting   | NHITS / TFT / PatchTST off-the-shelf               |
| 5 | From-scratch JAX     | TabTransformer + LSTM hand-rolled                  |
| 6 | Ensemble             | stacking the best of each family                   |

The output is a single comparison table + per-segment breakdown (by
position, age band, league tier, series length). The research question is:

> *When does each family win, and what does that tell us about the structure
> of the data?*

## Consequences

- More work than picking one model, but the result is a much stronger
  portfolio piece. It demonstrates **methodology**, not just "I trained
  something".
- Different families have very different tooling: statistical via
  `statsforecast`, ML via `mlforecast`/LightGBM, neural off-the-shelf via
  `neuralforecast`, and from-scratch via JAX. The project needs to handle
  this plurality cleanly.
- The from-scratch JAX model is included as the "craft" piece (continuity
  with [cronica-jax](https://github.com/DanielRegaladoUMiami/cronica-jax)).
  It is NOT expected to win on absolute metrics; if it does, that's a
  surprising result; if not, the comparison itself is the contribution.
