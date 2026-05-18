"""Gradio Space for player-value-ml.

Forecasts the 6-month log-change in a football player's Transfermarkt
market value using a LightGBM Booster trained in
`scripts/train_lgbm.py`. The user can either:

  1. Pick an existing player from the test split (most recent row), OR
  2. Manually adjust the most impactful features.

The model predicts y_log_ratio = log(value_{T+1} / value_T). We invert
that to EUR:  predicted_value = current_value * exp(y_log_ratio),
and expose a ±test-MAE band as a crude uncertainty interval.
"""
from __future__ import annotations

import json
from pathlib import Path

import gradio as gr
import lightgbm as lgb
import numpy as np
import pandas as pd

# ---------------------------------------------------------------- paths
ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "results" / "models" / "lgbm" / "model.lgb"
METRICS_PATH = ROOT / "results" / "models" / "lgbm" / "metrics.json"
IMPORTANCE_PATH = ROOT / "results" / "models" / "lgbm" / "feature_importance.csv"
TEST_PARQUET = ROOT / "data" / "features" / "test.parquet"

GITHUB_URL = "https://github.com/DanielRegaladoUMiami/player-value-ml"
CRONICA_URL = "https://github.com/DanielRegaladoUMiami/cronica-jax"

# ---------------------------------------------------------------- load
print(f"Loading model from {MODEL_PATH}")
MODEL = lgb.Booster(model_file=str(MODEL_PATH))
FEATURES: list[str] = MODEL.feature_name()
print(f"  num_features = {len(FEATURES)}")

CATEGORICAL = ["position", "sub_position", "foot", "career_stage",
               "country_of_citizenship"]

METRICS = json.loads(METRICS_PATH.read_text()) if METRICS_PATH.exists() else {}
TEST_MAE = float(METRICS.get("test_mae_log", 0.21))
TEST_R2 = float(METRICS.get("test_r2", 0.20))
print(f"  test MAE (log) = {TEST_MAE:.4f}  R² = {TEST_R2:+.4f}")

print(f"Loading test parquet from {TEST_PARQUET}")
TEST_DF = pd.read_parquet(TEST_PARQUET)
print(f"  rows = {len(TEST_DF):,}  cols = {TEST_DF.shape[1]}")

# Most recent row per player_id
LATEST = (
    TEST_DF.sort_values(["player_id", "date"])
    .groupby("player_id", as_index=False)
    .tail(1)
    .reset_index(drop=True)
)
# Top-200 by current market value
LATEST = LATEST.sort_values("market_value_in_eur", ascending=False).head(200)
LATEST = LATEST.reset_index(drop=True)

# Build dropdown labels.  We don't have player names in the feature table,
# so we identify players by id + position + age + current value.
def _label(row: pd.Series) -> str:
    val_m = row["market_value_in_eur"] / 1e6
    return (
        f"#{int(row['player_id'])} | {row.get('position', '?')} "
        f"({row.get('sub_position', '?')}) | "
        f"age {row.get('age', 0):.1f} | "
        f"€{val_m:.1f}M"
    )

LATEST["_label"] = LATEST.apply(_label, axis=1)
PLAYER_CHOICES = LATEST["_label"].tolist()
LABEL_TO_IDX = {lbl: i for i, lbl in enumerate(PLAYER_CHOICES)}

# Global top-15 feature importances (used as the "explanation" table)
if IMPORTANCE_PATH.exists():
    IMPORTANCE = pd.read_csv(IMPORTANCE_PATH)
else:
    IMPORTANCE = pd.DataFrame(columns=["feature", "importance_gain"])

# Build the categorical level vocabularies from the test split, so the
# dropdowns offer realistic options.
def _levels(col: str) -> list[str]:
    if col not in TEST_DF.columns:
        return []
    vals = TEST_DF[col].dropna().astype(str).unique().tolist()
    vals.sort()
    return vals

POSITIONS = _levels("position")
SUB_POSITIONS = _levels("sub_position")
FEET = _levels("foot")
CAREER_STAGES = _levels("career_stage")
COUNTRIES = _levels("country_of_citizenship")


# ---------------------------------------------------------------- helpers
def _row_to_X(row: dict) -> pd.DataFrame:
    """Coerce a dict of feature values into a 1-row DataFrame
    in the exact feature order LightGBM expects, with categoricals cast."""
    data = {}
    for f in FEATURES:
        data[f] = [row.get(f, np.nan)]
    X = pd.DataFrame(data)
    for c in CATEGORICAL:
        if c in X.columns:
            X[c] = X[c].astype("category")
    return X


def predict_from_dict(row: dict) -> float:
    X = _row_to_X(row)
    y = float(MODEL.predict(X)[0])
    return y


def explanation_table(top_n: int = 5) -> pd.DataFrame:
    if IMPORTANCE.empty:
        return pd.DataFrame({"feature": [], "importance_gain": []})
    return (
        IMPORTANCE.head(top_n)[["feature", "importance_gain"]]
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------- core
def predict_for_player(label: str,
                       override_current_value_m: float | None,
                       override_age: float | None,
                       override_goals_6mo: float | None,
                       override_assists_6mo: float | None,
                       override_minutes_6mo: float | None,
                       override_position: str | None,
                       override_sub_position: str | None,
                       override_foot: str | None,
                       override_country: str | None,
                       override_career_stage: str | None):
    if not label or label not in LABEL_TO_IDX:
        return ("Please pick a player from the dropdown.",
                None, None, None, None)

    base_row = LATEST.iloc[LABEL_TO_IDX[label]].to_dict()
    row = {k: base_row.get(k) for k in FEATURES}

    # Apply optional overrides (only when the user provided a non-None
    # value; we keep the player's real feature otherwise).
    if override_age is not None and override_age > 0:
        row["age"] = float(override_age)
    if override_goals_6mo is not None and override_goals_6mo >= 0:
        row["goals_6mo"] = float(override_goals_6mo)
    if override_assists_6mo is not None and override_assists_6mo >= 0:
        row["assists_6mo"] = float(override_assists_6mo)
    if override_minutes_6mo is not None and override_minutes_6mo >= 0:
        row["minutes_played_6mo"] = float(override_minutes_6mo)
    if override_position:
        row["position"] = override_position
    if override_sub_position:
        row["sub_position"] = override_sub_position
    if override_foot:
        row["foot"] = override_foot
    if override_country:
        row["country_of_citizenship"] = override_country
    if override_career_stage:
        row["career_stage"] = override_career_stage

    # current value: prefer the override, else the latest in the parquet
    if override_current_value_m is not None and override_current_value_m > 0:
        current_value = float(override_current_value_m) * 1e6
    else:
        current_value = float(base_row["market_value_in_eur"])

    y_log_ratio = predict_from_dict(row)
    multiplier = float(np.exp(y_log_ratio))
    predicted_value = current_value * multiplier

    lo = current_value * float(np.exp(y_log_ratio - TEST_MAE))
    hi = current_value * float(np.exp(y_log_ratio + TEST_MAE))

    summary_md = (
        f"### Forecast\n\n"
        f"| Quantity | Value |\n"
        f"|---|---|\n"
        f"| Current market value | **€{current_value/1e6:,.2f}M** |\n"
        f"| Predicted log-ratio  | `{y_log_ratio:+.4f}` |\n"
        f"| Implied multiplier   | `×{multiplier:.3f}` |\n"
        f"| **Forecast (6 mo)**  | **€{predicted_value/1e6:,.2f}M** |\n"
        f"| 68% band (±MAE)      | €{lo/1e6:,.2f}M  …  €{hi/1e6:,.2f}M |\n\n"
        f"_Model test MAE (log) = {TEST_MAE:.3f}, R² = {TEST_R2:+.3f}._"
    )

    return (summary_md,
            round(current_value / 1e6, 2),
            round(predicted_value / 1e6, 2),
            round(y_log_ratio, 4),
            explanation_table())


def autofill_from_player(label: str):
    """When the user picks a player, populate the override widgets
    with that player's real feature values so they can tweak from there."""
    if not label or label not in LABEL_TO_IDX:
        return [None] * 10
    r = LATEST.iloc[LABEL_TO_IDX[label]].to_dict()
    def _g(k, default=None):
        v = r.get(k, default)
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return default
        return v
    return [
        round(float(_g("market_value_in_eur", 0.0)) / 1e6, 2),
        round(float(_g("age", 25.0)), 1),
        float(_g("goals_6mo", 0.0)),
        float(_g("assists_6mo", 0.0)),
        float(_g("minutes_played_6mo", 0.0)),
        str(_g("position", "")) or None,
        str(_g("sub_position", "")) or None,
        str(_g("foot", "")) or None,
        str(_g("country_of_citizenship", "")) or None,
        str(_g("career_stage", "")) or None,
    ]


# ---------------------------------------------------------------- UI
with gr.Blocks(title="player-value-ml", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        f"# Player Market Value Forecasting — 6-month horizon\n"
        f"LightGBM on engineered Transfermarkt features. "
        f"[GitHub repo]({GITHUB_URL}) · "
        f"test MAE_log = {TEST_MAE:.3f} · R² = {TEST_R2:+.3f}"
    )

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 1. Pick a player")
            player_dd = gr.Dropdown(
                choices=PLAYER_CHOICES,
                label="Player (id | position | age | current value)",
                value=PLAYER_CHOICES[0] if PLAYER_CHOICES else None,
            )

            gr.Markdown("### 2. (Optional) Override features")
            with gr.Group():
                current_value_m = gr.Number(label="Current market value (€M)",
                                            value=None, precision=2)
                age_in = gr.Number(label="Age (years)",
                                   value=None, precision=1)
                goals_6mo_in = gr.Number(label="Goals (last 6 months)",
                                         value=None, precision=0)
                assists_6mo_in = gr.Number(label="Assists (last 6 months)",
                                           value=None, precision=0)
                minutes_6mo_in = gr.Number(label="Minutes played (last 6 mo)",
                                           value=None, precision=0)
                position_in = gr.Dropdown(POSITIONS, label="Position",
                                          value=None, allow_custom_value=False)
                sub_position_in = gr.Dropdown(SUB_POSITIONS, label="Sub-position",
                                              value=None, allow_custom_value=False)
                foot_in = gr.Dropdown(FEET, label="Foot",
                                      value=None, allow_custom_value=False)
                country_in = gr.Dropdown(COUNTRIES,
                                         label="Country of citizenship",
                                         value=None, allow_custom_value=False)
                career_stage_in = gr.Dropdown(CAREER_STAGES,
                                              label="Career stage",
                                              value=None,
                                              allow_custom_value=False)

            predict_btn = gr.Button("Forecast 6-month value",
                                    variant="primary")

        with gr.Column(scale=1):
            gr.Markdown("### 3. Forecast")
            summary_out = gr.Markdown()
            with gr.Row():
                current_out = gr.Number(label="Current (€M)", interactive=False)
                forecast_out = gr.Number(label="Forecast (€M)",
                                         interactive=False)
                logratio_out = gr.Number(label="Predicted log-ratio",
                                         interactive=False)
            gr.Markdown("### Top global features (importance by gain)")
            importance_out = gr.Dataframe(
                value=explanation_table(),
                headers=["feature", "importance_gain"],
                interactive=False,
                wrap=True,
            )

    # When the player changes, prefill the overrides with that player's row.
    player_dd.change(
        autofill_from_player,
        inputs=[player_dd],
        outputs=[current_value_m, age_in, goals_6mo_in, assists_6mo_in,
                 minutes_6mo_in, position_in, sub_position_in, foot_in,
                 country_in, career_stage_in],
    )

    predict_btn.click(
        predict_for_player,
        inputs=[player_dd, current_value_m, age_in, goals_6mo_in,
                assists_6mo_in, minutes_6mo_in, position_in, sub_position_in,
                foot_in, country_in, career_stage_in],
        outputs=[summary_out, current_out, forecast_out, logratio_out,
                 importance_out],
    )

    gr.Markdown(
        f"---\n"
        f"Trained on Kaggle Transfermarkt valuations (~507k rows). "
        f"Sibling project: [cronica-jax]({CRONICA_URL}) — a JAX-from-scratch "
        f"Spanish news LM. Built by Daniel Regalado."
    )


if __name__ == "__main__":
    demo.launch()
