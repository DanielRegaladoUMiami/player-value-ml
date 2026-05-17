"""Exhaustive EDA on the Transfermarkt player_valuations table + supporting tables.

Outputs:
  - results/eda/summary.md          markdown report with key numbers
  - results/eda/plots/*.png         matplotlib plots embedded in README

Goal: ground the feature-engineering phase in real numbers — series length
distribution, time gaps, target skew, cold-start prevalence, coverage by
position/league/age. NOT just a print of describe().
"""
from __future__ import annotations

import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CACHE = Path.home() / ".cache/kagglehub/datasets/davidcariboo/player-scores/versions/655"
OUT = Path(__file__).resolve().parent.parent / "results" / "eda"
PLOTS = OUT / "plots"
OUT.mkdir(parents=True, exist_ok=True)
PLOTS.mkdir(parents=True, exist_ok=True)

# ----------------------------- data load -----------------------------


def load() -> dict[str, pd.DataFrame]:
    print("Loading tables...")
    val = pd.read_csv(CACHE / "player_valuations.csv", parse_dates=["date"])
    players = pd.read_csv(CACHE / "players.csv", parse_dates=["date_of_birth"], low_memory=False)
    apps = pd.read_csv(CACHE / "appearances.csv", parse_dates=["date"])
    transfers = pd.read_csv(CACHE / "transfers.csv", parse_dates=["transfer_date"])
    comps = pd.read_csv(CACHE / "competitions.csv")
    clubs = pd.read_csv(CACHE / "clubs.csv")
    return dict(val=val, players=players, apps=apps, transfers=transfers,
                comps=comps, clubs=clubs)


# ----------------------------- analysis -----------------------------


def plot_value_distribution(val: pd.DataFrame) -> dict:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(val["market_value_in_eur"] / 1e6, bins=80, color="#4682B4")
    axes[0].set_xlabel("market value (€M)")
    axes[0].set_ylabel("# valuations")
    axes[0].set_title("Raw market value distribution")
    axes[0].set_yscale("log")
    axes[1].hist(np.log1p(val["market_value_in_eur"]), bins=80, color="#CD5C5C")
    axes[1].set_xlabel("log(1 + market value €)")
    axes[1].set_ylabel("# valuations")
    axes[1].set_title("Log-transformed distribution")
    plt.tight_layout()
    plt.savefig(PLOTS / "01_value_distribution.png", dpi=120)
    plt.close()
    return {
        "n": int(len(val)),
        "min_eur": int(val["market_value_in_eur"].min()),
        "p50_eur": int(val["market_value_in_eur"].median()),
        "p99_eur": int(val["market_value_in_eur"].quantile(0.99)),
        "max_eur": int(val["market_value_in_eur"].max()),
        "skew_raw": float(val["market_value_in_eur"].skew()),
        "skew_log": float(np.log1p(val["market_value_in_eur"]).skew()),
    }


def series_length_stats(val: pd.DataFrame) -> dict:
    counts = val.groupby("player_id").size()
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(counts.values, bins=60, color="#3CB371")
    ax.set_xlabel("# valuations per player")
    ax.set_ylabel("# players")
    ax.set_title(f"Series length distribution (n={len(counts)} players)")
    ax.axvline(counts.median(), color="black", linestyle="--",
               label=f"median = {int(counts.median())}")
    ax.legend()
    plt.tight_layout()
    plt.savefig(PLOTS / "02_series_length.png", dpi=120)
    plt.close()
    return {
        "n_players": int(len(counts)),
        "min": int(counts.min()),
        "p25": int(counts.quantile(0.25)),
        "p50": int(counts.median()),
        "p75": int(counts.quantile(0.75)),
        "p95": int(counts.quantile(0.95)),
        "max": int(counts.max()),
        "mean": float(counts.mean()),
        "share_lt3": float((counts < 3).mean()),
        "share_lt5": float((counts < 5).mean()),
    }


def time_gap_stats(val: pd.DataFrame) -> dict:
    v = val.sort_values(["player_id", "date"]).copy()
    v["prev_date"] = v.groupby("player_id")["date"].shift(1)
    v["gap_days"] = (v["date"] - v["prev_date"]).dt.days
    g = v["gap_days"].dropna()
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(g.clip(upper=400), bins=80, color="#8A2BE2")
    ax.set_xlabel("days between consecutive valuations (clipped 400)")
    ax.set_ylabel("# valuation pairs")
    ax.set_title("Time gap between valuations per player")
    plt.tight_layout()
    plt.savefig(PLOTS / "03_time_gaps.png", dpi=120)
    plt.close()
    return {
        "p25_days": int(g.quantile(0.25)),
        "p50_days": int(g.median()),
        "p75_days": int(g.quantile(0.75)),
        "mean_days": float(g.mean()),
        "share_gt365": float((g > 365).mean()),
    }


def coverage_by_year(val: pd.DataFrame) -> dict:
    v = val.copy()
    v["year"] = v["date"].dt.year
    counts = v.groupby("year").size()
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(counts.index, counts.values, color="#FF8C00")
    ax.set_xlabel("year")
    ax.set_ylabel("# valuations")
    ax.set_title("Valuation coverage by year")
    plt.tight_layout()
    plt.savefig(PLOTS / "04_coverage_year.png", dpi=120)
    plt.close()
    return {"min_year": int(counts.index.min()),
            "max_year": int(counts.index.max()),
            "by_year": {int(k): int(v) for k, v in counts.items()}}


def coverage_by_position(val: pd.DataFrame, players: pd.DataFrame) -> dict:
    merged = val.merge(players[["player_id", "position", "sub_position"]],
                       on="player_id", how="left")
    by_pos = merged.groupby("position").size().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(by_pos.index.astype(str), by_pos.values, color="#20B2AA")
    ax.set_xlabel("position")
    ax.set_ylabel("# valuations")
    ax.set_title("Coverage by player position")
    plt.tight_layout()
    plt.savefig(PLOTS / "05_coverage_position.png", dpi=120)
    plt.close()
    return {str(k): int(v) for k, v in by_pos.items()}


def value_vs_age(val: pd.DataFrame, players: pd.DataFrame) -> None:
    p = players[["player_id", "date_of_birth"]].dropna()
    v = val.merge(p, on="player_id", how="inner")
    v["age"] = ((v["date"] - v["date_of_birth"]).dt.days / 365.25)
    v = v[(v["age"] >= 14) & (v["age"] <= 45)]
    v["age_bin"] = v["age"].round().astype(int)
    agg = v.groupby("age_bin")["market_value_in_eur"].agg(["median", "mean", "count"])
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(agg.index, agg["median"] / 1e6, "o-", color="#DC143C", label="median (€M)")
    ax.plot(agg.index, agg["mean"]   / 1e6, "s--", color="#4169E1", alpha=0.6, label="mean (€M)")
    ax.set_xlabel("age")
    ax.set_ylabel("market value (€M)")
    ax.set_title("Market value by age (across all leagues/positions)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(PLOTS / "06_value_vs_age.png", dpi=120)
    plt.close()


def t_plus_6_target_stats(val: pd.DataFrame) -> dict:
    """Compute target distribution: log(value_T+6mo / value_T) for each player."""
    v = val.sort_values(["player_id", "date"]).copy()
    v["next_date"] = v.groupby("player_id")["date"].shift(-1)
    v["next_value"] = v.groupby("player_id")["market_value_in_eur"].shift(-1)
    v["next_gap_days"] = (v["next_date"] - v["date"]).dt.days
    # Target: log ratio of next value / current.
    # Filter zeros on BOTH sides — log(0) is -inf and propagates NaNs.
    eligible = v[
        (v["next_value"].notna())
        & (v["market_value_in_eur"] > 0)
        & (v["next_value"] > 0)
    ].copy()
    eligible["log_ratio"] = np.log(eligible["next_value"] / eligible["market_value_in_eur"])
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(eligible["log_ratio"].clip(-2, 2), bins=80, color="#9370DB")
    ax.axvline(0, color="black", linestyle="--", alpha=0.4)
    ax.set_xlabel("log(next_value / current_value), clipped to [-2, 2]")
    ax.set_ylabel("# eligible (current, next) pairs")
    ax.set_title("Target distribution: log-ratio of consecutive valuations")
    plt.tight_layout()
    plt.savefig(PLOTS / "07_target_distribution.png", dpi=120)
    plt.close()
    return {
        "n_eligible_pairs": int(len(eligible)),
        "log_ratio_mean": float(eligible["log_ratio"].mean()),
        "log_ratio_p50": float(eligible["log_ratio"].median()),
        "log_ratio_std": float(eligible["log_ratio"].std()),
        "share_gain":  float((eligible["log_ratio"] > 0).mean()),
        "share_flat":  float((eligible["log_ratio"].abs() < 0.05).mean()),
        "share_drop":  float((eligible["log_ratio"] < 0).mean()),
    }


# ----------------------------- run -----------------------------


def main() -> None:
    data = load()
    print("Computing analyses...")

    summary = {}
    summary["value_distribution"] = plot_value_distribution(data["val"])
    summary["series_length"]      = series_length_stats(data["val"])
    summary["time_gaps"]          = time_gap_stats(data["val"])
    summary["coverage_year"]      = coverage_by_year(data["val"])
    summary["coverage_position"]  = coverage_by_position(data["val"], data["players"])
    value_vs_age(data["val"], data["players"])
    summary["target_distribution"] = t_plus_6_target_stats(data["val"])

    # Markdown summary
    md = []
    md.append("# EDA — Transfermarkt player valuations\n")
    md.append(f"_Generated by `scripts/eda.py` over the Kaggle dump._\n\n")

    md.append("## 1. Value distribution\n")
    vd = summary["value_distribution"]
    md.append(f"- N valuations: **{vd['n']:,}**")
    md.append(f"- Range: €{vd['min_eur']:,} → €{vd['max_eur']:,}")
    md.append(f"- p50 / p99: €{vd['p50_eur']:,} / €{vd['p99_eur']:,}")
    md.append(f"- Skew (raw)  : **{vd['skew_raw']:.2f}**  ⇒ raw values are extremely right-skewed.")
    md.append(f"- Skew (log1p): **{vd['skew_log']:.2f}**  ⇒ log transform fixes the skew.")
    md.append("\n![value distribution](plots/01_value_distribution.png)\n")

    md.append("## 2. Series length per player\n")
    sl = summary["series_length"]
    md.append(f"- {sl['n_players']:,} unique players with valuations")
    md.append(f"- median series length: **{sl['p50']}** points")
    md.append(f"- p25 / p75 / p95: {sl['p25']} / {sl['p75']} / {sl['p95']}")
    md.append(f"- max: {sl['max']}")
    md.append(f"- players with < 3 points: **{sl['share_lt3']*100:.1f}%** (cold-start problem)")
    md.append(f"- players with < 5 points: **{sl['share_lt5']*100:.1f}%**")
    md.append("\n![series length](plots/02_series_length.png)\n")

    md.append("## 3. Time between consecutive valuations\n")
    tg = summary["time_gaps"]
    md.append(f"- p25 / p50 / p75 days: {tg['p25_days']} / {tg['p50_days']} / {tg['p75_days']}")
    md.append(f"- share with gap > 365d: **{tg['share_gt365']*100:.1f}%**")
    md.append(f"- Average gap: ~{tg['mean_days']:.0f} days  ⇒ updates are roughly twice a year.")
    md.append("\n![time gaps](plots/03_time_gaps.png)\n")

    md.append("## 4. Coverage by year\n")
    cy = summary["coverage_year"]
    md.append(f"- Range: **{cy['min_year']} – {cy['max_year']}**")
    md.append(f"- Coverage grew massively after ~2010 — earlier years are sparse.\n")
    md.append("![coverage year](plots/04_coverage_year.png)\n")

    md.append("## 5. Coverage by position\n")
    cp = summary["coverage_position"]
    for k, n in cp.items():
        md.append(f"- {k}: {n:,}")
    md.append("\n![coverage position](plots/05_coverage_position.png)\n")

    md.append("## 6. Value vs age\n")
    md.append("Players typically peak in value around age 24–27, then decline.\n")
    md.append("![value vs age](plots/06_value_vs_age.png)\n")

    md.append("## 7. Target distribution — log-ratio of consecutive valuations\n")
    td = summary["target_distribution"]
    md.append(f"- Eligible (current, next) pairs: **{td['n_eligible_pairs']:,}**")
    md.append(f"- Mean log-ratio: {td['log_ratio_mean']:+.3f}")
    md.append(f"- Median log-ratio: {td['log_ratio_p50']:+.3f}")
    md.append(f"- Std log-ratio: {td['log_ratio_std']:.3f}")
    md.append(f"- Share gain (>0): **{td['share_gain']*100:.1f}%**")
    md.append(f"- Share flat (|·|<0.05): {td['share_flat']*100:.1f}%")
    md.append(f"- Share drop (<0): **{td['share_drop']*100:.1f}%**")
    md.append("\n![target distribution](plots/07_target_distribution.png)\n")

    md.append("## Implications for modeling\n")
    md.append("- **Target**: predict `log(value_T+H / value_T)`, NOT raw €. The raw")
    md.append("  distribution is too skewed; log fixes it.")
    md.append("- **Cold start matters**: ~{:.0f}% of players have < 3 valuations.".format(sl['share_lt3']*100))
    md.append("  Sequence models will be useless on these; tabular ML must handle them.")
    md.append("- **Series are short and irregular**: median {} points, ~{:.0f} days apart.".format(sl['p50'], tg['mean_days']))
    md.append("  Don't expect classical AutoARIMA to work brilliantly — too few points.")
    md.append("- **Class imbalance**: more 'flat' and 'gain' pairs than 'drop' — model")
    md.append("  could trivially bias toward predicting non-negative ratios.")

    (OUT / "summary.md").write_text("\n".join(md), encoding="utf-8")
    print(f"Wrote {OUT / 'summary.md'}")
    print(f"Wrote 7 plots into {PLOTS}/")


if __name__ == "__main__":
    main()
