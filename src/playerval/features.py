"""Feature engineering pipeline — the heart of the project.

Builds ~75 features per (player_id, as_of_date) anchored at Transfermarkt
valuation timestamps. Every feature is computed using ONLY information
available BEFORE the as_of_date (no leakage).

Feature groups
--------------
  A. LAG / TRAJECTORY        (~15 features)  the spine of forecasting
  B. PERFORMANCE ABSOLUTE    (~12 features)  raw goals/assists/minutes
  C. PERFORMANCE vs TEAM     (~8 features)   player as % of team
  D. PERFORMANCE vs POSITION (~6 features)   z-scores within position+league
  E. COMPETITION-TIER WEIGHTED (~6 features) goals weighted by league quality
  F. PLAYER STATIC           (~10 features)  age, position, foot, etc.
  G. CLUB / LEAGUE STATIC    (~8 features)
  H. TEMPORAL / SEASONAL     (~5 features)
  I. AGE × POSITION INTERACTION (~3 features)
  J. XG / SHOT-QUALITY PROXIES (~7 features) approximations without event data

The target is y = log(value_T_plus_next / value_T), where T_plus_next is the
next observed valuation for the same player. y_horizon_days is also kept
as a feature so the model can learn how forecast horizon varies.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from playerval.refdata import (
    TIER_WEIGHT, career_stage, competition_tier, peak_age_for,
)
from playerval.xg_proxies import compute_xg_proxies

CACHE = Path.home() / ".cache/kagglehub/datasets/davidcariboo/player-scores/versions/655"


# ============================ load ============================


def _read(name: str, **kwargs) -> pd.DataFrame:
    return pd.read_csv(CACHE / f"{name}.csv", low_memory=False, **kwargs)


def load_all() -> dict[str, pd.DataFrame]:
    print("Loading raw tables...")
    val = _read("player_valuations", parse_dates=["date"])
    players = _read("players", parse_dates=["date_of_birth"])
    apps = _read("appearances", parse_dates=["date"])
    games = _read("games", parse_dates=["date"])
    transfers = _read("transfers", parse_dates=["transfer_date"])
    club_games = _read("club_games")
    events = _read("game_events", parse_dates=["date"])
    clubs = _read("clubs")
    print(f"  valuations={len(val):,} apps={len(apps):,} "
          f"events={len(events):,} games={len(games):,}")
    return dict(val=val, players=players, apps=apps, games=games,
                transfers=transfers, club_games=club_games, events=events,
                clubs=clubs)


# ============================ helpers ============================


def _rolling_window_sum(
    series: pd.DataFrame,
    date_col: str,
    value_cols: list[str],
    window_days: int,
    group_col: str = "player_id",
) -> pd.DataFrame:
    """Per-group rolling sum over the LAST `window_days` days (exclusive of T itself).

    Returns a frame indexed by the same (group_col, date_col) keys, with one
    column per value_col holding the trailing-window sum BEFORE that date.
    """
    s = series.sort_values([group_col, date_col]).copy()
    s[date_col] = pd.to_datetime(s[date_col])
    out_parts = []
    for g, sub in s.groupby(group_col, sort=False):
        sub = sub.set_index(date_col)
        # rolling on a time index includes the current row; we want strictly
        # before, so we shift by 1 minute back. Simpler: compute including
        # current, then subtract current row's values.
        rolling = sub[value_cols].rolling(f"{window_days}D").sum()
        # remove current-row contribution to make it strictly past
        rolling_excl = rolling - sub[value_cols]
        rolling_excl = rolling_excl.reset_index()
        rolling_excl[group_col] = g
        out_parts.append(rolling_excl)
    out = pd.concat(out_parts, ignore_index=True)
    out = out[[group_col, date_col] + value_cols]
    return out


# ============================ feature builders ============================


def build_valuation_traj(val: pd.DataFrame) -> pd.DataFrame:
    """Group A: lag and trajectory features over a player's own valuation history."""
    v = val.sort_values(["player_id", "date"]).copy()
    v["log_value"] = np.log1p(v["market_value_in_eur"].clip(lower=1.0))

    g = v.groupby("player_id", sort=False)

    for k in range(1, 6):
        v[f"value_lag_{k}"] = g["market_value_in_eur"].shift(k)
        v[f"log_value_lag_{k}"] = g["log_value"].shift(k)
    for k in range(1, 4):
        v[f"value_diff_{k}"] = v["log_value"] - v[f"log_value_lag_{k}"]

    # Growth over various horizons (in TIME, not in valuation index)
    v["prev_date"]  = g["date"].shift(1)
    v["gap_to_prev_days"] = (v["date"] - v["prev_date"]).dt.days

    # Career-wide stats up to (but not including) current row
    expanding = g["log_value"].expanding()
    v["log_value_career_mean"] = expanding.mean().shift(1).reset_index(drop=True)
    v["log_value_career_std"]  = expanding.std().shift(1).reset_index(drop=True)
    v["log_value_career_max"]  = expanding.max().shift(1).reset_index(drop=True)
    v["months_since_peak"] = np.nan  # filled per-group below
    # consecutive monotonicity streaks
    v["delta_sign"] = np.sign(v["value_diff_1"]).fillna(0)
    v["consec_increases"] = 0
    v["consec_decreases"] = 0

    pieces = []
    for pid, sub in v.groupby("player_id", sort=False):
        sub = sub.copy()
        # months_since_peak using strictly past peak
        log_vals = sub["log_value"].values
        past_peak_idx = np.full(len(sub), -1)
        running_max = -np.inf
        max_idx = -1
        for i, lv in enumerate(log_vals):
            past_peak_idx[i] = max_idx
            if lv > running_max:
                running_max = lv
                max_idx = i
        # months_since_peak: relative to row whose past peak index is past_peak_idx[i]
        dates = sub["date"].values
        msp = np.full(len(sub), np.nan)
        for i, pp in enumerate(past_peak_idx):
            if pp >= 0:
                delta = (pd.Timestamp(dates[i]) - pd.Timestamp(dates[pp])).days
                msp[i] = delta / 30.0
        sub["months_since_peak"] = msp

        # streaks of consecutive increases / decreases
        inc = np.zeros(len(sub), dtype=int)
        dec = np.zeros(len(sub), dtype=int)
        for i, s in enumerate(sub["delta_sign"].values):
            if i == 0:
                continue
            if s > 0:
                inc[i] = inc[i - 1] + 1
            elif s < 0:
                dec[i] = dec[i - 1] + 1
        # use streak BEFORE current row (use shift)
        sub["consec_increases"] = pd.Series(inc, index=sub.index).shift(1).fillna(0)
        sub["consec_decreases"] = pd.Series(dec, index=sub.index).shift(1).fillna(0)
        pieces.append(sub)
    v = pd.concat(pieces, ignore_index=True)

    cols = ["player_id", "date", "market_value_in_eur", "log_value",
            "gap_to_prev_days",
            *[f"value_lag_{k}" for k in range(1, 6)],
            *[f"log_value_lag_{k}" for k in range(1, 6)],
            *[f"value_diff_{k}" for k in range(1, 4)],
            "log_value_career_mean", "log_value_career_std",
            "log_value_career_max",
            "months_since_peak",
            "consec_increases", "consec_decreases",
            ]
    return v[cols]


def build_perf_rolling(apps: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    """Group B raw + Group C team-relative aggregates by player x date."""
    # Attach competition_tier to each appearance from games (apps already has competition_id)
    a = apps.copy()
    a["date"] = pd.to_datetime(a["date"])
    a["tier"] = a["competition_id"].apply(competition_tier)
    a["tier_weight"] = a["tier"].map(TIER_WEIGHT).fillna(0.3)

    # Per-appearance derived columns
    a["played"] = (a["minutes_played"] > 0).astype(int)
    a["started"] = (a["minutes_played"] >= 60).astype(int)   # crude but practical
    a["full90"] = (a["minutes_played"] >= 90).astype(int)
    a["weighted_goals"] = a["goals"] * a["tier_weight"]
    a["weighted_assists"] = a["assists"] * a["tier_weight"]
    a["weighted_minutes"] = a["minutes_played"] * a["tier_weight"]
    a["g_plus_a"] = a["goals"] + a["assists"]

    # Group-B/E columns we want rolling
    value_cols = [
        "goals", "assists", "g_plus_a", "minutes_played", "played",
        "started", "full90", "yellow_cards", "red_cards",
        "weighted_goals", "weighted_assists", "weighted_minutes",
    ]
    rolling_6mo = _rolling_window_sum(a, "date", value_cols, 180)
    rolling_6mo = rolling_6mo.rename(columns={c: f"{c}_6mo" for c in value_cols})
    rolling_12mo = _rolling_window_sum(a, "date", value_cols, 365)
    rolling_12mo = rolling_12mo.rename(columns={c: f"{c}_12mo" for c in value_cols})

    perf = rolling_6mo.merge(rolling_12mo, on=["player_id", "date"], how="outer")
    return perf


def build_team_context(
    apps: pd.DataFrame,
    games: pd.DataFrame,
) -> pd.DataFrame:
    """Group C: team-relative shares. Compute, per (club_id, date),
    rolling sums of team-total goals/assists/minutes, then express player
    contributions as shares.
    """
    a = apps.copy()
    a["date"] = pd.to_datetime(a["date"])

    # Aggregate to (club_id, date, game_id) — team total for that game = sum of player appearances
    team_per_game = (
        a.groupby(["player_club_id", "date"], as_index=False)
         .agg(team_goals=("goals", "sum"),
              team_assists=("assists", "sum"),
              team_minutes=("minutes_played", "sum"))
    )
    team_per_game = team_per_game.rename(columns={"player_club_id": "club_id"})

    # Rolling team aggregates per club over 6mo
    team_per_game = team_per_game.sort_values(["club_id", "date"])
    team_per_game["date"] = pd.to_datetime(team_per_game["date"])
    parts = []
    for cid, sub in team_per_game.groupby("club_id", sort=False):
        sub = sub.set_index("date")
        roll6 = sub.rolling("180D").sum()
        roll6 = roll6.rename(columns=lambda c: f"team_{c}_6mo".replace("team_team_", "team_"))
        roll6 = roll6.reset_index()
        roll6["club_id"] = cid
        parts.append(roll6)
    team_roll = pd.concat(parts, ignore_index=True)

    # Now we want, at each (player_id, date), the team_*_6mo from the player's current club.
    # The simplest: merge_asof on (club_id, date) backward, using apps as the
    # bridge to know which club a player was at on which date.
    apps_club = (
        a.sort_values(["player_id", "date"])
         [["player_id", "date", "player_club_id"]]
         .rename(columns={"player_club_id": "club_id"})
    )

    team_roll = team_roll.sort_values(["club_id", "date"])
    apps_club = apps_club.sort_values(["club_id", "date"])

    merged = pd.merge_asof(
        apps_club.sort_values("date"),
        team_roll.sort_values("date"),
        by="club_id", on="date", direction="backward",
    )
    return merged


def build_team_aggregates_per_player_date(apps: pd.DataFrame) -> pd.DataFrame:
    """Group C: team-relative shares.

    For each (player_id, date), compute the team's totals (over the past 180d
    at the player's current club) so we can derive shares.
    """
    a = apps.copy()
    a["date"] = pd.to_datetime(a["date"])
    a = a.rename(columns={"player_club_id": "club_id"})

    # Per (club_id, date) per-appearance row count + sums to compute team totals.
    # First: per game per club totals
    per_game = (
        a.groupby(["club_id", "game_id", "date"], as_index=False)
         .agg(team_goals=("goals", "sum"),
              team_assists=("assists", "sum"),
              team_minutes=("minutes_played", "sum"))
    )

    # Rolling sum per club over 180 days
    per_game = per_game.sort_values(["club_id", "date"])
    parts = []
    for cid, sub in per_game.groupby("club_id", sort=False):
        sub = sub.set_index("date")
        roll = sub[["team_goals", "team_assists", "team_minutes"]].rolling("180D").sum()
        roll = roll.rename(columns=lambda c: f"{c}_6mo")
        roll = roll.reset_index()
        roll["club_id"] = cid
        parts.append(roll)
    team_roll = pd.concat(parts, ignore_index=True)

    # apps_club: for each player-date, which club were they at?
    apps_club = (
        a.sort_values(["player_id", "date"])
         [["player_id", "date", "club_id"]]
         .drop_duplicates(subset=["player_id", "date"])
    )
    apps_club = apps_club.sort_values("date").reset_index(drop=True)
    team_roll = team_roll.sort_values("date").reset_index(drop=True)

    merged = pd.merge_asof(
        apps_club,
        team_roll,
        by="club_id", on="date", direction="backward",
    )
    return merged


def build_position_baselines_at(
    feats: pd.DataFrame,
    apps: pd.DataFrame,
    players: pd.DataFrame,
) -> pd.DataFrame:
    """Group D: per (position, calendar_year) baselines for z-scoring.

    For each feature row, attach mean/std of goals_6mo, assists_6mo, etc
    over all players in the SAME position in the SAME year.
    """
    a = apps.copy()
    a["date"] = pd.to_datetime(a["date"])
    a["season"] = a["date"].dt.year
    a = a.merge(players[["player_id", "position"]], on="player_id", how="left")

    # Per-season per-player totals
    season_tot = (
        a.groupby(["player_id", "position", "season"], as_index=False)
         .agg(season_goals=("goals", "sum"),
              season_assists=("assists", "sum"),
              season_minutes=("minutes_played", "sum"))
    )
    # Then per (position, season) baselines
    baselines = (
        season_tot.groupby(["position", "season"])
         .agg(mean_goals=("season_goals", "mean"),
              std_goals=("season_goals", "std"),
              mean_assists=("season_assists", "mean"),
              std_assists=("season_assists", "std"),
              mean_minutes=("season_minutes", "mean"),
              std_minutes=("season_minutes", "std"))
         .reset_index()
    )
    return baselines


def build_club_static(clubs: pd.DataFrame, transfers: pd.DataFrame) -> pd.DataFrame:
    """Group G: per-player career-level aggregates from transfers + club static."""
    # Per-player transfer count & last transfer info
    t = transfers.copy()
    t["transfer_date"] = pd.to_datetime(t["transfer_date"])
    # Drop weird future dates (>2026, those are contract expiries)
    t = t[t["transfer_date"] < pd.Timestamp("2027-01-01")]
    return t


# ============================ xG proxies (Group J) ============================


def build_xg_proxy_features(
    events: pd.DataFrame,
    apps: pd.DataFrame,
) -> pd.DataFrame:
    """Group J: per (player_id, date) xG-style proxies from goal events.

    chance_quality_index_6mo: average chance_weight of goals scored in past 180d
    skill_share_6mo: fraction of goals from free kicks / long distance
    conversion_efficiency_6mo: goals / (minutes/90)
    """
    from playerval.xg_proxies import classify_shot_type, CHANCE_QUALITY_WEIGHT

    g_events = events[events["type"] == "Goals"].copy()
    g_events["date"] = pd.to_datetime(g_events["date"])
    g_events["shot_type"] = g_events["description"].apply(classify_shot_type)
    g_events["chance_weight"] = g_events["shot_type"].map(CHANCE_QUALITY_WEIGHT).fillna(0.9)
    g_events["is_skill_shot"] = g_events["shot_type"].isin(["free_kick", "long_distance"]).astype(int)

    # We want at each (player_id, date) the rolling 6mo + 12mo:
    #   sum of chance_weight (call it cw_sum)
    #   count of goals (call it n_goals)
    #   sum of is_skill_shot
    g_events = g_events.rename(columns={"player_id": "player_id"})
    g_events["n_goal"] = 1

    cols = ["chance_weight", "n_goal", "is_skill_shot"]
    roll_6 = _rolling_window_sum(g_events, "date", cols, 180)
    roll_6 = roll_6.rename(columns={
        "chance_weight": "cw_sum_6mo",
        "n_goal":        "n_event_goals_6mo",
        "is_skill_shot": "n_skill_goals_6mo",
    })
    return roll_6


# ============================ Z-score helper ============================


def attach_zscores(feats: pd.DataFrame, baselines: pd.DataFrame) -> pd.DataFrame:
    """Add z-scores of goals_6mo/assists_6mo/minutes_played_6mo vs (position, year)
    baseline. Robust to baseline std=0 (returns 0)."""
    f = feats.copy()
    f["_season"] = f["date"].dt.year
    f = f.merge(baselines, left_on=["position", "_season"],
                right_on=["position", "season"], how="left")
    for kpi, mean_col, std_col, src in [
        ("goals_6mo", "mean_goals", "std_goals", "goals_6mo"),
        ("assists_6mo", "mean_assists", "std_assists", "assists_6mo"),
        ("minutes_played_6mo", "mean_minutes", "std_minutes", "minutes_played_6mo"),
    ]:
        std_safe = f[std_col].replace(0, np.nan)
        f[f"{kpi}_zscore_vs_pos"] = (f[src] - f[mean_col]) / std_safe
    f.drop(columns=["_season", "season", "mean_goals", "std_goals",
                    "mean_assists", "std_assists",
                    "mean_minutes", "std_minutes"], inplace=True, errors="ignore")
    return f


# ============================ main assembly ============================


def assemble_features(
    out_dir: Path,
    min_valuations_per_player: int = 3,
) -> pd.DataFrame:
    """Build the full feature matrix and write parquets to out_dir."""
    data = load_all()
    val, players, apps, games, transfers, club_games, events, clubs = (
        data[k] for k in ("val", "players", "apps", "games", "transfers",
                          "club_games", "events", "clubs")
    )

    # Filter to players with enough history
    counts = val.groupby("player_id").size()
    keep_ids = counts[counts >= min_valuations_per_player].index
    val = val[val["player_id"].isin(keep_ids)].copy()
    print(f"Players after min-{min_valuations_per_player} filter: {val['player_id'].nunique():,}")

    # Group A: valuation trajectory
    print("Group A: valuation trajectory...")
    A = build_valuation_traj(val)

    # Group B + E: rolling performance aggregates
    print("Group B+E: rolling performance...")
    perf = build_perf_rolling(apps, games)

    # Merge perf onto A by (player_id, nearest past date).
    # merge_asof requires the `on` column sorted GLOBALLY.
    A = A.sort_values(["date", "player_id"]).reset_index(drop=True)
    perf = perf.sort_values(["date", "player_id"]).reset_index(drop=True)
    # also drop perf rows with NaT date (can happen from empty groups)
    perf = perf[perf["date"].notna()].copy()
    A = A[A["date"].notna()].copy()
    feats = pd.merge_asof(A, perf, on="date", by="player_id", direction="backward")

    # Group F: player static (age, position, etc) — joined onto each row
    print("Group F: player static + age...")
    p = players[["player_id", "date_of_birth", "position", "sub_position",
                 "foot", "height_in_cm", "country_of_citizenship",
                 "agent_name", "international_caps", "international_goals"]].copy()
    feats = feats.merge(p, on="player_id", how="left")
    feats["age"] = (feats["date"] - feats["date_of_birth"]).dt.days / 365.25
    feats["age"] = feats["age"].clip(15, 45)
    feats["has_agent"] = feats["agent_name"].notna().astype(int)
    feats["career_stage"] = feats["age"].apply(career_stage)
    feats.drop(columns=["agent_name"], inplace=True)

    # Group I: age × position interaction
    print("Group I: age x position interaction...")
    feats["peak_age_for_position"] = feats.apply(
        lambda r: peak_age_for(r["position"], r["sub_position"]), axis=1
    )
    feats["age_minus_position_peak"] = feats["age"] - feats["peak_age_for_position"]
    feats["pre_peak"] = (feats["age_minus_position_peak"] < -2).astype(int)
    feats["post_peak"] = (feats["age_minus_position_peak"] > 2).astype(int)

    # Group H: temporal
    print("Group H: temporal...")
    feats["as_of_year"] = feats["date"].dt.year
    feats["as_of_month"] = feats["date"].dt.month
    feats["is_summer_window"] = feats["as_of_month"].between(6, 8).astype(int)
    feats["is_winter_window"] = feats["as_of_month"].isin([1, 2]).astype(int)
    first_val = feats.groupby("player_id")["date"].transform("min")
    feats["days_since_first_valuation"] = (feats["date"] - first_val).dt.days

    # Group C: team-relative shares
    print("Group C: team-relative aggregates...")
    team_agg = build_team_aggregates_per_player_date(apps)
    feats = feats.sort_values(["date", "player_id"]).reset_index(drop=True)
    team_agg = team_agg.sort_values(["date", "player_id"]).reset_index(drop=True)
    feats = pd.merge_asof(
        feats, team_agg[["player_id", "date", "team_goals_6mo",
                         "team_assists_6mo", "team_minutes_6mo", "club_id"]],
        on="date", by="player_id", direction="backward",
    )
    # Compute shares (safe denom)
    feats["goal_share_team_6mo"] = (
        feats["goals_6mo"] / feats["team_goals_6mo"].replace(0, np.nan)
    )
    feats["assist_share_team_6mo"] = (
        feats["assists_6mo"] / feats["team_assists_6mo"].replace(0, np.nan)
    )
    feats["minutes_share_team_6mo"] = (
        feats["minutes_played_6mo"] / feats["team_minutes_6mo"].replace(0, np.nan)
    )
    # Cap shares to [0, 1] for sanity (a player can't have > 100% of team)
    for c in ["goal_share_team_6mo", "assist_share_team_6mo", "minutes_share_team_6mo"]:
        feats[c] = feats[c].clip(0, 1)

    # Group D: position-relative z-scores
    print("Group D: position-relative z-scores...")
    baselines = build_position_baselines_at(feats, apps, players)
    feats = attach_zscores(feats, baselines)

    # Group J: xG proxies
    print("Group J: xG-proxy features...")
    xg_proxy = build_xg_proxy_features(events, apps)
    xg_proxy = xg_proxy.sort_values(["date", "player_id"]).reset_index(drop=True)
    feats = pd.merge_asof(
        feats, xg_proxy, on="date", by="player_id", direction="backward",
    )
    # Derived xG features
    feats["chance_quality_index_6mo"] = (
        feats["cw_sum_6mo"] / feats["n_event_goals_6mo"].replace(0, np.nan)
    )
    feats["skill_goal_share_6mo"] = (
        feats["n_skill_goals_6mo"] / feats["n_event_goals_6mo"].replace(0, np.nan)
    )
    # goals per 90 (raw efficiency)
    feats["goals_per_90_6mo"] = (
        feats["goals_6mo"] * 90.0 / feats["minutes_played_6mo"].replace(0, np.nan)
    )
    feats["assists_per_90_6mo"] = (
        feats["assists_6mo"] * 90.0 / feats["minutes_played_6mo"].replace(0, np.nan)
    )

    # Group G: career static from transfers
    print("Group G: career transfer info...")
    t = transfers.copy()
    t["transfer_date"] = pd.to_datetime(t["transfer_date"])
    t = t[t["transfer_date"] < pd.Timestamp("2027-01-01")]
    # career transfer count up to (but not including) as_of_date
    t = t.sort_values(["player_id", "transfer_date"])
    t["transfer_idx"] = t.groupby("player_id").cumcount() + 1
    # For each feats row, last transfer before date:
    feats = feats.sort_values(["date", "player_id"]).reset_index(drop=True)
    t_sorted = t.rename(columns={"transfer_date": "date"}).sort_values(
        ["date", "player_id"]).reset_index(drop=True)
    feats = pd.merge_asof(
        feats,
        t_sorted[["player_id", "date", "transfer_idx",
                  "transfer_fee", "market_value_in_eur"]].rename(
            columns={"transfer_fee": "last_transfer_fee_eur",
                     "market_value_in_eur": "last_transfer_market_value_eur",
                     "transfer_idx": "transfers_to_date"}
        ),
        on="date", by="player_id", direction="backward",
    )
    feats["transfers_to_date"] = feats["transfers_to_date"].fillna(0)

    # Target: log(next_value / current_value)
    print("Target: log-ratio to next valuation...")
    feats = feats.sort_values(["player_id", "date"])
    feats["next_value"] = feats.groupby("player_id")["market_value_in_eur"].shift(-1)
    feats["next_date"]  = feats.groupby("player_id")["date"].shift(-1)
    feats["y_horizon_days"] = (feats["next_date"] - feats["date"]).dt.days
    # safe log ratio
    valid = (feats["next_value"] > 0) & (feats["market_value_in_eur"] > 0)
    feats["y_log_ratio"] = np.where(
        valid,
        np.log(feats["next_value"].fillna(0) /
               feats["market_value_in_eur"].replace(0, np.nan)),
        np.nan,
    )

    # Drop rows with no target (last valuation per player)
    feats_with_target = feats[feats["y_log_ratio"].notna()].copy()

    print(f"Final feature rows: {len(feats_with_target):,}")
    print(f"Final feature columns: {len(feats_with_target.columns)}")

    return feats_with_target


def temporal_splits(feats: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Train: date < 2023-01-01; Val: 2023; Test: >= 2024-01-01."""
    d = feats["date"]
    train = feats[d < "2023-01-01"].copy()
    val   = feats[(d >= "2023-01-01") & (d < "2024-01-01")].copy()
    test  = feats[d >= "2024-01-01"].copy()
    return {"train": train, "val": val, "test": test}


def main() -> None:
    out_dir = Path(__file__).resolve().parent.parent.parent / "data" / "features"
    out_dir.mkdir(parents=True, exist_ok=True)

    feats = assemble_features(out_dir)
    splits = temporal_splits(feats)

    print()
    print("=" * 60)
    print("  Split sizes")
    print("=" * 60)
    for name, df in splits.items():
        n_players = df["player_id"].nunique()
        print(f"  {name:6s}  rows={len(df):>7,}  players={n_players:>6,}  "
              f"dates {df['date'].min().date()} → {df['date'].max().date()}")

    for name, df in splits.items():
        path = out_dir / f"{name}.parquet"
        df.to_parquet(path, index=False, compression="snappy")
        print(f"  Wrote {path}  ({path.stat().st_size/1e6:.1f} MB)")

    # Save the feature dictionary for reference
    cols = list(feats.columns)
    (out_dir / "feature_dict.txt").write_text(
        "\n".join(cols) + f"\n\nTotal: {len(cols)} columns\n"
    )
    print(f"  Wrote {out_dir / 'feature_dict.txt'}")


if __name__ == "__main__":
    main()
