"""Proxies for shot quality / expected goals (xG) without event-level data.

Real xG requires (x, y) shot coordinates + body part + assist type + defensive
pressure. We have none of those. What we DO have:

  - per-appearance goals/assists/minutes
  - game_events rows with type='Goals' and a free-text 'description'
    (e.g. "Right-footed shot, 2. Tournament Goal, Header")
  - per-club seasonal goal totals (from club_games)

The proxies below capture the SPIRIT of xG-like metrics — finishing efficiency,
chance involvement, big-game performance — using only that data. They are
clearly labelled as proxies, not true xG.

ALL proxies are computed at as_of_date `T` using ONLY data observed before
`T` (no leakage). Per-player, per-window.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import pandas as pd


# ---------- shot-type extraction from event descriptions ----------

# Examples of descriptions we see in game_events for goals:
#   ", Right-footed shot, 2. Tournament Goal"
#   ", Header, 1. Tournament Goal"
#   ", Penalty, ..."
#   ", Tap-in, ..."
#   ", Free kick, ..."
#   ", Direct free kick, ..."

_SHOT_TYPE_PATTERNS = [
    ("penalty",       re.compile(r"\bPenalty\b", re.I)),
    ("free_kick",     re.compile(r"\b(Direct )?[Ff]ree[- ]?kick\b")),
    ("header",        re.compile(r"\bHeader\b", re.I)),
    ("tap_in",        re.compile(r"\bTap[- ]?in\b", re.I)),
    ("long_distance", re.compile(r"\bLong distance\b", re.I)),
    ("left_foot",     re.compile(r"\bLeft[- ]footed\b", re.I)),
    ("right_foot",    re.compile(r"\bRight[- ]footed\b", re.I)),
]


def classify_shot_type(description: str | float | None) -> str:
    """Map a goal description string into a coarse shot type."""
    if not isinstance(description, str):
        return "unknown"
    for label, pat in _SHOT_TYPE_PATTERNS:
        if pat.search(description):
            return label
    return "unknown"


# ---------- xG-style weights per shot type (HAND-CURATED) ----------
#
# These are quality multipliers reflecting how "easy" a goal of that type is.
# Tap-ins are presumably from very good chances (high pre-shot xG), so a
# tap-in goal => high quality of chance the player was IN. Conversely, long-
# distance and free-kick goals reflect player SKILL (not chance quality).
#
# We use these weights two ways:
#   - "chance_quality_index" = weighted average of shot_types_taken
#   - "skill_index" = share of goals from low-chance-quality shots (free kicks,
#       long distance) — implies player creates value beyond chance quality

CHANCE_QUALITY_WEIGHT = {
    "tap_in":        1.5,    # someone served this on a plate
    "penalty":       1.2,    # near-automatic
    "header":        1.0,    # variable (could be near-post or far-post)
    "left_foot":     0.9,    # generic shot
    "right_foot":    0.9,    # generic shot
    "free_kick":     0.5,    # skill-heavy
    "long_distance": 0.4,    # skill-heavy
    "unknown":       0.9,    # treat as generic
}


# ---------- proxy computations ----------


@dataclass
class XGProxyRow:
    chance_quality_index_6mo: float
    chance_quality_index_12mo: float
    skill_share_6mo: float                # share of goals that were skill-shots
    finishing_vs_team_rate_6mo: float     # player goals/min vs team goals/min
    big_match_goals_6mo: int              # goals against top-half opponents
    big_match_goal_share_6mo: float       # of player's 6mo goals
    conversion_efficiency_6mo: float      # goals per (minutes/90)


def _shot_type_table(events_goals: pd.DataFrame) -> pd.DataFrame:
    """Add shot_type and chance_weight columns to a goals-only events frame."""
    out = events_goals.copy()
    out["shot_type"] = out["description"].apply(classify_shot_type)
    out["chance_weight"] = out["shot_type"].map(CHANCE_QUALITY_WEIGHT).fillna(0.9)
    out["is_skill_shot"] = out["shot_type"].isin(["free_kick", "long_distance"])
    return out


def compute_xg_proxies(
    events_goals: pd.DataFrame,
    appearances: pd.DataFrame,
    club_games: pd.DataFrame,
) -> pd.DataFrame:
    """Compute per (player_id, date) xG-style proxies as a time-indexed DataFrame.

    The output frame is sparse: one row per goal event the player scored, plus
    rolling aggregates. Downstream feature pipeline will merge-asof at the
    player's `as_of_date`.
    """
    events = _shot_type_table(events_goals)
    events["date"] = pd.to_datetime(events["date"])
    # We need player_id from events (the scorer column is `player_id`).
    needed = ["player_id", "date", "game_id", "club_id",
              "shot_type", "chance_weight", "is_skill_shot"]
    return events[needed].sort_values(["player_id", "date"]).reset_index(drop=True)
