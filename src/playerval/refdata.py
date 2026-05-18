"""Football-domain reference data used by the feature pipeline.

Encodes expert knowledge:
  - competition tier weights (a goal in Premier League ≠ a goal in MLS)
  - position-specific peak ages (CBs peak later than wingers)
  - position-specific KPI emphasis (we will surface DIFFERENT features per
    position group via position-aware z-scores)

These constants are deliberately hand-curated based on widely-accepted
football-analytics conventions, NOT learned from the data. Documenting them
here in one place makes the engineering decisions explicit and reviewable.
"""
from __future__ import annotations

# ---------------------------------------------------------------------
# Competition tier weights — used in weighted_goals_6mo etc.
# Key is competition_id (Transfermarkt code). Tier S = top weight 1.0.
# ---------------------------------------------------------------------

TIER_S = {
    # Top-5 European leagues
    "GB1": "Premier League",
    "ES1": "La Liga",
    "L1":  "Bundesliga",
    "IT1": "Serie A",
    "FR1": "Ligue 1",
    # Top UEFA continental
    "CL":  "UEFA Champions League",
}

TIER_A = {
    # Other major UEFA leagues
    "NL1": "Eredivisie",
    "PO1": "Liga Portugal",
    "BE1": "Pro League Belgium",
    "TR1": "Süper Lig",
    # Top CONMEBOL
    "AR1N": "Liga Profesional Argentina",
    "BRA1": "Brasileirao Serie A",
    # Europa
    "EL":  "Europa League",
    "UCOL": "Conference League",
}

TIER_B = {
    # Smaller European top divisions
    "GR1": "Super League Greece",
    "SC1": "Scottish Premiership",
    "RU1": "Russian Premier League",
    "UKR1": "Ukrainian Premier League",
    "DK1": "Danish Superliga",
    "SE1": "Allsvenskan",
    "NO1": "Eliteserien",
    "A1":  "Austrian Bundesliga",
    "C1":  "Swiss Super League",
    # MLS / Liga MX
    "MLS1": "Major League Soccer",
    # National-team competitions worth weighting
    "WM":  "World Cup",
    "EM":  "European Championship",
}

# Everything else falls into tier C (smaller leagues + low-tier domestic + cups)


def competition_tier(competition_id: str | None) -> str:
    """Map competition_id to a tier label in {'S','A','B','C'}."""
    if not competition_id:
        return "C"
    if competition_id in TIER_S:
        return "S"
    if competition_id in TIER_A:
        return "A"
    if competition_id in TIER_B:
        return "B"
    return "C"


TIER_WEIGHT = {"S": 1.0, "A": 0.7, "B": 0.5, "C": 0.3}


# ---------------------------------------------------------------------
# Position-specific peak ages (years).
# Source: standard football-analytics convention. Centre-backs and
# goalkeepers peak later (28-30); wingers and full-backs peak earlier
# (24-26); centre-forwards in between (26-28).
# ---------------------------------------------------------------------

POSITION_PEAK_AGE = {
    # Outfield by sub_position
    "Goalkeeper":          30,
    "Centre-Back":         29,
    "Left-Back":           26,
    "Right-Back":          26,
    "Defensive Midfield":  28,
    "Central Midfield":    27,
    "Attacking Midfield":  27,
    "Left Midfield":       26,
    "Right Midfield":      26,
    "Left Winger":         25,
    "Right Winger":        25,
    "Second Striker":      27,
    "Centre-Forward":      27,
}

# Coarse fallback by position
COARSE_POSITION_PEAK_AGE = {
    "Goalkeeper": 30,
    "Defender":   28,
    "Midfield":   27,
    "Attack":     26,
    "Missing":    27,
}


def peak_age_for(position: str | None, sub_position: str | None) -> float:
    if sub_position and sub_position in POSITION_PEAK_AGE:
        return float(POSITION_PEAK_AGE[sub_position])
    if position and position in COARSE_POSITION_PEAK_AGE:
        return float(COARSE_POSITION_PEAK_AGE[position])
    return 27.0


# ---------------------------------------------------------------------
# Position groups — used to compute per-group z-scores and to expose
# "what KPI matters for this position".
# ---------------------------------------------------------------------

POSITION_GROUP = {
    "Goalkeeper": "GK",
    "Defender":   "DEF",
    "Midfield":   "MID",
    "Attack":     "ATT",
}


# ---------------------------------------------------------------------
# Career-stage buckets — strictly age-based, applied AFTER computing
# age. The 'expected' direction is monotonic over stages:
#  emerging:  value usually grows
#  prime:     value plateaus around peak
#  aging:     value declines slowly
#  veteran:   value drops sharply
# ---------------------------------------------------------------------

def career_stage(age: float) -> str:
    if age < 22:
        return "emerging"
    if age < 29:
        return "prime"
    if age < 32:
        return "aging"
    return "veteran"
