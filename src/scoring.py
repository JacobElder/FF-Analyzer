"""
League scoring.

Fantasy points are computed from raw nflverse box-score columns using the
league's Sleeper-style scoring keys, so 6-pt passing TDs, TE premium, etc. are
honored. Everything is vectorized over the whole DataFrame (no row-wise apply).
"""

import pandas as pd

# Sleeper scoring keys. League settings are layered on top of these.
DEFAULT_SCORING = {
    "pass_yd": 0.04,
    "pass_td": 4.0,
    "pass_int": -2.0,
    "pass_2pt": 2.0,
    "rush_yd": 0.1,
    "rush_td": 6.0,
    "rush_2pt": 2.0,
    "rec": 0.5,
    "rec_yd": 0.1,
    "rec_td": 6.0,
    "rec_2pt": 2.0,
    "fum_lost": -2.0,
    "st_td": 6.0,
    "fgm_0_19": 3.0,
    "fgm_20_29": 3.0,
    "fgm_30_39": 3.0,
    "fgm_40_49": 4.0,
    "fgm_50p": 5.0,
    "fgmiss": -1.0,
    "xpm": 1.0,
    "xpmiss": -1.0,
}

# (scoring key, nflverse column(s)) — multiple columns are summed.
_LINEAR_TERMS = [
    ("pass_yd",   ["passing_yards"]),
    ("pass_td",   ["passing_tds"]),
    ("pass_int",  ["passing_interceptions", "interceptions"]),  # name changed across releases
    ("pass_2pt",  ["passing_2pt_conversions"]),
    ("rush_yd",   ["rushing_yards"]),
    ("rush_td",   ["rushing_tds"]),
    ("rush_2pt",  ["rushing_2pt_conversions"]),
    ("rec",       ["receptions"]),
    ("rec_yd",    ["receiving_yards"]),
    ("rec_td",    ["receiving_tds"]),
    ("rec_2pt",   ["receiving_2pt_conversions"]),
    ("fum_lost",  ["rushing_fumbles_lost", "receiving_fumbles_lost", "sack_fumbles_lost"]),
    ("st_td",     ["special_teams_tds"]),
    ("fgm_0_19",  ["fg_made_0_19"]),
    ("fgm_20_29", ["fg_made_20_29"]),
    ("fgm_30_39", ["fg_made_30_39"]),
    ("fgm_40_49", ["fg_made_40_49"]),
    ("fgmiss",    ["fg_missed"]),
    ("xpm",       ["pat_made"]),
    ("xpmiss",    ["pat_missed"]),
]

_POSITION_REC_BONUS = {"RB": "bonus_rec_rb", "WR": "bonus_rec_wr", "TE": "bonus_rec_te"}


def effective_scoring(league_scoring: dict | None) -> dict:
    """League scoring layered over defaults (keys the league omits keep defaults)."""
    merged = dict(DEFAULT_SCORING)
    for k, v in (league_scoring or {}).items():
        try:
            merged[k] = float(v)
        except (TypeError, ValueError):
            continue
    return merged


def scoring_key(scoring: dict) -> tuple:
    """Hashable, order-independent representation (for cache keys)."""
    return tuple(sorted((k, float(v)) for k, v in scoring.items()))


def scoring_label(scoring: dict) -> str:
    rec = scoring.get("rec", 0)
    if rec >= 1:
        return "PPR"
    if rec >= 0.5:
        return "Half-PPR"
    return "Standard"


def _col(df: pd.DataFrame, names: list[str]) -> pd.Series:
    for name in names:
        if name in df.columns:
            return pd.to_numeric(df[name], errors="coerce").fillna(0.0)
    return pd.Series(0.0, index=df.index)


def compute_fantasy_points(df: pd.DataFrame, scoring: dict) -> pd.Series:
    """Fantasy points for every row of an nflverse weekly player-stats frame."""
    pts = pd.Series(0.0, index=df.index)
    for key, cols in _LINEAR_TERMS:
        weight = scoring.get(key, 0.0)
        if not weight:
            continue
        if key == "fum_lost":
            total = sum(_col(df, [c]) for c in cols)
        else:
            total = _col(df, cols)
        pts += weight * total

    # 50+ yard field goals: Sleeper may split 50-59 / 60+, or use a single 50p key
    fg50 = _col(df, ["fg_made_50_59"])
    fg60 = _col(df, ["fg_made_60_"])
    w50 = scoring.get("fgm_50_59", scoring.get("fgm_50p", 0.0))
    w60 = scoring.get("fgm_60p", scoring.get("fgm_50p", 0.0))
    pts += w50 * fg50 + w60 * fg60

    # Position-specific reception bonuses (e.g. TE premium)
    if "position" in df.columns:
        receptions = _col(df, ["receptions"])
        for pos, key in _POSITION_REC_BONUS.items():
            bonus = scoring.get(key, 0.0)
            if bonus:
                pts += bonus * receptions * (df["position"] == pos)

    return pts.round(2)
