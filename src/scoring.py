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

# Sleeper scoring key → nflverse columns. Each entry lists alternatives tried in
# order (column names change across nflverse releases); the first alternative with
# any column present is used and its columns are summed. Absent columns count as 0.
# Every key is scored independently, exactly like score_projection: a league that
# sets both an aggregate key (fgmiss) and a split (fgmiss_30_39) gets both, as in Sleeper.
_LINEAR_TERMS = [
    ("pass_yd",     [["passing_yards"]]),
    ("pass_td",     [["passing_tds"]]),
    ("pass_int",    [["passing_interceptions"], ["interceptions"]]),
    ("pass_2pt",    [["passing_2pt_conversions"]]),
    ("pass_fd",     [["passing_first_downs"]]),
    ("pass_sack",   [["sacks_suffered"], ["sacks"]]),
    ("pass_cmp",    [["completions"]]),
    ("pass_att",    [["attempts"]]),
    ("rush_yd",     [["rushing_yards"]]),
    ("rush_td",     [["rushing_tds"]]),
    ("rush_2pt",    [["rushing_2pt_conversions"]]),
    ("rush_fd",     [["rushing_first_downs"]]),
    ("rush_att",    [["carries"]]),
    ("rec",         [["receptions"]]),
    ("rec_yd",      [["receiving_yards"]]),
    ("rec_td",      [["receiving_tds"]]),
    ("rec_2pt",     [["receiving_2pt_conversions"]]),
    ("rec_fd",      [["receiving_first_downs"]]),
    ("rec_tgt",     [["targets"]]),
    ("fum",         [["fumbles_total"], ["rushing_fumbles", "receiving_fumbles", "sack_fumbles"]]),
    ("fum_lost",    [["rushing_fumbles_lost", "receiving_fumbles_lost", "sack_fumbles_lost"]]),
    ("st_td",       [["special_teams_tds"]]),
    ("fgm",         [["fg_made"]]),
    ("fgm_yds",     [["fg_made_distance"]]),
    ("fgm_0_19",    [["fg_made_0_19"]]),
    ("fgm_20_29",   [["fg_made_20_29"]]),
    ("fgm_30_39",   [["fg_made_30_39"]]),
    ("fgm_40_49",   [["fg_made_40_49"]]),
    ("fgm_50p",     [["fg_made_50_59", "fg_made_60_"]]),
    ("fgm_50_59",   [["fg_made_50_59"]]),
    ("fgm_60p",     [["fg_made_60_"]]),
    ("fgmiss",      [["fg_missed"]]),
    ("fgmiss_0_19", [["fg_missed_0_19"]]),
    ("fgmiss_20_29", [["fg_missed_20_29"]]),
    ("fgmiss_30_39", [["fg_missed_30_39"]]),
    ("fgmiss_40_49", [["fg_missed_40_49"]]),
    ("fgmiss_50p",  [["fg_missed_50_59", "fg_missed_60_"]]),
    ("fgmiss_50_59", [["fg_missed_50_59"]]),
    ("fgmiss_60p",  [["fg_missed_60_"]]),
    ("xpm",         [["pat_made"]]),
    ("xpmiss",      [["pat_missed"]]),
]

_POSITION_REC_BONUS = {"RB": "bonus_rec_rb", "WR": "bonus_rec_wr", "TE": "bonus_rec_te"}
# Per-position first-down bonuses, applied to rushing + receiving first downs
_POSITION_FD_BONUS = {"QB": "bonus_fd_qb", "RB": "bonus_fd_rb", "WR": "bonus_fd_wr", "TE": "bonus_fd_te"}

# Default keys that a league's own split keys replace: a league that scores 50-59 / 60+
# field goals separately has no single 50+ key, so the default one must not stack on top.
_SPLIT_REPLACES_DEFAULT = {
    "fgm_50p": ("fgm_50_59", "fgm_60p"),
    "fgmiss": ("fgmiss_0_19", "fgmiss_20_29", "fgmiss_30_39", "fgmiss_40_49",
               "fgmiss_50p", "fgmiss_50_59", "fgmiss_60p"),
}


def effective_scoring(league_scoring: dict | None) -> dict:
    """
    League scoring layered over defaults (keys the league omits keep defaults),
    except that a league defining only split kicker keys (fgm_50_59/fgm_60p,
    fgmiss_*) drops the matching default aggregate so it isn't counted twice.
    """
    merged = dict(DEFAULT_SCORING)
    league = {}
    for k, v in (league_scoring or {}).items():
        try:
            f = float(v)
        except (TypeError, ValueError):
            continue
        if f == f and abs(f) != float("inf"):  # skip NaN/inf, which would poison every score
            league[k] = f
    for default_key, splits in _SPLIT_REPLACES_DEFAULT.items():
        if default_key not in league and any(k in league for k in splits):
            merged.pop(default_key)
    merged.update(league)
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


def _term(df: pd.DataFrame, alternatives: list[list[str]]) -> pd.Series:
    """Sum of the first alternative's columns that has any of them present (else 0)."""
    for cols in alternatives:
        if any(c in df.columns for c in cols):
            return sum(_col(df, [c]) for c in cols)
    return pd.Series(0.0, index=df.index)


def compute_fantasy_points(df: pd.DataFrame, scoring: dict) -> pd.Series:
    """Fantasy points for every row of an nflverse weekly player-stats frame."""
    pts = pd.Series(0.0, index=df.index)
    for key, alternatives in _LINEAR_TERMS:
        weight = scoring.get(key, 0.0)
        if weight:
            pts += weight * _term(df, alternatives)

    # Incompletions are not a column of their own
    weight = scoring.get("pass_inc", 0.0)
    if weight:
        pts += weight * (_col(df, ["attempts"]) - _col(df, ["completions"])).clip(lower=0)

    # Position-specific reception bonuses (e.g. TE premium)
    if "position" in df.columns:
        receptions = _col(df, ["receptions"])
        for pos, key in _POSITION_REC_BONUS.items():
            bonus = scoring.get(key, 0.0)
            if bonus:
                pts += bonus * receptions * (df["position"] == pos)
        first_downs = _col(df, ["rushing_first_downs"]) + _col(df, ["receiving_first_downs"])
        for pos, key in _POSITION_FD_BONUS.items():
            bonus = scoring.get(key, 0.0)
            if bonus:
                pts += bonus * first_downs * (df["position"] == pos)

    return pts.round(2)
