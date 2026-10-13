"""
Opponent defense adjustment.
Computes average fantasy points allowed per game by each defense, by position,
directly from the already-loaded nfl_data_py weekly stats DataFrame.
"""

import pandas as pd

SKILL_POSITIONS = {"QB", "RB", "WR", "TE"}


def compute_defense_rankings(weekly_df: pd.DataFrame, scoring: dict) -> pd.DataFrame:
    """
    Compute average fantasy points allowed per game, grouped by (defteam, position).

    Rank 1 = fewest points allowed (best defense for that position).
    Rank 32 = most points allowed (easiest matchup).

    Returns DataFrame with columns:
        defteam, position, pts_allowed_pg, rank (1–32), rank_label, league_avg
    """
    df = weekly_df[
        weekly_df["position"].isin(SKILL_POSITIONS)
        & weekly_df["opponent_team"].notna()
        & (weekly_df["opponent_team"] != "")
    ].copy()

    if df.empty:
        return pd.DataFrame()

    # Choose scoring column
    rec = scoring.get("rec", 0.5)
    if rec >= 1.0:
        fp_col = "fantasy_points_ppr"
    else:
        fp_col = "fantasy_points"

    if fp_col not in df.columns:
        fp_col = [c for c in df.columns if "fantasy_points" in c][0]

    df[fp_col] = pd.to_numeric(df[fp_col], errors="coerce").fillna(0)

    # Sum per (opponent, position, week), then average across weeks
    weekly_allowed = (
        df.groupby(["opponent_team", "position", "week"])[fp_col]
        .sum()
        .reset_index()
    )
    avg_allowed = (
        weekly_allowed.groupby(["opponent_team", "position"])[fp_col]
        .mean()
        .reset_index()
        .rename(columns={fp_col: "pts_allowed_pg", "opponent_team": "defteam"})
    )

    # Add league average per position
    pos_avg = avg_allowed.groupby("position")["pts_allowed_pg"].mean().rename("league_avg")
    avg_allowed = avg_allowed.join(pos_avg, on="position")

    # Rank within position (1 = stingiest = toughest matchup)
    avg_allowed["rank"] = (
        avg_allowed.groupby("position")["pts_allowed_pg"]
        .rank(ascending=True, method="min")
        .astype(int)
    )
    n_teams = avg_allowed.groupby("position")["defteam"].transform("count")
    avg_allowed["rank_pct"] = avg_allowed["rank"] / n_teams

    avg_allowed["rank_label"] = avg_allowed["rank_pct"].apply(
        lambda p: "Tough (top 25%)" if p <= 0.25
        else ("Average" if p <= 0.75 else "Favorable (bottom 25%)")
    )

    return avg_allowed.reset_index(drop=True)


def get_matchup_info(opponent_team: str, position: str,
                     defense_df: pd.DataFrame) -> dict:
    """
    Return matchup context for a given opponent and position.
    multiplier > 1.0 = favorable (weak D), < 1.0 = tough (strong D).
    """
    empty = {"multiplier": 1.0, "rank": None, "rank_label": "Unknown",
             "pts_allowed": None, "league_avg": None}

    if defense_df is None or defense_df.empty or not opponent_team:
        return empty

    row = defense_df[
        (defense_df["defteam"] == opponent_team)
        & (defense_df["position"] == position)
    ]
    if row.empty:
        return empty

    r = row.iloc[0]
    pts = float(r["pts_allowed_pg"])
    avg = float(r["league_avg"])
    multiplier = pts / avg if avg > 0 else 1.0

    return {
        "multiplier":  round(multiplier, 3),
        "rank":        int(r["rank"]),
        "rank_label":  r["rank_label"],
        "pts_allowed": round(pts, 1),
        "league_avg":  round(avg, 1),
    }
