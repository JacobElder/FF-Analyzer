"""
Opponent defense adjustment.
Average fantasy points allowed per game by each defense, by position, computed
from the weekly stats frame using the league's own scoring (`fp` column).
"""

import pandas as pd

SKILL_POSITIONS = {"QB", "RB", "WR", "TE"}

EMPTY_MATCHUP = {"multiplier": 1.0, "rank": None, "rank_label": "Unknown",
                 "pts_allowed": None, "league_avg": None}


def compute_defense_rankings(weekly_df: pd.DataFrame) -> pd.DataFrame:
    """
    Rank 1 = fewest points allowed (toughest matchup), rank 32 = most (easiest).

    Returns columns: defteam, position, pts_allowed_pg, league_avg, rank, rank_pct, rank_label
    """
    if weekly_df.empty or "fp" not in weekly_df.columns:
        return pd.DataFrame()

    df = weekly_df[
        weekly_df["position"].isin(SKILL_POSITIONS)
        & weekly_df["opponent_team"].notna()
        & (weekly_df["opponent_team"] != "")
    ]
    if df.empty:
        return pd.DataFrame()

    # Sum per (opponent, position, week), then average across weeks
    weekly_allowed = df.groupby(["opponent_team", "position", "week"])["fp"].sum().reset_index()
    avg_allowed = (
        weekly_allowed.groupby(["opponent_team", "position"])["fp"]
        .mean()
        .reset_index()
        .rename(columns={"fp": "pts_allowed_pg", "opponent_team": "defteam"})
    )

    pos_avg = avg_allowed.groupby("position")["pts_allowed_pg"].mean().rename("league_avg")
    avg_allowed = avg_allowed.join(pos_avg, on="position")

    avg_allowed["rank"] = (
        avg_allowed.groupby("position")["pts_allowed_pg"].rank(ascending=True, method="min").astype(int)
    )
    n_teams = avg_allowed.groupby("position")["defteam"].transform("count")
    avg_allowed["rank_pct"] = avg_allowed["rank"] / n_teams
    avg_allowed["rank_label"] = avg_allowed["rank_pct"].apply(
        lambda p: "Tough (top 25%)" if p <= 0.25 else ("Average" if p <= 0.75 else "Favorable (bottom 25%)")
    )
    return avg_allowed.reset_index(drop=True)


def build_matchup_table(defense_df: pd.DataFrame) -> dict:
    """(defteam, position) → matchup info dict, for O(1) lookups."""
    table = {}
    if defense_df is None or defense_df.empty:
        return table
    for r in defense_df.itertuples(index=False):
        avg = float(r.league_avg)
        pts = float(r.pts_allowed_pg)
        table[(r.defteam, r.position)] = {
            "multiplier": round(pts / avg, 3) if avg > 0 else 1.0,
            "rank": int(r.rank),
            "rank_label": r.rank_label,
            "pts_allowed": round(pts, 1),
            "league_avg": round(avg, 1),
        }
    return table


def get_matchup_info(opponent_team: str, position: str, matchup_table: dict) -> dict:
    """
    Matchup context for an opponent and position.
    multiplier > 1.0 = favorable (weak D), < 1.0 = tough (strong D).
    """
    if not matchup_table or not opponent_team:
        return dict(EMPTY_MATCHUP)
    return dict(matchup_table.get((opponent_team, position), EMPTY_MATCHUP))
