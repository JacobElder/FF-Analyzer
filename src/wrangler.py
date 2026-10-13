import os
import time
import pickle
from datetime import date
import pandas as pd

CACHE_DIR = "data"
CACHE_TTL = 86400  # 24 hours


def current_nfl_season() -> int:
    """
    Return the most recently completed (or in-progress) NFL season year.
    NFL seasons are labeled by the year they START (e.g. 2025 season = Sept 2025–Feb 2026).
    Sep–Dec: current calendar year is the active season.
    Jan–Aug: previous calendar year was the most recent season.
    """
    today = date.today()
    return today.year if today.month >= 9 else today.year - 1


def _cache_path(year):
    return os.path.join(CACHE_DIR, f"weekly_stats_{year}.pkl")


NFLVERSE_WEEKLY_URL = "https://github.com/nflverse/nflverse-data/releases/download/stats_player/stats_player_week_{year}.parquet"


def fetch_weekly_stats(year: int | None = None) -> pd.DataFrame:
    """
    Fetch nflverse weekly player stats with 24h file cache.
    Defaults to current NFL season; falls back to previous season if data unavailable.
    """
    if year is None:
        year = current_nfl_season()
    os.makedirs(CACHE_DIR, exist_ok=True)

    def _load(yr: int) -> pd.DataFrame:
        path = _cache_path(yr)
        if os.path.exists(path):
            age = time.time() - os.path.getmtime(path)
            if age < CACHE_TTL:
                with open(path, "rb") as f:
                    return pickle.load(f)
        try:
            df = pd.read_parquet(NFLVERSE_WEEKLY_URL.format(year=yr), engine="auto")
        except Exception as e:
            print(f"nflverse: no weekly data for {yr} ({e})")
            return pd.DataFrame()
        if not df.empty:
            with open(path, "wb") as f:
                pickle.dump(df, f)
        return df

    df = _load(year)
    if df.empty:
        # Season year exists but nfl_data_py has no data yet — use prior season
        df = _load(year - 1)
    return df


def _pick_fp_col(df: pd.DataFrame, scoring: dict) -> str:
    """Return the fantasy_points column name appropriate for the scoring settings."""
    rec = scoring.get("rec", 0.5)
    if rec >= 1.0:
        return "fantasy_points_ppr"
    elif rec <= 0.0:
        return "fantasy_points"
    else:
        # Half-PPR or custom: interpolate between standard and PPR
        return "_fp_half"  # signal to caller to compute


def _compute_fp(row, scoring: dict) -> float:
    """Compute fantasy points for a single row using scoring settings."""
    rec = scoring.get("rec", 0.5)
    std = row.get("fantasy_points", 0) or 0
    ppr = row.get("fantasy_points_ppr", 0) or 0
    if rec >= 1.0:
        return ppr
    elif rec <= 0.0:
        return std
    else:
        # interpolate: standard + rec * receptions
        receptions = row.get("receptions", 0) or 0
        return std + rec * receptions


def _fuzzy_match_name(target: str, candidates: list) -> str | None:
    """Simple fuzzy match: exact → last-name → contains."""
    target_lower = target.lower()
    # Exact
    for c in candidates:
        if c.lower() == target_lower:
            return c
    # Last name match
    target_last = target_lower.split()[-1]
    matches = [c for c in candidates if c.lower().split()[-1] == target_last]
    if len(matches) == 1:
        return matches[0]
    # Contains
    contains = [c for c in candidates if target_last in c.lower()]
    if len(contains) == 1:
        return contains[0]
    return None


def get_player_stats(name: str, df: pd.DataFrame, scoring: dict, recent_weeks: int = 4) -> dict:
    """Return season and recent stats dict for a player name."""
    col_names = df["player_display_name"].dropna().unique().tolist()
    matched = _fuzzy_match_name(name, col_names)
    if not matched:
        return {"season": {}, "recent": {}, "weekly": []}

    player_df = df[df["player_display_name"] == matched].copy()
    player_df = player_df.sort_values("week")

    fp_col = _pick_fp_col(player_df, scoring)
    if fp_col == "_fp_half":
        player_df["_fp"] = player_df.apply(lambda r: _compute_fp(r.to_dict(), scoring), axis=1)
        fp_col = "_fp"

    all_pts = player_df[fp_col].fillna(0).tolist()
    games_played = len([p for p in all_pts if p > 0])
    season_ppg = round(sum(all_pts) / games_played, 2) if games_played else 0

    recent_pts = all_pts[-recent_weeks:]
    recent_games = len([p for p in recent_pts if p > 0])
    recent_ppg = round(sum(recent_pts) / recent_games, 2) if recent_games else 0

    return {
        "season": {"ppg": season_ppg, "total": round(sum(all_pts), 1), "games": games_played},
        "recent": {"ppg": recent_ppg, "weeks": recent_weeks},
        "weekly": [round(p, 1) for p in all_pts],
    }


def get_usage_trends(name: str, df: pd.DataFrame, recent_weeks: int = 4) -> dict:
    """
    Compute target share and air yards share trends for a player.
    Returns recent averages and direction vs. season average.
    trend > 0 = rising usage, < 0 = falling.
    """
    col_names = df["player_display_name"].dropna().unique().tolist()
    matched = _fuzzy_match_name(name, col_names)
    if not matched:
        return {}

    pdf = df[df["player_display_name"] == matched].sort_values("week").copy()

    result = {}
    for col, label in [("target_share", "target_share"), ("air_yards_share", "air_yards_share"),
                        ("wopr", "wopr"), ("racr", "racr")]:
        if col not in pdf.columns:
            continue
        vals = pdf[col].fillna(0).tolist()
        if not vals:
            continue
        season_avg = round(sum(vals) / len(vals), 3)
        recent = vals[-recent_weeks:]
        recent_avg = round(sum(recent) / len(recent), 3) if recent else 0.0
        result[label] = {
            "season_avg": season_avg,
            "recent_avg": recent_avg,
            "trend": round(recent_avg - season_avg, 3),
        }

    # For RBs: carry share via carries / team_carries proxy using rush_attempts
    if "carries" in pdf.columns and not pdf["carries"].isna().all():
        carries = pdf["carries"].fillna(0).tolist()
        recent_c = carries[-recent_weeks:]
        result["recent_carries_pg"] = round(sum(recent_c) / max(len(recent_c), 1), 1)

    return result


def rank_available_players(players: list, df: pd.DataFrame, scoring: dict) -> list:
    """
    Bulk-join nfl_data_py stats onto list of player dicts.
    Each player dict needs at least 'name'. Adds season_ppg and recent_ppg.
    """
    # Build a lookup: display_name -> aggregated stats
    fp_col = _pick_fp_col(df, scoring)
    work_df = df.copy()
    if fp_col == "_fp_half":
        work_df["_fp"] = work_df.apply(lambda r: _compute_fp(r.to_dict(), scoring), axis=1)
        fp_col = "_fp"

    grouped = work_df.groupby("player_display_name")[fp_col].apply(list).to_dict()
    col_names = list(grouped.keys())

    enriched = []
    for p in players:
        name = p.get("name", "")
        matched = _fuzzy_match_name(name, col_names)
        pts = grouped.get(matched, []) if matched else []
        pts = [x for x in pts if x is not None]
        games = len([x for x in pts if x > 0])
        season_ppg = round(sum(pts) / games, 2) if games else 0.0
        recent_pts = pts[-4:]
        recent_games = len([x for x in recent_pts if x > 0])
        recent_ppg = round(sum(recent_pts) / recent_games, 2) if recent_games else 0.0
        enriched.append({**p, "season_ppg": season_ppg, "recent_ppg": recent_ppg})

    return enriched
