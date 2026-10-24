"""
NFL weekly stats (nflverse) and the per-player stats index.

- Regular-season games only (playoff weeks are dropped at load).
- Team abbreviations normalized to Sleeper's (LA → LAR).
- Players keyed by nflverse player_id (= Sleeper gsis_id), with a strict
  normalized-name fallback (see src/players.py).
- Every game a player recorded stats in counts as a game, including 0-point
  and negative games. "Recent" means the last N NFL weeks, not the last N rows.
"""

import logging
from datetime import date

import numpy as np
import pandas as pd

from src.players import normalize_name, norm_team
from src.storage import data_path, is_fresh, write_parquet_atomic

log = logging.getLogger(__name__)

CACHE_TTL = 86400  # 24 hours
RECENT_WEEKS = 6   # "recent form" window; backtested (see src/backtest.py)
NFLVERSE_WEEKLY_URL = "https://github.com/nflverse/nflverse-data/releases/download/stats_player/stats_player_week_{year}.parquet"


def current_nfl_season() -> int:
    """
    The most recently started NFL season. Seasons are labeled by the year they
    start (2025 season = Sept 2025–Feb 2026), so Jan–Aug maps to last year.
    """
    today = date.today()
    return today.year if today.month >= 9 else today.year - 1


def _prepare(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    if "season_type" in df.columns:
        df = df[df["season_type"] == "REG"]
    df = df.copy()
    for col in ("team", "opponent_team"):
        if col in df.columns:
            df[col] = df[col].map(norm_team)
    return df.sort_values(["player_id", "week"]).reset_index(drop=True)


def _load_season(year: int, force: bool) -> pd.DataFrame:
    path = data_path(f"weekly_stats_{year}.parquet")
    if not force and is_fresh(path, CACHE_TTL):
        return pd.read_parquet(path)
    try:
        df = pd.read_parquet(NFLVERSE_WEEKLY_URL.format(year=year))
        if not df.empty:
            write_parquet_atomic(df, path)
        return df
    except Exception as e:
        log.warning("nflverse: no weekly data for %s (%s)", year, e)
        if path.exists():  # stale cache beats nothing
            log.warning("nflverse: using stale cache for %s", year)
            return pd.read_parquet(path)
        return pd.DataFrame()


def fetch_weekly_stats(year: int | None = None, force: bool = False) -> pd.DataFrame:
    """
    Regular-season weekly player stats for `year` (default: current season),
    falling back to the previous season if the current one has no data yet.
    """
    if year is None:
        year = current_nfl_season()
    df = _prepare(_load_season(year, force))
    if df.empty:
        df = _prepare(_load_season(year - 1, force))
    return df


# ── Stats index ───────────────────────────────────────────────────────────────

_EMPTY_STATS = {
    "season": {"ppg": 0.0, "total": 0.0, "games": 0},
    "recent": {"ppg": None, "weeks": 0, "games": 0},
    "weekly": [],
    "weeks": [],
    "found": False,
}

_USAGE_COLS = ["target_share", "air_yards_share", "wopr", "racr"]


_POSITION_GROUPS = {"FB": "RB", "HB": "RB"}


def _same_position(indexed, wanted) -> bool:
    """Positions match (FB/HB count as RB). A missing indexed position can't rule a match out."""
    if not isinstance(indexed, str) or not indexed:
        return True
    norm = lambda p: _POSITION_GROUPS.get(p.upper(), p.upper())
    return norm(indexed) == norm(str(wanted))


class StatsIndex:
    """
    Pre-grouped per-player stats built once per (stats data, scoring) pair.
    `df` must already have an `fp` column (see src.scoring.compute_fantasy_points).
    """

    def __init__(self, df: pd.DataFrame):
        if not df.empty:
            # Row lookups below are positional, and "latest team" means the last week
            df = df.sort_values(["player_id", "week"], kind="stable").reset_index(drop=True)
        self.df = df
        self.season = int(df["season"].iloc[0]) if not df.empty and "season" in df.columns else None
        self.last_week = int(df["week"].max()) if not df.empty else 0
        self.by_id: dict[str, dict] = {}
        self.by_name: dict[str, list[str]] = {}
        if df.empty:
            return

        weeks = df["week"].to_numpy()
        fps = df["fp"].to_numpy(dtype=float)
        for pid, rows in df.groupby("player_id", sort=False).indices.items():
            last = rows[-1]
            name = df.at[last, "player_display_name"]
            self.by_id[pid] = {
                "name": name,
                "position": df.at[last, "position"],
                "team": df.at[last, "team"],
                "rows": rows,
                "weeks": weeks[rows].tolist(),
                "pts": fps[rows].tolist(),
            }
            self.by_name.setdefault(normalize_name(name), []).append(pid)

    # -- lookup -----------------------------------------------------------------

    def resolve(self, player) -> str | None:
        """nflverse player_id for a player dict (or bare name), or None."""
        if isinstance(player, str):
            player = {"name": player}
        gid = player.get("gsis_id")
        if gid and gid in self.by_id:
            return gid
        ids = self.by_name.get(normalize_name(player.get("name", "")), [])
        pos, team = player.get("position"), player.get("team")
        if pos:
            # A name match at another position is a different person (WR vs CB namesakes)
            ids = [i for i in ids if _same_position(self.by_id[i]["position"], pos)]
        if len(ids) > 1 and team:
            ids = [i for i in ids if self.by_id[i]["team"] == team] or ids
        if gid and team and len(ids) == 1 and self.by_id[ids[0]]["team"] != team:
            return None  # has its own ID that isn't in the stats, and the namesake plays elsewhere
        return ids[0] if len(ids) == 1 else None

    def get_player_stats(self, player, recent_weeks: int = RECENT_WEEKS) -> dict:
        pid = self.resolve(player)
        if not pid:
            return {k: (dict(v) if isinstance(v, dict) else list(v) if isinstance(v, list) else v)
                    for k, v in _EMPTY_STATS.items()}
        entry = self.by_id[pid]
        pts, wks = entry["pts"], entry["weeks"]
        games = len(pts)
        total = sum(pts)

        cutoff = self.last_week - recent_weeks
        recent = [p for p, w in zip(pts, wks) if w > cutoff]
        recent_ppg = round(sum(recent) / len(recent), 2) if recent else None

        return {
            "season": {"ppg": round(total / games, 2) if games else 0.0, "total": round(total, 1), "games": games},
            "recent": {"ppg": recent_ppg, "weeks": recent_weeks, "games": len(recent)},
            "weekly": [round(p, 1) for p in pts],
            "weeks": list(wks),
            "found": True,
            "gsis_id": pid,
            "matched_name": entry["name"],
            "position": entry["position"],
            "team": entry["team"],
        }

    def get_usage_trends(self, player, recent_weeks: int = 4) -> dict:
        """Target share / air yards share / WOPR: recent average vs. season average."""
        pid = self.resolve(player)
        if not pid:
            return {}
        pdf = self.df.iloc[self.by_id[pid]["rows"]]
        recent_mask = pdf["week"] > self.last_week - recent_weeks

        result = {}
        for col in _USAGE_COLS:
            if col not in pdf.columns:
                continue
            vals = pd.to_numeric(pdf[col], errors="coerce").fillna(0)
            recent = vals[recent_mask]
            season_avg = round(float(vals.mean()), 3)
            recent_avg = round(float(recent.mean()), 3) if len(recent) else season_avg
            result[col] = {"season_avg": season_avg, "recent_avg": recent_avg,
                           "trend": round(recent_avg - season_avg, 3)}

        if "carries" in pdf.columns and pdf["carries"].notna().any():
            carries = pd.to_numeric(pdf["carries"], errors="coerce").fillna(0)[recent_mask]
            if len(carries):
                result["recent_carries_pg"] = round(float(carries.mean()), 1)
        return result

    def rank_available_players(self, players: list) -> list:
        """Attach season_ppg / recent_ppg / games / stats_found to each player dict."""
        out = []
        for p in players:
            s = self.get_player_stats(p)
            out.append({**p, "season_ppg": s["season"]["ppg"], "recent_ppg": s["recent"]["ppg"],
                        "games": s["season"]["games"], "stats_found": s["found"]})
        return out

    def season_ppg_table(self, min_games: int = 1) -> pd.DataFrame:
        """One row per player: position, games, ppg (used for replacement levels)."""
        rows = [{"player_id": pid, "position": e["position"], "games": len(e["pts"]),
                 "ppg": float(np.mean(e["pts"]))}
                for pid, e in self.by_id.items() if len(e["pts"]) >= min_games]
        return pd.DataFrame(rows, columns=["player_id", "position", "games", "ppg"])
