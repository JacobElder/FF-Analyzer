"""
Weekly player projections from Sleeper.

Uses the (unofficial, undocumented) endpoint the Sleeper app itself calls:
    https://api.sleeper.com/projections/nfl/{season}/{week}?season_type=regular&position[]=...
Projected stat lines use Sleeper scoring keys, so they're scored with the
league's own settings (DEF and K included). If the endpoint changes or fails,
callers fall back to history-based values.
"""

import logging

import requests

from src.storage import data_path, is_fresh, read_json, write_json_atomic

log = logging.getLogger(__name__)

PROJECTIONS_URL = "https://api.sleeper.com/projections/nfl/{season}/{week}"
POSITIONS = ["QB", "RB", "WR", "TE", "K", "DEF"]
CURRENT_TTL = 6 * 3600       # upcoming weeks get re-projected during the week
PAST_TTL = 365 * 86400       # completed weeks never change

_POINT_KEYS = ("pts_std", "pts_half_ppr", "pts_ppr")


def fetch_week_projections(season: int, week: int, completed: bool = False, force: bool = False) -> dict:
    """
    {sleeper_player_id: projected stat dict} for one week, file-cached.
    Players Sleeper explicitly projects at 0 are kept (that is information:
    usually inactive or a backup); rows with no point totals at all are dropped.
    """
    path = data_path(f"projections_{season}_{week}.json")
    if not force and is_fresh(path, PAST_TTL if completed else CURRENT_TTL):
        return read_json(path)
    try:
        r = requests.get(PROJECTIONS_URL.format(season=season, week=week),
                         params={"season_type": "regular", "position[]": POSITIONS}, timeout=20)
        r.raise_for_status()
        out = {}
        for row in r.json():
            stats = row.get("stats") or {}
            if any(isinstance(stats.get(k), (int, float)) for k in _POINT_KEYS) and row.get("player_id"):
                out[row["player_id"]] = stats
        write_json_atomic(out, path)
        return out
    except Exception as e:
        log.warning("Projections for %s week %s unavailable: %s", season, week, e)
        return read_json(path) if path.exists() else {}


def score_projection(stats: dict, scoring: dict) -> float:
    """Projected fantasy points: projected stats · league scoring weights."""
    pts = sum(stats[k] * w for k, w in scoring.items() if k in stats and isinstance(stats[k], (int, float)))
    if pts == 0:
        # League scoring lacks the relevant keys (e.g. no DEF keys) — use Sleeper's own total
        rec = scoring.get("rec", 0)
        pts = stats.get("pts_ppr" if rec >= 1 else "pts_half_ppr" if rec >= 0.5 else "pts_std") or 0.0
    return round(float(pts), 2)


class Projections:
    """Scored projections {week: {player_id: points}} for a set of weeks."""

    def __init__(self, season: int, weeks: list, scoring: dict, completed_through: int = 0, force: bool = False):
        self.weeks = list(weeks)
        self.points: dict[int, dict[str, float]] = {}
        for w in self.weeks:
            raw = fetch_week_projections(season, w, completed=w <= completed_through, force=force)
            self.points[w] = {pid: score_projection(st, scoring) for pid, st in raw.items()}

    @property
    def available(self) -> bool:
        return any(self.points.values())

    def get(self, player_id: str | None, week: int) -> float | None:
        if not player_id:
            return None
        return self.points.get(week, {}).get(player_id)
