"""
Live-scoring helpers: which NFL games have kicked off.

Kickoff times come from the nflverse schedule (src/schedule.fetch_schedules),
whose `gameday` (YYYY-MM-DD) and `gametime` (HH:MM) are US Eastern time.
Pair with the synced matchup's `players_points` / `starters` (Sleeper) to show
points scored so far and which starters are still to play.
"""

from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pandas as pd

from src.players import norm_team

_ET = ZoneInfo("America/New_York")
RECENT_GAME_WINDOW = timedelta(days=6)  # with no week given: a kickoff this recent counts as this week's


def _kickoff(row) -> datetime | None:
    try:
        local = datetime.strptime(f"{row.get('gameday')} {row.get('gametime')}", "%Y-%m-%d %H:%M")
    except (TypeError, ValueError):
        return None
    return local.replace(tzinfo=_ET).astimezone(timezone.utc)


def kickoff_times(schedule: pd.DataFrame, week: int | None = None) -> dict:
    """
    {team: [kickoff (UTC datetime), ...]} sorted, for `week` (every week when None).
    Rows with no usable date/time are skipped.
    """
    if schedule is None or schedule.empty:
        return {}
    games = schedule if week is None or "week" not in schedule.columns else schedule[schedule["week"] == int(week)]
    out: dict = {}
    for _, row in games.iterrows():
        ko = _kickoff(row)
        if ko is None:
            continue
        for team in (row.get("home_team"), row.get("away_team")):
            if team:
                out.setdefault(norm_team(team), []).append(ko)
    return {t: sorted(v) for t, v in out.items()}


def game_started(team: str, now: datetime, schedule: pd.DataFrame, week: int | None = None) -> bool:
    """
    True once `team`'s game has kicked off: its game in `week` when given,
    otherwise its latest game that kicked off within the last 6 days.
    False on a bye, for an unknown team, or without a schedule.
    `now` should be timezone-aware (naive is taken as UTC).
    """
    now = now if now.tzinfo else now.replace(tzinfo=timezone.utc)
    kickoffs = kickoff_times(schedule, week).get(norm_team(team), [])
    started = [ko for ko in kickoffs if ko <= now]
    if not started:
        return False
    return week is not None or now - started[-1] < RECENT_GAME_WINDOW
