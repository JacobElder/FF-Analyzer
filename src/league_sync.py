"""
Keep the configured default leagues in sync with Sleeper.

A league is re-synced when its saved data is older than the configured age.
Failed attempts are remembered (in `attempts`) so an outage doesn't trigger a
re-sync on every Streamlit rerun; the last good copy keeps being used.
"""

import time
from datetime import datetime, timezone

from src.sleeper_integration import sync_sleeper_league
from src.state_manager import load_league, save_league


def sync_age_seconds(data: dict, now: float | None = None) -> float | None:
    synced_at = data.get("synced_at")
    if not synced_at:
        return None
    now = now if now is not None else time.time()
    return now - datetime.fromisoformat(synced_at).astimezone(timezone.utc).timestamp()


def is_sync_due(data: dict, max_age_s: float, now: float | None = None) -> bool:
    age = sync_age_seconds(data, now)
    return age is None or age > max_age_s


def leagues_due(league_ids: list, max_age_s: float, attempts: dict, now: float | None = None) -> list:
    """League IDs whose data is stale and that weren't attempted within max_age_s."""
    now = now if now is not None else time.time()
    return [lid for lid in league_ids
            if is_sync_due(load_league(lid), max_age_s, now) and now - attempts.get(lid, 0) >= max_age_s]


def auto_sync(username: str, league_ids: list, max_age_s: float, attempts: dict,
              sync_fn=sync_sleeper_league, now: float | None = None) -> dict:
    """
    Sync every due league. Returns {league_id: error} for failures.
    `attempts` ({league_id: last attempt time}) is updated in place.
    """
    now = now if now is not None else time.time()
    errors = {}
    if not username:
        return errors
    for lid in leagues_due(league_ids, max_age_s, attempts, now):
        attempts[lid] = now
        data, err = sync_fn(username, lid)
        if data:
            save_league(data)
        else:
            errors[lid] = err
    return errors
