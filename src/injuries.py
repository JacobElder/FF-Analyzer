"""
Injury report from Sleeper's /players/nfl endpoint.
Fetches directly (bypassing the general 24h player cache) so injury status
is never more than 4h stale during the season.
"""

import os
import time
import pickle
import requests

CACHE_PATH = "data/injury_report.pkl"
SEASON_TTL = 14400    # 4 hours during season
OFFSEASON_TTL = 86400  # 24 hours offseason

GAME_STATUS_ORDER = {"Out": 4, "Doubtful": 3, "Questionable": 2, "Probable": 1}

SLEEPER_PLAYERS_URL = "https://api.sleeper.app/v1/players/nfl"


def _is_season() -> bool:
    from datetime import date
    m = date.today().month
    return m >= 9 or m <= 1  # Sep–Jan = active season


def fetch_injury_report(year: int | None = None) -> dict:
    """
    Returns {player_full_name: injury_status} using a direct Sleeper API fetch.
    Statuses: 'Out', 'Doubtful', 'Questionable', 'IR', 'Sus', etc.
    """
    ttl = SEASON_TTL if _is_season() else OFFSEASON_TTL
    os.makedirs("data", exist_ok=True)

    if os.path.exists(CACHE_PATH):
        if time.time() - os.path.getmtime(CACHE_PATH) < ttl:
            with open(CACHE_PATH, "rb") as f:
                return pickle.load(f)

    try:
        resp = requests.get(SLEEPER_PLAYERS_URL, timeout=60)
        if resp.status_code != 200:
            raise ValueError(f"Sleeper returned {resp.status_code}")
        players = resp.json()

        result = {}
        for p_info in players.values():
            name = p_info.get("full_name", "")
            status = p_info.get("injury_status", "")
            if not name or not status or status == "NA":
                continue
            result[name] = status

        with open(CACHE_PATH, "wb") as f:
            pickle.dump(result, f)
        return result

    except Exception as e:
        print(f"Injury report fetch failed: {e}")
        if os.path.exists(CACHE_PATH):
            with open(CACHE_PATH, "rb") as f:
                return pickle.load(f)
        return {}


def injury_badge(name: str, injury_map: dict) -> str | None:
    """Return status string for display, or None if healthy/unknown."""
    status = injury_map.get(name)
    if status and status != "NA":
        return status
    return None
