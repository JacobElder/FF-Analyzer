"""
League files in leagues/<key>.json.

Keys are sanitized (letters, digits, - and _) so user input can never escape
the leagues directory. Synced leagues are keyed by league_id, so re-syncing by
name or by ID updates the same file. Reads never write.
"""

import re

from src.storage import LEAGUES_DIR, read_json, write_json_atomic


def safe_key(raw: str) -> str:
    key = re.sub(r"[^A-Za-z0-9_-]+", "_", str(raw or "")).strip("_")
    return key or "league"


def _path(key: str):
    return LEAGUES_DIR / f"{safe_key(key)}.json"


def list_leagues() -> list[tuple[str, str]]:
    """
    [(key, label)] for every saved league, one entry per league_id (files named
    after the league_id win over legacy duplicates named after the team).
    """
    if not LEAGUES_DIR.exists():
        return []
    by_league: dict[str, tuple[str, str]] = {}
    for path in sorted(LEAGUES_DIR.glob("*.json")):
        key = path.stem
        try:
            data = read_json(path)
        except (OSError, ValueError):
            continue
        name = data.get("league_name") or data.get("team_name") or key
        season = data.get("season")
        label = f"{name} ({season})" if season else name
        ident = str(data.get("league_id") or key)
        if ident in by_league and by_league[ident][0] == ident:
            continue  # canonical file already listed
        by_league[ident] = (key, label)
    return sorted(by_league.values(), key=lambda kv: kv[1].lower())


def load_league(key: str) -> dict:
    path = _path(key)
    if not path.exists():
        return {"team_name": key, "budget": 100, "roster": [], "waivers": []}
    return read_json(path)


def save_league(data: dict, key: str | None = None) -> str:
    """Write league data atomically; returns the key it was saved under."""
    key = safe_key(key or data.get("league_id") or data.get("league_name") or data.get("team_name"))
    write_json_atomic(data, _path(key), indent=2)
    return key
