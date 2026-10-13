"""
Sleeper API integration: league sync and the shared player database.

The full player database (~15 MB) is downloaded at most once per TTL and shared
by sync, injuries, and ID lookups — Sleeper asks clients to call /players/nfl
no more than about once a day.
"""

import logging
from datetime import datetime, timezone

import requests

from src.players import RELEVANT_POSITIONS, sleeper_to_player
from src.storage import data_path, file_age_seconds, is_fresh, read_json, write_json_atomic

log = logging.getLogger(__name__)

BASE_URL = "https://api.sleeper.app/v1"
TIMEOUT = 10
PLAYERS_TTL = 12 * 3600
FAAB_WAIVER_TYPE = 2  # Sleeper settings.waiver_type: 0 rolling, 1 reverse standings, 2 FAAB


def _players_cache():
    return data_path("sleeper_players.json")


def _get(path: str, timeout: int = TIMEOUT):
    """GET a Sleeper endpoint → (json, None) or (None, error message)."""
    try:
        r = requests.get(f"{BASE_URL}{path}", timeout=timeout)
    except requests.RequestException as e:
        return None, f"Network error contacting Sleeper: {e}"
    if r.status_code != 200:
        return None, f"Sleeper returned HTTP {r.status_code} for {path}"
    try:
        return r.json(), None
    except ValueError:
        return None, f"Sleeper returned invalid JSON for {path}"


# ── Player database ───────────────────────────────────────────────────────────

def fetch_all_players(force: bool = False) -> dict:
    """All Sleeper NFL players {player_id: info}, file-cached for PLAYERS_TTL."""
    path = _players_cache()
    if not force and is_fresh(path, PLAYERS_TTL):
        return read_json(path)

    players, err = _get("/players/nfl", timeout=60)
    if players:
        write_json_atomic(players, path)
        return players

    log.warning("Player database fetch failed: %s", err)
    return read_json(path) if path.exists() else {}


def players_db_age_seconds() -> float | None:
    return file_age_seconds(_players_cache())


# ── League lookups ────────────────────────────────────────────────────────────

def get_nfl_season() -> str:
    """Current NFL season per Sleeper (/state/nfl), falling back to the calendar."""
    state, _ = _get("/state/nfl")
    if state and state.get("season"):
        return str(state["season"])
    from src.wrangler import current_nfl_season
    return str(current_nfl_season())


def get_trending_players(kind: str = "add", hours: int = 24) -> list:
    data, err = _get(f"/players/nfl/trending/{kind}?lookback_hours={hours}")
    if err:
        log.warning("Trending players unavailable: %s", err)
    return data or []


def parse_faab_budget(league: dict, roster: dict) -> int:
    """Remaining FAAB = league waiver_budget − this roster's waiver_budget_used."""
    total = (league.get("settings") or {}).get("waiver_budget", 100) or 0
    used = (roster.get("settings") or {}).get("waiver_budget_used", 0) or 0
    return max(0, int(total) - int(used))


def build_waiver_pool(rosters: list, players_db: dict, trend_map: dict) -> list:
    """
    Every active, team-affiliated player at a fantasy position who isn't on a
    roster. Injured players are kept (useful stashes); the UI filters them.
    """
    owned = set()
    for r in rosters:
        owned.update(r.get("players") or [])
        owned.update(r.get("reserve") or [])
        owned.update(r.get("taxi") or [])

    available = []
    for p_id, p_info in players_db.items():
        if p_id in owned or not p_info.get("active") or not p_info.get("team"):
            continue
        if p_info.get("position") not in RELEVANT_POSITIONS:
            continue
        p = sleeper_to_player(p_id, p_info)
        p["trend_count"] = trend_map.get(p_id, 0)
        available.append(p)

    available.sort(key=lambda x: (-x["trend_count"], x["name"]))
    return available


def _find_league(user_id: str, league_id_or_name: str):
    """Returns (league, error, current_season)."""
    season = get_nfl_season()
    if league_id_or_name.isdigit() and len(league_id_or_name) > 10:
        league, err = _get(f"/league/{league_id_or_name}")
        return league, (None if league else err or "League not found."), season

    for s in (season, str(int(season) - 1)):
        leagues, err = _get(f"/user/{user_id}/leagues/nfl/{s}")
        if err:
            return None, err, season
        match = next((l for l in leagues or [] if l.get("name", "").lower() == league_id_or_name.lower()), None)
        if match:
            return match, None, season
    return None, f"Could not find a league named '{league_id_or_name}' in {season} or {int(season) - 1}.", season


def sync_sleeper_league(username: str, league_id_or_name: str):
    """Returns (league_data, None) on success or (None, error message)."""
    if not username or not league_id_or_name:
        return None, "Please enter both username and league identifier."

    user, err = _get(f"/user/{username}")
    if not user:
        return None, err if err and "HTTP 404" not in err else f"Sleeper user '{username}' not found."
    user_id = user.get("user_id")

    league, err, current_season = _find_league(user_id, league_id_or_name)
    if not league:
        return None, err

    league_id = league["league_id"]
    rosters, err = _get(f"/league/{league_id}/rosters")
    if rosters is None:
        return None, err

    my_roster = next(
        (r for r in rosters if r.get("owner_id") == user_id or user_id in (r.get("co_owners") or [])),
        None,
    )
    if not my_roster:
        return None, f"'{username}' doesn't own a roster in {league.get('name', league_id)}."

    players_db = fetch_all_players()
    if not players_db:
        return None, "Couldn't load the Sleeper player database. Try again shortly."

    roster_ids = my_roster.get("players") or []
    roster = [sleeper_to_player(p_id, players_db.get(p_id, {})) for p_id in roster_ids]
    reserve = set(my_roster.get("reserve") or [])
    for p in roster:
        if p["player_id"] in reserve:
            p["slot"] = "IR"

    trend_map = {t["player_id"]: t.get("count", 0) for t in get_trending_players("add")}
    settings = league.get("settings") or {}

    data = {
        "league_id": league_id,
        "league_name": league.get("name", ""),
        "team_name": league.get("name", ""),  # kept for older UI code / labels
        "season": str(league.get("season", "")),
        "current_season": current_season,
        "platform": "Sleeper",
        "budget": parse_faab_budget(league, my_roster),
        "faab_enabled": settings.get("waiver_type") == FAAB_WAIVER_TYPE,
        "total_rosters": league.get("total_rosters") or len(rosters),
        "playoff_week_start": settings.get("playoff_week_start"),
        "roster": roster,
        "waivers": build_waiver_pool(rosters, players_db, trend_map),
        "scoring_settings": league.get("scoring_settings", {}),
        "roster_positions": league.get("roster_positions", []),
        "synced_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    return data, None

