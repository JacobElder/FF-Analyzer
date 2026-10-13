import requests
import json
import os
import time

BASE_URL = "https://api.sleeper.app/v1"
TIMEOUT = 10
CACHE_PATH = "data/sleeper_players.json"

RELEVANT_POSITIONS = {"QB", "RB", "WR", "TE", "K", "DEF"}


def _ensure_data_dir():
    if not os.path.exists("data"):
        os.makedirs("data")


def fetch_all_players():
    """Fetches all players from Sleeper and caches them locally (24h TTL)."""
    _ensure_data_dir()

    if os.path.exists(CACHE_PATH):
        age = time.time() - os.path.getmtime(CACHE_PATH)
        if age < 86400:
            with open(CACHE_PATH, "r") as f:
                return json.load(f)

    print("Fetching full Sleeper player database...")
    try:
        response = requests.get(f"{BASE_URL}/players/nfl", timeout=60)
        if response.status_code == 200:
            players = response.json()
            with open(CACHE_PATH, "w") as f:
                json.dump(players, f)
            return players
    except Exception as e:
        print(f"Error fetching player database: {e}")

    if os.path.exists(CACHE_PATH):
        with open(CACHE_PATH, "r") as f:
            return json.load(f)
    return {}


def get_user_id(username):
    try:
        r = requests.get(f"{BASE_URL}/user/{username}", timeout=TIMEOUT)
        if r.status_code == 200:
            return r.json().get("user_id")
    except Exception as e:
        print(f"Error fetching user ID: {e}")
    return None


def get_league_info(league_id):
    try:
        r = requests.get(f"{BASE_URL}/league/{league_id}", timeout=TIMEOUT)
        if r.status_code == 200:
            return r.json()
    except Exception as e:
        print(f"Error fetching league info: {e}")
    return None


def get_leagues(user_id, season="2025"):
    try:
        r = requests.get(f"{BASE_URL}/user/{user_id}/leagues/nfl/{season}", timeout=TIMEOUT)
        if r.status_code == 200:
            return r.json()
    except Exception as e:
        print(f"Error fetching leagues: {e}")
    return []


def get_roster_data(league_id, user_id):
    try:
        r = requests.get(f"{BASE_URL}/league/{league_id}/rosters", timeout=TIMEOUT)
        if r.status_code == 200:
            for roster in r.json():
                if roster["owner_id"] == user_id:
                    return roster
    except Exception as e:
        print(f"Error fetching roster: {e}")
    return None


def get_trending_players(type="add", hours=24):
    try:
        r = requests.get(
            f"{BASE_URL}/players/nfl/trending/{type}?lookback_hours={hours}",
            timeout=TIMEOUT,
        )
        if r.status_code == 200:
            return r.json()
    except Exception as e:
        print(f"Error fetching trending players: {e}")
    return []


def get_available_waivers(league_id, all_players_cache):
    """
    Returns all available (unrostered) active NFL players.
    No hard cap — UI handles filtering.
    """
    try:
        r = requests.get(f"{BASE_URL}/league/{league_id}/rosters", timeout=TIMEOUT)
        if r.status_code != 200:
            return []

        # Collect all owned player IDs (including IR)
        owned_ids = set()
        for roster in r.json():
            owned_ids.update(roster.get("players") or [])

        # Get trending adds for context
        trending_raw = get_trending_players("add")
        trend_map = {item["player_id"]: item["count"] for item in trending_raw}

        available = []
        for p_id, p_info in all_players_cache.items():
            # Skip owned
            if p_id in owned_ids:
                continue
            # Active players with a team
            if not p_info.get("active"):
                continue
            team = p_info.get("team")
            if not team:
                continue
            pos = p_info.get("position", "")
            if pos not in RELEVANT_POSITIONS:
                continue
            # Skip known cut/injured statuses
            status = p_info.get("injury_status", "")
            if status in ("PUP", "IR", "Out", "Doubtful"):
                continue

            available.append({
                "name": p_info.get("full_name", "Unknown"),
                "position": pos,
                "team": team,
                "player_id": p_id,
                "trend_count": trend_map.get(p_id, 0),
            })

        # Sort: trending first, then alphabetical
        available.sort(key=lambda x: (-x["trend_count"], x["name"]))
        return available

    except Exception as e:
        print(f"Error identifying waivers: {e}")
    return []


def sync_sleeper_league(username, league_id_or_name):
    if not username or not league_id_or_name:
        return None, "Please enter both username and league identifier."

    user_id = get_user_id(username)
    if not user_id:
        return None, f"Sleeper user '{username}' not found."

    target_league = None
    if league_id_or_name.isdigit() and len(league_id_or_name) > 10:
        target_league = get_league_info(league_id_or_name)

    if not target_league:
        for season in ["2025", "2024"]:
            leagues = get_leagues(user_id, season)
            target_league = next(
                (l for l in leagues
                 if l["name"].lower() == league_id_or_name.lower()
                 or l["league_id"] == league_id_or_name),
                None,
            )
            if target_league:
                break

    if not target_league:
        return None, f"Could not find league '{league_id_or_name}'."

    league_id = target_league["league_id"]
    display_name = target_league["name"]

    roster_raw = get_roster_data(league_id, user_id)
    if not roster_raw:
        return None, "Roster not found in league."

    all_players = fetch_all_players()
    player_ids = roster_raw.get("players") or []

    formatted_roster = []
    for p_id in player_ids:
        p_info = all_players.get(p_id, {})
        name = p_info.get("full_name") or p_info.get("last_name") or p_id
        pos = p_info.get("position") or "UNK"
        team = p_info.get("team") or ""
        formatted_roster.append({"name": name, "position": pos, "team": team})

    waivers = get_available_waivers(league_id, all_players)

    formatted_data = {
        "team_name": display_name,
        "league_id": league_id,
        "platform": "Sleeper",
        "budget": int(roster_raw.get("settings", {}).get("fleaflicker_budget", 100)),
        "roster": formatted_roster,
        "waivers": waivers,
        "scoring_settings": target_league.get("scoring_settings", {}),
        "roster_positions": target_league.get("roster_positions", []),
    }
    return formatted_data, None
