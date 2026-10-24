"""
Vegas implied team totals via The Odds API (free tier, optional).
Set ODDS_API_KEY in .env to enable. Gracefully returns empty dict if not configured.
Free tier: 500 requests/month — cached 1 hour to stay well within limits.

Sign up at https://the-odds-api.com to get a free key.
"""

import logging
import os

import requests

from src.storage import data_path, is_fresh, read_json, write_json_atomic

log = logging.getLogger(__name__)

CACHE_TTL = 3600  # 1 hour

BASE_URL = "https://api.the-odds-api.com/v4"
SPORT = "americanfootball_nfl"
REGIONS = "us"
MARKETS = "totals,spreads"

# Odds API team names → Sleeper abbreviations
TEAM_NAME_MAP = {
    "Arizona Cardinals": "ARI", "Atlanta Falcons": "ATL", "Baltimore Ravens": "BAL",
    "Buffalo Bills": "BUF", "Carolina Panthers": "CAR", "Chicago Bears": "CHI",
    "Cincinnati Bengals": "CIN", "Cleveland Browns": "CLE", "Dallas Cowboys": "DAL",
    "Denver Broncos": "DEN", "Detroit Lions": "DET", "Green Bay Packers": "GB",
    "Houston Texans": "HOU", "Indianapolis Colts": "IND", "Jacksonville Jaguars": "JAX",
    "Kansas City Chiefs": "KC", "Las Vegas Raiders": "LV", "Los Angeles Chargers": "LAC",
    "Los Angeles Rams": "LAR", "Miami Dolphins": "MIA", "Minnesota Vikings": "MIN",
    "New England Patriots": "NE", "New Orleans Saints": "NO", "New York Giants": "NYG",
    "New York Jets": "NYJ", "Philadelphia Eagles": "PHI", "Pittsburgh Steelers": "PIT",
    "San Francisco 49ers": "SF", "Seattle Seahawks": "SEA", "Tampa Bay Buccaneers": "TB",
    "Tennessee Titans": "TEN", "Washington Commanders": "WAS",
}


def parse_implied_totals(games: list) -> dict:
    """
    {team_abbr: implied_total} from Odds API game objects, using each team's
    EARLIEST upcoming game (the feed spans several weeks).

    team_implied = (over_under - team_spread) / 2   (spread negative for favorite)
    """
    totals: dict[str, float] = {}

    for game in sorted(games, key=lambda g: g.get("commence_time") or ""):
        home, away = game.get("home_team", ""), game.get("away_team", "")
        home_abbr, away_abbr = TEAM_NAME_MAP.get(home), TEAM_NAME_MAP.get(away)
        if not home_abbr or not away_abbr:
            continue
        if home_abbr in totals and away_abbr in totals:
            continue  # already have both teams' next games

        total_val = home_spread = None
        for bookmaker in game.get("bookmakers", []):  # first bookmaker that has each market
            for market in bookmaker.get("markets", []):
                if market.get("key") == "totals" and total_val is None:
                    for o in market.get("outcomes", []):
                        if o.get("name") == "Over":
                            total_val = o.get("point")
                elif market.get("key") == "spreads" and home_spread is None:
                    for o in market.get("outcomes", []):
                        if o.get("name") == home:
                            home_spread = o.get("point")
            if total_val is not None and home_spread is not None:
                break

        if total_val is not None and home_spread is not None:
            totals.setdefault(home_abbr, round((total_val - home_spread) / 2, 1))
            totals.setdefault(away_abbr, round((total_val + home_spread) / 2, 1))
    return totals


def fetch_implied_totals(force: bool = False) -> dict:
    """{team_abbr: implied_total}; empty if no API key, offseason, or fetch failure."""
    api_key = os.getenv("ODDS_API_KEY")
    if not api_key:
        return {}

    path = data_path("vegas_totals.json")
    if not force and is_fresh(path, CACHE_TTL):
        return read_json(path)

    try:
        resp = requests.get(
            f"{BASE_URL}/sports/{SPORT}/odds",
            params={"apiKey": api_key, "regions": REGIONS, "markets": MARKETS},
            timeout=10,
        )
        if resp.status_code != 200:
            log.warning("Odds API returned %s", resp.status_code)
            return _stale(path)
        totals = parse_implied_totals(resp.json())
        write_json_atomic(totals, path)
        return totals
    except Exception as e:
        # Exception text can include the request URL, which carries the API key
        log.warning("Vegas fetch failed: %s", str(e).replace(api_key, "***"))
        return _stale(path)


def _stale(path) -> dict:
    """Last cached totals (even if old), or {} when there is none."""
    try:
        return read_json(path) if path.exists() else {}
    except (OSError, ValueError):
        return {}
