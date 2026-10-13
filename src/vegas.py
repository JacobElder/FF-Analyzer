"""
Vegas implied team totals via The Odds API (free tier, optional).
Set ODDS_API_KEY in .env to enable. Gracefully returns empty dict if not configured.
Free tier: 500 requests/month — cached 1 hour to stay well within limits.

Sign up at https://the-odds-api.com to get a free key.
"""

import os
import time
import json
import requests
from dotenv import load_dotenv

load_dotenv()

CACHE_PATH = "data/vegas_totals.json"
CACHE_TTL = 3600  # 1 hour

BASE_URL = "https://api.the-odds-api.com/v4"
SPORT = "americanfootball_nfl"
REGIONS = "us"
MARKETS = "totals,spreads"

# Map Odds API team names → nfl_data_py abbreviations
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


def fetch_implied_totals() -> dict:
    """
    Returns {team_abbr: implied_total} for all games with current lines.
    Returns empty dict if no API key or offseason (no active lines).

    Implied total formula:
        favored_implied = (over_under + |spread|) / 2
        underdog_implied = (over_under - |spread|) / 2
    Equivalently: team_implied = (over_under - team_spread) / 2
    (spread is negative for the favored team)
    """
    api_key = os.getenv("ODDS_API_KEY")
    if not api_key:
        return {}

    os.makedirs("data", exist_ok=True)

    if os.path.exists(CACHE_PATH):
        if time.time() - os.path.getmtime(CACHE_PATH) < CACHE_TTL:
            with open(CACHE_PATH, "r") as f:
                return json.load(f)

    try:
        resp = requests.get(
            f"{BASE_URL}/sports/{SPORT}/odds",
            params={"apiKey": api_key, "regions": REGIONS, "markets": MARKETS},
            timeout=10,
        )
        if resp.status_code != 200:
            return {}

        games = resp.json()
        totals = {}

        for game in games:
            home = game.get("home_team", "")
            away = game.get("away_team", "")
            home_abbr = TEAM_NAME_MAP.get(home)
            away_abbr = TEAM_NAME_MAP.get(away)
            if not home_abbr or not away_abbr:
                continue

            total_val = None
            home_spread = None

            for bookmaker in game.get("bookmakers", [])[:1]:  # use first bookmaker
                for market in bookmaker.get("markets", []):
                    if market["key"] == "totals":
                        for o in market["outcomes"]:
                            if o["name"] == "Over":
                                total_val = o["point"]
                    elif market["key"] == "spreads":
                        for o in market["outcomes"]:
                            if o["name"] == home:
                                home_spread = o["point"]

            if total_val is not None and home_spread is not None:
                home_implied = (total_val - home_spread) / 2
                away_implied = (total_val + home_spread) / 2
                totals[home_abbr] = round(home_implied, 1)
                totals[away_abbr] = round(away_implied, 1)

        with open(CACHE_PATH, "w") as f:
            json.dump(totals, f)
        return totals

    except Exception as e:
        print(f"Vegas fetch failed: {e}")
        return {}
