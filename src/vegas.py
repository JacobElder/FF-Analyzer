"""
Vegas implied team totals via The Odds API (free tier, optional).
Set ODDS_API_KEY in .env to enable. Gracefully returns empty dict if not configured.

Quota: the free tier is 500 credits/month and each request costs one credit
per market — totals + spreads = 2 credits. Results are file-cached for
CACHE_TTL (4 h → at most ~360 credits/month from one deployment), fetching
pauses when the API reports fewer than MIN_REMAINING credits left, and totals
older than MAX_STALE_S are dropped rather than served silently stale.

Sign up at https://the-odds-api.com to get a free key.
"""

import logging
import os
import time
from datetime import datetime, timedelta, timezone

import requests

from src.storage import data_path, read_json, write_json_atomic

log = logging.getLogger(__name__)

CACHE_TTL = 4 * 3600         # 4 hours between fetches
MAX_STALE_S = 24 * 3600      # never serve totals older than this
MIN_REMAINING = 20           # credits: below this, stop fetching...
LOW_QUOTA_BACKOFF_S = 24 * 3600  # ...except one check a day (the quota resets monthly)
# NFL weeks run Thursday–Monday; a new week's window opens Tuesday morning (after MNF ends)
WEEK_BOUNDARY_WEEKDAY = 1    # Tuesday
WEEK_BOUNDARY_HOUR_UTC = 10

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


def _parse_time(value) -> datetime | None:
    try:
        ts = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    return ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)


def nfl_week_window(now: datetime) -> tuple[datetime, datetime]:
    """[start, end) of the NFL week containing `now`: Tuesday 10:00 UTC to the next one."""
    now = now if now.tzinfo else now.replace(tzinfo=timezone.utc)
    start = now.astimezone(timezone.utc).replace(hour=WEEK_BOUNDARY_HOUR_UTC, minute=0, second=0, microsecond=0)
    start -= timedelta(days=(start.weekday() - WEEK_BOUNDARY_WEEKDAY) % 7)
    if start > now:
        start -= timedelta(days=7)
    return start, start + timedelta(days=7)


def parse_implied_totals(games: list, now: datetime | None = None) -> dict:
    """
    {team_abbr: implied_total} from Odds API game objects, using each team's
    game in the current NFL week (the feed spans several weeks). Teams that
    already played this week get no total rather than next week's line.

    team_implied = (over_under - team_spread) / 2   (spread negative for favorite)
    """
    start, end = nfl_week_window(now or datetime.now(timezone.utc))
    totals: dict[str, float] = {}

    for game in sorted(games, key=lambda g: g.get("commence_time") or ""):
        kickoff = _parse_time(game.get("commence_time"))
        if kickoff is None or not start <= kickoff < end:
            continue
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


def _load_cache(path) -> dict:
    """{"fetched_at", "totals", "remaining", "checked_at"}; tolerates old flat-dict files."""
    if not path.exists():
        return {}
    try:
        data = read_json(path)
    except OSError:
        return {}
    if not isinstance(data, dict):
        return {}
    if "totals" not in data:  # legacy format: the totals themselves, dated by the file
        return {"totals": data, "fetched_at": path.stat().st_mtime}
    return data


def _usable(cache: dict, now: float) -> dict:
    """Cached totals if younger than MAX_STALE_S, else {}."""
    fetched = cache.get("fetched_at")
    if fetched is None or now - fetched > MAX_STALE_S:
        return {}
    return cache.get("totals") or {}


def _remaining(resp):
    try:
        return int(float(resp.headers.get("x-requests-remaining")))
    except (AttributeError, TypeError, ValueError):
        return None


def fetch_implied_totals(force: bool = False, now: float | None = None) -> dict:
    """
    {team_abbr: implied_total}; empty if no API key, offseason, fetch failure
    with nothing recent cached, or cached totals older than MAX_STALE_S.
    `force` skips the TTL but never the low-quota backoff.
    """
    api_key = os.getenv("ODDS_API_KEY")
    if not api_key:
        return {}

    now = now if now is not None else time.time()
    path = data_path("vegas_totals.json")
    cache = _load_cache(path)
    fetched = cache.get("fetched_at")
    if not force and fetched is not None and now - fetched < CACHE_TTL:
        return cache.get("totals") or {}
    remaining = cache.get("remaining")
    if remaining is not None and remaining < MIN_REMAINING and now - (cache.get("checked_at") or 0) < LOW_QUOTA_BACKOFF_S:
        log.warning("Odds API quota low (%s credits left); not fetching", remaining)
        return _usable(cache, now)

    try:
        resp = requests.get(
            f"{BASE_URL}/sports/{SPORT}/odds",
            params={"apiKey": api_key, "regions": REGIONS, "markets": MARKETS},
            timeout=10,
        )
        remaining = _remaining(resp)
        if remaining is not None:
            cache.update(remaining=remaining, checked_at=now)
        if resp.status_code != 200:
            log.warning("Odds API returned %s", resp.status_code)
            if remaining is not None:
                write_json_atomic(cache, path)
            return _usable(cache, now)
        totals = parse_implied_totals(resp.json(), datetime.fromtimestamp(now, timezone.utc))
        cache.update(totals=totals, fetched_at=now)
        write_json_atomic(cache, path)
        return totals
    except Exception as e:
        # Exception text can include the request URL, which carries the API key
        log.warning("Vegas fetch failed: %s", str(e).replace(api_key, "***"))
        return _usable(cache, now)


def vegas_status() -> dict:
    """{"fetched_at": epoch | None, "remaining": credits | None} from the cache file, for display."""
    cache = _load_cache(data_path("vegas_totals.json"))
    return {"fetched_at": cache.get("fetched_at"), "remaining": cache.get("remaining")}
