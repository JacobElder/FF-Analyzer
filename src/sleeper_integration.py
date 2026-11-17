"""
Sleeper API integration: league sync and the shared player database.

The full player database (~15 MB) is downloaded at most once per TTL and shared
by sync, injuries, and ID lookups — Sleeper asks clients to call /players/nfl
no more than about once a day.
"""

import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import requests

from src.players import RELEVANT_POSITIONS, sleeper_to_player
from src.state_manager import load_league
from src.storage import DataUnavailable, data_path, file_age_seconds, is_fresh, read_json, write_json_atomic

log = logging.getLogger(__name__)

BASE_URL = "https://api.sleeper.app/v1"
TIMEOUT = 10
PLAYERS_TTL = 12 * 3600
GAMEDAY_PLAYERS_TTL = 4 * 3600  # injury news moves fast on game days (Thu/Sun/Mon, Sat in Dec–Jan)
LAST_REGULAR_WEEK = 18
_ET = ZoneInfo("America/New_York")
_players_lock = threading.Lock()  # parallel league syncs share one download
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

def players_ttl(now: datetime | None = None) -> int:
    """How long the player database (and so injury statuses) may be cached right now."""
    et = (now or datetime.now(timezone.utc)).astimezone(_ET)
    game_day = et.weekday() in (0, 3, 6) or (et.weekday() == 5 and et.month in (12, 1))
    return GAMEDAY_PLAYERS_TTL if game_day else PLAYERS_TTL


def _read_players(path) -> dict | None:
    data = read_json(path) if path.exists() else None
    return data if isinstance(data, dict) and data else None


def fetch_all_players(force: bool = False) -> dict:
    """
    All Sleeper NFL players {player_id: info}, file-cached for players_ttl().
    Falls back to a stale cache when the download fails; raises
    DataUnavailable when there is no usable cached copy either.
    """
    path = _players_cache()
    requested_at = time.time()
    ttl = players_ttl()
    if not force and is_fresh(path, ttl) and (cached := _read_players(path)):
        return cached

    with _players_lock:
        # Another thread may have downloaded it while this one waited for the lock
        just_written = path.exists() and path.stat().st_mtime >= requested_at
        if (just_written or (not force and is_fresh(path, ttl))) and (cached := _read_players(path)):
            return cached

        players, err = _get("/players/nfl", timeout=60)
        if isinstance(players, dict) and players:
            write_json_atomic(players, path)
            return players

        log.warning("Player database fetch failed: %s", err)
        cached = _read_players(path)
        if cached:
            return cached
        raise DataUnavailable(f"Couldn't load the Sleeper player database: {err or 'empty response'}")


def players_db_age_seconds() -> float | None:
    return file_age_seconds(_players_cache())


def players_db_due(now: datetime | None = None) -> bool:
    """True when the cached player database is missing or older than players_ttl(now)."""
    age = players_db_age_seconds()
    return age is None or age >= players_ttl(now)


# ── League lookups ────────────────────────────────────────────────────────────

def _fetch_nfl_state():
    """(state, None) from Sleeper's /state/nfl, or (None, error)."""
    state, err = _get("/state/nfl")
    if state and state.get("season"):
        # "week" is the upcoming/in-progress week (the one waiver claims and lineups are for);
        # Sleeper's display_week lags behind it until mid-week.
        week = state.get("week") if state.get("season_type") == "regular" else None
        return {"season": str(state["season"]), "week": week, "season_type": state.get("season_type")}, None
    return None, err or "Sleeper returned no NFL state."


def get_nfl_state() -> dict:
    """Current NFL season/week per Sleeper (/state/nfl), falling back to the calendar."""
    state, _ = _fetch_nfl_state()
    if state:
        return state
    from src.wrangler import current_nfl_season
    return {"season": str(current_nfl_season()), "week": None, "season_type": None}


def get_nfl_season() -> str:
    return get_nfl_state()["season"]


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


NON_STARTING_SLOTS = {"BN", "IR", "TAXI"}


def assign_slots(starters: list, roster_positions: list, reserve: list, taxi: list) -> dict:
    """
    {player_id: lineup slot} for your current Sleeper lineup. `starters` is
    ordered like the starting slots in roster_positions; "0" marks an empty slot.
    """
    starting_slots = [pos for pos in roster_positions if pos not in NON_STARTING_SLOTS]
    slots = {pid: slot for slot, pid in zip(starting_slots, starters or []) if pid and pid != "0"}
    for pid in reserve or []:
        slots[pid] = "IR"
    for pid in taxi or []:
        slots[pid] = "TAXI"
    return slots


def _team_names(users: list) -> dict:
    """user_id → team name (custom team name, else display name)."""
    return {u["user_id"]: (u.get("metadata") or {}).get("team_name") or u.get("display_name") or "Unknown"
            for u in users or [] if u.get("user_id")}


def _record(roster: dict) -> dict:
    st = roster.get("settings") or {}
    return {
        "wins": st.get("wins", 0), "losses": st.get("losses", 0), "ties": st.get("ties", 0),
        "points_for": round((st.get("fpts") or 0) + (st.get("fpts_decimal") or 0) / 100, 2),
        "waiver_position": st.get("waiver_position"),
    }


def _side(m: dict) -> dict:
    return {
        "points": round(m.get("points") or 0, 2),
        "starters": [pid for pid in m.get("starters") or [] if pid and pid != "0"],
        # points scored so far by every rostered player this week (live during games)
        "players_points": {pid: round(pts or 0, 2) for pid, pts in (m.get("players_points") or {}).items()},
    }


def _current_matchup(league_id: str, week, my_roster: dict, rosters: list, team_names: dict):
    """(matchup | None, error | None) for `week`."""
    if not week:
        return None, None
    matchups, err = _get(f"/league/{league_id}/matchups/{week}")
    if not matchups:
        if err:
            log.warning("Matchup unavailable: %s", err)
        return None, err
    mine = next((m for m in matchups if m.get("roster_id") == my_roster.get("roster_id")), None)
    if not mine or mine.get("matchup_id") is None:
        return {"week": week, "bye": True}, None
    opp = next((m for m in matchups if m.get("matchup_id") == mine["matchup_id"]
                and m.get("roster_id") != mine["roster_id"]), None)
    opp_roster = next((r for r in rosters if opp and r.get("roster_id") == opp["roster_id"]), {})
    me, them = _side(mine), _side(opp or {})
    return {
        "week": week,
        **me,
        "opponent": team_names.get(opp_roster.get("owner_id"), "Unknown") if opp else None,
        "opponent_roster_id": opp.get("roster_id") if opp else None,
        "opponent_points": them["points"] if opp else None,
        "opponent_starters": them["starters"],
        "opponent_players_points": them["players_points"],
        "opponent_record": _record(opp_roster) if opp_roster else None,
    }, None


def _schedule(league_id: str, week, playoff_week_start):
    """
    ({week: [[roster_id, roster_id], ...]}, complete) for the rest of the regular
    season. Sleeper publishes every week's pairings in advance; weeks without
    pairings are left out. `complete` is False when any week's request failed.
    """
    if not week:
        return {}, True
    weeks = list(range(int(week), int(playoff_week_start or 15)))
    with ThreadPoolExecutor(max_workers=8) as pool:
        responses = list(pool.map(lambda wk: _get(f"/league/{league_id}/matchups/{wk}"), weeks))
    out, complete = {}, True
    for wk, (rows, err) in zip(weeks, responses):
        if err:
            log.warning("Schedule for week %s unavailable: %s", wk, err)
            complete = False
            continue
        pairs: dict = {}
        for m in rows or []:
            if m.get("matchup_id") is not None:
                pairs.setdefault(m["matchup_id"], []).append(m.get("roster_id"))
        games = [ids for ids in pairs.values() if len(ids) == 2]
        if games:
            out[str(wk)] = games
    return out, complete


def _league_rosters(rosters: list, league: dict, team_names: dict) -> list:
    """Compact view of every team: players, FAAB left, waiver order, record."""
    out = []
    for r in rosters:
        reserve = r.get("reserve") or []
        taxi = r.get("taxi") or []
        out.append({
            "roster_id": r.get("roster_id"),
            "owner_id": r.get("owner_id"),
            "team_name": team_names.get(r.get("owner_id"), f"Team {r.get('roster_id')}"),
            "players": list(dict.fromkeys((r.get("players") or []) + reserve + taxi)),
            "reserve": reserve,
            "taxi": taxi,  # taxi-squad players: rostered but not eligible to start (slot "TAXI")
            "faab_remaining": parse_faab_budget(league, r),
            "record": _record(r),
        })
    return out


def _transactions(league_id: str, through_week: int | None):
    """(compact waiver / free-agent transactions for weeks 1..through_week, [weeks that failed])."""
    out, failed = [], []
    weeks = range(1, (through_week or 0) + 1)
    with ThreadPoolExecutor(max_workers=8) as pool:  # one request per week, fetched in parallel
        responses = list(pool.map(lambda wk: _get(f"/league/{league_id}/transactions/{wk}"), weeks))
    for wk, (txs, err) in zip(weeks, responses):
        if err:
            log.warning("Transactions for week %s unavailable: %s", wk, err)
            failed.append(wk)
            continue
        for t in txs or []:
            if t.get("type") not in ("waiver", "free_agent"):
                continue
            out.append({
                "week": t.get("leg", wk),
                "type": t["type"],
                "status": t.get("status"),
                "bid": (t.get("settings") or {}).get("waiver_bid"),
                "roster_id": (t.get("roster_ids") or [None])[0],
                "adds": list((t.get("adds") or {}).keys()),
                "drops": list((t.get("drops") or {}).keys()),
                "note": (t.get("metadata") or {}).get("notes"),
                "created": t.get("created"),
            })
    return out, failed


def _transactions_through(week, league: dict, nfl_state: dict) -> int | None:
    """
    Last week to read transactions for: the current week in season; the whole
    regular season once the league is complete or the NFL is in its
    postseason (Sleeper's week is None then); None before the season.
    """
    if week:
        return week
    same_season = str(league.get("season", "")) == nfl_state["season"]
    if league.get("status") == "complete" or (same_season and nfl_state.get("season_type") == "post"):
        return LAST_REGULAR_WEEK
    return None


def _find_league(user_id: str, league_id_or_name: str):
    """Returns (league, error, nfl_state)."""
    state, err = _fetch_nfl_state()
    if not state:
        # Without the current week a sync would save a week-less league over good data
        return None, f"Couldn't read the current NFL week from Sleeper: {err}", None
    season = state["season"]
    if league_id_or_name.isdigit() and len(league_id_or_name) > 10:
        league, err = _get(f"/league/{league_id_or_name}")
        return league, (None if league else err or "League not found."), state

    for s in (season, str(int(season) - 1)):
        leagues, err = _get(f"/user/{user_id}/leagues/nfl/{s}")
        if err:
            return None, err, state
        match = next((l for l in leagues or [] if (l.get("name") or "").lower() == league_id_or_name.lower()), None)
        if match:
            return match, None, state
    return None, f"Could not find a league named '{league_id_or_name}' in {season} or {int(season) - 1}.", state


def _carry_forward(previous: dict, league_id: str, season: str) -> dict:
    """The previously saved sync of this league and season, or {} (nothing to reuse)."""
    if previous.get("league_id") == league_id and str(previous.get("season", "")) == season:
        return previous
    return {}


def sync_sleeper_league(username: str, league_id_or_name: str, players_db: dict | None = None,
                        previous: dict | None = None):
    """
    Returns (league_data, None) on success or (None, error message).

    A failed /state/nfl, user, league, rosters or player-database request fails
    the sync (so the last good file is kept). Failed secondary requests (users,
    matchups, transactions, schedule weeks) don't: their values are carried
    forward from `previous` (default: the saved file for this league) and named
    in league_data["partial"], so the caller can retry sooner.
    """
    if not username or not league_id_or_name:
        return None, "Please enter both username and league identifier."

    user, err = _get(f"/user/{username}")
    if not user:
        return None, err if err and "HTTP 404" not in err else f"Sleeper user '{username}' not found."
    user_id = user.get("user_id")

    league, err, nfl_state = _find_league(user_id, league_id_or_name)
    if not league:
        return None, err

    league_id = league["league_id"]
    rosters, err = _get(f"/league/{league_id}/rosters")
    if rosters is None:
        return None, err or "Sleeper returned no rosters for this league."

    my_roster = next(
        (r for r in rosters if r.get("owner_id") == user_id or user_id in (r.get("co_owners") or [])),
        None,
    )
    if not my_roster:
        return None, f"'{username}' doesn't own a roster in {league.get('name', league_id)}."

    if players_db is None:
        try:
            players_db = fetch_all_players()
        except DataUnavailable as e:
            log.warning("%s", e)
            players_db = {}
    if not players_db:
        return None, "Couldn't load the Sleeper player database. Try again shortly."

    league_season = str(league.get("season", ""))
    prev = _carry_forward(load_league(league_id) if previous is None else previous, league_id, league_season)
    partial = []

    roster_positions = league.get("roster_positions") or []
    reserve = my_roster.get("reserve") or []
    taxi = my_roster.get("taxi") or []
    roster_ids = list(dict.fromkeys((my_roster.get("players") or []) + reserve + taxi))
    slots = assign_slots(my_roster.get("starters") or [], roster_positions, reserve, taxi)
    roster = []
    for p_id in roster_ids:
        p = sleeper_to_player(p_id, players_db.get(p_id, {}))
        p["slot"] = slots.get(p_id, "BN")
        roster.append(p)

    users, err = _get(f"/league/{league_id}/users")
    team_names = _team_names(users)
    if err:
        log.warning("League users unavailable: %s", err)
        partial.append("users")
        team_names = {r["owner_id"]: r["team_name"] for r in prev.get("league_rosters") or []
                      if r.get("owner_id") and r.get("team_name")}
    trend_map = {t["player_id"]: t.get("count", 0) for t in get_trending_players("add")}
    settings = league.get("settings") or {}
    week = nfl_state.get("week") if league_season == nfl_state["season"] else None

    def matchup_for(wk, field):
        result, m_err = _current_matchup(league_id, wk, my_roster, rosters, team_names)
        if m_err:
            partial.append(field)
            old = prev.get(field)
            return old if old and old.get("week") == wk else None
        return result

    matchup = matchup_for(week, "matchup")
    last_result = matchup_for(week - 1, "last_result") if week and week > 1 else None

    transactions, failed_weeks = _transactions(league_id, _transactions_through(week, league, nfl_state))
    if failed_weeks:
        partial.append("transactions")
        transactions += [t for t in prev.get("transactions") or [] if t.get("week") in failed_weeks]
        transactions.sort(key=lambda t: (t.get("week") or 0, t.get("created") or 0))

    schedule, schedule_complete = _schedule(league_id, week, settings.get("playoff_week_start"))
    if not schedule_complete:
        partial.append("schedule")
        kept = {wk: games for wk, games in (prev.get("schedule") or {}).items() if int(wk) >= int(week)}
        schedule = {**kept, **schedule}

    data = {
        "league_id": league_id,
        "league_name": league.get("name", ""),
        # The team's name belongs to its owner (co-owners share it), as for every other team
        "team_name": team_names.get(my_roster.get("owner_id")) or team_names.get(user_id)
                     or user.get("display_name") or username,
        "username": username,
        "season": league_season,
        "current_season": nfl_state["season"],
        "status": league.get("status"),
        "week": week,
        "platform": "Sleeper",
        "budget": parse_faab_budget(league, my_roster),
        "waiver_budget_total": settings.get("waiver_budget", 100),
        "faab_enabled": settings.get("waiver_type") == FAAB_WAIVER_TYPE,
        "total_rosters": league.get("total_rosters") or len(rosters),
        "playoff_week_start": settings.get("playoff_week_start"),
        "record": _record(my_roster),
        "roster_id": my_roster.get("roster_id"),
        "matchup": matchup,
        "last_result": last_result,
        "league_rosters": _league_rosters(rosters, league, team_names),
        "transactions": transactions,
        "waiver_settings": {k: settings.get(k) for k in
                            ("waiver_type", "waiver_day_of_week", "waiver_clear_days", "waiver_bid_min", "daily_waivers")},
        "playoff_teams": settings.get("playoff_teams"),
        "playoff_round_type": settings.get("playoff_round_type"),  # 0 = 1 wk/round, 1 = 2-wk final, 2 = 2 wks/round
        "median_game": bool(settings.get("league_average_match")),  # each week also scored vs the league median
        "trade_deadline": settings.get("trade_deadline"),
        "schedule": schedule,
        "schedule_complete": schedule_complete,
        "roster": roster,
        "waivers": build_waiver_pool(rosters, players_db, trend_map),
        "scoring_settings": league.get("scoring_settings", {}),
        "roster_positions": roster_positions,
        # parts whose request failed and were carried forward from the last sync (retry soon)
        "partial": partial,
        "synced_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    return data, None
