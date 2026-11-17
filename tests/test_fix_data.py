"""Data-layer audit fixes: partial syncs, taxi, live scoring, player-db TTL, loaders that raise, odds quota."""

import os
import time
from datetime import datetime, timedelta, timezone
from io import BytesIO

import pandas as pd
import pytest
import requests

import src.league_sync as ls
import src.live as live
import src.sleeper_integration as si
import src.state_manager as sm
import src.storage as storage
import src.vegas as vegas
import src.wrangler as wrangler
from src.injuries import build_injury_map, player_injury_status
from http_helpers import FakeResponse, _route
from test_http_integrations import LID, PLAYERS, _book, _game, _sleeper_routes


def _utc(*args):
    return datetime(*args, tzinfo=timezone.utc)


# ── storage ──────────────────────────────────────────────────────────────────

def test_read_json_tolerates_truncated_file(tmp_path):
    path = tmp_path / "x.json"
    path.write_text('{"a": [1, ')
    assert storage.read_json(path) is None
    assert storage.read_json(path, default={}) == {}


def test_atomic_write_does_not_touch_umask_and_uses_normal_permissions(tmp_path):
    before = os.umask(0o022)
    try:
        storage.write_json_atomic({"a": 1}, tmp_path / "y.json")
        assert os.umask(0o022) == 0o022  # unchanged by the write
        assert (tmp_path / "y.json").stat().st_mode & 0o777 == 0o644
    finally:
        os.umask(before)


def test_unreadable_league_file_is_skipped(leagues_dir):
    (leagues_dir / "bad.json").mkdir()  # a directory: opening it raises OSError
    assert sm.list_leagues() == []
    assert sm.load_league("bad")["roster"] == []


# ── injuries ─────────────────────────────────────────────────────────────────

def test_roster_status_used_when_injury_status_empty():
    assert player_injury_status({"injury_status": None, "status": "Injured Reserve"}) == "IR"
    assert player_injury_status({"injury_status": "NA", "status": "Suspended"}) == "Sus"
    assert player_injury_status({"injury_status": "Questionable", "status": "Injured Reserve"}) == "Questionable"
    assert player_injury_status({"status": "Active"}) is None
    db = {"1": {"full_name": "Pup Guy", "position": "RB", "status": "Physically Unable to Perform"},
          "2": {"full_name": "Gone", "position": "WR", "active": False, "status": "Inactive"}}
    assert build_injury_map(db) == {"by_id": {"1": "PUP"}, "by_name": {"pup guy": "PUP"}}


# ── wrangler ─────────────────────────────────────────────────────────────────

def test_download_parquet_uses_timeout_and_raises_on_http_error(monkeypatch):
    buf = BytesIO()
    pd.DataFrame({"a": [1]}).to_parquet(buf)
    seen = {}

    def get(url, timeout=None):
        seen["timeout"] = timeout
        resp = FakeResponse(200 if "ok" in url else 404)
        resp.content = buf.getvalue()
        return resp
    monkeypatch.setattr(wrangler.requests, "get", get)
    assert wrangler.download_parquet("http://ok")["a"].tolist() == [1]
    assert seen["timeout"] == wrangler.DOWNLOAD_TIMEOUT
    with pytest.raises(requests.HTTPError):
        wrangler.download_parquet("http://missing")


def test_weekly_stats_fall_back_to_last_season_then_raise(monkeypatch):
    frames = {2025: pd.DataFrame({"player_id": ["a"], "week": [1], "season": [2025]})}

    def download(url):
        year = int(url.rsplit("_", 1)[1].split(".")[0])
        if year in frames:
            return frames[year]
        raise OSError("404")
    monkeypatch.setattr(wrangler, "download_parquet", download)
    assert wrangler.fetch_weekly_stats(2026)["season"].tolist() == [2025]  # new season not published yet
    frames.clear()
    with pytest.raises(storage.DataUnavailable):
        wrangler.fetch_weekly_stats(2027, force=True)


# ── player database ──────────────────────────────────────────────────────────

@pytest.mark.parametrize("now,ttl", [
    (_utc(2026, 11, 15, 18), si.GAMEDAY_PLAYERS_TTL),   # Sunday
    (_utc(2026, 11, 17, 18), si.PLAYERS_TTL),           # Tuesday
    (_utc(2026, 12, 19, 18), si.GAMEDAY_PLAYERS_TTL),   # Saturday in December
    (_utc(2026, 10, 17, 18), si.PLAYERS_TTL),           # Saturday in October
])
def test_players_ttl_is_shorter_on_game_days(now, ttl):
    assert si.players_ttl(now) == ttl


def test_players_db_due(monkeypatch):
    assert si.players_db_due()
    storage.write_json_atomic(PLAYERS, si._players_cache())
    assert not si.players_db_due()
    assert si.players_ttl() in (si.PLAYERS_TTL, si.GAMEDAY_PLAYERS_TTL)


def test_players_db_reuses_a_download_finished_while_waiting(monkeypatch):
    path = si._players_cache()
    storage.write_json_atomic(PLAYERS, path)
    future = time.time() + 60
    os.utime(path, (future, future))  # written "after" this request started
    monkeypatch.setattr(si.requests, "get", lambda *a, **k: pytest.fail("should reuse the fresh download"))
    assert si.fetch_all_players(force=True) == PLAYERS


@pytest.mark.parametrize("cache", [None, "{truncated"])
def test_players_db_raises_with_empty_response_and_no_usable_cache(monkeypatch, cache):
    if cache:
        si._players_cache().write_text(cache)
    _route(monkeypatch, si, si.BASE_URL, {"/players/nfl": {}})
    with pytest.raises(storage.DataUnavailable, match="empty response"):
        si.fetch_all_players(force=True)


# ── league sync: hard failures, partial syncs, taxi, live points ─────────────

def test_failed_nfl_state_fails_the_sync(monkeypatch):
    _route(monkeypatch, si, si.BASE_URL, _sleeper_routes(**{"/state/nfl": FakeResponse(500)}))
    data, err = si.sync_sleeper_league("me", LID)
    assert data is None and "current NFL week" in err


def test_sync_records_taxi_and_live_points(monkeypatch):
    routes = _sleeper_routes(**{
        f"/league/{LID}/matchups/3": [
            {"roster_id": 1, "matchup_id": 1, "points": 10.5, "starters": ["qb1", "0"],
             "players_points": {"qb1": 10.456, "wr1": 0}},
            {"roster_id": 2, "matchup_id": 1, "points": 8.0, "starters": ["opp"], "players_points": {"opp": 8}}],
        f"/league/{LID}/transactions/3": [],
    })
    for wk in range(4, 15):
        routes[f"/league/{LID}/matchups/{wk}"] = []
    _route(monkeypatch, si, si.BASE_URL, routes)
    data, err = si.sync_sleeper_league("me", LID, players_db=PLAYERS, previous={})
    assert err is None and data["partial"] == [] and data["schedule_complete"] is True
    m = data["matchup"]
    assert m["starters"] == ["qb1"] and m["players_points"] == {"qb1": 10.46, "wr1": 0}
    assert m["opponent_starters"] == ["opp"] and m["opponent_players_points"] == {"opp": 8}
    mine = next(r for r in data["league_rosters"] if r["roster_id"] == 1)
    assert mine["taxi"] == ["te1"] and "te1" in mine["players"]
    assert next(p for p in data["roster"] if p["player_id"] == "te1")["slot"] == "TAXI"


def _previous():
    return {
        "league_id": LID, "season": "2026",
        "league_rosters": [{"roster_id": 1, "owner_id": "someone", "team_name": "Old Crew"},
                           {"roster_id": 2, "owner_id": "u2", "team_name": "Old Rival"},
                           {"roster_id": 3, "owner_id": None, "team_name": "Team 3"}],
        "matchup": {"week": 3, "points": 1.0},
        "last_result": {"week": 1, "points": 99.0},  # a different week: not reused
        "transactions": [{"week": 3, "bid": 40, "created": 5}, {"week": 2, "bid": 7, "created": 1}],
        "schedule": {"2": [[1, 2]], "3": [[2, 1]], "9": [[1, 2]]},
    }


def test_failed_secondary_requests_carry_forward_and_mark_partial(monkeypatch):
    routes = _sleeper_routes(**{
        f"/league/{LID}/users": FakeResponse(500),
        f"/league/{LID}/matchups/3": FakeResponse(503),   # current matchup + schedule week 3
        f"/league/{LID}/matchups/2": FakeResponse(503),
    })
    _route(monkeypatch, si, si.BASE_URL, routes)
    data, err = si.sync_sleeper_league("me", LID, players_db=PLAYERS, previous=_previous())
    assert err is None
    assert data["partial"] == ["users", "matchup", "last_result", "transactions", "schedule"]
    assert data["team_name"] == "Old Crew"
    assert data["matchup"] == {"week": 3, "points": 1.0} and data["last_result"] is None
    assert [t["bid"] for t in data["transactions"]] == [12, 40]  # week 3 failed: kept from the last sync
    assert data["schedule_complete"] is False
    assert data["schedule"] == {"3": [[2, 1]], "9": [[1, 2]]}  # past week 2 dropped


def test_sync_loads_previous_file_and_ignores_another_season(monkeypatch):
    sm.save_league({**_previous(), "season": "2025"})
    routes = _sleeper_routes(**{f"/league/{LID}/users": FakeResponse(500)})
    _route(monkeypatch, si, si.BASE_URL, routes)
    data, _ = si.sync_sleeper_league("me", LID)
    assert data["team_name"] == "me" and "users" in data["partial"]


@pytest.mark.parametrize("league,state,expected", [
    ({"status": "complete", "season": "2025"}, {"season": "2026", "season_type": "pre"}, 18),
    ({"status": "in_season", "season": "2026"}, {"season": "2026", "season_type": "post"}, 18),
    ({"status": "pre_draft", "season": "2026"}, {"season": "2026", "season_type": "pre"}, None),
])
def test_offseason_transactions_cover_the_whole_season(league, state, expected):
    assert si._transactions_through(None, league, state) == expected
    assert si._transactions_through(5, league, state) == 5


def test_postseason_sync_keeps_transactions(monkeypatch):
    routes = _sleeper_routes(**{"/state/nfl": {"season": "2026", "week": 19, "season_type": "post"}})
    _route(monkeypatch, si, si.BASE_URL, routes)
    data, _ = si.sync_sleeper_league("me", LID, players_db=PLAYERS, previous={})
    assert data["week"] is None and [t["bid"] for t in data["transactions"]] == [12]


def test_sync_with_empty_players_db_fails(monkeypatch):
    _route(monkeypatch, si, si.BASE_URL, _sleeper_routes())
    assert "player database" in si.sync_sleeper_league("me", LID, players_db={})[1]


def test_partial_sync_is_retried_sooner():
    now = time.time()
    synced = datetime.fromtimestamp(now - 600, timezone.utc).isoformat()
    sm.save_league({"league_id": "p", "synced_at": synced, "partial": ["schedule"]})
    sm.save_league({"league_id": "ok", "synced_at": synced, "partial": []})
    assert ls.effective_max_age({"partial": ["users"]}, 1800) == ls.PARTIAL_RETRY_S
    assert ls.leagues_due(["p", "ok"], 1800, {}, now) == ["p"]
    assert ls.leagues_due(["p"], 1800, {"p": now - 60}, now) == []  # just attempted


# ── live scoring ─────────────────────────────────────────────────────────────

def _sched():
    return pd.DataFrame([
        {"week": 11, "gameday": "2026-11-15", "gametime": "13:00", "home_team": "BUF", "away_team": "KC"},
        {"week": 11, "gameday": "2026-11-16", "gametime": "20:15", "home_team": "LA", "away_team": "NYJ"},
        {"week": 11, "gameday": None, "gametime": None, "home_team": "MIA", "away_team": "NE"},
        {"week": 10, "gameday": "2026-11-09", "gametime": "13:00", "home_team": "MIA", "away_team": "DAL"},
        {"week": 12, "gameday": "2026-11-22", "gametime": "13:00", "home_team": "TBD", "away_team": None},
    ])


def test_kickoff_times_convert_eastern_to_utc():
    ko = live.kickoff_times(_sched(), 11)
    assert ko["BUF"] == [_utc(2026, 11, 15, 18)] and ko["LAR"] == [_utc(2026, 11, 17, 1, 15)]
    assert "MIA" not in ko  # unusable date skipped
    assert live.kickoff_times(pd.DataFrame(), 11) == {} and live.kickoff_times(None) == {}
    assert live.kickoff_times(_sched().drop(columns="week"), 11)["MIA"] == [_utc(2026, 11, 9, 18)]


def test_game_started():
    s, sunday = _sched(), _utc(2026, 11, 15, 20)
    assert live.game_started("BUF", sunday, s, 11)
    assert not live.game_started("NYJ", sunday, s, 11)          # Monday night: not yet
    assert not live.game_started("SEA", sunday, s, 11)          # bye / unknown
    assert live.game_started("KC", datetime(2026, 11, 15, 20), s)  # naive = UTC, no week: recent game
    assert not live.game_started("DAL", _utc(2026, 11, 16), s)  # last game too long ago without a week
    assert not live.game_started("BUF", sunday, pd.DataFrame(), 11)


# ── odds ─────────────────────────────────────────────────────────────────────

GAME_DAY = _utc(2026, 10, 25, 12).timestamp()


def _odds(monkeypatch, response):
    monkeypatch.setenv("ODDS_API_KEY", "k")
    return _route(monkeypatch, vegas, vegas.BASE_URL, {"/sports/americanfootball_nfl/odds": response})


def _ok(remaining=None):
    resp = FakeResponse(200, [_game("Buffalo Bills", "New York Jets", [_book(total=44, spread=-7)])])
    resp.headers = {} if remaining is None else {"x-requests-remaining": str(remaining)}
    return resp


def test_week_window_runs_tuesday_to_tuesday():
    assert vegas.nfl_week_window(_utc(2026, 10, 27, 9)) == (_utc(2026, 10, 20, 10), _utc(2026, 10, 27, 10))
    assert vegas.nfl_week_window(datetime(2026, 10, 27, 11))[0] == _utc(2026, 10, 27, 10)
    bad = {"home_team": "Buffalo Bills", "away_team": "New York Jets", "commence_time": "soon", "bookmakers": []}
    assert vegas.parse_implied_totals([bad], _utc(2026, 10, 22)) == {}


def test_odds_cache_records_fetch_time_and_quota(monkeypatch):
    _odds(monkeypatch, _ok(remaining="480.0"))
    assert vegas.fetch_implied_totals(now=GAME_DAY) == {"BUF": 25.5, "NYJ": 18.5}
    assert vegas.vegas_status() == {"fetched_at": GAME_DAY, "remaining": 480}
    router = _odds(monkeypatch, _ok())
    assert vegas.fetch_implied_totals(now=GAME_DAY + 60)["BUF"] == 25.5 and router.calls == []  # within TTL


def test_low_quota_skips_fetching_for_a_day(monkeypatch):
    _odds(monkeypatch, _ok(remaining=5))
    vegas.fetch_implied_totals(now=GAME_DAY)
    router = _odds(monkeypatch, _ok(remaining=5))
    assert vegas.fetch_implied_totals(force=True, now=GAME_DAY + 3600)["BUF"] == 25.5
    assert router.calls == []
    vegas.fetch_implied_totals(force=True, now=GAME_DAY + vegas.LOW_QUOTA_BACKOFF_S + 1)
    assert len(router.calls) == 1  # a daily check notices the monthly reset


def test_failed_fetch_records_quota_and_drops_stale_totals(monkeypatch):
    _odds(monkeypatch, _ok())
    vegas.fetch_implied_totals(now=GAME_DAY)
    limited = FakeResponse(429)
    limited.headers = {"x-requests-remaining": "0"}
    _odds(monkeypatch, limited)
    assert vegas.fetch_implied_totals(force=True, now=GAME_DAY + 10)["BUF"] == 25.5
    assert vegas.vegas_status()["remaining"] == 0
    later = GAME_DAY + vegas.MAX_STALE_S + vegas.LOW_QUOTA_BACKOFF_S
    assert vegas.fetch_implied_totals(now=later) == {}  # day-old totals aren't served


@pytest.mark.parametrize("content", ["[1, 2]", "{broken"])
def test_odds_cache_that_is_not_an_object_is_ignored(monkeypatch, content):
    storage.data_path("vegas_totals.json").write_text(content)
    _odds(monkeypatch, FakeResponse(500))
    assert vegas.fetch_implied_totals(now=GAME_DAY) == {}


def test_unreadable_odds_cache_is_ignored(monkeypatch):
    storage.data_path("vegas_totals.json").mkdir()
    assert vegas.vegas_status() == {"fetched_at": None, "remaining": None}
    assert vegas._usable({"totals": {"BUF": 1}}, GAME_DAY) == {}
