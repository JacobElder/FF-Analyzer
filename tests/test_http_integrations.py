"""
External APIs with mocked HTTP: Sleeper, The Odds API, Sleeper projections and
Open-Meteo. Payloads are shaped like the real responses (trimmed).
"""

import logging
from datetime import date, timedelta

import pytest
import requests

import src.projections as proj
import src.schedule as schedule
import src.sleeper_integration as si
import src.storage as storage
import src.vegas as vegas
from http_helpers import FakeResponse, _route


# ── Sleeper: low-level GET ───────────────────────────────────────────────────

@pytest.mark.parametrize("response,expected", [
    (requests.Timeout("timed out"), "Network error"),
    (FakeResponse(500), "HTTP 500"),
    (FakeResponse(200, text="<html>"), "invalid JSON"),
])
def test_sleeper_get_reports_errors(monkeypatch, response, expected):
    _route(monkeypatch, si, si.BASE_URL, {"/x": response})
    data, err = si._get("/x")
    assert data is None and expected in err


def test_sleeper_get_returns_null_body_without_error(monkeypatch):
    _route(monkeypatch, si, si.BASE_URL, {"/x": FakeResponse(200, None)})
    assert si._get("/x") == (None, None)  # Sleeper's "not found" for some endpoints


# ── Sleeper: player database cache ───────────────────────────────────────────

DB = {"4046": {"full_name": "Patrick Mahomes", "position": "QB", "team": "KC", "active": True}}


def test_players_db_fetched_then_served_from_cache(monkeypatch):
    router = _route(monkeypatch, si, si.BASE_URL, {"/players/nfl": DB})
    assert si.fetch_all_players() == DB
    assert si.fetch_all_players() == DB
    assert router.calls == ["/players/nfl"]  # second call hit the file cache
    assert si.players_db_age_seconds() is not None


def test_players_db_falls_back_to_stale_cache(monkeypatch):
    storage.write_json_atomic(DB, si._players_cache())
    _route(monkeypatch, si, si.BASE_URL, {"/players/nfl": FakeResponse(503)})
    assert si.fetch_all_players(force=True) == DB


def test_players_db_empty_without_cache_or_network(monkeypatch):
    _route(monkeypatch, si, si.BASE_URL, {"/players/nfl": requests.ConnectionError("down")})
    assert si.fetch_all_players() == {}


# ── Sleeper: NFL state ───────────────────────────────────────────────────────

@pytest.mark.parametrize("season_type,week", [("regular", 7), ("pre", None), ("off", None)])
def test_nfl_state_week_only_in_regular_season(monkeypatch, season_type, week):
    _route(monkeypatch, si, si.BASE_URL, {"/state/nfl": {"season": 2026, "week": 7, "season_type": season_type}})
    assert si.get_nfl_state() == {"season": "2026", "week": week, "season_type": season_type}


def test_nfl_state_falls_back_to_calendar(monkeypatch):
    _route(monkeypatch, si, si.BASE_URL, {"/state/nfl": FakeResponse(500)})
    state = si.get_nfl_state()
    assert state["week"] is None and state["season"].isdigit()


# ── Sleeper: full league sync ────────────────────────────────────────────────

LID = "1180000000000000001"
PLAYERS = {
    "qb1": {"full_name": "Joe Burrow", "position": "QB", "team": "CIN", "active": True},
    "wr1": {"full_name": "Ja'Marr Chase", "position": "WR", "team": "CIN", "active": True, "gsis_id": "00-1"},
    "rb1": {"full_name": "Hurt Back", "position": "RB", "team": "NYJ", "active": True},
    "te1": {"full_name": "Taxi Squad", "position": "TE", "team": "DAL", "active": True},
    "opp": {"full_name": "Rival Guy", "position": "WR", "team": "BUF", "active": True},
    "fa1": {"full_name": "Free Agent", "position": "WR", "team": "MIA", "active": True},
    "fa2": {"full_name": "Retired Guy", "position": "WR", "team": None, "active": False},
}


def _sleeper_routes(**overrides):
    routes = {
        "/user/me": {"user_id": "u1", "display_name": "me"},
        "/state/nfl": {"season": "2026", "week": 3, "season_type": "regular"},
        f"/league/{LID}": {"league_id": LID, "name": "The League", "season": "2026", "status": "in_season",
                           "total_rosters": 2,
                           "roster_positions": ["QB", "WR", "FLEX", "BN", "BN", "IR"],
                           "settings": {"waiver_budget": 100, "waiver_type": 2, "playoff_week_start": 15,
                                        "waiver_bid_min": 1, "playoff_teams": 1},
                           "scoring_settings": {"rec": 1.0}},
        f"/league/{LID}/rosters": [
            {"roster_id": 1, "owner_id": "someone", "co_owners": ["u1"], "players": ["qb1", "wr1", "rb1"],
             "starters": ["qb1", "wr1", "0"], "reserve": ["rb1"], "taxi": ["te1"],
             "settings": {"wins": 2, "losses": 0, "fpts": 250, "fpts_decimal": 50, "waiver_budget_used": 15,
                          "waiver_position": 2}},
            {"roster_id": 2, "owner_id": "u2", "players": ["opp"], "starters": ["opp"],
             "settings": {"wins": 0, "losses": 2, "fpts": 180, "waiver_position": 1}},
        ],
        f"/league/{LID}/users": [{"user_id": "someone", "metadata": {"team_name": "Co-owned Crew"}},
                                 {"user_id": "u2", "display_name": "rival"}],
        f"/league/{LID}/matchups/3": [{"roster_id": 1, "matchup_id": 1, "points": 10.5, "starters": ["qb1"]},
                                      {"roster_id": 2, "matchup_id": 1, "points": 8.0, "starters": ["opp"]}],
        f"/league/{LID}/matchups/2": [{"roster_id": 1, "matchup_id": None}, {"roster_id": 2, "matchup_id": None}],
        f"/league/{LID}/transactions/1": [{"type": "waiver", "status": "complete", "leg": 1, "roster_ids": [2],
                                           "adds": {"opp": 2}, "settings": {"waiver_bid": 12}, "created": 1}],
        f"/league/{LID}/transactions/2": [{"type": "trade", "status": "complete"}],
        f"/league/{LID}/transactions/3": FakeResponse(500),
        "/players/nfl/trending/add": [{"player_id": "fa1", "count": 300}],
        "/players/nfl": PLAYERS,
    }
    routes.update(overrides)
    return routes


def test_full_sync_builds_league_data(monkeypatch):
    _route(monkeypatch, si, si.BASE_URL, _sleeper_routes())
    data, err = si.sync_sleeper_league("me", LID)
    assert err is None
    assert data["team_name"] == "Co-owned Crew"  # co-owner finds the roster; custom team name wins
    assert data["budget"] == 85 and data["faab_enabled"] and data["week"] == 3
    assert data["record"] == {"wins": 2, "losses": 0, "ties": 0, "points_for": 250.5, "waiver_position": 2}
    slots = {p["player_id"]: p["slot"] for p in data["roster"]}
    assert slots == {"qb1": "QB", "wr1": "WR", "rb1": "IR", "te1": "Taxi"}
    assert data["matchup"]["opponent"] == "rival" and data["matchup"]["opponent_points"] == 8.0
    assert data["last_result"] == {"week": 2, "bye": True}
    assert [p["player_id"] for p in data["waivers"]] == ["fa1"]  # inactive / rostered excluded
    assert data["waivers"][0]["trend_count"] == 300
    assert [t["bid"] for t in data["transactions"]] == [12]  # trades skipped, failed week tolerated
    assert data["waiver_settings"]["waiver_bid_min"] == 1
    assert data["synced_at"].endswith("+00:00")


def test_sync_finds_league_by_name_in_previous_season(monkeypatch):
    routes = _sleeper_routes(**{
        "/user/u1/leagues/nfl/2026": [{"league_id": "x", "name": None}],  # unnamed league must not crash
        "/user/u1/leagues/nfl/2025": [{"league_id": LID, "name": "The League"}],
    })
    routes[f"/league/{LID}"] = routes[f"/league/{LID}"]
    _route(monkeypatch, si, si.BASE_URL, routes)
    league, err, _ = si._find_league("u1", "the league")
    assert err is None and league["league_id"] == LID


@pytest.mark.parametrize("overrides,message", [
    ({"/user/me": None}, "not found"),
    ({f"/league/{LID}/rosters": FakeResponse(200, None)}, "no rosters"),
    ({f"/league/{LID}/rosters": [{"roster_id": 9, "owner_id": "x"}]}, "doesn't own a roster"),
    ({"/players/nfl": FakeResponse(500)}, "player database"),
])
def test_sync_failures_return_messages(monkeypatch, overrides, message):
    _route(monkeypatch, si, si.BASE_URL, _sleeper_routes(**overrides))
    data, err = si.sync_sleeper_league("me", LID)
    assert data is None and message in err


def test_sync_requires_inputs():
    assert si.sync_sleeper_league("", "x")[0] is None


# ── The Odds API ─────────────────────────────────────────────────────────────

def _game(home, away, books, t="2026-10-25T17:00:00Z"):
    return {"home_team": home, "away_team": away, "commence_time": t, "bookmakers": books}


def _book(total=None, spread=None, home="Buffalo Bills"):
    markets = []
    if total is not None:
        markets.append({"key": "totals", "outcomes": [{"name": "Over", "point": total}]})
    if spread is not None:
        markets.append({"key": "spreads", "outcomes": [{"name": home, "point": spread}]})
    return {"markets": markets}


def test_vegas_uses_later_bookmaker_when_first_lacks_a_market():
    totals = vegas.parse_implied_totals([_game("Buffalo Bills", "New York Jets", [_book(total=44), _book(spread=-7)])])
    assert totals == {"BUF": 25.5, "NYJ": 18.5}


def test_vegas_without_key_makes_no_request(monkeypatch):
    monkeypatch.setattr(vegas.requests, "get", lambda *a, **k: pytest.fail("no request expected"))
    assert vegas.fetch_implied_totals() == {}


def test_vegas_fetch_caches_and_falls_back_to_stale(monkeypatch):
    monkeypatch.setenv("ODDS_API_KEY", "secret-key")
    games = [_game("Buffalo Bills", "New York Jets", [_book(total=44, spread=-7)])]
    _route(monkeypatch, vegas, vegas.BASE_URL, {"/sports/americanfootball_nfl/odds": games})
    assert vegas.fetch_implied_totals()["BUF"] == 25.5
    _route(monkeypatch, vegas, vegas.BASE_URL, {"/sports/americanfootball_nfl/odds": FakeResponse(429)})
    assert vegas.fetch_implied_totals(force=True)["BUF"] == 25.5  # rate-limited: keep last good data


def test_vegas_error_log_never_contains_api_key(monkeypatch, caplog):
    monkeypatch.setenv("ODDS_API_KEY", "secret-key")

    def boom(url, params=None, **_):
        raise requests.ConnectionError(f"Max retries exceeded with url: /odds?apiKey={params['apiKey']}")
    monkeypatch.setattr(vegas.requests, "get", boom)
    with caplog.at_level(logging.WARNING):
        assert vegas.fetch_implied_totals() == {}
    assert "secret-key" not in caplog.text and "***" in caplog.text


# ── Sleeper projections ──────────────────────────────────────────────────────

def test_projections_keep_explicit_zeros_and_drop_empty_rows(monkeypatch):
    rows = [{"player_id": "a", "stats": {"pts_ppr": 12.5}},
            {"player_id": "b", "stats": {"pts_ppr": 0.0, "pts_std": 0.0}},  # projected not to play
            {"player_id": "c", "stats": {}},
            {"player_id": None, "stats": {"pts_ppr": 5}}]
    monkeypatch.setattr(proj.requests, "get", lambda *a, **k: FakeResponse(200, rows))
    out = proj.fetch_week_projections(2026, 3)
    assert set(out) == {"a", "b"}
    p = proj.Projections(2026, [3], {"rec": 1.0})
    assert p.get("b", 3) == 0.0 and p.get("c", 3) is None and p.available


def test_projections_fall_back_to_stale_file_or_empty(monkeypatch):
    monkeypatch.setattr(proj.requests, "get", lambda *a, **k: FakeResponse(500))
    assert proj.fetch_week_projections(2026, 4) == {}
    storage.write_json_atomic({"a": {"pts_ppr": 3}}, storage.data_path("projections_2026_4.json"))
    assert proj.fetch_week_projections(2026, 4, force=True) == {"a": {"pts_ppr": 3}}


# ── Weather ──────────────────────────────────────────────────────────────────

def _future():
    return str(date.today() + timedelta(days=2))


def test_weather_dome_and_past_games_skip_the_request(monkeypatch):
    monkeypatch.setattr(schedule.requests, "get", lambda *a, **k: pytest.fail("no request expected"))
    assert schedule.get_weather_for_game("DET", _future())["dome"]
    assert schedule.get_weather_for_game("BUF", "2000-01-01") is None
    assert schedule.get_weather_for_game("BUF", "None") is None
    assert schedule.get_weather_for_game("XXX", _future()) is None


@pytest.mark.parametrize("daily,flags", [
    ({"temperature_2m_max": [50], "precipitation_sum": [0.0], "windspeed_10m_max": [5]}, 0),
    ({"temperature_2m_max": [34], "precipitation_sum": [0.5], "windspeed_10m_max": [15]}, 3),
    ({}, 0),
])
def test_weather_flags_at_thresholds(monkeypatch, daily, flags):
    monkeypatch.setattr(schedule.requests, "get", lambda *a, **k: FakeResponse(200, {"daily": daily}))
    out = schedule.get_weather_for_game("BUF", _future())
    assert out["dome"] is False and len(out["flags"]) == flags


def test_weather_http_error_is_none(monkeypatch):
    monkeypatch.setattr(schedule.requests, "get", lambda *a, **k: FakeResponse(503))
    assert schedule.get_weather_for_game("BUF", _future()) is None
