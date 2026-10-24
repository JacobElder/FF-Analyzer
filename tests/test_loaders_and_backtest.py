"""File-cached loaders (schedule, weekly stats, odds) and the backtest, with downloads stubbed."""

import pandas as pd
import pytest

import src.backtest as backtest
import src.schedule as schedule
import src.sleeper_integration as si
import src.vegas as vegas
import src.wrangler as wrangler
from src.players import normalize_name, sleeper_display_name
from src.storage import data_path, write_json_atomic
from http_helpers import FakeResponse, _route


def _games():
    return pd.DataFrame([
        {"season": 2026, "game_type": "REG", "week": 1, "home_team": "LA", "away_team": "OAK"},
        {"season": 2026, "game_type": "POST", "week": 19, "home_team": "BUF", "away_team": "KC"},
        {"season": 2025, "game_type": "REG", "week": 1, "home_team": "BUF", "away_team": "NYJ"},
    ])


def test_fetch_schedules_filters_normalizes_and_caches(monkeypatch):
    calls = []
    real = pd.read_parquet

    def read(src, *a, **k):
        calls.append(str(src))
        return _games() if str(src).startswith("http") else real(src, *a, **k)
    monkeypatch.setattr(schedule.pd, "read_parquet", read)
    df = schedule.fetch_schedules(2026)
    assert df[["week", "home_team", "away_team"]].values.tolist() == [[1, "LAR", "LV"]]
    schedule.fetch_schedules(2026)  # served from the file cache
    assert sum(c.startswith("http") for c in calls) == 1


def test_fetch_schedules_falls_back_to_stale_cache_or_empty(monkeypatch):
    real = pd.read_parquet

    def read(src, *a, **k):
        if str(src).startswith("http"):
            raise OSError("offline")
        return real(src, *a, **k)
    monkeypatch.setattr(schedule.pd, "read_parquet", read)
    assert schedule.fetch_schedules(2026).empty
    _games().to_parquet(data_path("schedules_2026.parquet"))
    assert len(schedule.fetch_schedules(2026, force=True)) == 3


def test_load_season_downloads_caches_and_skips_empty(monkeypatch):
    frames = {2026: pd.DataFrame({"player_id": ["a"], "week": [1]}), 2027: pd.DataFrame()}
    real = pd.read_parquet

    def read(src, *a, **k):
        s = str(src)
        if s.startswith("http"):
            return frames[2026 if "2026" in s else 2027]
        return real(src, *a, **k)
    monkeypatch.setattr(wrangler.pd, "read_parquet", read)
    assert len(wrangler._load_season(2026, force=False)) == 1
    assert data_path("weekly_stats_2026.parquet").exists()
    assert len(wrangler._load_season(2026, force=False)) == 1  # fresh cache
    assert wrangler._load_season(2027, force=False).empty
    assert not data_path("weekly_stats_2027.parquet").exists()  # empty downloads aren't cached


def test_load_season_with_nothing_available(monkeypatch):
    monkeypatch.setattr(wrangler.pd, "read_parquet", lambda *a, **k: (_ for _ in ()).throw(OSError("offline")))
    assert wrangler._load_season(2026, force=True).empty


def test_vegas_fresh_cache_skips_request(monkeypatch):
    monkeypatch.setenv("ODDS_API_KEY", "k")
    write_json_atomic({"BUF": 24.0}, data_path("vegas_totals.json"))
    monkeypatch.setattr(vegas.requests, "get", lambda *a, **k: pytest.fail("cache should be used"))
    assert vegas.fetch_implied_totals() == {"BUF": 24.0}


def test_vegas_skips_unknown_teams_and_later_games():
    def g(home, away, t, total=44, spread=-3):
        return {"home_team": home, "away_team": away, "commence_time": t,
                "bookmakers": [{"markets": [{"key": "totals", "outcomes": [{"name": "Over", "point": total}]},
                                            {"key": "spreads", "outcomes": [{"name": home, "point": spread}]}]}]}
    totals = vegas.parse_implied_totals([
        g("London Monarchs", "Buffalo Bills", "2026-10-01"),
        g("Buffalo Bills", "Miami Dolphins", "2026-10-02"),
        g("Miami Dolphins", "Buffalo Bills", "2026-10-09", total=60),  # both already set: ignored
    ])
    assert totals == {"BUF": 23.5, "MIA": 20.5}


def test_trending_players_error_returns_empty(monkeypatch):
    _route(monkeypatch, si, si.BASE_URL, {"/players/nfl/trending/add": FakeResponse(500)})
    assert si.get_trending_players() == []


def test_find_league_errors(monkeypatch):
    routes = {"/state/nfl": {"season": "2026", "week": 1, "season_type": "regular"},
              "/user/u1/leagues/nfl/2026": [], "/user/u1/leagues/nfl/2025": []}
    _route(monkeypatch, si, si.BASE_URL, routes)
    assert si.get_nfl_season() == "2026"
    league, err, _ = si._find_league("u1", "Missing League")
    assert league is None and "Could not find a league named 'Missing League'" in err
    _route(monkeypatch, si, si.BASE_URL, {**routes, "/user/u1/leagues/nfl/2026": FakeResponse(500)})
    assert "HTTP 500" in si._find_league("u1", "x")[1]
    _route(monkeypatch, si, si.BASE_URL, {**routes, "/league/1180000000000000009": FakeResponse(404)})
    assert si._find_league("u1", "1180000000000000009")[1]


def test_display_names():
    assert normalize_name("") == ""
    assert sleeper_display_name({"first_name": "Buffalo", "last_name": "Bills"}) == "Buffalo Bills"
    assert sleeper_display_name({}, fallback="BUF") == "BUF"


# ── backtest ─────────────────────────────────────────────────────────────────

class FakeProjections:
    def __init__(self, season, weeks, scoring, completed_through=0):
        self.weeks = list(weeks)

    def get(self, sid, week):
        return 14.0 if sid == "s1" else None


def _season_stats():
    rows = [("g1", "Steady Guy", "WR", w, 0, 0, 5, 80, 0) for w in range(1, 10)]
    rows += [("g2", "Name Match", "RB", w, 60, 0, 2, 10, 0) for w in range(1, 10)]
    rows += [("g3", "Kicker Guy", "K", w, 0, 0, 0, 0, 0) for w in range(1, 10)]
    df = pd.DataFrame(rows, columns=["player_id", "player_display_name", "position", "week", "rushing_yards",
                                     "rushing_tds", "receptions", "receiving_yards", "receiving_tds"])
    df["season"] = 2025
    return df


def test_backtest_scores_history_and_blend_models(monkeypatch):
    monkeypatch.setattr(backtest, "fetch_weekly_stats", lambda season: _season_stats())
    monkeypatch.setattr(backtest, "fetch_all_players", lambda: {
        "s1": {"full_name": "Steady Guy", "position": "WR", "gsis_id": "g1", "team": "NYJ", "active": True},
        "s2": {"full_name": "Name Match", "position": "RB", "team": "NYJ", "active": True},
    })
    monkeypatch.setattr(backtest, "Projections", FakeProjections)
    results = backtest.run_backtest(2025, first_week=4, last_week=8)
    assert set(results) == {"next_week", "next_4"}
    kinds = {k[0] for k, _, _ in results["next_week"]}
    assert kinds == {"history", "blend"}  # g1 has a Sleeper projection, so blends are scored
    maes = [mae for _, mae, _ in results["next_week"]]
    assert maes == sorted(maes)
    best_history = next(r for r in results["next_week"] if r[0][0] == "history")
    assert best_history[1] == pytest.approx(0.0) and best_history[2] == 2 * 5  # flat lines: perfect history


def test_backtest_main_prints_report(monkeypatch, capsys):
    monkeypatch.setattr(backtest, "run_backtest", lambda season: {
        "next_week": [(("history", (0.35, 6)), 4.2, 100), (("blend", 0.8), 3.9, 90)]})
    monkeypatch.setattr(backtest.sys, "argv", ["backtest", "2024"])
    backtest.main()
    out = capsys.readouterr().out
    assert "history recent_weight=0.35 window=6" in out and "blend α=0.8" in out and "n=  100" in out
