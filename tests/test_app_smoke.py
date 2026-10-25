"""
Streamlit app tests: the whole script runs offline against a temp data dir with
every data source stubbed (no network, no .env). An in-season league fixture
gets each tab past its empty states; the tests then click through the main
interactions and assert nothing raises.
"""

from datetime import date, timedelta
from pathlib import Path

import dotenv
import pandas as pd
import pytest

pytest.importorskip("streamlit.testing.v1")
import streamlit as st  # noqa: E402
from streamlit.testing.v1 import AppTest  # noqa: E402

import src.config  # noqa: E402
import src.projections  # noqa: E402
import src.schedule  # noqa: E402
import src.sleeper_integration  # noqa: E402
import src.state_manager as sm  # noqa: E402
import src.vegas  # noqa: E402
import src.wrangler  # noqa: E402

pytestmark = pytest.mark.app

APP = str(Path(__file__).resolve().parent.parent / "app.py")
SEASON, WEEK = 2026, 3

# Sleeper players: id → (name, position, team, gsis_id, injury)
PLAYERS = {
    "qb1": ("Joe Burrow", "QB", "CIN", "00-1", None),
    "wr1": ("Ja'Marr Chase", "WR", "CIN", "00-2", None),
    "rb1": ("Bijan Robinson", "RB", "ATL", "00-4", None),
    "te1": ("Travis Kelce", "TE", "KC", "00-5", None),
    "wr2": ("Bench Guy", "WR", "NYJ", "00-6", "Questionable"),
    "rb2": ("Hurt Back", "RB", "NYJ", "00-9", "IR"),
    "fa1": ("Free Agent", "WR", "MIA", "00-3", None),
    "fa2": ("Waiver Back", "RB", "MIA", "00-7", "Out"),
    "fa3": ("Rookie Nobody", "TE", "NYJ", None, None),    # NYJ plays at KC: no forecast yet
    "fa6": ("Seattle Guy", "WR", "SEA", None, None),      # team not on the schedule
    "k1": ("Kicker Guy", "K", "BUF", None, None),
    "opp1": ("Rival Receiver", "WR", "BUF", "00-8", None),
    "fa4": ("Next Man Up", "RB", "MIA", "00-10", None),    # backup behind an Out starter: promotion
    "fa5": ("Cuff Back", "RB", "ATL", "00-11", None),      # backup to your RB1: handcuff
    "rk1": ("Roster Rookie", "WR", "SEA", None, None),     # no stats, team not on the schedule
}
PROJ = {"qb1": 22, "wr1": 18, "rb1": 16, "te1": 11, "wr2": 9, "rb2": 0, "fa1": 15, "fa2": 7, "fa3": 4,
        "k1": 8, "opp1": 12, "d1": 7, "fa4": 10, "fa5": 6, "rk1": 3, "fa6": 2}
PAIRS = [("CIN", "BUF"), ("ATL", "MIA"), ("KC", "NYJ")]


def _players_db():
    db = {pid: {"full_name": n, "position": pos, "team": t, "gsis_id": g, "injury_status": inj, "active": True}
          for pid, (n, pos, t, g, inj) in PLAYERS.items()}
    db["d1"] = {"first_name": "Buffalo", "last_name": "Bills", "position": "DEF", "team": "BUF", "active": True}
    db["fa2"].update({"depth_chart_order": 1, "depth_chart_position": "RB"})
    db["rb1"].update({"depth_chart_order": 1, "depth_chart_position": "RB"})
    db["fa4"].update({"depth_chart_order": 2, "depth_chart_position": "RB"})
    db["fa5"].update({"depth_chart_order": 2, "depth_chart_position": "RB"})
    return db


def _weekly_stats():
    rows = []
    for week in (1, 2):
        for pid, (n, pos, team, gid, _) in PLAYERS.items():
            if not gid:
                continue
            opp = next(b if a == team else a for a, b in PAIRS if team in (a, b))
            base = PROJ[pid] / 10
            rows.append({"player_id": gid, "player_display_name": n, "position": pos, "team": team,
                         "opponent_team": opp, "week": week, "season": SEASON, "season_type": "REG",
                         "passing_yards": 250 * base if pos == "QB" else 0, "passing_tds": 2 if pos == "QB" else 0,
                         "rushing_yards": 60 * base if pos == "RB" else 0,
                         "receptions": 5 * base if pos in ("WR", "TE") else 1,
                         "receiving_yards": 60 * base * week if pos in ("WR", "TE") else 5,
                         "receiving_tds": week - 1, "target_share": 0.2, "carries": 10 if pos == "RB" else None})
    return pd.DataFrame(rows).sort_values(["player_id", "week"]).reset_index(drop=True)


def _schedules():
    rows, start = [], date.today() + timedelta(days=3)
    for week in range(1, 18):
        for i, (a, b) in enumerate(PAIRS):
            home, away = (a, b) if week % 2 else (b, a)
            played = week < WEEK
            rows.append({"season": SEASON, "game_type": "REG", "week": week, "home_team": home, "away_team": away,
                         "home_score": 20.0 if played else None,
                         "gameday": str(start + timedelta(days=7 * (week - WEEK))), "gametime": "13:00"})
    return pd.DataFrame(rows)


def _league(**over):
    slots = {"qb1": "QB", "rb1": "RB", "wr1": "WR", "te1": "TE", "wr2": "FLEX", "k1": "K", "rb2": "IR",
             "rk1": "BN"}
    roster = [{"name": PLAYERS[pid][0], "position": PLAYERS[pid][1], "team": PLAYERS[pid][2], "player_id": pid,
               "gsis_id": PLAYERS[pid][3], "slot": slot} for pid, slot in slots.items()]
    waivers = [{"name": PLAYERS[pid][0], "position": PLAYERS[pid][1], "team": PLAYERS[pid][2], "player_id": pid,
                "gsis_id": PLAYERS[pid][3], "trend_count": t} for pid, t in (("fa1", 900), ("fa2", 40), ("fa3", 0),
                                                                          ("fa4", 300), ("fa5", 20), ("fa6", 1))]
    waivers.append({"name": "Buffalo Bills", "position": "DEF", "team": "BUF", "player_id": "d1", "trend_count": 5})
    data = {
        "league_id": "111", "league_name": "Smoke League", "season": str(SEASON), "current_season": str(SEASON),
        "team_name": "Mine", "username": "me", "week": WEEK, "status": "in_season", "platform": "Sleeper",
        "budget": 60, "waiver_budget_total": 100, "faab_enabled": True, "total_rosters": 2,
        "playoff_week_start": 15, "playoff_teams": 1, "roster_id": 1,
        "record": {"wins": 1, "losses": 1, "ties": 0, "points_for": 210.5, "waiver_position": 2},
        "roster_positions": ["QB", "RB", "WR", "TE", "FLEX", "K", "DEF", "BN", "BN", "IR"],
        "roster": roster, "waivers": waivers, "scoring_settings": {"rec": 1, "pass_td": 4},
        "waiver_settings": {"waiver_type": 2, "waiver_day_of_week": 2, "waiver_bid_min": 0},
        "league_rosters": [
            {"roster_id": 1, "owner_id": "me", "team_name": "Mine", "players": list({p["player_id"] for p in roster}),
             "reserve": ["rb2"], "faab_remaining": 60, "record": {"wins": 1, "losses": 1, "points_for": 210.5,
                                                                 "waiver_position": 2}},
            {"roster_id": 2, "owner_id": "u2", "team_name": "Rival", "players": ["opp1"], "reserve": [],
             "faab_remaining": 80, "record": {"wins": 2, "losses": 0, "points_for": 250.0, "waiver_position": 1}},
        ],
        "transactions": [
            {"week": 1, "type": "waiver", "status": "complete", "bid": 12, "roster_id": 1, "adds": ["wr2"],
             "drops": [], "note": None},
            {"week": 2, "type": "waiver", "status": "complete", "bid": 30, "roster_id": 2, "adds": ["opp1"],
             "drops": [], "note": None},
            {"week": 2, "type": "waiver", "status": "failed", "bid": 20, "roster_id": 1, "adds": ["opp1"],
             "drops": [], "note": "claimed by another owner"},
        ],
        "matchup": {"week": WEEK, "points": 0, "starters": ["qb1", "rb1", "wr1", "te1"], "opponent": "Rival",
                    "opponent_roster_id": 2, "opponent_points": 0, "opponent_starters": ["opp1"],
                    "opponent_record": {"wins": 2, "losses": 0, "ties": 0, "points_for": 250.0}},
        "last_result": {"week": 2, "points": 120.5, "opponent": "Rival", "opponent_points": 99.0,
                        "starters": [], "opponent_starters": []},
        "synced_at": "2026-10-24T00:00:00+00:00",
    }
    data.update(over)
    return data


def _weather(home, day):
    if home == "ATL":
        return {"dome": True, "note": "Indoor"}
    if home == "KC":
        return None  # beyond the forecast window
    return {"dome": False, "flags": ["⚠️ Wind 20 mph"], "note": "⚠️ Wind 20 mph", "temp_f": 40,
            "wind_mph": 20, "precip_mm": 0}


@pytest.fixture(autouse=True)
def offline_app(monkeypatch):
    st.cache_data.clear()
    st.cache_resource.clear()
    monkeypatch.setattr(dotenv, "load_dotenv", lambda *a, **k: False)
    monkeypatch.delenv("AI_PASSWORD", raising=False)
    monkeypatch.setattr(src.config, "load_config",
                        lambda *a, **k: {"username": "", "league_ids": [], "auto_sync_minutes": 30.0})
    monkeypatch.setattr(src.wrangler, "fetch_weekly_stats", lambda *a, **k: _weekly_stats())
    monkeypatch.setattr(src.wrangler, "current_nfl_season", lambda: SEASON)
    monkeypatch.setattr(src.schedule, "fetch_schedules", lambda *a, **k: _schedules())
    monkeypatch.setattr(src.schedule, "get_weather_for_game", _weather)
    monkeypatch.setattr(src.sleeper_integration, "fetch_all_players", lambda *a, **k: _players_db())
    monkeypatch.setattr(src.sleeper_integration, "players_db_age_seconds", lambda: 7200.0)
    monkeypatch.setattr(src.vegas, "fetch_implied_totals", lambda *a, **k: {"BUF": 24.5, "CIN": 27.0, "MIA": 20.0})
    monkeypatch.setattr(src.projections, "fetch_week_projections",
                        lambda season, week, **k: {pid: {"pts_ppr": float(v)} for pid, v in PROJ.items()})
    yield
    st.cache_data.clear()
    st.cache_resource.clear()


def _run(at=None):
    at = at or AppTest.from_file(APP, default_timeout=60)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def _widget(widgets, label):
    for w in widgets:
        if w.label == label:
            return w
    raise AssertionError(f"no widget labelled {label!r}; have {[w.label for w in widgets]}")


def _text(at) -> str:
    parts = [m.value for m in at.markdown] + [c.value for c in at.caption] + [i.value for i in at.info] \
        + [w.value for w in at.warning] + [e.value for e in at.error] + [s.value for s in at.success]
    return "\n".join(str(p) for p in parts)


# ── tests ────────────────────────────────────────────────────────────────────

def test_app_runs_with_no_leagues():
    at = _run()
    assert not at.error, [e.value for e in at.error]
    assert at.title[0].value == "🏈 FF Analyzer"
    assert "No leagues yet" in "\n".join(i.value for i in at.sidebar.info)


def test_app_without_stats_stops_with_a_message(monkeypatch, leagues_dir):
    monkeypatch.setattr(src.wrangler, "fetch_weekly_stats", lambda *a, **k: pd.DataFrame())
    sm.save_league(_league())
    at = _run()
    assert "Couldn't load NFL stats" in at.error[0].value and not at.tabs


def test_in_season_league_renders_every_tab(leagues_dir):
    sm.save_league(_league())
    at = _run()
    assert [t.label for t in at.tabs][:6] and len(at.tabs) >= 6
    text = _text(at)
    assert "Bid recommendations" in text            # waiver engine ran
    assert "Claims to submit for week 3" in text     # planner ran
    assert at.sidebar.selectbox[0].value == "111"


def test_start_sit_compare(leagues_dir):
    sm.save_league(_league())
    at = _run()
    _widget(at.multiselect, "Select 2–4 players to compare").select("Ja'Marr Chase").select("Bench Guy")
    _run(at)
    _widget(at.button, "⚖️ Compare").click()
    _run(at)
    assert "START: " in _text(at)
    for style in ("Safe", "Upside"):
        _widget(at.radio, "Start/sit style").set_value(style)
        _run(at)
        assert "START: " in _text(at)


def test_trade_analysis_with_and_without_roster_context(leagues_dir):
    sm.save_league(_league())
    at = _run()
    _widget(at.multiselect, "You Give (from your roster)").select("Bench Guy")
    _widget(at.multiselect, "You Receive").select("opp1")
    _run(at)
    _widget(at.button, "🤝 Analyze Trade").click()
    _run(at)
    assert any(v in _text(at) for v in ("ACCEPT", "DECLINE", "NEGOTIATE"))


def test_waiver_browsing_and_deep_dive(leagues_dir):
    sm.save_league(_league())
    at = _run()
    _widget(at.checkbox, "Hide Out / IR / Doubtful").uncheck()
    _widget(at.multiselect, "Filter by Position").select("DEF")
    _run(at)
    for sort in ("Season PPG", "Recent PPG"):
        _widget(at.radio, "Sort by").set_value(sort)
        _run(at)
    pick = _widget(at.selectbox, "Select player")
    for i in range(len(pick.options)):
        _widget(at.selectbox, "Select player").set_value(i)
        _run(at)
    assert "Player Deep Dive" in "\n".join(s.value for s in at.subheader)


def test_ai_buttons_without_keys_show_setup_warning(leagues_dir):
    sm.save_league(_league())
    at = _run()
    _widget(at.multiselect, "Select 2–4 players to compare").select("Ja'Marr Chase").select("Bench Guy")
    _widget(at.multiselect, "You Give (from your roster)").select("Bench Guy")
    _run(at)
    _widget(at.button, "⚖️ Compare").click()
    _run(at)
    _widget(at.button, "🤝 Analyze Trade").click()
    _run(at)
    ai_buttons = [b for b in at.button if "AI" in b.label]
    assert ai_buttons
    for label in [b.label for b in ai_buttons]:
        _widget(at.button, label).click()
        _run(at)
    assert "API key not configured" in _text(at)



def test_ai_password_locks_ai_buttons_until_entered(monkeypatch, leagues_dir):
    monkeypatch.setenv("AI_PASSWORD", "letmein")
    sm.save_league(_league())
    at = _run()
    _widget(at.multiselect, "Select 2–4 players to compare").select("Ja'Marr Chase").select("Bench Guy")
    _run(at)
    _widget(at.button, "⚖️ Compare").click()
    _run(at)
    assert not [b for b in at.button if "AI" in b.label]
    assert "Enter the AI password" in _text(at)

    _widget(at.sidebar.text_input, "🔒 AI password").input("wrong")
    _run(at)
    assert "Wrong password." in [e.value for e in at.sidebar.error]

    _widget(at.sidebar.text_input, "🔒 AI password").input("letmein")
    _run(at)
    assert not [w for w in at.sidebar.text_input if w.label == "🔒 AI password"]
    _widget(at.button, "⚖️ Compare").click()
    _run(at)
    ai_buttons = [b for b in at.button if "AI" in b.label]
    assert ai_buttons
    _widget(at.button, ai_buttons[0].label).click()
    _run(at)
    assert "API key not configured" in _text(at)

def test_budget_change_and_zero_budget(leagues_dir):
    sm.save_league(_league())
    at = _run()
    _widget(at.number_input, "Remaining Budget ($)").set_value(0)
    _run(at)
    assert "$0 left" in _text(at)


def test_offseason_league_uses_history_views(leagues_dir):
    sm.save_league(_league(week=None, matchup=None, last_result=None))
    at = _run()
    text = _text(at)
    assert "planner needs an in-season league" in text
    _widget(at.multiselect, "Select 2–4 players to compare").select("Ja'Marr Chase").select("Joe Burrow")
    _run(at)
    _widget(at.button, "⚖️ Compare").click()
    _run(at)
    _widget(at.multiselect, "You Give (from your roster)").select("Bench Guy")
    _widget(at.multiselect, "You Receive").select("opp1")
    _run(at)
    _widget(at.button, "🤝 Analyze Trade").click()
    _run(at)
    assert "START: " in _text(at)


def test_non_faab_league_and_empty_roster(leagues_dir):
    sm.save_league(_league(faab_enabled=False, league_id="222", league_name="Priority League"))
    sm.save_league({"league_id": "333", "league_name": "Empty League", "roster": [], "waivers": []})
    at = _run()
    at.sidebar.selectbox[0].set_value("222")
    _run(at)
    assert "doesn't use FAAB" in "\n".join(c.value for c in at.sidebar.caption)
    at.sidebar.selectbox[0].set_value("333")
    _run(at)
    assert "Sync a league from the sidebar." in _text(at)


def test_sidebar_sync_saves_league(monkeypatch, leagues_dir):
    monkeypatch.setattr(src.sleeper_integration, "sync_sleeper_league",
                        lambda user, league: (_league(league_id="444", league_name="New League"), None)
                        if user == "me" else (None, "Sleeper user 'bad' not found."))
    at = _run()
    _widget(at.sidebar.text_input, "Sleeper Username").input("bad")
    _widget(at.sidebar.text_input, "League ID or Name").input("New League")
    _widget(at.sidebar.button, "Sync").click()
    _run(at)
    assert "not found" in at.sidebar.error[0].value
    _widget(at.sidebar.text_input, "Sleeper Username").input("me")
    _widget(at.sidebar.button, "Sync").click()
    _run(at)
    assert (leagues_dir / "444.json").exists()


def test_refresh_data_button(leagues_dir):
    sm.save_league(_league())
    at = _run()
    _widget(at.sidebar.button, "🔄 Refresh data").click()
    _run(at)
    assert len(at.tabs) >= 6


# ── scenario coverage: each test reaches a set of otherwise-unrendered branches ──

def _db_with(**changes):
    db = _players_db()
    for pid, fields in changes.items():
        db[pid].update(fields)
    return db


def test_injured_starter_gets_flagged_and_fixed(monkeypatch, leagues_dir):
    monkeypatch.setattr(src.sleeper_integration, "fetch_all_players",
                        lambda *a, **k: _db_with(wr1={"injury_status": "Out"}))
    sm.save_league(_league())
    at = _run()
    text = _text(at)
    assert "Lineup fixes for week 3" in text
    assert any(m.label == "Win chance with fixes below" for m in at.metric)


def test_offseason_flags_an_inactive_starter(monkeypatch, leagues_dir):
    monkeypatch.setattr(src.sleeper_integration, "fetch_all_players",
                        lambda *a, **k: _db_with(wr1={"injury_status": "Out"}))
    sm.save_league(_league(week=None, roster_positions=["QB", "RB", "WR", "TE", "FLEX", "K", "BN", "IR"]))
    at = _run()
    text = _text(at)
    assert "Starting an inactive player: Ja'Marr Chase" in text and "empty starting slot" not in text


def test_bye_week_and_an_already_optimal_full_lineup(leagues_dir):
    sm.save_league(_league(matchup={"week": WEEK, "bye": True},
                           roster_positions=["QB", "RB", "WR", "TE", "FLEX", "K", "BN", "IR"]))
    at = _run()
    text = _text(at)
    assert "Week 3: bye" in text
    assert "lineup is already the best by projection" in text
    _widget(at.selectbox, "Select player").set_value(0)  # top waiver add: roster is full, so a drop is named
    _run(at)
    assert "✂️ Drop:" in _text(at)


def test_league_without_rivals_history_or_standings(leagues_dir):
    sm.save_league(_league(league_rosters=[], transactions=[], budget=1000, playoff_teams=None))
    at = _run()
    text = _text(at)
    assert "rivals not modeled" in text and "Rival bids weren't modeled" in text
    assert "No winning FAAB claims yet" in text and "No completed adds yet" in text
    assert not any("Standings" in e.label for e in at.expander)


def test_standings_without_playoff_info_and_no_weak_positions(leagues_dir):
    data = _league(playoff_teams=None, roster_positions=["QB", "WR", "BN", "BN", "BN"])
    data["roster"] = [p for p in data["roster"] if p["player_id"] in ("qb1", "wr1", "wr2")]
    sm.save_league(data)
    at = _run()
    assert any(e.label == "🏆 Standings" for e in at.expander)
    assert "No critical positional weaknesses" in _text(at)
    assert "No 2026 stats found for" not in _text(at)


def test_waiver_pool_with_nothing_worth_adding(leagues_dir):
    d1 = next(w for w in _league()["waivers"] if w["player_id"] == "d1")
    sm.save_league(_league(waivers=[d1], roster_positions=["QB", "RB", "WR", "TE", "FLEX", "K", "BN", "BN", "IR"]))
    at = _run()
    text = _text(at)
    assert "save your FAAB this week" in text
    assert "No claim improves your lineup enough" in text
    assert "No available players found" in text


def test_filters_that_match_nobody(leagues_dir):
    fa2 = next(w for w in _league()["waivers"] if w["player_id"] == "fa2")  # Out, hidden by default
    sm.save_league(_league(waivers=[fa2]))
    at = _run()
    assert "No players match the current filters." in _text(at)


def test_degraded_data_sources(monkeypatch, leagues_dir):
    monkeypatch.setattr(src.schedule, "fetch_schedules", lambda *a, **k: pd.DataFrame())
    monkeypatch.setattr(src.sleeper_integration, "fetch_all_players", lambda *a, **k: {})
    monkeypatch.setattr(src.sleeper_integration, "players_db_age_seconds", lambda: None)
    monkeypatch.setattr(src.projections, "fetch_week_projections", lambda *a, **k: {})
    monkeypatch.setattr(src.wrangler, "current_nfl_season", lambda: SEASON + 1)
    sm.save_league(_league())
    at = _run()
    text = _text(at)
    assert "Schedule unavailable" in text and "Injuries as of" not in text
    assert f"No {SEASON + 1} stats yet" in text
    assert "player database unavailable" in text
    assert "Sleeper projections are unavailable" in text


def test_sidebar_sync_now_and_stale_warnings(monkeypatch, leagues_dir):
    results = iter([(None, "Sleeper is down"), (_league(league_name="Smoke League v2"), None)])
    monkeypatch.setattr(src.sleeper_integration, "sync_sleeper_league", lambda user, lid: next(results))
    sm.save_league(_league(season=str(SEASON - 1), synced_at="2020-01-01T00:00:00+00:00"))
    at = _run()
    side = "\n".join(w.value for w in at.sidebar.warning)
    assert "may be stale" in side and f"This is the {SEASON - 1} league" in side
    _widget(at.sidebar.button, "🔄 Sync this league now").click()
    _run(at)
    assert "Sleeper is down" in at.sidebar.error[0].value
    _widget(at.sidebar.button, "🔄 Sync this league now").click()
    _run(at)
    assert sm.load_league("111")["league_name"] == "Smoke League v2"


def test_league_file_without_sync_time(leagues_dir):
    data = _league()
    data.pop("synced_at")
    sm.save_league(data)
    at = _run()
    assert "Sync time unknown" in "\n".join(w.value for w in at.sidebar.warning)


def test_auto_sync_errors_show_for_the_failed_league(monkeypatch, leagues_dir):
    monkeypatch.setattr(src.config, "load_config",
                        lambda *a, **k: {"username": "me", "league_ids": ["111", "555"], "auto_sync_minutes": 30.0})
    sm.save_league(_league(synced_at="2020-01-01T00:00:00+00:00"))  # stale: due for auto-sync
    calls = []

    def fake_auto_sync(user, ids, max_age, attempts, *a, **k):
        calls.append(user)
        for lid in ids:
            attempts[lid] = 1e18  # attempted "now": not due again this session
        return {"111": "League not found."}  # 111 failed, 555 succeeded
    monkeypatch.setattr(src.league_sync, "auto_sync", fake_auto_sync)
    at = _run()
    assert calls == ["me"]
    warnings = "\n".join(w.value for w in at.sidebar.warning)
    assert "Auto-sync failed: League not found. Showing the last saved data." in warnings
    assert "auto-updates every 30 min" in "\n".join(c.value for c in at.sidebar.caption)
    assert "may be stale" not in warnings  # default leagues are kept fresh automatically


def test_season_over_and_offseason_matchup(leagues_dir):
    sm.save_league(_league(week=18))  # past the fantasy championship: no weeks left to project
    at = _run()
    assert "planner needs an in-season league" in _text(at)
    sm.save_league(_league(week=None, league_id="111"))  # live Sleeper points, no projection engine
    at = _run()
    assert "Live points from Sleeper" in _text(at)


def test_three_way_start_sit_with_a_player_missing_stats(leagues_dir):
    sm.save_league(_league())
    at = _run()
    pick = _widget(at.multiselect, "Select 2–4 players to compare")
    pick.select("Ja'Marr Chase").select("Bench Guy").select("Roster Rookie")
    _run(at)
    _widget(at.button, "⚖️ Compare").click()
    _run(at)
    text = _text(at)
    assert "No 2026 stats for: Roster Rookie" in text and "No upcoming game" in text
    assert "only available for 2-player comparisons" in text


def test_trade_without_a_roster_uses_vorp(leagues_dir):
    data = _league(roster=[])
    data.pop("synced_at")  # no sync time and no roster: no stale-data warning either
    sm.save_league(data)
    at = _run()
    assert "Sync a league first." in _text(at)
    _widget(at.multiselect, "You Receive").select("fa3").select("opp1")
    _run(at)
    _widget(at.button, "🤝 Analyze Trade").click()
    _run(at)
    text = _text(at)
    assert "Value over replacement" in text and "valued at 0" in text


def test_roster_without_waiver_pool(leagues_dir):
    sm.save_league(_league(waivers=[]))
    at = _run()
    assert "load the waiver pool for recommendations" in _text(at)


def test_refresh_clears_cached_projections(leagues_dir):
    import src.storage as storage
    sm.save_league(_league())
    at = _run()
    stale = storage.data_path("projections_2026_3.json")
    stale.write_text("{}")
    _widget(at.sidebar.button, "🔄 Refresh data").click()
    _run(at)
    assert not stale.exists()


_SCHEDULE = {str(w): [[1, 2]] for w in range(WEEK, 15)}


def test_playoff_odds_median_game_and_trade_finder(monkeypatch, leagues_dir):
    import src.season
    sm.save_league(_league(schedule=_SCHEDULE, median_game=True, trade_deadline=11))
    monkeypatch.setattr(src.season, "find_trades", lambda *a, **k: [
        {"team": "Rival", "roster_id": 2, "give": "Bench Guy", "give_pos": "WR", "get": "Rival Receiver",
         "get_pos": "WR", "gain_me": 4.2, "gain_them": 1.1}])
    at = _run()
    assert "Median game: top-half finish" in [m.label for m in at.metric]
    assert any("playoff chances" in e.label for e in at.expander)
    standings_df = next(d.value for d in at.dataframe if "Playoff chances" in d.value.columns)
    assert standings_df["Playoff chances"].between(0, 1).all()
    text = _text(at)
    assert "Playoff chances simulate" in text and "Trade deadline: week 11, 8 week(s) away." in text
    assert "Playoff weeks count by your" in text
    _widget(at.button, "Find trades").click()
    _run(at)
    ideas = next(d.value for d in at.dataframe if "Trade with" in d.value.columns)
    assert ideas.iloc[0]["You get"] == "Rival Receiver (WR)"


def test_trade_finder_with_no_ideas(monkeypatch, leagues_dir):
    import src.season
    sm.save_league(_league(schedule=_SCHEDULE, trade_deadline=WEEK))
    monkeypatch.setattr(src.season, "find_trades", lambda *a, **k: [])
    at = _run()
    assert f"This is the trade deadline week (week {WEEK})." in _text(at)
    _widget(at.button, "Find trades").click()
    _run(at)
    assert "No 1-for-1 trade improves both lineups right now." in _text(at)


def test_trades_closed_after_the_deadline(leagues_dir):
    sm.save_league(_league(trade_deadline=WEEK - 1))
    at = _run()
    assert "has passed" in _text(at)
    assert not [b for b in at.button if b.label == "Find trades"]


def test_trade_analysis_shows_rest_of_season_projection(leagues_dir):
    sm.save_league(_league())
    at = _run()
    _widget(at.multiselect, "You Give (from your roster)").select("Bench Guy")
    _run(at)
    _widget(at.button, "🤝 Analyze Trade").click()
    _run(at)
    assert "Rest of season, from weekly projections" in [m.label for m in at.metric]
