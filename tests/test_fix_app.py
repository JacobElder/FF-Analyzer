"""App branches added with the audit fixes: data errors, live scoring, no-FAAB views, AI limits, trades."""

import pytest

pytest.importorskip("streamlit.testing.v1")

import src.live  # noqa: E402
import src.sleeper_integration  # noqa: E402
import src.state_manager as sm  # noqa: E402
import src.vegas  # noqa: E402
import src.waiver_analysis  # noqa: E402
import src.wrangler  # noqa: E402
from src.storage import DataUnavailable  # noqa: E402
from test_app_smoke import _db_with, _fake_ai, _league, _run, _text, _widget, offline_app  # noqa: E402,F401

pytestmark = pytest.mark.app


def _raise(*a, **k):
    raise DataUnavailable("Couldn't load it: offline")


def test_stats_download_failure_stops_with_retry_hint(monkeypatch, leagues_dir):
    monkeypatch.setattr(src.wrangler, "fetch_weekly_stats", _raise)
    at = _run()
    assert "Couldn't load it: offline" in at.error[0].value and "Refresh data" in at.error[0].value
    assert not at.tabs


def test_player_db_failure_keeps_the_app_running(monkeypatch, leagues_dir):
    monkeypatch.setattr(src.sleeper_integration, "fetch_all_players", _raise)
    sm.save_league(_league())
    at = _run()
    text = _text(at)
    assert "Couldn't load it: offline" in text and "player database unavailable" in text


def test_odds_status_partial_sync_and_no_deadline(monkeypatch, leagues_dir):
    import time
    monkeypatch.setattr(src.vegas, "vegas_status", lambda: {"fetched_at": time.time() - 7200, "remaining": 412})
    sm.save_league(_league(partial=["users"], schedule_complete=False, trade_deadline=99))
    at = _run()
    text = _text(at)
    assert "Odds as of 2 hours ago (412 API credits left)" in text
    assert "couldn't load: schedule, users" in text and "Sync this league now" in text
    assert "no trade deadline" in text


def test_live_scoring_locks_started_players(monkeypatch, leagues_dir):
    # CIN has kicked off: Chase (Out) can't be benched any more, Burrow's points are final
    monkeypatch.setattr(src.live, "game_started", lambda team, now, sched, week=None: team == "CIN")
    monkeypatch.setattr(src.sleeper_integration, "fetch_all_players",
                        lambda *a, **k: _db_with(wr1={"injury_status": "Out"}))
    mu = dict(_league()["matchup"], points=31.5, players_points={"qb1": 31.5, "wr1": 0})
    sm.save_league(_league(matchup=mu))
    at = _run()
    labels = [m.label for m in at.metric]
    assert "Week 3 — You (live + projected)" in labels
    text = _text(at)
    assert "games have already kicked off" in text and "Lineup fixes" not in text
    assert "Points already scored" in text


def test_matchup_without_opponent_lineup_shows_sleeper_points(leagues_dir):
    sm.save_league(_league(matchup=dict(_league()["matchup"], opponent_starters=[])))
    at = _run()
    assert "Live points from Sleeper" in _text(at)


def test_unmodeled_rival_bids(monkeypatch, leagues_dir):
    monkeypatch.setattr(src.waiver_analysis, "RIVAL_CANDIDATES", 0)
    sm.save_league(_league())
    at = _run()
    text = _text(at)
    assert "rivals not modeled" in text and "Rival bids weren't modeled" in text


def test_ai_password_lockout_and_session_cap(monkeypatch, leagues_dir):
    _fake_ai(monkeypatch)
    monkeypatch.setenv("AI_PASSWORD", "a-long-random-password")
    sm.save_league(_league())
    at = _run()
    for _ in range(5):
        _widget(at.sidebar.text_input, "🔒 AI password").input("wrong")
        _run(at)
    assert "Too many wrong passwords" in at.sidebar.error[0].value
    assert not [w for w in at.sidebar.text_input if w.label == "🔒 AI password"]
    assert at.session_state["ai_pw_fails"] == 5

    at.session_state["ai_locked_until"] = 0  # lockout over
    _run(at)
    _widget(at.sidebar.text_input, "🔒 AI password").input("a-long-random-password")
    _run(at)
    at.session_state["ai_calls"] = 30
    at.button(key="ai_waiver_btn").click()
    _run(at)
    assert "AI limit reached for this session (30 requests)" in _text(at)


def test_offseason_priority_league_deep_dive_with_simulation(leagues_dir):
    sm.save_league(_league(week=None, faab_enabled=False))
    at = _run()
    text = _text(at)
    assert "No FAAB in this league — claims go by waiver priority." in text
    _widget(at.toggle, "🎲 Monte Carlo simulation").set_value(True)
    _run(at)
    assert "Floor (10th pct)" in [m.label for m in at.metric]


def test_full_roster_simulation_compares_the_engine_drop(leagues_dir):
    sm.save_league(_league(roster_positions=["QB", "RB", "WR", "TE", "FLEX", "K", "BN", "IR"]))
    at = _run()
    _widget(at.selectbox, "Select player").select_index(0)
    _run(at)
    _widget(at.toggle, "🎲 Monte Carlo simulation").set_value(True)
    _run(at)
    assert "Centered on each player's week 3 projection" in _text(at)


def test_trade_that_overfills_the_roster_names_the_cut(leagues_dir):
    sm.save_league(_league())
    at = _run()
    _widget(at.multiselect, "You Receive").select("opp1").select("fa1").select("fa4")
    _run(at)
    _widget(at.button, "🤝 Analyze Trade").click()
    _run(at)
    assert "To stay within your 9-player roster you'd have to cut" in _text(at)


def test_lineup_without_position_slots_has_nothing_to_grade(leagues_dir):
    sm.save_league(_league(roster_positions=["FLEX", "FLEX", "BN", "BN", "IR"]))
    at = _run()
    assert "no position-specific slots to grade" in _text(at)
