"""
Streamlit smoke tests: the whole app script runs without exceptions, offline,
against a temp data directory. Data sources are stubbed; no .env is loaded.
"""

from pathlib import Path

import dotenv
import pandas as pd
import pytest

pytest.importorskip("streamlit.testing.v1")
from streamlit.testing.v1 import AppTest  # noqa: E402

import src.config
import src.schedule
import src.sleeper_integration
import src.state_manager as sm
import src.vegas
import src.wrangler

def _weekly_stats():
    rows = []
    for week in range(1, 7):
        rows += [("00-1", "Joe Burrow", "QB", "CIN", "BUF", week, 250, 2, 0, 0, 0),
                 ("00-2", "Ja'Marr Chase", "WR", "CIN", "BUF", week, 0, 0, 7, 95, 1),
                 ("00-3", "Free Agent", "WR", "MIA", "NYJ", week, 0, 0, 3, 30, 0)]
    cols = ["player_id", "player_display_name", "position", "team", "opponent_team", "week",
            "passing_yards", "passing_tds", "receptions", "receiving_yards", "receiving_tds"]
    df = pd.DataFrame(rows, columns=cols)
    df["season"] = 2026
    df["season_type"] = "REG"
    return df.sort_values(["player_id", "week"]).reset_index(drop=True)


APP = str(Path(__file__).resolve().parent.parent / "app.py")


@pytest.fixture(autouse=True)
def offline_app(monkeypatch):
    monkeypatch.setattr(dotenv, "load_dotenv", lambda *a, **k: False)
    monkeypatch.setattr(src.config, "load_config",
                        lambda *a, **k: {"username": "", "league_ids": [], "auto_sync_minutes": 30.0})
    monkeypatch.setattr(src.wrangler, "fetch_weekly_stats", lambda *a, **k: _weekly_stats())
    monkeypatch.setattr(src.schedule, "fetch_schedules", lambda *a, **k: pd.DataFrame())
    monkeypatch.setattr(src.sleeper_integration, "fetch_all_players", lambda *a, **k: {})
    monkeypatch.setattr(src.vegas, "fetch_implied_totals", lambda *a, **k: {})


def _run():
    at = AppTest.from_file(APP, default_timeout=60)
    at.run()
    return at


def test_app_runs_with_no_leagues():
    at = _run()
    assert not at.exception, [e.value for e in at.exception]
    assert not at.error, [e.value for e in at.error]
    assert at.title[0].value == "🏈 FF Analyzer"


def test_app_renders_a_saved_league(leagues_dir):
    sm.save_league({
        "league_id": "1", "league_name": "Smoke League", "season": "2026", "team_name": "Mine",
        "budget": 50, "waiver_budget_total": 100, "faab_enabled": True, "week": None,
        "roster_positions": ["QB", "RB", "WR", "TE", "FLEX", "BN"],
        "roster": [{"name": "Joe Burrow", "position": "QB", "team": "CIN", "player_id": "1", "slot": "QB"},
                   {"name": "Ja'Marr Chase", "position": "WR", "team": "CIN", "player_id": "2", "slot": "WR"}],
        "waivers": [{"name": "Free Agent", "position": "WR", "team": "MIA", "player_id": "3", "trend_count": 5}],
        "scoring_settings": {"rec": 1}, "league_rosters": [], "transactions": [],
    })
    at = _run()
    assert not at.exception, [e.value for e in at.exception]
    labels = [t.label for t in at.tabs]
    assert len(labels) >= 6, labels  # got past data loading and rendered every tab
