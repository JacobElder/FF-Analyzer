from datetime import datetime, timedelta, timezone

import pytest

import src.league_sync as ls
import src.state_manager as sm
from src.config import load_config
from src.sleeper_integration import assign_slots


def _iso(minutes_ago):
    return (datetime.now(timezone.utc) - timedelta(minutes=minutes_ago)).isoformat(timespec="seconds")


def test_assign_slots_maps_starters_to_lineup_order():
    positions = ["QB", "RB", "RB", "FLEX", "DEF", "BN", "BN", "IR"]
    slots = assign_slots(["q", "r1", "0", "f", "d"], positions, reserve=["hurt"], taxi=["kid"])
    assert slots == {"q": "QB", "r1": "RB", "f": "FLEX", "d": "DEF", "hurt": "IR", "kid": "Taxi"}


def test_sync_due_logic():
    assert ls.is_sync_due({}, 1800)                                  # never synced
    assert not ls.is_sync_due({"synced_at": _iso(5)}, 1800)
    assert ls.is_sync_due({"synced_at": _iso(45)}, 1800)


def test_auto_sync_syncs_only_stale_leagues_and_saves():
    sm.save_league({"league_id": "fresh", "synced_at": _iso(5)})
    sm.save_league({"league_id": "stale", "synced_at": _iso(60)})
    calls = []

    def fake_sync(user, lid):
        calls.append(lid)
        return {"league_id": lid, "league_name": lid, "synced_at": _iso(0)}, None

    errors = ls.auto_sync("me", ["fresh", "stale", "new"], 1800, {}, sync_fn=fake_sync)
    assert errors == {} and calls == ["stale", "new"]
    assert sm.load_league("new")["league_name"] == "new"


def test_failed_sync_keeps_old_data_and_backs_off():
    sm.save_league({"league_id": "x", "league_name": "old", "synced_at": _iso(60)})
    attempts, calls = {}, []

    def failing(user, lid):
        calls.append(lid)
        return None, "Sleeper is down"

    assert ls.auto_sync("me", ["x"], 1800, attempts, sync_fn=failing) == {"x": "Sleeper is down"}
    assert sm.load_league("x")["league_name"] == "old"
    ls.auto_sync("me", ["x"], 1800, attempts, sync_fn=failing)  # immediate rerun: no retry
    assert calls == ["x"]


def test_config_file_and_env_override(tmp_path, monkeypatch):
    cfg = tmp_path / "config.toml"
    cfg.write_text('[sleeper]\nusername = "me"\nleague_ids = ["1", "2"]\nauto_sync_minutes = 15\n')
    for var in ("SLEEPER_USERNAME", "SLEEPER_LEAGUE_IDS", "SLEEPER_AUTO_SYNC_MINUTES"):
        monkeypatch.delenv(var, raising=False)
    assert load_config(cfg) == {"username": "me", "league_ids": ["1", "2"], "auto_sync_minutes": 15.0}
    monkeypatch.setenv("SLEEPER_LEAGUE_IDS", "9, 8")
    assert load_config(cfg)["league_ids"] == ["9", "8"]
