import json

import pytest

import src.state_manager as sm
import src.storage as storage
from src.sleeper_integration import build_waiver_pool, parse_faab_budget
from src.vegas import parse_implied_totals


def test_faab_budget_is_league_budget_minus_used():
    assert parse_faab_budget({"settings": {"waiver_budget": 200}}, {"settings": {"waiver_budget_used": 37}}) == 163
    assert parse_faab_budget({}, {}) == 100


def test_waiver_pool_keeps_injured_and_excludes_rostered():
    db = {
        "1": {"full_name": "Owned", "position": "RB", "team": "KC", "active": True},
        "2": {"full_name": "Hurt Stash", "position": "WR", "team": "KC", "active": True, "injury_status": "IR"},
        "3": {"full_name": "No Team", "position": "WR", "team": None, "active": True},
        "4": {"full_name": "Lineman", "position": "OL", "team": "KC", "active": True},
    }
    pool = build_waiver_pool([{"players": ["1"]}], db, {"2": 50})
    assert [p["name"] for p in pool] == ["Hurt Stash"]
    assert pool[0]["trend_count"] == 50 and pool[0]["player_id"] == "2"


def _game(t, home, away, total, home_spread):
    return {"commence_time": t, "home_team": home, "away_team": away, "bookmakers": [{"markets": [
        {"key": "totals", "outcomes": [{"name": "Over", "point": total}]},
        {"key": "spreads", "outcomes": [{"name": home, "point": home_spread}]},
    ]}]}


def test_vegas_uses_each_teams_earliest_game():
    games = [
        _game("2026-10-25T17:00:00Z", "Buffalo Bills", "Miami Dolphins", 40.0, -2.0),   # later week
        _game("2026-10-18T17:00:00Z", "Buffalo Bills", "New York Jets", 47.0, -7.0),    # next week
    ]
    totals = parse_implied_totals(games)
    assert totals["BUF"] == 27.0 and totals["NYJ"] == 20.0
    assert totals["MIA"] == 19.0  # MIA's next game is still used even though BUF's was set earlier


@pytest.fixture
def leagues_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(sm, "LEAGUES_DIR", tmp_path)
    return tmp_path


def test_safe_key_blocks_path_traversal(leagues_dir):
    assert "/" not in sm.safe_key("../../etc/passwd") and ".." not in sm.safe_key("../../etc/passwd")
    key = sm.save_league({"team_name": "x"}, "../evil")
    assert (leagues_dir / f"{key}.json").exists()


def test_save_by_league_id_and_dedupe_legacy_files(leagues_dir):
    key = sm.save_league({"league_id": "123", "league_name": "Mixer", "season": "2026", "roster": []})
    assert key == "123"
    (leagues_dir / "Mixer.json").write_text(json.dumps({"league_id": "123", "team_name": "Mixer"}))
    assert sm.list_leagues() == [("123", "Mixer (2026)")]


def test_load_missing_league_does_not_create_file(leagues_dir):
    sm.load_league("nope")
    assert not any(leagues_dir.iterdir())


def test_atomic_write_leaves_no_temp_files(tmp_path):
    storage.write_json_atomic({"a": 1}, tmp_path / "x.json")
    assert [p.name for p in tmp_path.iterdir()] == ["x.json"]
