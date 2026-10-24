"""Bad files, bad timestamps, bad config: the app should degrade, not crash."""

import json
import os
import threading
import time

import pandas as pd
import pytest

import src.league_sync as ls
import src.state_manager as sm
import src.storage as storage
from src.config import DEFAULT_SYNC_MINUTES, load_config


@pytest.mark.parametrize("content", ['{"league_id": "1", "roster": [', "[1, 2, 3]", "42", ""])
def test_corrupt_league_file_loads_as_empty(leagues_dir, content):
    (leagues_dir / "123.json").write_text(content)
    data = sm.load_league("123")
    assert data["roster"] == [] and data["team_name"] == "123"
    assert sm.list_leagues() == []  # skipped, not a crash


def test_corrupt_default_league_does_not_break_auto_sync(leagues_dir):
    (leagues_dir / "123.json").write_text("{not json")
    calls = []

    def sync_fn(user, lid):
        calls.append(lid)
        return {"league_id": lid, "league_name": "Fixed", "roster": []}, None

    assert ls.auto_sync("me", ["123"], 1800, {}, sync_fn) == {}
    assert calls == ["123"] and sm.load_league("123")["league_name"] == "Fixed"


def test_list_leagues_without_directory_is_empty(isolated):
    assert not sm.LEAGUES_DIR.exists()
    assert sm.list_leagues() == []


@pytest.mark.parametrize("value", ["garbage", "2026-13-45T00:00:00", 12345])
def test_malformed_sync_timestamp_means_never_synced(value):
    assert ls.sync_age_seconds({"synced_at": value}) is None
    assert ls.is_sync_due({"synced_at": value}, 1800)


def test_naive_timestamp_is_treated_as_utc():
    assert ls.sync_age_seconds({"synced_at": "2026-01-01T00:00:00"}, now=1767225600 + 60) == 60


def test_one_failing_league_does_not_stop_the_others(leagues_dir):
    def sync_fn(user, lid):
        if lid == "bad":
            raise RuntimeError("boom")
        return {"league_id": lid, "roster": []}, None

    errors = ls.auto_sync("me", ["bad", "good"], 1800, {}, sync_fn)
    assert "boom" in errors["bad"] and (leagues_dir / "good.json").exists()


def test_auto_sync_without_username_does_nothing():
    assert ls.auto_sync("", ["1"], 1800, {}, lambda *a: pytest.fail("should not sync")) == {}


def test_config_defaults_without_file(tmp_path):
    cfg = load_config(tmp_path / "missing.toml")
    assert cfg == {"username": "", "league_ids": [], "auto_sync_minutes": DEFAULT_SYNC_MINUTES}


@pytest.mark.parametrize("env,expected", [("abc", 45.0), ("-5", 45.0), ("0", 45.0), ("10", 10.0)])
def test_invalid_sync_minutes_fall_back(tmp_path, monkeypatch, env, expected):
    cfg = tmp_path / "config.toml"
    cfg.write_text('[sleeper]\nauto_sync_minutes = 45\n')
    monkeypatch.setenv("SLEEPER_AUTO_SYNC_MINUTES", env)
    assert load_config(cfg)["auto_sync_minutes"] == expected


def test_invalid_toml_minutes_use_default(tmp_path):
    cfg = tmp_path / "config.toml"
    cfg.write_text('[sleeper]\nauto_sync_minutes = "soon"\n')
    assert load_config(cfg)["auto_sync_minutes"] == DEFAULT_SYNC_MINUTES


# ── storage ──────────────────────────────────────────────────────────────────

def test_data_path_creates_directory(isolated):
    path = storage.data_path("x.json")
    assert path.parent.is_dir() and path.parent == storage.DATA_DIR


def test_freshness_and_age(tmp_path):
    path = tmp_path / "f.json"
    assert not storage.is_fresh(path, 60) and storage.file_age_seconds(path) is None
    path.write_text("{}")
    old = time.time() - 120
    os.utime(path, (old, old))
    assert not storage.is_fresh(path, 60) and storage.is_fresh(path, 600)
    assert 119 <= storage.file_age_seconds(path) < 130


def test_failed_write_keeps_old_file_and_no_temp(tmp_path):
    path = tmp_path / "x.json"
    storage.write_json_atomic({"ok": 1}, path)
    with pytest.raises(TypeError):
        storage.write_json_atomic({"bad": object()}, path)
    assert json.loads(path.read_text()) == {"ok": 1}
    assert [p.name for p in tmp_path.iterdir()] == ["x.json"]


def test_parquet_round_trip(tmp_path):
    df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    storage.write_parquet_atomic(df, tmp_path / "d.parquet")
    pd.testing.assert_frame_equal(pd.read_parquet(tmp_path / "d.parquet"), df)


def test_concurrent_writes_always_leave_valid_json(tmp_path):
    path = tmp_path / "shared.json"

    def writer(i):
        for _ in range(20):
            storage.write_json_atomic({"writer": i, "payload": list(range(200))}, path)

    threads = [threading.Thread(target=writer, args=(i,)) for i in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert json.loads(path.read_text())["payload"] == list(range(200))
    assert [p.name for p in tmp_path.iterdir()] == ["shared.json"]
