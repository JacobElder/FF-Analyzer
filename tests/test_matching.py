import pandas as pd

from src.players import build_sleeper_name_index, enrich_with_ids, normalize_name
from src.wrangler import StatsIndex


def _df(rows):
    df = pd.DataFrame(rows, columns=["player_id", "player_display_name", "position", "team", "week", "fp"])
    df["season"] = 2025
    return df.sort_values(["player_id", "week"]).reset_index(drop=True)


def test_normalize_strips_suffixes_accents_punctuation():
    assert normalize_name("Calvin Austin III") == "calvin austin"
    assert normalize_name("Brian Thomas Jr.") == "brian thomas"
    assert normalize_name("Audric Estimé") == "audric estime"
    assert normalize_name("Ja'Marr Chase") == "jamarr chase"


def test_no_last_name_fallback_to_a_different_player():
    idx = StatsIndex(_df([
        ("00-1", "Alex Austin", "CB", "NE", 1, 3.0),
        ("00-2", "Devin Lloyd", "LB", "JAX", 1, 5.0),
    ]))
    assert not idx.get_player_stats({"name": "Calvin Austin", "position": "WR"})["found"]
    assert not idx.get_player_stats({"name": "MarShawn Lloyd", "position": "RB"})["found"]


def test_suffix_variant_matches():
    idx = StatsIndex(_df([("00-3", "Calvin Austin III", "WR", "PIT", 1, 9.0)]))
    s = idx.get_player_stats({"name": "Calvin Austin", "position": "WR"})
    assert s["found"] and s["matched_name"] == "Calvin Austin III"


def test_gsis_id_wins_and_namesakes_disambiguated_by_position():
    idx = StatsIndex(_df([
        ("00-10", "Josh Allen", "QB", "BUF", 1, 25.0),
        ("00-11", "Josh Allen", "LB", "JAX", 1, 2.0),
    ]))
    assert idx.get_player_stats({"name": "Josh Allen", "gsis_id": "00-11"})["season"]["ppg"] == 2.0
    assert idx.get_player_stats({"name": "Josh Allen", "position": "QB"})["season"]["ppg"] == 25.0
    assert not idx.get_player_stats({"name": "Josh Allen"})["found"]  # ambiguous → not guessed


def test_zero_point_games_count_and_recent_is_by_week():
    idx = StatsIndex(_df([
        ("00-5", "Some Back", "RB", "KC", 1, 20.0),
        ("00-5", "Some Back", "RB", "KC", 2, 0.0),
        ("00-5", "Some Back", "RB", "KC", 3, 10.0),
        ("00-6", "Other Guy", "WR", "KC", 10, 5.0),  # sets last_week = 10
    ]))
    s = idx.get_player_stats("Some Back")
    assert s["season"]["games"] == 3
    assert s["season"]["ppg"] == 10.0
    # weeks 1–3 are not within the last 4 weeks (7–10)
    assert s["recent"]["ppg"] is None and s["recent"]["games"] == 0


def test_enrich_with_ids_fills_sleeper_ids():
    db = {
        "1": {"full_name": "Brian Thomas", "position": "WR", "team": "JAX", "active": True},
        "2": {"full_name": "Josh Allen", "position": "QB", "team": "BUF", "active": True, "gsis_id": " 00-9 "},
        "3": {"full_name": "Josh Allen", "position": "G", "team": None, "active": False},
    }
    out = enrich_with_ids([{"name": "Josh Allen", "position": "QB"}, {"name": "Brian Thomas", "position": "WR"}],
                          db, build_sleeper_name_index(db))
    assert out[0]["player_id"] == "2" and out[0]["gsis_id"] == "00-9"
    assert out[1]["player_id"] == "1" and out[1]["team"] == "JAX"
