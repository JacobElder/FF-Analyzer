"""Stats loading and matching, defense ranks, schedule, injuries, backtest helper."""

import pandas as pd
import pytest

import src.wrangler as wrangler
from src.backtest import _history_pred
from src.defense import SHRINK_GAMES, build_matchup_table, compute_defense_rankings, get_matchup_info
from src.injuries import build_injury_map
from src.players import enrich_with_ids, build_sleeper_name_index, player_label
from src.schedule import get_next_game, get_playoff_schedule_strength, next_game_map, playoff_weeks_for
from src.wrangler import StatsIndex


def _df(rows, extra=None):
    df = pd.DataFrame(rows, columns=["player_id", "player_display_name", "position", "team", "week", "fp"])
    df["season"] = 2025
    for k, v in (extra or {}).items():
        df[k] = v
    return df


# ── stats matching ───────────────────────────────────────────────────────────

def test_single_namesake_at_another_position_is_not_matched():
    idx = StatsIndex(_df([("00-1", "Calvin Austin", "CB", "NE", 1, 3.0)]))
    assert not idx.get_player_stats({"name": "Calvin Austin III", "position": "WR"})["found"]


def test_player_with_unknown_id_does_not_take_a_namesakes_stats():
    idx = StatsIndex(_df([("00-1", "Mike Williams", "WR", "PIT", 1, 9.0)]))
    rookie = {"name": "Mike Williams", "position": "WR", "team": "NYJ", "gsis_id": "00-new"}
    assert not idx.get_player_stats(rookie)["found"]
    assert idx.get_player_stats({**rookie, "team": "PIT"})["found"]  # same name, same team: same player


def test_fullback_matches_running_back_and_missing_position_is_allowed():
    idx = StatsIndex(_df([("00-1", "Pat Freiermuth", "FB", "PIT", 1, 4.0),
                          ("00-2", "No Position", None, "PIT", 1, 4.0)]))
    assert idx.get_player_stats({"name": "Pat Freiermuth", "position": "RB"})["found"]
    assert idx.get_player_stats({"name": "No Position", "position": "WR"})["found"]


def test_index_handles_filtered_and_unsorted_frames():
    df = _df([("00-1", "A Guy", "WR", "NYJ", 2, 5.0), ("00-1", "A Guy", "WR", "BUF", 5, 9.0),
              ("00-2", "B Guy", "WR", "MIA", 1, 1.0), ("00-1", "A Guy", "WR", "NYJ", 1, 3.0)])
    filtered = df[df["player_id"] == "00-1"]  # non-contiguous index labels
    s = StatsIndex(filtered).get_player_stats({"name": "A Guy", "position": "WR"})
    assert s["weeks"] == [1, 2, 5] and s["weekly"] == [3.0, 5.0, 9.0]
    assert s["team"] == "BUF"  # latest week's team, even though rows arrived unsorted


def test_usage_trends_recent_vs_season():
    df = _df([("00-1", "A Guy", "WR", "NYJ", w, 5.0) for w in range(1, 9)],
             extra={"target_share": [0.1] * 4 + [0.3] * 4, "carries": [0] * 8})
    trends = StatsIndex(df).get_usage_trends({"name": "A Guy", "position": "WR"}, recent_weeks=4)
    assert trends["target_share"] == {"season_avg": 0.2, "recent_avg": 0.3, "trend": 0.1}
    assert trends["recent_carries_pg"] == 0.0
    assert StatsIndex(df).get_usage_trends({"name": "Nobody"}) == {}


def test_prepare_keeps_regular_season_and_normalizes_teams():
    df = _df([("00-1", "A", "WR", "LA", 1, 5.0), ("00-1", "A", "WR", "LA", 19, 9.0)],
             extra={"season_type": ["REG", "POST"], "opponent_team": ["LV", "OAK"]})
    out = wrangler._prepare(df)
    assert out["week"].tolist() == [1] and out["team"].tolist() == ["LAR"]


def test_fetch_weekly_stats_falls_back_to_previous_season(monkeypatch):
    seasons = {2025: _df([("00-1", "A", "WR", "NYJ", 1, 5.0)]), 2026: pd.DataFrame()}
    monkeypatch.setattr(wrangler, "_load_season", lambda year, force: seasons.get(year, pd.DataFrame()))
    assert wrangler.fetch_weekly_stats(2026)["season"].unique().tolist() == [2025]


def test_load_season_uses_stale_cache_when_download_fails(monkeypatch):
    cached = _df([("00-1", "A", "WR", "NYJ", 1, 5.0)])
    path = wrangler.data_path("weekly_stats_2025.parquet")
    cached.to_parquet(path)
    monkeypatch.setattr(wrangler.pd, "read_parquet",
                        lambda src, *a, **k: pd.DataFrame(cached) if str(src) == str(path) else (_ for _ in ()).throw(OSError("down")))
    assert len(wrangler._load_season(2025, force=True)) == 1


# ── defense ──────────────────────────────────────────────────────────────────

def _def_frame():
    rows = []
    for week in range(1, 5):
        rows += [("WR", "NYJ", "BUF", week, 10.0), ("WR", "MIA", "KC", week, 30.0)]
    return pd.DataFrame(rows, columns=["position", "team", "opponent_team", "week", "fp"])


def test_defense_rank_direction_and_shrinkage():
    ranks = compute_defense_rankings(_def_frame())
    by_team = ranks.set_index("defteam")
    assert by_team.loc["BUF", "rank"] == 1 and by_team.loc["KC", "rank"] == 2  # BUF allows least
    table = build_matchup_table(ranks)
    buf, kc = table[("BUF", "WR")], table[("KC", "WR")]
    assert buf["raw_multiplier"] == 0.5 and kc["raw_multiplier"] == 1.5
    weight = 4 / (4 + SHRINK_GAMES)
    assert buf["multiplier"] == pytest.approx(1 - 0.5 * weight) and kc["multiplier"] == pytest.approx(1 + 0.5 * weight)


def test_defense_empty_inputs():
    assert compute_defense_rankings(pd.DataFrame()).empty
    assert compute_defense_rankings(_def_frame().drop(columns="fp")).empty
    assert build_matchup_table(None) == {}
    assert get_matchup_info("BUF", "WR", {})["multiplier"] == 1.0
    assert get_matchup_info("ZZZ", "WR", {("BUF", "WR"): {"multiplier": 2}})["rank_label"] == "Unknown"


# ── schedule ─────────────────────────────────────────────────────────────────

def _sched():
    return pd.DataFrame([
        {"week": 1, "home_team": "BUF", "away_team": "NYJ", "home_score": 20.0, "gameday": "2026-09-10"},
        {"week": 2, "home_team": "MIA", "away_team": "BUF", "home_score": None, "gameday": "2026-09-17"},
        {"week": 15, "home_team": "BUF", "away_team": "KC", "home_score": None, "gameday": "2026-12-14"},
        {"week": 17, "home_team": "NE", "away_team": "BUF", "home_score": None, "gameday": "2026-12-28"},
    ])


def test_next_game_skips_played_games():
    g = get_next_game("BUF", _sched())
    assert g["week"] == 2 and g["opponent"] == "MIA" and not g["is_home"]
    assert get_next_game("BUF", pd.DataFrame()) is None and get_next_game("", _sched()) is None
    assert set(next_game_map(_sched())) == {"BUF", "NYJ", "MIA", "KC", "NE"}
    assert next_game_map(_sched())["NYJ"] is None  # season over for them in this frame


def test_playoff_schedule_strength_marks_byes_and_grades():
    table = {("KC", "WR"): {"multiplier": 1.3, "rank_label": "Favorable"},
             ("NE", "WR"): {"multiplier": 1.2, "rank_label": "Favorable"}}
    out = get_playoff_schedule_strength("BUF", _sched(), table, "WR", (15, 16, 17))
    assert [w["opponent"] for w in out["weeks"]] == ["KC", "BYE", "NE"]
    assert out["avg_multiplier"] == 1.25 and out["grade"] == "A"
    assert get_playoff_schedule_strength("BUF", _sched(), {}, "WR")["grade"] == "N/A"


@pytest.mark.parametrize("mult,grade", [(1.20, "A"), (1.08, "B"), (0.92, "C"), (0.80, "D"), (0.79, "F")])
def test_playoff_grade_boundaries(mult, grade):
    table = {("KC", "WR"): {"multiplier": mult, "rank_label": "x"}}
    assert get_playoff_schedule_strength("BUF", _sched(), table, "WR", (15,))["grade"] == grade


def test_playoff_weeks_for():
    assert playoff_weeks_for(14) == (14, 15, 16) and playoff_weeks_for(None) == (15, 16, 17)


# ── injuries, players, backtest ──────────────────────────────────────────────

def test_injury_map_ignores_na_and_ambiguous_names():
    db = {"1": {"full_name": "A Guy", "position": "WR", "injury_status": "Out"},
          "2": {"full_name": "Twin", "position": "RB", "injury_status": "IR"},
          "3": {"full_name": "Twin", "position": "WR", "injury_status": None, "active": True},
          "4": {"full_name": "Healthy", "position": "TE", "injury_status": "NA"},
          "5": {"full_name": "A Guy", "position": "WR", "active": False},  # retired namesake
          "6": {"full_name": "Kicker Guy", "position": "OL", "injury_status": "Out"}}
    m = build_injury_map(db)
    assert m["by_id"] == {"1": "Out", "2": "IR"}
    assert m["by_name"] == {"a guy": "Out"}  # Twin is ambiguous; retired A Guy doesn't count


def test_enrich_breaks_ties_by_team_then_active_status():
    db = {"1": {"full_name": "Mike Williams", "position": "WR", "team": "PIT", "active": True},
          "2": {"full_name": "Mike Williams", "position": "WR", "team": "NYJ", "active": True},
          "3": {"full_name": "Sam Smith", "position": "RB", "team": None, "active": False},
          "4": {"full_name": "Sam Smith", "position": "RB", "team": "DAL", "active": True}}
    index = build_sleeper_name_index(db)
    out = enrich_with_ids([{"name": "Mike Williams", "position": "WR", "team": "NYJ"},
                           {"name": "Mike Williams", "position": "WR"},
                           {"name": "Sam Smith", "position": "RB"}], db, index)
    assert [p.get("player_id") for p in out] == ["2", None, "4"]
    assert player_label({"name": "X", "position": "WR", "team": "NYJ"}) == "X (WR, NYJ)"
    assert player_label({"name": "X"}) == "X"


def test_history_pred_uses_only_prior_games():
    games = [(1, 10.0), (2, 20.0), (3, 30.0), (4, 100.0)]
    assert _history_pred(games, 2, 0.5, 2) is None  # fewer than 2 prior games
    assert _history_pred(games, 4, 0.0, 2) == pytest.approx(20.0)  # season-to-date only; week 4 unseen
    assert _history_pred(games, 4, 1.0, 1) == pytest.approx(30.0)  # last game only
    assert _history_pred([(1, 10.0), (2, 20.0)], 9, 0.5, 2) == pytest.approx(15.0)  # no recent → season
