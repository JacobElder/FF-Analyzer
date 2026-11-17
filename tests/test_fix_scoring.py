"""History scoring covers every Sleeper key projections use; kicker projection derivations; backtest rows."""

import pandas as pd
import pytest

from src import backtest
from src.projections import _derive_kicker_stats, score_projection
from src.scoring import compute_fantasy_points, effective_scoring


def _pts(scoring: dict, **stats) -> float:
    df = pd.DataFrame([{"position": stats.pop("position", "WR"), **stats}])
    return compute_fantasy_points(df, scoring).iloc[0]


# ── history scoring: new keys ────────────────────────────────────────────────

@pytest.mark.parametrize("key,column", [
    ("pass_fd", "passing_first_downs"), ("rush_fd", "rushing_first_downs"), ("rec_fd", "receiving_first_downs"),
    ("pass_cmp", "completions"), ("pass_att", "attempts"), ("rush_att", "carries"), ("rec_tgt", "targets"),
    ("fgm", "fg_made"), ("fgm_yds", "fg_made_distance"), ("fgm_50_59", "fg_made_50_59"),
    ("fgm_60p", "fg_made_60_"), ("fgmiss_0_19", "fg_missed_0_19"), ("fgmiss_20_29", "fg_missed_20_29"),
    ("fgmiss_30_39", "fg_missed_30_39"), ("fgmiss_40_49", "fg_missed_40_49"),
    ("fgmiss_50_59", "fg_missed_50_59"), ("fgmiss_60p", "fg_missed_60_"), ("pass_sack", "sacks_suffered"),
])
def test_each_sleeper_key_scores_its_nflverse_column(key, column):
    assert _pts({key: 0.5}, **{column: 4}) == pytest.approx(2.0)


def test_missing_columns_count_as_zero():
    assert _pts({"rush_fd": 0.5, "fum": -2, "pass_inc": -1, "bonus_fd_rb": 1}, position="RB") == 0.0


def test_fumbles_prefer_total_column_and_fall_back_to_components():
    sc = {"fum": -2}
    assert _pts(sc, fumbles_total=2, rushing_fumbles=5) == pytest.approx(-4.0)
    assert _pts(sc, rushing_fumbles=1, receiving_fumbles=1, sack_fumbles=1) == pytest.approx(-6.0)


def test_older_sacks_column_and_incompletions():
    assert _pts({"pass_sack": -1}, sacks=3) == pytest.approx(-3.0)
    assert _pts({"pass_inc": -0.5}, attempts=30, completions=20) == pytest.approx(-5.0)


def test_long_kick_splits_sum_for_50_plus_keys():
    sc = {"fgm_50p": 5, "fgmiss_50p": -1}
    assert _pts(sc, fg_made_50_59=1, fg_made_60_=1, fg_missed_50_59=1, fg_missed_60_=2) == pytest.approx(7.0)


def test_first_down_bonus_is_position_masked():
    sc = {"bonus_fd_rb": 1.0, "bonus_fd_qb": 0.5}
    line = {"rushing_first_downs": 3, "receiving_first_downs": 2, "passing_first_downs": 10}
    assert _pts(sc, position="RB", **line) == pytest.approx(5.0)
    assert _pts(sc, position="QB", **line) == pytest.approx(2.5)
    assert _pts(sc, position="WR", **line) == 0.0


def test_first_down_league_matches_projection_scoring():
    # League 1389377870526177280 scores rush_fd / rec_fd at 0.5 and fum at -2
    sc = effective_scoring({"rush_fd": 0.5, "rec_fd": 0.5, "fum": -2, "fum_lost": 0, "rec": 0.5})
    hist = _pts(sc, position="RB", rushing_first_downs=4, receiving_first_downs=2, fumbles_total=1,
                rushing_yards=50, receptions=3, receiving_yards=20)
    proj = score_projection({"rush_fd": 4, "rec_fd": 2, "fum": 1, "rush_yd": 50, "rec": 3, "rec_yd": 20}, sc)
    assert hist == pytest.approx(proj) == pytest.approx(3 - 2 + 5 + 1.5 + 2)


# ── effective_scoring: split kicker keys replace default aggregates ──────────

def test_split_keys_drop_default_aggregates():
    sc = effective_scoring({"fgm_50_59": 5, "fgm_60p": 6, "fgmiss_30_39": -1})
    assert "fgm_50p" not in sc and "fgmiss" not in sc
    assert _pts(sc, position="K", fg_made_50_59=1, fg_missed=1, fg_missed_30_39=1) == pytest.approx(4.0)


def test_league_defining_aggregate_and_split_applies_both_like_sleeper():
    sc = effective_scoring({"fgmiss": -1, "fgmiss_30_39": -1, "fgm_50p": 5, "fgm_50_59": 1})
    assert sc["fgmiss"] == -1 and sc["fgm_50p"] == 5
    assert _pts(sc, position="K", fg_missed=1, fg_missed_30_39=1, fg_made_50_59=1) == pytest.approx(4.0)
    proj = score_projection({"fgmiss_30_39": 1, "fgm_50p": 1}, sc)
    assert proj == pytest.approx(4.0)  # projections agree with history


def test_defaults_kept_without_split_keys():
    sc = effective_scoring({"rec": 1})
    assert sc["fgm_50p"] == 5 and sc["fgmiss"] == -1


# ── projections: derived kicker keys ─────────────────────────────────────────

def test_fgmiss_derived_from_split_misses():
    stats = {"fgmiss_30_39": 0.1, "fgmiss_40_49": 0.2, "fgmiss_50p": 0.3, "fgm": 2}
    assert score_projection(stats, {"fgmiss": -1, "fgm": 3}) == pytest.approx(6 - 0.6)
    assert "fgmiss" not in stats  # caller's dict isn't mutated


def test_existing_aggregate_and_absent_splits_are_left_alone():
    assert _derive_kicker_stats({"fgmiss": 1, "fgmiss_30_39": 5}, {"fgmiss": -1}) == {"fgmiss": 1, "fgmiss_30_39": 5}
    assert _derive_kicker_stats({"fgm": 1}, {"fgmiss": -1}) == {"fgm": 1}
    assert _derive_kicker_stats({"fgmiss_30_39": 1}, {"fgm": 3}) == {"fgmiss_30_39": 1}


def test_fgm_50p_takes_the_50_59_weight_when_league_splits():
    sc = effective_scoring({"fgm_50_59": 5, "fgm_60p": 6})
    assert score_projection({"fgm_50p": 1.0}, sc) == pytest.approx(5.0)  # not 5 (default 50p) + 5
    assert _derive_kicker_stats({"fgm_50p": 1, "fgm_50_59": 0.5}, sc) == {"fgm_50p": 1, "fgm_50_59": 0.5}
    assert _derive_kicker_stats({"fgm_50p": 1, "fgm_60p": 0.5}, sc) == {"fgm_50p": 1, "fgm_60p": 0.5}
    assert _derive_kicker_stats({"fgm": 1}, sc) == {"fgm": 1}


# ── backtest: common rows, DNP = 0, missing projection = 0, defense multiplier ──

class CoveredProjections:
    def __init__(self, season, weeks, scoring, completed_through=0):
        pass

    def covers(self, week):
        return True

    def get(self, sid, week):
        return 12.0 if sid == "s1" else None


def _frame():
    rows = [("g1", "Steady Guy", "WR", w, "NYJ", "BUF", 5, 75) for w in (1, 2, 3, 4, 6)]  # DNP week 5
    rows += [("g2", "Other Guy", "WR", w, "MIA", "NE", 5, 75) for w in range(1, 7)]
    rows += [("g3", "Unmapped Teammate", "TE", w, "NYJ", "BUF", 1, 5) for w in range(1, 7)]  # no projection
    df = pd.DataFrame(rows, columns=["player_id", "player_display_name", "position", "week", "team",
                                     "opponent_team", "receptions", "receiving_yards"])
    df["season"] = 2025
    return df  # 10 half-PPR points every game


def _run(monkeypatch, multiplier=None):
    monkeypatch.setattr(backtest, "fetch_weekly_stats", lambda season: _frame())
    monkeypatch.setattr(backtest, "fetch_all_players", lambda: {
        "s1": {"full_name": "Steady Guy", "position": "WR", "gsis_id": "g1", "team": "NYJ", "active": True},
        "s2": {"full_name": "Other Guy", "position": "WR", "gsis_id": "g2", "team": "MIA", "active": True}})
    monkeypatch.setattr(backtest, "Projections", CoveredProjections)
    if multiplier:
        monkeypatch.setattr(backtest, "get_matchup_info", lambda opp, pos, table: {"multiplier": multiplier})
    res = backtest.run_backtest(2025, first_week=4, last_week=6)
    return {target: {k: (mae, n) for k, mae, n in rows} for target, rows in res.items()}


def test_backtest_common_rows_dnp_zero_and_defense_multiplier(monkeypatch):
    res = _run(monkeypatch, multiplier=2.0)["next_week"]
    hist_mae, hist_n = res[("history", backtest.BLEND_HISTORY)]
    blend_mae, blend_n = res[("blend", 1.0)]
    # g1 weeks 4-6 (5 projected, not played) + g2 weeks 4-6; unprojected g3 is left out of every model
    assert hist_n == blend_n == 6
    # history = 10 ppg × 2.0 multiplier = 20; actuals g1: 10, 0, 10; g2: 10 ×3
    assert hist_mae == pytest.approx((10 + 20 + 10 + 30) / 6)
    # blend α=1: g1 projects 12; g2 is left out of covered weeks → 0
    assert blend_mae == pytest.approx((2 + 12 + 2 + 30) / 6)


def test_backtest_real_matchup_tables_and_next_4(monkeypatch):
    res = _run(monkeypatch)
    hist_mae, n = res["next_4"][("history", backtest.BLEND_HISTORY)]
    assert n == 6
    # single defense per position → multiplier 1; g1 next_4 actuals: mean(10,0,10), mean(0,10), 10
    assert hist_mae == pytest.approx((10 - 20 / 3 + 5 + 0) / 6)


def test_team_opponents_and_tables_without_columns():
    df = pd.DataFrame({"week": [1, 1], "team": ["NYJ", None], "opponent_team": ["BUF", "MIA"]})
    assert backtest._team_opponents(df) == {("NYJ", 1): "BUF"}
    assert backtest._team_opponents(df.drop(columns="opponent_team")) == {}
    assert backtest._matchup_tables(df.drop(columns="opponent_team"), [4, 5]) == {4: {}, 5: {}}


def test_projections_covers_only_published_weeks(monkeypatch):
    from src import projections
    monkeypatch.setattr(projections, "fetch_week_projections",
                        lambda season, w, completed=False, force=False: {"s1": {"fgm_50p": 1}} if w == 4 else {})
    proj = projections.Projections(2025, [4, 5], effective_scoring({"fgm_50_59": 5}))
    assert proj.covers(4) and not proj.covers(5) and proj.get("s1", 4) == 5.0
