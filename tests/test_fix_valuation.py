"""Audit fixes: playoff bracket weeks, streamer drop costs, per-week gains, simulator centers, season guards."""

import numpy as np
import pytest

from src import simulator
from src.schedule import playoff_byes, playoff_rounds, playoff_weeks, playoff_weeks_for
from src.season import all_week_weights, round_alive, simulate_season, team_roster, week_weights
from src.simulator import SCORE_CV, simulate_lineup_change, simulate_player, simulate_trade
from src.valuation import ValuationContext, evaluate_add, fantasy_weeks, handcuff_bonus
from waiver_helpers import NoHistory, _ctx, _p

# ── playoff bracket ──────────────────────────────────────────────────────────


@pytest.mark.parametrize("settings,expected", [
    (None, [[15], [16], [17]]),
    ({"playoff_teams": 1}, [[15], [16], [17]]),  # unknown / degenerate size: 3 rounds
    ({"playoff_teams": 4, "playoff_week_start": 14}, [[14], [15]]),
    ({"playoff_teams": 6, "playoff_round_type": 1}, [[15], [16], [17, 18]]),
    ({"playoff_teams": 6, "playoff_round_type": 2}, [[15, 16], [17, 18]]),  # round 3 would be past week 18
    ({"playoff_teams": 8, "playoff_round_type": 2, "playoff_week_start": 14}, [[14, 15], [16, 17], [18]]),
    ({"playoff_teams": 12}, [[15], [16], [17], [18]]),
])
def test_playoff_rounds(settings, expected):
    assert playoff_rounds(settings) == expected
    assert playoff_weeks(settings) == [w for r in expected for w in r]


def test_playoff_byes_and_tuple_helper():
    assert [playoff_byes({"playoff_teams": n}) for n in (None, 1, 4, 5, 6, 7, 8, 10)] == [0, 0, 0, 3, 2, 1, 0, 6]
    assert playoff_weeks_for(16, {"playoff_teams": 4}) == (16, 17)
    assert playoff_weeks_for(None) == (15, 16, 17)


def test_fantasy_weeks_follow_the_bracket():
    assert fantasy_weeks(14, 15, settings={"playoff_teams": 4}) == [14, 15, 16]
    assert fantasy_weeks(13, None, settings={"playoff_week_start": 14, "playoff_teams": 6,
                                             "playoff_round_type": 2}) == [13, 14, 15, 16, 17, 18]
    assert fantasy_weeks(17, 19) == [17, 18]  # nonsense start: through the last NFL week


# ── week weights with byes ───────────────────────────────────────────────────

def test_round_alive():
    assert round_alive(0.8, 3) == [0.8, 0.4, 0.2]
    assert round_alive(0.8, 3, byes=2, playoff_teams=6, p_bye=0.5) == pytest.approx([0.3, 0.65, 0.325])
    assert round_alive(0.6, 2, byes=2, playoff_teams=6) == pytest.approx([0.4, 0.4])  # p_bye ≈ 0.6 × 2/6
    assert round_alive(0.5, 2, byes=2, playoff_teams=6, p_bye=0.9)[0] == 0.0  # p_bye clipped to p_playoffs


def test_week_weights_with_bracket_settings():
    six = {"playoff_teams": 6, "playoff_week_start": 15}
    assert week_weights([14, 15, 16, 17], None, 0.8, six, p_bye=0.5) == {14: 1.0, 15: 0.3, 16: 0.65, 17: 0.325}
    # 4-team bracket: two rounds, nothing after the championship
    assert week_weights([15, 16, 17], 15, 1.0, {"playoff_teams": 4}) == {15: 1.0, 16: 0.5, 17: 0.0}
    # two-week rounds share a weight
    assert week_weights([15, 16, 17, 18], 15, 1.0, {"playoff_teams": 4, "playoff_round_type": 2}) == \
        {15: 1.0, 16: 1.0, 17: 0.5, 18: 0.5}


def test_all_week_weights_uses_simulated_bye_odds():
    sim = {"p_playoffs": {1: 0.8}, "p_bye": {1: 0.5}}
    assert all_week_weights(sim, [15, 16], 15, {"playoff_teams": 6}) == {1: {15: 0.3, 16: 0.65}}


# ── season simulation ────────────────────────────────────────────────────────

DB = {pid: {"full_name": pid, "position": pos, "team": "T1"}
      for pid, pos in [("qa", "QB"), ("wa", "WR"), ("qb", "QB"), ("wb", "WR"), ("tx", "WR")]}
POS = ["QB", "WR", "BN", "IR"]


def _league(wins=(1, 1), **over):
    league = {
        "league_id": "L", "playoff_teams": 3, "playoff_week_start": 8,
        "league_rosters": [
            {"roster_id": 1, "players": ["qa", "wa"], "record": {"wins": wins[0], "losses": 0}},
            {"roster_id": 2, "players": ["qb", "wb"], "record": {"wins": wins[1], "losses": 0}},
            {"roster_id": 3, "players": [], "record": {}},
        ],
        "schedule": {"6": [[1, 2]], "7": [[1, 2]]},
    }
    league.update(over)
    return league


def test_team_roster_puts_taxi_players_in_the_taxi_slot():
    roster = team_roster({"players": ["qa", "tx"], "taxi": ["tx"]}, DB)
    assert {p["player_id"]: p["slot"] for p in roster} == {"qa": "BN", "tx": "TAXI"}


def test_simulate_season_reports_bye_odds():
    ctx = _ctx({"qa": 25, "wa": 15, "qb": 8, "wb": 5})
    sim = simulate_season(_league(), ctx, DB, POS, n_sims=300)  # 3-team bracket: 1 bye
    assert sim["p_bye"][1] > 0.95 and sim["p_bye"][2] < 0.05 and sim["p_bye"][3] == 0.0
    assert sim["weeks"] == [6, 7]


def test_simulate_season_skips_a_current_week_already_in_the_records():
    ctx = _ctx({"qa": 25, "wa": 15, "qb": 8, "wb": 5})
    sim = simulate_season(_league(wins=(6, 0)), ctx, DB, POS, n_sims=100)  # 6 games through week 6
    assert sim["weeks"] == [7]
    median = simulate_season(_league(wins=(6, 0), median_game=True), ctx, DB, POS, n_sims=100)
    assert median["weeks"] == [6, 7]  # 6 games of 12 with a median game: week 6 not final yet


# ── streamers: the drop's loss counts every week ─────────────────────────────

STREAM_POS = ["RB", "DEF", "BN"]


def _stream_case(rb2_pts):
    ctx = _ctx({"rb1": 12, "rb2": rb2_pts, "d_old": 4, "d_new": 10}, byes={("T1", 8)})
    roster = [_p("rb1", "RB", slot="RB"), _p("rb2", "RB", team="T2"), _p("d_old", "DEF", team="T2", slot="DEF")]
    return ctx, roster


def test_streamer_drop_loses_the_bye_week_fill_in():
    ctx, roster = _stream_case(3)  # rb2 (9 ROS) is cheaper than d_old (12 ROS)
    e = evaluate_add(_p("d_new", "DEF", team="T3"), roster, ctx, STREAM_POS, max_drop_candidates=1)
    # +6 in each streaming week, but rb2 would have started in rb1's week-8 bye
    assert e["drop"]["name"] == "rb2" and e["weekly_gain"] == {6: 6.0, 7: 6.0, 8: -3.0}
    assert e["gain_total"] == 9.0 and e["gain_per_week"] == 4.5  # over the 2 streaming weeks


def test_dropping_a_def_for_a_streamer_costs_nothing_later():
    ctx, roster = _stream_case(3)
    e = evaluate_add(_p("d_new", "DEF", team="T3"), roster, ctx, STREAM_POS)
    assert e["drop"]["name"] == "d_old" and e["weekly_gain"] == {6: 6.0, 7: 6.0, 8: 0.0}
    assert {o["name"]: o["gain_total"] for o in e["drop_options"]} == {"d_old": 12.0, "rb2": 9.0, "rb1": -6.0}


def test_gain_per_week_divides_by_summed_weights():
    ctx = _ctx({"qb": 20, "te": 8, "te_new": 12})
    roster = [_p("qb", "QB", slot="QB"), _p("te", "TE", slot="TE")]
    e = evaluate_add(_p("te_new", "TE"), roster, ctx, ["QB", "TE", "BN"], weights={8: 0.5})
    assert e["gain_total"] == 10.0 and e["gain_per_week"] == 4.0  # 10 / 2.5 weighted weeks
    zero = evaluate_add(_p("te_new", "TE"), roster, ctx, ["QB", "TE", "BN"], weights={6: 0, 7: 0, 8: 0})
    assert zero["gain_total"] == 0.0 and zero["gain_per_week"] == 0.0


def test_handcuff_bonus_is_weighted_like_the_gain():
    db = {"rb1": {"team": "T1", "depth_chart_order": 1, "depth_chart_position": "RB", "active": True},
          "rb2": {"team": "T1", "depth_chart_order": 2, "depth_chart_position": "RB", "active": True}}
    ctx = ValuationContext(NoHistory(), _ctx({"rb1": 20.0, "rb2": 4.0}).projections,
                           {"T1": {w: "XX" for w in (6, 7, 8)}}, {}, {"by_id": {}, "by_name": {}}, [6, 7, 8],
                           players_db=db)
    starter = _p("rb1", "RB", slot="RB")
    full, _ = handcuff_bonus(_p("rb2", "RB"), [starter], ctx)
    half, _ = handcuff_bonus(_p("rb2", "RB"), [starter], ctx, {6: 0.5, 7: 0.5, 8: 0.5})
    assert half == pytest.approx(full / 2, abs=0.01)


def test_return_week_is_not_before_the_first_unpublished_week():
    class Proj:
        available = True

        def get(self, pid, week):
            return {6: {"x": 1}, 7: {"x": 1}}.get(week, {}).get(pid)

        def covers(self, week):
            return week in (6, 7)

    ctx = ValuationContext(NoHistory(), Proj(), {"T1": {w: "XX" for w in (6, 7, 8)}}, {},
                           {"by_id": {"hurt": "Out"}, "by_name": {}}, [6, 7, 8])
    p = _p("hurt", "WR")
    # Out (1-week estimate → 7), but Sleeper left him out of week 7 too; week 8 is unpublished
    assert ctx.return_week(p) == 8 and ctx.week_values(p)[7] == 0.0


# ── simulator ────────────────────────────────────────────────────────────────

STATS = {"weekly": [12, 0, 18, 9, 22, 14, 7, 16], "season": {"ppg": 12.25}, "recent": {"ppg": 14.0}}
FEW = {"weekly": [10, 14], "season": {"ppg": 12.0}}


def test_sparse_fallback_is_a_gamma_with_score_cv():
    s = simulate_player(FEW, "rookie")
    assert s.min() >= 0 and s.mean() == pytest.approx(12, rel=0.05)
    assert s.std() / s.mean() == pytest.approx(SCORE_CV, rel=0.1)


def test_center_sets_the_mean():
    assert simulate_player(STATS, "A", center=20).mean() == pytest.approx(20, rel=1e-6)
    assert simulate_player(FEW, "rookie", center=6).mean() == pytest.approx(6, rel=0.05)
    assert np.all(simulate_player(STATS, "A", center=0) == 0)
    assert np.all(simulate_player(FEW, "rookie", center=0) == 0)
    assert simulate_player(STATS, "A", center=10, multiplier=0.5).mean() == pytest.approx(5, rel=1e-6)


def test_failed_fit_recenters_on_the_fallback(monkeypatch):
    monkeypatch.setattr(simulator.sp_stats.gamma, "fit", lambda *a, **k: (0.0, 0, 1.0))
    assert simulate_player(STATS, "A", center=8).mean() == pytest.approx(8, rel=0.05)


def test_lineup_change_and_trade_use_centers():
    stats = {"A": STATS, "B": STATS}
    bye = simulate_lineup_change(["A"], ["B"], stats, centers={"A": 15, "B": 0})
    assert bye["p_better"] < 0.1 and bye["after"]["mean"] == 0
    assert simulate_trade({"A": STATS}, {"B": STATS}, centers={"A": 0})["verdict"] == "ACCEPT"
