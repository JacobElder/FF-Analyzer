"""Season model (playoff odds, median game, week weights, trade finder) and the projection-coverage rules."""

import pytest

from src.season import (
    all_week_weights, combined_risk_prob, find_trades, league_dists, lineup_dists, median_win_prob,
    simulate_season, team_roster, team_week_dist, week_weights, weeks_until_deadline,
)
from src.valuation import ValuationContext, evaluate_add
from waiver_helpers import NoHistory, _ctx, _p

WEEKS = [6, 7, 8]


class WeeklyProjections:
    """Per-week projections that, like Sleeper's, say which weeks were published."""
    available = True

    def __init__(self, by_week):
        self.by_week = by_week

    def get(self, pid, week):
        return self.by_week.get(week, {}).get(pid)

    def covers(self, week):
        return bool(self.by_week.get(week))


class TenPpgHistory:
    """Every player averaged 10 points over 4 games."""
    def get_player_stats(self, p):
        return {"found": True, "season": {"games": 4, "ppg": 10.0}, "recent": {"ppg": 10.0}}


def _weekly_ctx(by_week, injuries=None, history=None, weeks=WEEKS):
    opponents = {"T1": {w: "XX" for w in weeks}}
    inj = {"by_id": injuries or {}, "by_name": {}}
    return ValuationContext(history or NoHistory(), WeeklyProjections(by_week), opponents, {}, inj, list(weeks))


# ── projection coverage: unprojected players and injury timing ───────────────

def test_unprojected_player_is_zero_in_covered_weeks_and_uses_history_after():
    ctx = _weekly_ctx({6: {"other": 5}, 7: {"other": 5}}, history=TenPpgHistory())
    assert ctx.week_values(_p("backup", "QB")) == {6: 0.0, 7: 0.0, 8: 10.0}


def test_player_without_a_sleeper_id_keeps_his_history_value():
    ctx = _weekly_ctx({6: {"other": 5}}, history=TenPpgHistory())
    unmatched = {"name": "Mystery", "position": "WR", "team": "T1", "slot": "BN"}
    assert ctx.week_values(unmatched)[6] == 10.0


def test_injury_timing_follows_sleeper_projections_where_published():
    ctx = _weekly_ctx({6: {"hurt": 9}, 7: {"hurt": 9}}, injuries={"hurt": "IR"})
    # Out this week by designation; Sleeper projects him back in week 7; week 8 isn't published,
    # so the 4-week IR estimate still applies there
    assert ctx.week_values(_p("hurt", "WR")) == {6: 0.0, 7: 9.0, 8: 0.0}
    assert ctx.return_week(_p("hurt", "WR")) == 7


def test_return_week_when_sleeper_never_projects_him_back():
    ctx = _weekly_ctx({6: {"x": 1}, 7: {"x": 1}, 8: {"x": 1}}, injuries={"hurt": "Out"})
    assert ctx.return_week(_p("hurt", "WR")) is None


def test_return_week_falls_back_to_the_estimate_past_published_weeks():
    ctx = _weekly_ctx({6: {"x": 1}}, injuries={"hurt": "IR"})
    assert ctx.return_week(_p("hurt", "WR")) == 10  # week 6 + 4-week IR estimate
    no_id = {"name": "Hurt", "position": "WR", "team": "T1"}
    ctx_named = _weekly_ctx({6: {"x": 1}, 7: {"x": 1}, 8: {"x": 1}}, injuries={})
    ctx_named.injury_map["by_name"] = {}
    assert ctx_named.return_week(no_id) is None  # not injured
    single = _weekly_ctx({6: {"x": 1}}, injuries={"hurt": "Out"}, weeks=[6])
    assert single.return_week(_p("hurt", "WR")) == 7  # no later weeks to check


def test_return_week_without_a_player_id_uses_the_estimate():
    ctx = _weekly_ctx({6: {"x": 1}, 7: {"x": 1}}, injuries={})
    p = {"name": "Hurt", "position": "WR", "team": "T1"}
    ctx.injury = lambda _p: "Out"
    assert ctx.return_week(p) == 7


# ── week weights ─────────────────────────────────────────────────────────────

def test_week_weights_scale_playoff_weeks_by_playoff_chances():
    assert week_weights([13, 14, 15, 16, 17], 15, None) == {13: 1.0, 14: 1.0, 15: 1.0, 16: 1.0, 17: 1.0}
    assert week_weights([14, 15, 16, 17], 15, 0.4) == {14: 1.0, 15: 0.4, 16: 0.2, 17: 0.1}
    assert week_weights([15], None, 0.5) == {15: 0.5}  # playoffs start in week 15 by default


def test_weighted_add_values_playoff_weeks_less():
    ctx = _ctx({"qb": 20, "te": 8, "te_new": 12})
    roster = [_p("qb", "QB", slot="QB"), _p("te", "TE", slot="TE")]
    full = evaluate_add(_p("te_new", "TE"), roster, ctx, ["QB", "TE", "BN"])
    half = evaluate_add(_p("te_new", "TE"), roster, ctx, ["QB", "TE", "BN"], weights={8: 0.5})
    assert full["gain_total"] == 12 and half["gain_total"] == 10
    assert half["weekly_gain"] == full["weekly_gain"]  # per-week gains stay unweighted


def test_all_week_weights_per_team():
    assert all_week_weights(None, WEEKS, 8) == {}
    sim = {"p_playoffs": {1: 1.0, 2: 0.0}}
    assert all_week_weights(sim, WEEKS, 8) == {1: {6: 1.0, 7: 1.0, 8: 1.0}, 2: {6: 1.0, 7: 1.0, 8: 0.0}}


# ── trade deadline, risk ─────────────────────────────────────────────────────

@pytest.mark.parametrize("deadline,week,expected", [(None, 7, None), (14, None, None), (14, 7, 7), (11, 11, 0),
                                                     (11, 12, -1)])
def test_weeks_until_deadline(deadline, week, expected):
    assert weeks_until_deadline(deadline, week) == expected


def test_combined_risk_prob_averages_both_games():
    assert combined_risk_prob(None, 0.5) is None
    assert combined_risk_prob(0.6, None) == 0.6
    assert combined_risk_prob(0.6, 0.4) == 0.5


# ── team projections, median game, playoff odds ──────────────────────────────

DB = {pid: {"full_name": pid, "position": pos, "team": "T1"}
      for pid, pos in [("qa", "QB"), ("wa", "WR"), ("qb", "QB"), ("wb", "WR"), ("hurt", "WR")]}
POS = ["QB", "WR", "BN", "IR"]


def _league(**over):
    league = {
        "league_id": "L", "playoff_teams": 1, "playoff_week_start": 8,
        "league_rosters": [
            {"roster_id": 1, "team_name": "Strong", "players": ["qa", "wa", "hurt"], "reserve": ["hurt"],
             "record": {"wins": 1, "losses": 0, "points_for": 100.0}},
            {"roster_id": 2, "team_name": "Weak", "players": ["qb", "wb"], "reserve": [],
             "record": {"wins": 1, "losses": 0, "ties": 0, "points_for": 90.0}},
        ],
        "schedule": {"6": [[1, 2]], "7": [[1, 2], [1, 99]], "8": [[1, 2]]},
    }
    league.update(over)
    return league


def test_team_roster_and_lineup_dists():
    roster = team_roster(_league()["league_rosters"][0], DB)
    assert {p["player_id"]: p["slot"] for p in roster} == {"qa": "BN", "wa": "BN", "hurt": "IR"}
    ctx = _ctx({"qa": 20, "wa": 10, "hurt": 30})
    mean, var = lineup_dists(roster, ctx, POS)[6]
    assert mean == 30 and var == pytest.approx((0.6 * 20) ** 2 + (0.6 * 10) ** 2)  # IR player excluded


def test_median_win_prob():
    assert median_win_prob((10, 1), []) is None
    assert median_win_prob((100, 1), [(10, 1), (10, 1), (10, 1)]) == 1.0
    assert median_win_prob((10, 1), [(100, 1), (100, 1), (100, 1)]) == 0.0


def test_simulate_season_gives_the_stronger_team_the_playoff_spot():
    ctx = _ctx({"qa": 25, "wa": 15, "qb": 8, "wb": 5})
    sim = simulate_season(_league(), ctx, DB, POS, n_sims=500)
    assert sim["weeks"] == [6, 7]  # week 8 is the playoffs; the unknown roster 99 is ignored
    assert sim["p_playoffs"][1] > 0.95 and sim["p_playoffs"][2] < 0.05
    assert sim["exp_wins"][1] == pytest.approx(3.0, abs=0.2)


def test_simulate_season_counts_the_median_game_and_ties():
    ctx = _ctx({})  # nobody projected: every score is 0, so every game is a tie
    sim = simulate_season(_league(median_game=True), ctx, DB, POS, n_sims=200)
    # Two half-wins from ties, no median wins (0 is not above the median of 0s); points for breaks the tie
    assert sim["exp_wins"] == {1: 2.0, 2: 2.0} and sim["p_playoffs"] == {1: 1.0, 2: 0.0}
    dists = league_dists(_league(), ctx, DB, POS)
    assert simulate_season(_league(), ctx, DB, POS, n_sims=50, dists=dists)["weeks"] == [6, 7]


@pytest.mark.parametrize("over", [{"schedule": {}}, {"playoff_teams": None}, {"league_rosters": []}])
def test_simulate_season_needs_schedule_and_playoff_size(over):
    assert simulate_season(_league(**over), _ctx({}), DB, POS) is None


def test_team_week_dist():
    ctx = _ctx({"qa": 20, "wa": 10})
    mean, var = team_week_dist(["qa", "wa"], ctx, DB, 6)
    assert mean == 30 and var == pytest.approx(144 + 36)


# ── trade finder ─────────────────────────────────────────────────────────────

def test_find_trades_keeps_swaps_that_help_both_teams():
    # You have two good QBs and a weak WR; the rival has two good WRs and a weak QB
    pts = {"q1": 20, "q2": 18, "w1": 5, "r_w1": 15, "r_w2": 14, "r_q": 6}
    ctx = _ctx(pts)
    db = {pid: {"full_name": pid, "position": "QB" if "q" in pid.split("_")[-1] else "WR", "team": "T1"}
          for pid in pts}
    mine = [_p("q1", "QB", slot="QB"), _p("q2", "QB"), _p("w1", "WR", slot="WR")]
    rival = {"roster_id": 2, "team_name": "Rival", "players": ["r_w1", "r_w2", "r_q"]}
    trades = find_trades(mine, [rival], ctx, db, ["QB", "WR", "BN", "BN"], rival_weights={2: {8: 0.5}})
    best = trades[0]
    assert (best["give"], best["get"]) == ("q2", "r_w1")
    # You: 25 → 35 a week for 3 weeks. Rival: 21 → 32 a week, with week 8 at half weight
    assert best["gain_me"] == 30.0 and best["gain_them"] == 27.5
    # Giving your weak WR helps you but costs the rival: never suggested
    assert not any(t["give"] == "w1" for t in trades)
    assert find_trades(mine, [rival], ctx, db, ["QB", "WR", "BN", "BN"], min_gain=1000) == []
