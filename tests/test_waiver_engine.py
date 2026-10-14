from src.market import (
    bid_profile, market_price_pct, max_bid, plan_claims, playoff_position, recommend_bid, waiver_day, win_curve,
)
from src.projections import score_projection
from src.valuation import ValuationContext, evaluate_add, fantasy_weeks, lineup_suggestions
from src.waiver_analysis import build_claim_plan

WEEKS = [6, 7, 8]
POSITIONS = ["QB", "RB", "WR", "TE", "BN", "BN"]


class NoHistory:
    """StatsIndex stand-in: no past production, so values come from projections only."""
    def get_player_stats(self, p):
        return {"found": False, "season": {"games": 0, "ppg": 0}, "recent": {"ppg": None}}


class FlatProjections:
    available = True

    def __init__(self, pts):
        self.pts = pts

    def get(self, pid, week):
        return self.pts.get(pid)


def _ctx(pts, injuries=None, byes=None):
    teams = {"T1", "T2", "T3"}
    opponents = {t: {w: "XX" for w in WEEKS if (t, w) not in (byes or set())} for t in teams}
    inj = {"by_id": injuries or {}, "by_name": {}}
    return ValuationContext(NoHistory(), FlatProjections(pts), opponents, {}, inj, WEEKS)


def _p(pid, pos, team="T1", slot="BN"):
    return {"name": pid, "player_id": pid, "position": pos, "team": team, "slot": slot}


ROSTER = [_p("qb", "QB", slot="QB"), _p("rb", "RB", slot="RB"), _p("wr", "WR", slot="WR"),
          _p("te", "TE", slot="TE"), _p("wr2", "WR"), _p("rb2", "RB")]
PTS = {"qb": 20, "rb": 15, "wr": 14, "te": 8, "wr2": 6, "rb2": 3}


def test_fantasy_weeks_run_through_championship():
    assert fantasy_weeks(6, 15) == list(range(6, 18))
    assert fantasy_weeks(None, 15) == []


def test_week_values_zero_on_bye_and_while_injured():
    ctx = _ctx({"a": 10}, injuries={"a": "IR"}, byes={("T2", 7)})
    assert ctx.week_values(_p("a", "WR")) == {6: 0.0, 7: 0.0, 8: 0.0}  # IR: out an estimated 4 weeks
    ctx = _ctx({"b": 10}, byes={("T2", 7)})
    assert ctx.week_values(_p("b", "WR", team="T2")) == {6: 10, 7: 0.0, 8: 10}


def test_add_drops_the_cheapest_player_and_values_lineup_gain():
    ctx = _ctx({**PTS, "te_new": 12})
    e = evaluate_add(_p("te_new", "TE"), ROSTER, ctx, POSITIONS)
    assert e["drop"]["name"] == "rb2"
    assert e["gain_total"] == (12 - 8) * 3  # replaces the starting TE every week


def test_bench_player_adds_nothing():
    ctx = _ctx({**PTS, "wr_new": 5})
    assert evaluate_add(_p("wr_new", "WR"), ROSTER, ctx, POSITIONS)["gain_total"] == 0


def test_lineup_suggestions_name_the_replacement_for_an_out_starter():
    ctx = _ctx(PTS, injuries={"wr": "Out"})
    swaps = lineup_suggestions(ROSTER, ctx, POSITIONS)["swaps"]
    assert [(s["out"], s["in"]) for s in swaps] == [("wr", "wr2")]
    assert "Out" in swaps[0]["reason"]


def _tx(week, roster_id, bid, status="complete", adds=("p1",), note=None, type_="waiver"):
    return {"week": week, "type": type_, "status": status, "bid": bid, "roster_id": roster_id,
            "adds": list(adds), "note": note}


def test_bid_profile_and_market_price():
    txs = [_tx(1, 1, 10), _tx(1, 2, 4, "failed", note="claimed by another owner"),
           _tx(2, 1, 20, adds=("p2",)), _tx(2, 2, 0, adds=("p3",)), _tx(3, 3, 30, adds=("p4",)),
           _tx(3, 3, 5, adds=("p5",))]
    prof = bid_profile(txs, 100)
    assert prof["claims_won"] == 5 and prof["claims_outbid"] == 1
    assert prof["contested"] == 1 and prof["median_overbid"] == 6
    assert prof["zero_dollar_share"] == 0.2
    assert market_price_pct(prof, 1.0) == prof["win_pct_p50"]
    assert market_price_pct(prof, 0.1) == prof["win_pct_p25"] < market_price_pct(prof, 2.5)
    assert market_price_pct({"claims_won": 2}, 1.0) is None  # too little history


def test_max_bid_scales_with_gain_and_season_left():
    assert max_bid(0, 10, 100, 10) == 0
    assert max_bid(20, 10, 100, 10) > max_bid(10, 10, 100, 10)
    assert max_bid(10, 10, 100, 2) > max_bid(10, 10, 100, 12)  # fewer claims left → each worth more
    assert max_bid(1000, 10, 100, 10) == 100


def _rival(interested=True, est=20, budget=100, pos=3, aggression=1.0):
    return {"interested": interested, "est_bid": est, "budget": budget, "waiver_position": pos,
            "aggression": aggression}


def test_win_curve_uncontested_and_monotonic():
    assert all(p == 1.0 for _, p in win_curve([_rival(interested=False)], 1, 10))
    curve = win_curve([_rival(), _rival(est=35)], 5, 100, seed_key="x")
    probs = [p for _, p in curve]
    assert probs == sorted(probs) and probs[0] < 0.5 and probs[-1] > 0.95


def test_recommend_bid_targets_win_prob_within_value():
    curve = [(b, min(1.0, b / 40)) for b in range(61)]
    assert recommend_bid(50, curve)["suggested"] == 30  # 75% at $30
    capped = recommend_bid(20, curve)
    assert capped["suggested"] == 20 and "more than he's worth" in capped["note"]
    assert recommend_bid(50, [(b, 1.0) for b in range(11)], min_bid=1)["suggested"] == 1


def _cand(name, gain, bid, drops, max_pay=None, p_win=0.8):
    return {"name": name, "gain_total": gain, "suggested_bid": bid, "p_win": p_win, "max_pay": max_pay or bid * 2,
            "curve": [], "min_bid": 0, "drop_options": [{"name": d, "gain_total": gain} for d in drops]}


def test_plan_respects_budget_and_uses_each_drop_once():
    cands = [_cand("A", 30, 60, ["x", "y"]), _cand("B", 20, 30, ["x", "y"]), _cand("C", 25, 50, ["x"])]
    plan = plan_claims(cands, budget=100)
    assert [c["name"] for c in plan] == ["A", "B"]
    assert {c["drop_name"] for c in plan} == {"x", "y"}
    assert sum(c["suggested_bid"] for c in plan) <= 100


def test_plan_reprices_claims_that_lose_value_after_earlier_ones():
    cands = [_cand("TE1", 30, 40, ["x", "y"], max_pay=60), _cand("TE2", 28, 40, ["x", "y"], max_pay=56)]

    def reevaluate(c, plan_so_far):  # with TE1 on the roster, TE2 is only a backup
        return [{"name": d, "gain_total": 2} for d in ("x", "y")]

    plan = plan_claims(cands, budget=200, reevaluate=reevaluate)
    te2 = next(c for c in plan if c["name"] == "TE2")
    assert te2["gain_total"] == 2 and te2["standalone_gain"] == 28
    assert te2["suggested_bid"] <= 4  # 56 × 2/28


def test_conditional_plan_on_a_real_roster_never_stacks_one_slot():
    ctx = _ctx({**PTS, "te_a": 12, "te_b": 11})
    cands = []
    for pid in ("te_a", "te_b"):
        e = evaluate_add(_p(pid, "TE"), ROSTER, ctx, POSITIONS)
        cands.append({**_p(pid, "TE"), **e, "drop_name": e["drop"]["name"], "suggested_bid": 5, "p_win": 0.9,
                      "max_pay": 20, "curve": [], "min_bid": 0})
    analysis = {"candidates": cands, "open_spots": 0}
    plan = build_claim_plan(analysis, ROSTER, ctx, 100, POSITIONS)
    assert [c["name"] for c in plan] == ["te_a"]  # te_b adds nothing once te_a starts


def test_playoff_position_and_waiver_day():
    rec = lambda w, pf: {"wins": w, "losses": 5 - w, "ties": 0, "points_for": pf}
    rosters = [{"roster_id": i, "record": rec(w, 100 + i)} for i, w in enumerate([5, 4, 3, 2, 1, 0])]
    pp = playoff_position(rosters, 3, 2)
    assert pp["rank"] == 4 and not pp["in_position"] and pp["games_margin"] == -2
    assert waiver_day({"waiver_day_of_week": 2}) == "Wednesday"
    assert waiver_day({}) is None


def test_projection_scoring_uses_league_weights_with_fallback():
    assert score_projection({"rec": 5, "rec_yd": 60, "pts_ppr": 99}, {"rec": 0.5, "rec_yd": 0.1}) == 8.5
    assert score_projection({"pts_half_ppr": 7.5}, {"rec": 0.5}) == 7.5  # DEF: no matching keys
