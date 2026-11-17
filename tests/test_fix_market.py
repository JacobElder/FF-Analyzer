"""Audit fixes: trade scoring (IR/TAXI, roster limit, projections), no-FAAB waivers, bid legality, planner."""

import pytest
from waiver_helpers import POSITIONS, PTS, ROSTER, WEEKS, _ctx, _p, _rival

from src.evaluator import score_trade
from src.market import (
    RIVAL_BID_PROB,
    TYPICAL_ADD_PER_WEEK,
    bid_profile,
    plan_claims,
    priority_win_prob,
    recommend_bid,
    rival_interest,
    rivals_ahead,
    typical_gain,
    win_curve,
)
from src.waiver_analysis import (
    RIVAL_CANDIDATES,
    analyze_waiver_pool,
    build_claim_plan,
    faab_enabled,
)


def _stats(ppg):
    return {"season": {"ppg": ppg}, "recent": {"ppg": ppg}, "weekly": [ppg] * 6}


# ── score_trade ──────────────────────────────────────────────────────────────

LINEUP = ["QB", "RB", "WR", "BN"]


def test_trade_ignores_ir_and_taxi_players():
    roster = [_p("qb", "QB", slot="QB"), _p("rb", "RB", slot="RB"), _p("wr", "WR", slot="WR"),
              _p("ir_wr", "WR", slot="IR"), _p("taxi_rb", "RB", slot="TAXI")]
    stats = {"qb": _stats(20), "rb": _stats(10), "wr": _stats(8), "ir_wr": _stats(30), "taxi_rb": _stats(25),
             "new_rb": _stats(12)}
    t = score_trade(roster, ["rb"], [_p("new_rb", "RB")], stats, roster_positions=LINEUP)
    names = {s["name"] for s in t["before"]["starters"] + t["after"]["starters"]}
    assert not names & {"ir_wr", "taxi_rb"}
    assert sorted(t["excluded"]) == ["ir_wr", "taxi_rb"]
    assert t["basis"] == "history" and t["verdict"] == t["history_verdict"] == "ACCEPT"
    assert t["delta"] == t["history_delta"] and t["roster_limit"] == 4


def test_two_for_one_over_the_limit_drops_the_lowest_bench_player():
    roster = [_p("qb", "QB", slot="QB"), _p("rb", "RB", slot="RB"), _p("wr", "WR", slot="WR"), _p("bn", "WR")]
    stats = {"qb": _stats(20), "rb": _stats(10), "wr": _stats(8), "bn": _stats(5),
             "a": _stats(3), "b": _stats(9)}
    t = score_trade(roster, [], [_p("a", "RB"), _p("b", "WR")], stats, roster_positions=LINEUP)
    assert t["dropped"] == ["a", "bn"]  # two over the 4-man limit: the two worst non-starters
    assert {s["name"] for s in t["after"]["starters"]} == {"qb", "rb", "b"}


def test_forced_drop_can_be_a_starter_when_everyone_starts():
    roster = [_p("qb", "QB", slot="QB"), _p("rb", "RB", slot="RB")]
    stats = {"qb": _stats(20), "rb": _stats(10), "wr": _stats(8)}
    t = score_trade(roster, [], [_p("wr", "WR")], stats, roster_positions=["QB", "RB", "WR"][:2])
    assert t["dropped"] == ["wr"]


def test_no_limit_without_roster_positions_or_when_already_over():
    roster = [_p("qb", "QB", slot="QB"), _p("rb", "RB", slot="RB"), _p("x", "WR"), _p("y", "WR")]
    stats = {"qb": _stats(20), "rb": _stats(10), "x": _stats(1), "y": _stats(1), "z": _stats(2)}
    assert score_trade(roster, [], [_p("z", "WR")], stats)["dropped"] == []
    # 4 players on a 2-man limit: a 1-for-1 keeps the count, so nothing is cut
    t = score_trade(roster, ["x"], [_p("z", "WR")], stats, roster_positions=["QB", "RB"])
    assert t["dropped"] == [] and t["roster_limit"] == 2


def test_vorp_fallback_keeps_all_keys():
    t = score_trade([], [], [_p("a", "RB")], {"a": _stats(20)})
    assert t["method"] == "vorp" and t["basis"] == "history" and t["verdict"] == t["history_verdict"]
    assert t["dropped"] == [] and t["excluded"] == [] and t["roster_limit"] is None


def test_projection_basis_drives_the_verdict():
    # History says the received WR is worse; projections say he's much better
    ctx = _ctx({**PTS, "new_wr": 30})
    stats = {p["name"]: _stats(10) for p in ROSTER} | {"new_wr": _stats(1)}
    t = score_trade(ROSTER, ["wr2"], [_p("new_wr", "WR")], stats, roster_positions=POSITIONS, ctx=ctx)
    assert t["basis"] == "projection" and t["method"] == "lineup"
    assert t["ros_delta"] == (30 - 14) * len(WEEKS)
    assert t["ros_delta_per_week"] == t["delta"] == 16
    assert t["verdict"] == "ACCEPT" and t["history_verdict"] == "NEGOTIATE"
    assert set(t["ros_before"]) == set(WEEKS) and "before" in t and "after" in t


def test_projection_basis_drops_the_cheapest_player_to_fit():
    ctx = _ctx({**PTS, "new_wr": 30, "new_rb": 1})
    stats = {}
    t = score_trade(ROSTER, [], [_p("new_wr", "WR"), _p("new_rb", "RB")], stats,
                    roster_positions=POSITIONS, ctx=ctx)
    assert t["dropped"] == ["new_rb", "rb2"]
    assert t["ros_delta"] == (30 - 14) * len(WEEKS)


@pytest.mark.parametrize("before_pts,after_pts,verdict", [(0, 10, "ACCEPT"), (10, 0, "DECLINE"), (0, 0, "NEGOTIATE")])
def test_projection_basis_with_an_empty_lineup(before_pts, after_pts, verdict):
    ctx = _ctx({"me": before_pts, "you": after_pts})
    t = score_trade([_p("me", "QB", slot="QB")], ["me"], [_p("you", "QB")], {}, roster_positions=["QB"], ctx=ctx)
    assert t["verdict"] == verdict


# ── market ───────────────────────────────────────────────────────────────────

def test_typical_gain_floor_follows_week_weights():
    assert typical_gain([], 10) == TYPICAL_ADD_PER_WEEK * 10
    assert typical_gain([], 10, weight_sum=4.5) == TYPICAL_ADD_PER_WEEK * 4.5
    assert typical_gain([], 10, weight_sum=0.2) == TYPICAL_ADD_PER_WEEK  # never below one week


def test_rivals_ahead_and_priority_odds():
    rivals = [_rival(pos=1), _rival(pos=7), _rival(pos=None), _rival(interested=False, pos=1)]
    assert rivals_ahead(rivals, 5) == 2           # pos 1 and the unknown one
    assert rivals_ahead(rivals, None) == 3        # my position unknown: all interested count
    assert priority_win_prob(rivals, 5) == round((1 - RIVAL_BID_PROB) ** 2, 3)
    assert priority_win_prob([], 5) == 1.0
    assert win_curve(rivals, 5, 40, faab=False) == [(0, priority_win_prob(rivals, 5))]


def test_market_tie_break_uses_the_rival_holding_the_top_bid():
    # The big bidder sits behind me in waiver order; a token bidder sits ahead.
    me_ahead = [_rival(est=40, budget=50, pos=9), _rival(est=0, budget=50, pos=1)]
    both_ahead = [_rival(est=40, budget=50, pos=1), _rival(est=0, budget=50, pos=1)]
    a = dict(win_curve(me_ahead, 5, 50, seed_key="tb", market_price=20))
    b = dict(win_curve(both_ahead, 5, 50, seed_key="tb", market_price=20))
    assert all(a[x] >= b[x] for x in a) and any(a[x] > b[x] for x in a)


def test_rival_interest_taxi_own_typical_and_no_faab():
    ctx = _ctx({**PTS, "new_te": 14, "r_te": 2, "r_stash": 40})
    players_db = {"r_te": {"full_name": "r_te", "position": "TE", "team": "T1"},
                  "r_stash": {"full_name": "r_stash", "position": "TE", "team": "T1"}}
    rival = {"roster_id": 2, "team_name": "R", "players": ["r_te", "r_stash"], "reserve": [], "taxi": ["r_stash"],
             "faab_remaining": 100, "record": {"waiver_position": 2}}
    profile = bid_profile([], 100)
    add = _p("new_te", "TE")
    base = rival_interest(add, [dict(rival)], ctx, players_db, ["TE"], typical=10, profile=profile)[0]
    assert base["interested"]  # the 40-pt taxi stash doesn't start, so he still needs a TE
    cheap = rival_interest(add, [dict(rival)], ctx, players_db, ["TE"], typical=10, profile=profile,
                           rival_typicals={2: 100})[0]
    assert cheap["est_bid"] < base["est_bid"]  # priced against his own (bigger) typical add
    nofaab = rival_interest(add, [dict(rival)], ctx, players_db, ["TE"], typical=10, profile=profile, faab=False)[0]
    assert nofaab["interested"] and nofaab["est_bid"] == 0


def test_contested_worthless_player_is_excluded_under_a_minimum():
    rec = recommend_bid(0, [(0, 0.2), (1, 0.4)], min_bid=1, budget=50)
    assert rec["suggested"] == 0 and rec["excluded"] and not rec["below_value"]
    assert not recommend_bid(0, [(0, 0.2), (1, 0.4)], min_bid=0, budget=50)["excluded"]
    assert recommend_bid(0, [(0, 0.3)], min_bid=1, budget=0)["excluded"]


def test_below_value_flags():
    unc = recommend_bid(0, [(0, 1.0)], min_bid=2, budget=50)
    assert unc["suggested"] == 2 and unc["below_value"] and not unc["excluded"]
    curve = [(0, 0.1), (1, 0.5), (2, 0.8), (3, 0.9)]
    assert recommend_bid(1, curve, min_bid=3, budget=50)["below_value"]   # minimum beats his worth
    assert not recommend_bid(5, curve, min_bid=0, budget=50)["below_value"]


def _cand(name, gain, bid, max_pay, curve=(), min_bid=0, p_win=1.0, drop=None, **kw):
    return {"name": name, "gain_total": gain, "suggested_bid": bid, "p_win": p_win, "max_pay": max_pay,
            "curve": list(curve), "min_bid": min_bid,
            "drop_options": [{"name": drop or f"d_{name}", "gain_total": gain}], **kw}


def test_plan_skips_excluded_candidates():
    cands = [_cand("x", 50, 0, 0, excluded=True), _cand("ok", 10, 1, 5)]
    assert [c["name"] for c in plan_claims(cands, budget=10)] == ["ok"]


def test_plan_rebids_a_claim_that_no_longer_fits():
    curve = [(b, 0.8 if b >= 30 else 0.3) for b in range(81)]
    cands = [_cand("A", 100, 60, 80), _cand("B", 90, 50, 80, curve=curve, p_win=0.8)]
    plan = plan_claims(cands, budget=100)
    assert [(c["name"], c["suggested_bid"]) for c in plan] == [("A", 60), ("B", 30)]


def test_plan_never_emits_a_bid_below_the_minimum():
    curve = [(b, 0.3) for b in range(11)]
    cands = [_cand("A", 100, 5, 10, min_bid=5), _cand("B", 50, 5, 10, curve=curve, min_bid=5)]

    def reevaluate(c, plan_so_far):
        return [{"name": f"d_{c['name']}", "gain_total": 1}]  # after A, B is worth almost nothing

    plan = plan_claims(cands, budget=100, reevaluate=reevaluate)
    assert [c["name"] for c in plan] == ["A"]
    assert all(c["suggested_bid"] >= c["min_bid"] for c in plan)


def test_plan_without_curve_keeps_a_conditional_bid():
    cands = [_cand("A", 100, 5, 10), _cand("B", 40, 8, 10)]

    def reevaluate(c, plan_so_far):
        return [{"name": f"d_{c['name']}", "gain_total": 20}]

    plan = plan_claims(cands, budget=100, reevaluate=reevaluate)
    assert plan[1]["name"] == "B" and plan[1]["suggested_bid"] == 5  # 10 × 20/40


# ── waiver_analysis ──────────────────────────────────────────────────────────

POOL_PTS = {"fa_te": 14, "fa_wr": 18, "fa_rb": 2, "fa_qb": 25}
POOL = [_p(pid, pos) for pid, pos in (("fa_te", "TE"), ("fa_wr", "WR"), ("fa_rb", "RB"), ("fa_qb", "QB"))]
PLAYERS_DB = {"r_wr": {"full_name": "r_wr", "position": "WR", "team": "T1"}}


def _league(faab=True, budget_total=100, rival_pos=1, my_pos=4, rivals=True, min_bid=1):
    rosters = [{"roster_id": 1, "team_name": "Me", "players": [], "reserve": [], "faab_remaining": 0,
                "record": {"waiver_position": my_pos}}]
    if rivals:
        rosters.append({"roster_id": 2, "team_name": "Rival", "players": ["r_wr"], "reserve": [],
                        "faab_remaining": 0 if not faab else 60, "record": {"waiver_position": rival_pos}})
    return {"roster_id": 1, "record": {"waiver_position": my_pos}, "league_rosters": rosters,
            "faab_enabled": faab, "waiver_budget_total": budget_total,
            "waiver_settings": {"waiver_bid_min": min_bid}, "transactions": []}


def test_faab_enabled_detection():
    assert faab_enabled({}) and faab_enabled({"faab_enabled": True, "waiver_budget_total": 100})
    assert not faab_enabled({"faab_enabled": False})
    assert not faab_enabled({"faab_enabled": True, "waiver_budget_total": 0})


@pytest.mark.parametrize("rival_pos,expected", [(1, round(1 - RIVAL_BID_PROB, 3)), (9, 1.0)])
def test_no_faab_odds_follow_waiver_priority(rival_pos, expected):
    ctx = _ctx({**PTS, **POOL_PTS, "r_wr": 1})
    analysis = analyze_waiver_pool(ROSTER, POOL, ctx, _league(faab=False, rival_pos=rival_pos), PLAYERS_DB,
                                   100, POSITIONS)
    assert analysis["faab"] is False
    wr = next(c for c in analysis["candidates"] if c["name"] == "fa_wr")  # the rival wants a WR
    assert wr["faab"] is False and wr["suggested_bid"] == 0 and wr["max_pay"] == 0 and wr["min_bid"] == 0
    assert wr["p_win_known"] and wr["p_win"] == expected and wr["curve"] == [(0, expected)]
    assert ("could claim first" in wr["bid_note"]) == (expected < 1)
    plan = build_claim_plan(analysis, ROSTER, ctx, 100, POSITIONS)
    assert plan and all(c["suggested_bid"] == 0 for c in plan)


def test_no_faab_without_rivals_is_a_sure_claim():
    ctx = _ctx({**PTS, **POOL_PTS})
    analysis = analyze_waiver_pool(ROSTER, POOL, ctx, _league(faab=True, budget_total=0, rivals=False), {},
                                   0, POSITIONS)
    for c in analysis["candidates"]:
        if c["gain_total"] > 0:
            assert c["p_win"] == 1.0 and c["suggested_bid"] == 0 and not c["contested"]


def _long_pool():
    pts, pool = {**PTS, "r_wr": 1}, []
    for i in range(RIVAL_CANDIDATES + 2):
        pts[f"fa{i}"] = 30 - i * 0.1
        pool.append(_p(f"fa{i}", "WR"))
    return pts, pool


def test_no_faab_lower_ranked_candidates_get_placeholders():
    pts, pool = _long_pool()
    analysis = analyze_waiver_pool(ROSTER, pool, _ctx(pts), _league(faab=False), PLAYERS_DB, 100, POSITIONS)
    tail = analysis["candidates"][RIVAL_CANDIDATES:]
    assert tail and all(not c["p_win_known"] and c["suggested_bid"] == 0 for c in tail)


def test_lower_ranked_candidate_below_minimum_is_excluded():
    pts, pool = _long_pool()
    analysis = analyze_waiver_pool(ROSTER, pool, _ctx(pts), _league(min_bid=5), PLAYERS_DB, 2, POSITIONS)
    tail = analysis["candidates"][RIVAL_CANDIDATES:]
    assert tail and all(c["excluded"] and c["suggested_bid"] == 0 for c in tail)


@pytest.mark.parametrize("my_weight", [1.0, 0.0])
def test_rival_weights_price_against_their_own_typical(my_weight):
    ctx = _ctx({**PTS, **POOL_PTS, "r_wr": 1})
    weights = {w: my_weight for w in WEEKS}
    analysis = analyze_waiver_pool(ROSTER, POOL, ctx, _league(), PLAYERS_DB, 40, POSITIONS, weights=weights,
                                   rival_weights={2: {w: 0.5 for w in WEEKS}})
    assert analysis["typical_gain"] >= TYPICAL_ADD_PER_WEEK * max(my_weight * len(WEEKS), 1)
    assert all("below_value" in c and "excluded" in c for c in analysis["candidates"])


def test_faab_lower_ranked_candidates_get_a_placeholder_bid():
    pts, pool = _long_pool()
    analysis = analyze_waiver_pool(ROSTER, pool, _ctx(pts), _league(), PLAYERS_DB, 100, POSITIONS)
    tail = [c for c in analysis["candidates"][RIVAL_CANDIDATES:] if c["gain_total"] > 0]
    assert tail and all(not c["p_win_known"] and "not modeled" in c["bid_note"] for c in tail)
    assert all(c["suggested_bid"] == min(max(1, c["max_pay"] // 2), 100) for c in tail)
