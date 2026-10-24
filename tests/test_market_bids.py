"""FAAB bid math invariants, rival modeling and the claim planner (src/market.py)."""

import math
import random

import pytest

from src.market import (
    MIN_INTEREST_PER_WEEK, bid_profile, max_bid, plan_claims, recommend_bid, rival_interest, standings, win_curve,
)
from waiver_helpers import PTS, _ctx, _p, _rival, _tx


def _random_curve(rng, top):
    probs = sorted(round(rng.random(), 3) for _ in range(top + 1))
    return list(enumerate(probs))


@pytest.mark.parametrize("seed", range(200))
def test_recommended_bid_never_exceeds_budget_or_value(seed):
    rng = random.Random(seed)
    budget = rng.randint(0, 100)
    min_bid = rng.choice([0, 0, 1, 2, 5])
    max_pay = rng.randint(0, budget)
    curve = _random_curve(rng, rng.randint(0, max(budget, 1)))
    rec = recommend_bid(max_pay, curve, min_bid, budget)
    assert 0 <= rec["suggested"] <= budget
    if rec["suggested"] > 0:
        assert rec["suggested"] >= min_bid  # a non-zero bid is always a legal bid
        assert rec["suggested"] <= max(max_pay, min_bid) or not rec["contested"]


def test_no_bid_when_budget_is_below_the_league_minimum():
    rec = recommend_bid(0, [(0, 0.341)], min_bid=1, budget=0)
    assert rec["suggested"] == 0 and "minimum bid" in rec["note"]
    assert recommend_bid(10, [(b, 1.0) for b in range(5)], min_bid=2, budget=0)["suggested"] == 0


def test_player_worth_nothing_gets_no_bid_even_with_a_minimum():
    rec = recommend_bid(0, [(0, 0.2), (1, 0.4)], min_bid=1, budget=50)
    assert rec["suggested"] == 0 and rec["note"] == "Not worth a bid for your roster."


def test_max_bid_guards_against_bad_inputs():
    assert max_bid(5, 0, 100, 10) == max_bid(5, 5, 100, 10)  # no typical add: treat this one as typical
    assert max_bid(float("nan"), 10, 100, 10) == 0
    assert max_bid(10, float("nan"), 100, 10) >= 0
    assert max_bid(10, 10, 0, 10) == 0


@pytest.mark.parametrize("seed", range(30))
def test_max_bid_is_monotonic_in_gain_and_within_budget(seed):
    rng = random.Random(seed)
    budget, typical, weeks = rng.randint(1, 200), rng.uniform(1, 40), rng.randint(1, 17)
    gains = sorted(rng.uniform(0, 100) for _ in range(5))
    bids = [max_bid(g, typical, budget, weeks) for g in gains]
    assert bids == sorted(bids)
    assert all(0 <= b <= budget for b in bids)


def test_market_price_curve_is_monotonic_and_capped():
    rivals = [_rival(est=10, budget=30), _rival(est=5, budget=20)]
    curve = win_curve(rivals, 5, 60, seed_key="mp", market_price=15)
    probs = [p for _, p in curve]
    assert probs == sorted(probs)
    assert curve[30][1] < 1.0   # a bid equal to the richest budget can still lose the tie
    assert curve[31][1] == 1.0  # nobody can bid above the richest rival's budget


def test_more_interested_rivals_lower_your_odds():
    one = dict(win_curve([_rival()], 5, 40, seed_key="k", market_price=10))
    three = dict(win_curve([_rival(), _rival(), _rival()], 5, 40, seed_key="k", market_price=10))
    assert three[10] < one[10]


def test_tie_goes_to_better_waiver_position():
    rival = _rival(est=10, budget=10, pos=3)
    worse = dict(win_curve([rival], my_waiver_position=5, max_level=10, seed_key="t", market_price=10))
    better = dict(win_curve([rival], my_waiver_position=1, max_level=10, seed_key="t", market_price=10))
    assert better[10] > worse[10]


def test_standings_tolerate_missing_records():
    table = standings([{"roster_id": 1, "record": None},
                       {"roster_id": 2, "record": {"wins": 3, "ties": 1, "points_for": 500}}])
    assert [r["roster_id"] for r in table] == [2, 1]


def test_plan_uses_an_open_roster_spot_only_once():
    cands = [{"name": n, "gain_total": g, "suggested_bid": 5, "p_win": 1.0, "max_pay": 10, "curve": [],
              "min_bid": 0, "drop_options": [{"name": None, "gain_total": g}]} for n, g in (("A", 10), ("B", 8))]
    plan = plan_claims(cands, budget=100, open_spots=1)
    assert [c["name"] for c in plan] == ["A"] and plan[0]["drop_name"] is None


def test_plan_with_zero_budget_only_takes_free_claims():
    cands = [{"name": "free", "gain_total": 5, "suggested_bid": 0, "p_win": 0.6, "max_pay": 0, "curve": [],
              "min_bid": 0, "drop_options": [{"name": "x", "gain_total": 5}]},
             {"name": "paid", "gain_total": 50, "suggested_bid": 3, "p_win": 0.9, "max_pay": 10, "curve": [],
              "min_bid": 0, "drop_options": [{"name": "y", "gain_total": 50}]}]
    assert [c["name"] for c in plan_claims(cands, budget=0)] == ["free"]


def test_plan_respects_max_claims():
    cands = [{"name": f"c{i}", "gain_total": 10 - i, "suggested_bid": 1, "p_win": 1.0, "max_pay": 2, "curve": [],
              "min_bid": 0, "drop_options": [{"name": f"d{i}", "gain_total": 10 - i}]} for i in range(6)]
    assert len(plan_claims(cands, budget=100, max_claims=3)) == 3


def _league_rival(rid, players, faab=100):
    return {"roster_id": rid, "team_name": f"Team {rid}", "players": players, "reserve": [],
            "faab_remaining": faab, "record": {"waiver_position": rid}}


def test_rival_interest_threshold_and_budget_cap():
    ctx = _ctx({**PTS, "new_te": 14, "r_te": 2, "r_full": 30})
    players_db = {"r_te": {"full_name": "r_te", "position": "TE", "team": "T1"},
                  "r_full": {"full_name": "r_full", "position": "TE", "team": "T1"}}
    needy = _league_rival(2, ["r_te"], faab=7)       # starting TE scores 2: wants the 14-pt TE
    set_ = _league_rival(3, ["r_full"], faab=100)    # starting TE scores 30: doesn't care
    profile = bid_profile([], 100)
    out = {r["roster_id"]: r for r in rival_interest(_p("new_te", "TE"), [needy, set_], ctx, players_db,
                                                     ["TE"], typical=10, profile=profile)}
    assert out[2]["interested"] and out[2]["gain_per_week"] >= MIN_INTEREST_PER_WEEK
    assert 0 < out[2]["est_bid"] <= 7
    assert not out[3]["interested"] and out[3]["est_bid"] == 0


def test_bid_profile_handles_empty_history():
    prof = bid_profile([], 100)
    assert prof["claims_won"] == 0 and prof["win_pct_p50"] is None and prof["zero_dollar_share"] is None
    assert math.isclose(prof["league_avg_pct"], 0.0)
    assert bid_profile([_tx(1, 1, 10)], 0)["budget_total"] == 100  # 0 budget_total falls back to 100
