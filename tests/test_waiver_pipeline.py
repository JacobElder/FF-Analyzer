"""End-to-end: league data → waiver analysis → claim plan (src/waiver_analysis.py)."""

import pytest

from src.waiver_analysis import RIVAL_CANDIDATES, analyze_waiver_pool, build_claim_plan
from waiver_helpers import POSITIONS, PTS, ROSTER, _ctx, _p

POOL_PTS = {"fa_te": 14, "fa_wr": 18, "fa_rb": 2, "fa_qb": 25}
POOL = [_p(pid, pos) for pid, pos in (("fa_te", "TE"), ("fa_wr", "WR"), ("fa_rb", "RB"), ("fa_qb", "QB"))]


def _league(min_bid=1, rivals=True):
    rosters = [{"roster_id": 1, "team_name": "Me", "players": [], "reserve": [], "faab_remaining": 40,
                "record": {"waiver_position": 4}}]
    if rivals:
        rosters += [{"roster_id": 2, "team_name": "Rival", "players": ["r_wr"], "reserve": [],
                     "faab_remaining": 60, "record": {"waiver_position": 1}}]
    return {"roster_id": 1, "record": {"waiver_position": 4}, "league_rosters": rosters,
            "waiver_budget_total": 100, "waiver_settings": {"waiver_bid_min": min_bid}, "transactions": []}


PLAYERS_DB = {"r_wr": {"full_name": "r_wr", "position": "WR", "team": "T1"}}


@pytest.mark.parametrize("budget", [0, 1, 7, 40, 100])
def test_bids_stay_within_budget_and_plan_is_consistent(budget):
    ctx = _ctx({**PTS, **POOL_PTS, "r_wr": 1})
    analysis = analyze_waiver_pool(ROSTER, POOL, ctx, _league(), PLAYERS_DB, budget, POSITIONS)
    cands = {c["name"]: c for c in analysis["candidates"]}

    for c in cands.values():
        assert 0 <= c["suggested_bid"] <= budget
        assert 0 <= c["p_win"] <= 1
        if c["gain_total"] <= 0:
            assert c["suggested_bid"] == 0
    assert cands["fa_rb"]["gain_total"] == 0  # 2 pts never starts
    assert cands["fa_qb"]["gain_total"] == (25 - 20) * 3

    plan = build_claim_plan(analysis, ROSTER, ctx, budget, POSITIONS)
    assert sum(c["suggested_bid"] for c in plan) <= budget
    drops = [c["drop_name"] for c in plan if c["drop_name"]]
    assert len(drops) == len(set(drops))  # each roster player dropped at most once
    rostered = {p["name"] for p in ROSTER}
    assert not {c["name"] for c in plan} & rostered
    assert [c["priority"] for c in plan] == list(range(1, len(plan) + 1))


def test_without_rivals_bids_are_uncontested_minimums():
    ctx = _ctx({**PTS, **POOL_PTS})
    analysis = analyze_waiver_pool(ROSTER, POOL, ctx, _league(min_bid=1, rivals=False), {}, 50, POSITIONS)
    for c in analysis["candidates"]:
        if c["gain_total"] > 0:
            assert c["suggested_bid"] == min(max(1, c["max_pay"] // 2), 50)
            assert not c["p_win_known"] and c["p_win"] == 0.5


def test_lower_ranked_candidates_get_placeholder_odds():
    pts = {**PTS}
    pool = []
    for i in range(RIVAL_CANDIDATES + 3):
        pid = f"wr{i}"
        pts[pid] = 30 - i * 0.1  # all beat the starting WR, in a known order
        pool.append(_p(pid, "WR"))
    ctx = _ctx(pts)
    analysis = analyze_waiver_pool(ROSTER, pool, ctx, _league(), PLAYERS_DB, 100, POSITIONS)
    tail = analysis["candidates"][RIVAL_CANDIDATES:]
    assert tail and all(not c["p_win_known"] and c["rivals"] == [] for c in tail)


def test_open_spots_ignore_ir_and_taxi():
    ctx = _ctx(PTS)
    roster = ROSTER[:5] + [{**_p("hurt", "RB"), "slot": "IR"}]
    analysis = analyze_waiver_pool(roster, [], ctx, _league(), {}, 10, POSITIONS)
    assert analysis["open_spots"] == 1
