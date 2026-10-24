"""Lineup optimization, trade scoring, drop advice and waiver scores (src/evaluator.py)."""

import itertools
import math
import random

import pytest

from src.evaluator import (
    SLOT_ELIGIBILITY, _consistency, optimal_lineup, player_value, recommend_drop, score_trade, score_waiver,
)


def _st(ppg, games=4):
    """Stats with a flat weekly line, so adj_value == ppg exactly."""
    return {"found": True, "season": {"ppg": ppg, "games": games}, "recent": {"ppg": ppg},
            "weekly": [ppg] * games}


def _brute_force_total(players, slots):
    """Best possible lineup total by trying every assignment (small inputs only)."""
    best = 0.0
    idx = list(range(len(players)))
    for k in range(min(len(players), len(slots)) + 1):
        for chosen in itertools.permutations(idx, k):
            for slot_ids in itertools.combinations(range(len(slots)), k):
                if all(players[p]["position"] in SLOT_ELIGIBILITY[slots[s]] for p, s in zip(chosen, slot_ids)):
                    best = max(best, sum(players[p]["value"] for p in chosen))
    return round(best, 2)


def test_overlapping_flex_slots_are_filled_optimally():
    # Greedy filling put the WR in WRRB_FLEX and left REC_FLEX for the TE: 20 instead of 29
    players = [{"name": "w", "position": "WR", "value": 15}, {"name": "r", "position": "RB", "value": 14},
               {"name": "t", "position": "TE", "value": 5}]
    lineup = optimal_lineup(players, ["WRRB_FLEX", "REC_FLEX"])
    assert lineup["total"] == 29
    assert {s["name"] for s in lineup["starters"]} == {"w", "r"}


@pytest.mark.parametrize("seed", range(150))
def test_optimal_lineup_matches_brute_force(seed):
    rng = random.Random(seed)
    slot_pool = ["QB", "RB", "WR", "TE", "FLEX", "WRRB_FLEX", "REC_FLEX", "SUPER_FLEX"]
    slots = [rng.choice(slot_pool) for _ in range(rng.randint(1, 4))]
    if seed % 3 == 0:
        slots += ["WRRB_FLEX", "REC_FLEX"]  # the overlapping pair greedy filling gets wrong
    players = [{"name": f"p{i}", "position": rng.choice(["RB", "WR", "TE", "QB"]),
                "value": round(rng.uniform(0, 25), 1)} for i in range(rng.randint(0, 6))]
    lineup = optimal_lineup(players, slots)
    assert lineup["total"] == pytest.approx(_brute_force_total(players, slots))
    names = [s["name"] for s in lineup["starters"] if s["name"]]
    assert len(names) == len(set(names))  # nobody starts twice
    for s in lineup["starters"]:
        if s["name"]:
            assert s["position"] in SLOT_ELIGIBILITY[s["slot"]]


def test_unfillable_slots_stay_empty():
    lineup = optimal_lineup([{"name": "q", "position": "QB", "value": 20}], ["QB", "TE"])
    assert lineup["total"] == 20
    assert [s["name"] for s in lineup["starters"]] == ["q", None] or [s["name"] for s in lineup["starters"]] == [None, "q"]


def test_trade_into_an_empty_lineup_is_an_accept():
    roster = [{"name": "nobody", "position": "WR"}]
    out = score_trade(roster, ["nobody"], [{"name": "star", "position": "WR"}],
                      {"star": _st(20)}, roster_positions=["WR"])
    assert out["before"]["total"] == 0 and out["after"]["total"] == 20
    assert out["verdict"] == "ACCEPT"


def test_receiving_a_player_you_already_own_is_not_double_counted():
    roster = [{"name": "a", "position": "WR"}, {"name": "b", "position": "WR"}]
    stats = {"a": _st(15), "b": _st(5)}
    out = score_trade(roster, ["b"], [{"name": "a", "position": "WR"}], stats, roster_positions=["WR", "WR"])
    assert out["after"]["total"] == 15  # a once, not twice


def test_trade_is_antisymmetric():
    roster = [{"name": "a", "position": "RB"}, {"name": "b", "position": "WR"}]
    stats = {"a": _st(12), "b": _st(9), "c": _st(16)}
    positions = ["RB", "WR", "FLEX"]
    there = score_trade(roster, ["a"], [{"name": "c", "position": "WR"}], stats, roster_positions=positions)
    swapped = [{"name": "c", "position": "WR"}, {"name": "b", "position": "WR"}]
    back = score_trade(swapped, ["c"], [{"name": "a", "position": "RB"}], stats, roster_positions=positions)
    assert there["delta"] == pytest.approx(-back["delta"])


def test_recommend_drop_does_not_pick_a_stat_less_defense():
    roster = [{"name": "Bills D", "position": "DEF"}, {"name": "wr_ok", "position": "WR"},
              {"name": "rb_ok", "position": "RB"}]
    stats = {"wr_ok": _st(12), "rb_ok": _st(11)}
    drop = recommend_drop("WR", roster, stats)
    assert drop["name"] == "wr_ok"  # weakest same-position player; the DEF can't be valued
    assert recommend_drop("DEF", roster, stats)["name"] == "Bills D"  # swapping defenses is fine


def test_recommend_drop_priority_tiers():
    roster = [{"name": "wr_bad", "position": "WR"}, {"name": "rb_bad", "position": "RB"},
              {"name": "wr_good", "position": "WR"}]
    stats = {"wr_bad": _st(6), "rb_bad": _st(4), "wr_good": _st(15)}
    assert recommend_drop("WR", roster, stats)["name"] == "wr_bad"  # 1: same position, below replacement
    assert recommend_drop("TE", roster, stats)["name"] == "rb_bad"  # 2: weakest below replacement
    good = {"wr_bad": _st(12), "rb_bad": _st(11), "wr_good": _st(15)}
    assert recommend_drop("WR", roster, good)["name"] == "wr_bad"   # 3: weakest same position
    assert recommend_drop("TE", roster, good)["name"] == "rb_bad"   # 4: weakest overall
    assert recommend_drop("WR", [], stats) is None


@pytest.mark.parametrize("ppg,verdict", [(14.4, "STRONG ADD"), (11, "ADD"), (8, "STREAM"), (7.9, "AVOID")])
def test_score_waiver_verdict_boundaries(ppg, verdict):
    assert score_waiver(_st(ppg), "WR", replacement_levels={"WR": 10.0})["verdict"] == verdict


def test_score_waiver_replacement_level_is_fifty():
    assert score_waiver(_st(10), "WR", replacement_levels={"WR": 10.0})["score"] == 50.0


def test_consistency_ignores_nan_weeks():
    assert _consistency([10, float("nan"), 10, 10], 10) == 1.0
    assert math.isfinite(player_value({"season": {"ppg": 10}, "recent": {"ppg": 10},
                                       "weekly": [10, float("nan"), 12, 8]})["adj_value"])
