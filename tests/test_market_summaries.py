"""FAAB log, standings/playoff position, matchup projection and risk mode (src/market.py)."""

import math

import pytest

from src.injuries import injury_status, is_inactive
from src.market import faab_log, matchup_projection, playoff_position, plan_claims, risk_mode
from src.valuation import lineup_suggestions
from waiver_helpers import POSITIONS, PTS, ROSTER, _ctx, _p, _tx


class Index:
    def __init__(self, stats):
        self.stats = stats

    def get_player_stats(self, p):
        return self.stats.get(p["player_id"], {"found": False})


def test_faab_log_counts_points_since_added_and_cost_per_point():
    txs = [_tx(3, 1, 20, adds=("wr",)), _tx(5, 1, 0, adds=("def",), type_="free_agent"),
           _tx(4, 2, 50, adds=("other",)), _tx(2, 1, 9, status="failed", adds=("x",))]
    index = Index({"wr": {"found": True, "weekly": [5, 10, 15, 20], "weeks": [1, 3, 4, 6]}})
    db = {"wr": {"full_name": "Wide Out", "position": "WR"}, "def": {"first_name": "Buffalo", "last_name": "Bills",
                                                                     "position": "DEF"}}
    rows = faab_log(txs, 1, index, db)
    assert [r["Player"] for r in rows] == ["Wide Out", "Buffalo Bills"]  # sorted by week; others' claims excluded
    wr, d = rows
    assert wr["Pts since added"] == 45 and wr["$ per pt"] == round(20 / 45, 2) and wr["Type"] == "Waiver"
    assert d["Pts since added"] is None and d["$ per pt"] is None and d["Type"] == "Free agent"


def _rec(w, pf, ties=0):
    return {"wins": w, "losses": 0, "ties": ties, "points_for": pf}


def test_playoff_position_inside_the_cut_shows_lead_over_first_team_out():
    rosters = [{"roster_id": i, "record": _rec(w, 100 - i)} for i, w in enumerate([6, 5, 3, 1])]
    pp = playoff_position(rosters, 1, 2)
    assert pp["in_position"] and pp["games_margin"] == 2
    assert playoff_position(rosters, 0, 4)["games_margin"] is None  # everyone makes it: no first-out team


def test_playoff_position_edge_cases():
    rosters = [{"roster_id": 1, "record": _rec(3, 90)}]
    assert playoff_position([], 1, 4) is None and playoff_position(rosters, 1, None) is None
    assert playoff_position(rosters, 99, 1) is None
    assert playoff_position(rosters, 1, 6)["in_position"]  # fewer teams than playoff spots


def test_ties_count_half_a_win_in_standings():
    rosters = [{"roster_id": 1, "record": _rec(3, 90, ties=1)}, {"roster_id": 2, "record": _rec(3, 200)}]
    assert playoff_position(rosters, 1, 1)["rank"] == 1


def test_matchup_projection_favours_the_stronger_lineup():
    ctx = _ctx({"a": 20, "b": 15, "c": 10, "d": 5})
    db = {pid: {"full_name": pid, "position": "WR", "team": "T1"} for pid in "abcd"}
    m = matchup_projection(["a", "b"], ["c", "d"], ctx, db)
    assert m["mine"] == 35 and m["theirs"] == 15 and m["p_win"] > 0.8
    assert matchup_projection(["a"], ["a"], ctx, db)["p_win"] == 0.5
    assert matchup_projection([], ["a"], ctx, db) is None
    fresh = _ctx({"a": 20, "b": 15})
    assert matchup_projection(["a"], ["b"], fresh, {})["mine"] == 0  # unknown to Sleeper: no team, no points


def test_matchup_projection_with_zero_variance_does_not_divide_by_zero():
    ctx = _ctx({})
    assert matchup_projection(["x"], ["y"], ctx, {})["p_win"] == 0.5


@pytest.mark.parametrize("p,mode", [(None, "Balanced"), (0.65, "Safe"), (0.5, "Balanced"), (0.35, "Upside")])
def test_risk_mode(p, mode):
    assert risk_mode(p) == mode


def test_injury_status_lookup_and_inactive():
    m = {"by_id": {"1": "Out"}, "by_name": {"joe guy": "Questionable"}}
    assert injury_status({"player_id": "1"}, m) == "Out"
    assert injury_status("Joe Guy", m) == "Questionable"
    assert injury_status({"player_id": "2", "name": "Joe Guy"}, m) is None  # IDs are trusted over names
    assert injury_status("Joe Guy", {}) is None
    assert is_inactive("IR") and is_inactive("Doubtful") and not is_inactive("Questionable") and not is_inactive(None)


def test_reprice_with_a_curve_uses_recommend_bid_within_budget():
    curve = [(b, min(1.0, b / 20)) for b in range(31)]
    cands = [{"name": n, "gain_total": 30, "suggested_bid": 15, "p_win": 0.75, "max_pay": 30, "curve": curve,
              "min_bid": 1, "drop_options": [{"name": "x", "gain_total": 30}, {"name": "y", "gain_total": 30}]}
             for n in ("A", "B")]

    def reevaluate(c, plan_so_far):
        return [{"name": "y", "gain_total": 10}]

    plan = plan_claims(cands, budget=25, reevaluate=reevaluate)
    assert [c["name"] for c in plan] == ["A", "B"]
    b = plan[1]
    assert b["suggested_bid"] <= 10 and b["suggested_bid"] + plan[0]["suggested_bid"] <= 25
    assert b["p_win"] == pytest.approx(min(1.0, b["suggested_bid"] / 20))


def test_lineup_suggestions_flag_bye_and_empty_slot():
    ctx = _ctx(PTS, byes={("T1", 6)})
    roster = [_p("qb", "QB", slot="QB"), _p("rb", "RB", team="T2", slot="RB"), _p("wr", "WR", slot="WR"),
              _p("te", "TE", team="T2", slot="TE"), _p("wr2", "WR", team="T2"), _p("rb2", "RB", team="T2")]
    out = lineup_suggestions(roster, ctx, POSITIONS + ["FLEX"])
    reasons = {s["out"]: s["reason"] for s in out["swaps"]}
    assert reasons.get("wr") == "wr is on bye"
    assert any(s["reason"] == "Empty starting slot" for s in out["swaps"])
    assert lineup_suggestions(roster, _ctx(PTS).__class__(**{**_ctx(PTS).__dict__, "weeks": []}))["swaps"] == []
