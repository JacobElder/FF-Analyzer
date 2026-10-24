"""Projection/history blending, injuries, streamers, handcuffs (src/valuation.py) and roster grades."""

import pandas as pd
import pytest

from src.evaluator import analyze_roster_construction
from src.valuation import (
    CURRENT_WEEK_FACTORS, HANDCUFF_SHARE, PROJECTION_WEIGHT, STARTER_MISS_RATE, STREAM_HORIZON,
    ValuationContext, evaluate_add, handcuff_bonus, team_week_opponents,
)
from waiver_helpers import WEEKS, FlatProjections, NoHistory, _p


class History:
    """StatsIndex stand-in with a fixed season/recent ppg per player."""
    def __init__(self, ppg):
        self.ppg = ppg

    def get_player_stats(self, p):
        v = self.ppg.get(p["name"])
        if v is None:
            return {"found": False, "season": {"games": 0, "ppg": 0}, "recent": {"ppg": None}}
        return {"found": True, "season": {"games": 6, "ppg": v}, "recent": {"ppg": v}}


def _ctx(index, pts=None, injuries=None, vegas=None, players_db=None, matchup_table=None):
    opponents = {t: {w: "OPP" for w in WEEKS} for t in ("T1", "T2", "OPP")}
    return ValuationContext(index, FlatProjections(pts) if pts is not None else None, opponents,
                            matchup_table or {}, {"by_id": injuries or {}, "by_name": {}}, WEEKS,
                            vegas=vegas or {}, players_db=players_db or {})


def test_projection_and_history_are_blended():
    ctx = _ctx(History({"a": 10.0}), pts={"a": 20.0})
    assert ctx.week_values(_p("a", "WR"))[7] == pytest.approx(PROJECTION_WEIGHT * 20 + (1 - PROJECTION_WEIGHT) * 10)


def test_history_alone_uses_matchup_multiplier():
    table = {("OPP", "WR"): {"multiplier": 1.2, "rank_label": "Favorable"}}
    ctx = _ctx(History({"a": 10.0}), matchup_table=table)
    assert ctx.week_values(_p("a", "WR"))[7] == pytest.approx(12.0)


def test_questionable_discounted_only_this_week():
    ctx = _ctx(NoHistory(), pts={"a": 10.0}, injuries={"a": "Questionable"})
    vals = ctx.week_values(_p("a", "WR"))
    assert vals[6] == pytest.approx(10 * CURRENT_WEEK_FACTORS["Questionable"]) and vals[7] == 10.0
    assert ctx.return_week(_p("a", "WR")) is None


def test_out_player_return_week():
    ctx = _ctx(NoHistory(), pts={"a": 10.0}, injuries={"a": "Out"})
    assert ctx.return_week(_p("a", "WR")) == 7 and ctx.week_values(_p("a", "WR"))[6] == 0.0


def test_nan_projection_falls_back_instead_of_poisoning_totals():
    ctx = _ctx(History({"a": 10.0}), pts={"a": float("nan")})
    assert ctx.week_values(_p("a", "WR"))[7] == pytest.approx(10.0)
    assert ctx.ros_total(_p("a", "WR")) == pytest.approx(30.0)


def test_vegas_fallback_for_defense_and_kicker_this_week_only():
    ctx = _ctx(NoHistory(), vegas={"OPP": 17.0, "T1": 27.0})
    d = ctx.week_values(_p("d", "DEF"))
    k = ctx.week_values(_p("k", "K"))
    assert d[6] == pytest.approx(7 + (21 - 17) * 0.45) and d[7] == 0.0
    assert k[6] == pytest.approx(8 + (27 - 22) * 0.2)


def test_streamer_gain_counts_only_the_short_horizon():
    roster = [{**_p("d_old", "DEF"), "slot": "DEF"}]
    ctx = _ctx(NoHistory(), pts={"d_old": 5.0, "d_new": 9.0})
    e = evaluate_add(_p("d_new", "DEF"), roster, ctx, ["DEF"])
    assert e["streamer"] and e["gain_total"] == pytest.approx(4.0 * STREAM_HORIZON)
    assert list(e["weekly_gain"].values())[STREAM_HORIZON:] == [0.0] * (len(WEEKS) - STREAM_HORIZON)


def test_handcuff_bonus_only_for_a_starting_rb1_teammate():
    db = {"rb1": {"team": "T1", "depth_chart_order": 1, "depth_chart_position": "RB", "active": True},
          "rb2": {"team": "T1", "depth_chart_order": 2, "depth_chart_position": "RB", "active": True}}
    ctx = _ctx(NoHistory(), pts={"rb1": 20.0, "rb2": 4.0}, players_db=db)
    starter = {**_p("rb1", "RB"), "slot": "RB"}
    bonus, cuff = handcuff_bonus(_p("rb2", "RB"), [starter], ctx)
    assert cuff == "rb1" and bonus == pytest.approx(STARTER_MISS_RATE * HANDCUFF_SHARE * 20 * len(WEEKS), abs=0.01)
    assert handcuff_bonus(_p("rb2", "RB"), [{**starter, "slot": "BN"}], ctx) == (0.0, None)
    assert handcuff_bonus(_p("rb1", "RB"), [starter], ctx) == (0.0, None)  # an RB1 isn't a handcuff
    e = evaluate_add(_p("rb2", "RB"), [starter], ctx, ["RB", "BN"])
    assert e["gain_total"] == pytest.approx(sum(e["weekly_gain"].values()) + e["handcuff_bonus"])


def test_promotion_note_when_starter_ahead_is_out():
    db = {"rb1": {"full_name": "Lead Back", "team": "T1", "depth_chart_order": 1, "depth_chart_position": "RB",
                  "active": True},
          "rb2": {"team": "T1", "depth_chart_order": 2, "depth_chart_position": "RB", "active": True}}
    ctx = _ctx(NoHistory(), pts={}, injuries={"rb1": "IR"}, players_db=db)
    assert ctx.promotion_note(_p("rb2", "RB")) == "Lead Back (ahead on depth chart) is IR"
    assert ctx.promotion_note(_p("rb1", "RB")) is None


def test_team_week_opponents_both_directions():
    sched = pd.DataFrame([{"week": 1, "home_team": "BUF", "away_team": "NYJ"}])
    assert team_week_opponents(sched) == {"BUF": {1: "NYJ"}, "NYJ": {1: "BUF"}}
    assert team_week_opponents(pd.DataFrame()) == {}


# ── roster construction grades ───────────────────────────────────────────────

def _st(ppg):
    return {"found": True, "season": {"ppg": ppg, "games": 4}, "recent": {"ppg": ppg}, "weekly": [ppg] * 4}


@pytest.mark.parametrize("ppg,grade", [(14.0, "A"), (11.5, "B"), (9.0, "C"), (7.0, "D"), (6.9, "F")])
def test_position_grade_thresholds(ppg, grade):
    out = analyze_roster_construction([{"name": "w", "position": "WR"}], [], {"w": _st(ppg)}, {},
                                      replacement_levels={"WR": 10.0}, roster_positions=["WR"])
    assert out["position_grades"]["WR"]["grade"] == grade


def test_weak_or_thin_positions_are_weaknesses_and_targeted():
    roster = [{"name": "qb", "position": "QB"}, {"name": "wr", "position": "WR"}]
    stats = {"qb": _st(25), "wr": _st(5)}
    avail = [{"name": "fa_wr", "position": "WR"}, {"name": "fa_te", "position": "TE"},
             {"name": "fa_k", "position": "K"}]
    out = analyze_roster_construction(roster, avail, stats, {"fa_wr": _st(12), "fa_te": _st(9)},
                                      replacement_levels={"QB": 18, "WR": 10, "TE": 8},
                                      roster_positions=["QB", "WR", "TE"])
    assert set(out["weaknesses"]) == {"WR", "TE"}  # WR graded F; no TE at all
    assert "QB" not in out["weaknesses"]
    names = [p["name"] for p in out["targeted_adds"]]
    assert "fa_k" not in names  # no K slot in this league
    fa_wr = next(p for p in out["targeted_adds"] if p["name"] == "fa_wr")
    assert fa_wr["fills_need"] and fa_wr["need_adjusted_score"] == min(100.0, fa_wr["base_score"] + 20)


def test_targeted_adds_capped_at_twenty():
    avail = [{"name": f"p{i}", "position": "WR"} for i in range(30)]
    out = analyze_roster_construction([], avail, {}, {f"p{i}": _st(10 + i) for i in range(30)},
                                      roster_positions=["WR"])
    assert len(out["targeted_adds"]) == 20
    scores = [p["need_adjusted_score"] for p in out["targeted_adds"]]
    assert scores == sorted(scores, reverse=True)
