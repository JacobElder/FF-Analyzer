"""Remaining edge branches across src/: empty inputs, fallbacks, and failure paths."""

import json
import sys
import types

import numpy as np
import pandas as pd
import pytest

import src.schedule as schedule
import src.simulator as simulator
import src.sleeper_integration as si
import src.vegas as vegas
import src.wrangler as wrangler
from http_helpers import FakeResponse, _route
from src.agent import FantasyAgent
from src.defense import compute_defense_rankings
from src.evaluator import compute_positional_scarcity, rank_players, starter_counts
from src.market import market_price_pct, rival_interest
from src.players import build_sleeper_name_index, enrich_with_ids
from src.projections import Projections
from src.scoring import compute_fantasy_points
from src.storage import data_path, write_json_atomic
from src.valuation import _finite, lineup_suggestions
from src.waiver_analysis import analyze_waiver_pool
from src.wrangler import StatsIndex
from waiver_helpers import POSITIONS, PTS, ROSTER, WEEKS, _ctx, _p, _tx


def _stats_df(rows):
    df = pd.DataFrame(rows, columns=["player_id", "player_display_name", "position", "team", "week", "fp"])
    df["season"] = 2025
    return df


# ── agent ────────────────────────────────────────────────────────────────────

def test_missing_ai_package_is_reported(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    monkeypatch.setitem(sys.modules, "anthropic", None)  # import raises ImportError
    assert "AI package not installed" in FantasyAgent("claude").analyze("hi")


# ── defense / evaluator ──────────────────────────────────────────────────────

def test_defense_rankings_without_opponents_is_empty():
    df = pd.DataFrame({"position": ["WR"], "opponent_team": [""], "week": [1], "fp": [10.0]})
    assert compute_defense_rankings(df).empty


def test_positional_scarcity_uses_the_next_player_after_starters():
    rows = [(f"wr{i}", f"WR {i}", "WR", "T", w, 20.0 - i) for i in range(5) for w in range(1, 5)]
    idx = StatsIndex(_stats_df(rows))
    repl = compute_positional_scarcity(idx, {"WR": 2, "QB": 1})
    assert repl["WR"] == 18.0      # ppg 20, 19, 18…: with 2 starters, the 3rd-best sets the bar
    assert repl["QB"] == 18.0      # no QB data: static fallback
    assert set(compute_positional_scarcity(StatsIndex(pd.DataFrame()))) == set(starter_counts())


def test_rank_players_single_player_has_no_confidence():
    out = rank_players({"solo": {"season": {"ppg": 10}, "recent": {"ppg": 10}, "weekly": [10] * 4}})
    assert out[0]["recommendation"] == "START" and "confidence_pct" not in out[0]


# ── market ───────────────────────────────────────────────────────────────────

PROFILE = {"claims_won": 10, "win_pct_p25": 0.02, "win_pct_p50": 0.05, "win_pct_p75": 0.1, "win_pct_p90": 0.2,
           "budget_total": 100, "by_roster": {2: {"aggression": 1.5}}}


def test_market_price_beyond_top_knot_is_the_90th_percentile():
    assert market_price_pct(PROFILE, 5.0) == 0.2


def test_rival_bid_anchors_to_league_market_and_aggression():
    ctx = _ctx({"new_te": 14, "r_te": 2})
    rival = {"roster_id": 2, "team_name": "R", "players": ["r_te"], "reserve": [], "faab_remaining": 100,
             "record": None}
    db = {"r_te": {"full_name": "r_te", "position": "TE", "team": "T1"}}
    out = rival_interest(_p("new_te", "TE"), [rival], ctx, db, ["TE"], typical=36, profile=PROFILE)[0]
    assert out["interested"] and out["est_bid"] == round(PROFILE["win_pct_p50"] * 100 * 1.5)
    assert out["waiver_position"] is None


# ── players / projections / scoring ──────────────────────────────────────────

def test_name_index_skips_nameless_players_and_enrich_without_position():
    db = {"1": {"position": "WR"}, "2": {"full_name": "Twin", "position": "WR", "team": "A", "active": True},
          "3": {"full_name": "Twin", "position": "RB", "team": "B", "active": True}}
    index = build_sleeper_name_index(db)
    assert "" not in index
    assert enrich_with_ids([{"name": "Twin", "team": "B"}], db, index)[0]["player_id"] == "3"


def test_projection_lookup_without_player_id(monkeypatch):
    monkeypatch.setattr("src.projections.fetch_week_projections", lambda *a, **k: {"x": {"pts_ppr": 3}})
    p = Projections(2026, [1], {"rec": 1})
    assert p.get(None, 1) is None and p.get("", 1) is None and p.get("x", 1) == 3.0


def test_zero_weight_terms_are_skipped():
    df = pd.DataFrame([{"position": "WR", "receptions": 5}])
    assert compute_fantasy_points(df, {"rec": 0.0, "rec_yd": 0.1}).iloc[0] == 0.0


# ── schedule ─────────────────────────────────────────────────────────────────

def test_fetch_schedules_default_year_and_no_game_type(monkeypatch):
    monkeypatch.setattr(wrangler, "current_nfl_season", lambda: 2026)
    games = pd.DataFrame([{"season": 2026, "week": 1, "home_team": "BUF", "away_team": "NYJ"}])
    real = pd.read_parquet
    monkeypatch.setattr(schedule.pd, "read_parquet",
                        lambda src, *a, **k: games if str(src).startswith("http") else real(src, *a, **k))
    assert len(schedule.fetch_schedules()) == 1
    assert data_path("schedules_2026.parquet").exists()


def test_next_game_without_scores_and_empty_map():
    sched = pd.DataFrame([{"week": 2, "home_team": "BUF", "away_team": "NYJ"}])
    assert schedule.get_next_game("NYJ", sched)["opponent"] == "BUF"
    assert schedule.next_game_map(pd.DataFrame()) == {} and schedule.next_game_map(None) == {}


def test_weather_bad_payload_is_none(monkeypatch):
    from datetime import date, timedelta
    monkeypatch.setattr(schedule.requests, "get", lambda *a, **k: FakeResponse(200, text="not json"))
    assert schedule.get_weather_for_game("BUF", str(date.today() + timedelta(days=1))) is None


def test_playoff_strength_all_byes():
    sched = pd.DataFrame([{"week": 1, "home_team": "BUF", "away_team": "NYJ"}])
    out = schedule.get_playoff_schedule_strength("BUF", sched, {("NYJ", "WR"): {"multiplier": 1}}, "WR", (15, 16))
    assert out["grade"] == "N/A" and [w["opponent"] for w in out["weeks"]] == ["BYE", "BYE"]


# ── simulator ────────────────────────────────────────────────────────────────

def test_gamma_fit_failure_falls_back_to_normal(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("fit failed")
    monkeypatch.setattr(simulator.sp_stats.gamma, "fit", boom)
    s = simulator.simulate_player({"weekly": [10, 12, 14, 16], "season": {"ppg": 13}}, "fail")
    assert s.min() >= 0 and abs(s.mean() - 13) < 1


def test_degenerate_gamma_fit_falls_back_to_normal(monkeypatch):
    monkeypatch.setattr(simulator.sp_stats.gamma, "fit", lambda *a, **k: (0.0, 0, 1.0))
    s = simulator.simulate_player({"weekly": [10, 12, 14, 16]}, "degenerate")
    assert abs(s.mean() - 13) < 1


def test_clearly_worse_trade_is_declined():
    strong = {"weekly": [20, 22, 25, 18, 21, 24], "season": {"ppg": 21.7}}
    weak = {"weekly": [2, 3, 1, 4, 2, 3], "season": {"ppg": 2.5}}
    assert simulator.simulate_trade({"S": strong}, {"W": weak})["verdict"] == "DECLINE"


# ── sleeper ──────────────────────────────────────────────────────────────────

from test_http_integrations import LID, _sleeper_routes  # noqa: E402


def test_sync_without_current_week_or_users(monkeypatch):
    routes = _sleeper_routes(**{"/state/nfl": {"season": "2026", "week": None, "season_type": "off"},
                                f"/league/{LID}/users": FakeResponse(500)})
    _route(monkeypatch, si, si.BASE_URL, routes)
    data, err = si.sync_sleeper_league("me", LID)
    assert err is None and data["matchup"] is None and data["transactions"] == []
    assert data["team_name"] == "me"  # no users list: falls back to the display name


def test_sync_with_unavailable_matchups(monkeypatch):
    _route(monkeypatch, si, si.BASE_URL, _sleeper_routes(**{f"/league/{LID}/matchups/3": FakeResponse(500),
                                                            f"/league/{LID}/matchups/2": []}))
    data, _ = si.sync_sleeper_league("me", LID)
    assert data["matchup"] is None and data["last_result"] is None


def test_sync_reports_league_lookup_failure(monkeypatch):
    _route(monkeypatch, si, si.BASE_URL, _sleeper_routes(**{f"/league/{LID}": FakeResponse(404)}))
    data, err = si.sync_sleeper_league("me", LID)
    assert data is None and err


# ── valuation ────────────────────────────────────────────────────────────────

def test_finite_rejects_non_numbers():
    assert _finite("abc") is None and _finite([1]) is None and _finite("2.5") == 2.5


def test_vegas_fallback_ignores_other_positions_and_uses_depth_cache():
    from src.valuation import ValuationContext
    from waiver_helpers import FlatProjections, NoHistory
    db = {"a": {"team": "T1", "depth_chart_order": 1, "depth_chart_position": "RB", "active": True},
          "b": {"team": "T1", "depth_chart_order": 2, "depth_chart_position": "RB", "active": True},
          "c": {"team": "T1", "depth_chart_order": 3, "depth_chart_position": "RB", "active": True},
          "x": {"team": None, "depth_chart_order": 1, "active": True}}
    opponents = {"T1": {w: "OPP" for w in WEEKS}}
    ctx = ValuationContext(NoHistory(), FlatProjections({}), opponents, {},
                           {"by_id": {"a": "Questionable"}, "by_name": {}}, WEEKS, vegas={"OPP": 20}, players_db=db)
    assert ctx.week_values(_p("wr", "WR"))[6] == 0.0     # Vegas fallback is DEF/K only
    assert ctx.promotion_note(_p("c", "RB")) is None      # starter only Questionable: no promotion
    assert ctx.promotion_note(_p("b", "RB")) is None      # cached depth chart, loop breaks at own order


def test_lineup_suggestions_higher_projection_and_no_eligible_swap():
    ctx = _ctx({**PTS, "wr2": 16})
    out = lineup_suggestions(ROSTER, ctx, POSITIONS)
    assert [(s["out"], s["in"], s["reason"]) for s in out["swaps"]] == [("wr", "wr2", "Higher projection")]
    te_only = [_p("te", "TE", slot="TE"), _p("qb", "QB")]  # better QB on bench can't play TE
    ctx2 = _ctx({"te": 2, "qb": 30})
    assert lineup_suggestions(te_only, ctx2, ["TE", "BN"])["swaps"] == []


# ── vegas ────────────────────────────────────────────────────────────────────

def test_vegas_corrupt_stale_cache_is_ignored(monkeypatch, isolated):
    monkeypatch.setenv("ODDS_API_KEY", "k")
    path = data_path("vegas_totals.json")
    path.write_text("{broken")
    monkeypatch.setattr(vegas.requests, "get", lambda *a, **k: FakeResponse(503))
    assert vegas.fetch_implied_totals(force=True) == {}


def test_vegas_partial_markets_are_skipped():
    def game(markets, home="Buffalo Bills"):
        return {"home_team": home, "away_team": "Miami Dolphins", "commence_time": "2026-10-01",
                "bookmakers": [{"markets": markets}]}
    noisy = [{"key": "h2h", "outcomes": []},
             {"key": "totals", "outcomes": [{"name": "Under", "point": 40}, {"name": "Over", "point": 44}]},
             {"key": "spreads", "outcomes": [{"name": "Miami Dolphins", "point": 3},
                                             {"name": "Buffalo Bills", "point": -3}]}]
    assert vegas.parse_implied_totals([game(noisy)]) == {"BUF": 23.5, "MIA": 20.5}
    assert vegas.parse_implied_totals([{**game([]), "bookmakers": []}]) == {}
    totals_only = [{"key": "totals", "outcomes": [{"name": "Over", "point": 44}]}]
    assert vegas.parse_implied_totals([game(totals_only)]) == {}


# ── waiver pipeline ──────────────────────────────────────────────────────────

def test_pipeline_uses_league_market_history_and_respects_min_bid():
    history = [_tx(1, 2, b, adds=(f"p{b}",)) for b in (5, 8, 10, 12, 20, 25)]
    league = {"roster_id": 1, "record": {"waiver_position": 2}, "waiver_budget_total": 100,
              "waiver_settings": {"waiver_bid_min": 5}, "transactions": history,
              "league_rosters": [{"roster_id": 2, "team_name": "R", "players": ["r_te"], "reserve": [],
                                  "faab_remaining": 90, "record": {"waiver_position": 1}}]}
    db = {"r_te": {"full_name": "r_te", "position": "TE", "team": "T1"}}
    ctx = _ctx({**PTS, "fa_te": 14, "r_te": 2})
    with_market = analyze_waiver_pool(ROSTER, [_p("fa_te", "TE")], ctx, league, db, 50, POSITIONS)
    c = with_market["candidates"][0]
    assert with_market["profile"]["claims_won"] == 6 and c["market_pct"] is not None and c["p_win_known"]
    broke = analyze_waiver_pool(ROSTER, [_p("fa_te", "TE")], ctx, {**league, "league_rosters": []}, db, 3,
                                POSITIONS)
    assert broke["candidates"][0]["suggested_bid"] == 0
    assert "minimum bid" in broke["candidates"][0]["bid_note"]


# ── wrangler ─────────────────────────────────────────────────────────────────

def test_fetch_weekly_stats_default_year_without_fallback(monkeypatch):
    monkeypatch.setattr(wrangler, "current_nfl_season", lambda: 2026)
    seen = []

    def load(year, force):
        seen.append(year)
        return _stats_df([("00-1", "A", "WR", "NYJ", 1, 5.0)])
    monkeypatch.setattr(wrangler, "_load_season", load)
    assert len(wrangler.fetch_weekly_stats()) == 1 and seen == [2026]


def test_empty_index_and_team_disambiguation():
    empty = StatsIndex(pd.DataFrame())
    assert empty.season is None and empty.last_week == 0 and not empty.get_player_stats("x")["found"]
    idx = StatsIndex(_stats_df([("00-1", "Mike Williams", "WR", "PIT", 1, 9.0),
                                ("00-2", "Mike Williams", "WR", "NYJ", 1, 4.0)]))
    assert idx.get_player_stats({"name": "Mike Williams", "position": "WR", "team": "NYJ"})["season"]["ppg"] == 4.0


def test_usage_trends_without_carries_or_recent_games():
    df = _stats_df([("00-1", "A", "WR", "NYJ", w, 5.0) for w in (1, 2)])
    df["target_share"] = [0.1, 0.3]
    no_carries = StatsIndex(df).get_usage_trends("A")
    assert "recent_carries_pg" not in no_carries
    df["carries"] = [1, 2]
    none_recent = StatsIndex(df).get_usage_trends("A", recent_weeks=0)
    assert "recent_carries_pg" not in none_recent and none_recent["target_share"]["recent_avg"] == 0.2


def test_rank_available_players_attaches_stats():
    idx = StatsIndex(_stats_df([("00-1", "A Guy", "WR", "NYJ", 1, 6.0), ("00-1", "A Guy", "WR", "NYJ", 2, 8.0)]))
    out = idx.rank_available_players([{"name": "A Guy", "position": "WR"}, {"name": "Nobody"}])
    assert out[0]["season_ppg"] == 7.0 and out[0]["games"] == 2 and out[0]["stats_found"]
    assert not out[1]["stats_found"] and out[1]["season_ppg"] == 0.0


def test_swap_skipped_when_the_better_player_cant_fill_that_slot():
    # wr1 should sit, but the player who should start (an RB) can't play wr1's WR slot
    roster = [_p("wr1", "WR", slot="WR"), _p("wr2", "WR", slot="FLEX"), _p("rb", "RB")]
    out = lineup_suggestions(roster, _ctx({"wr1": 5, "wr2": 8, "rb": 20}), ["WR", "FLEX", "BN"])
    assert out["swaps"] == [] and out["optimal_total"] == 28


def test_no_promotion_when_everyone_ahead_is_healthy():
    from src.valuation import ValuationContext
    from waiver_helpers import FlatProjections, NoHistory
    db = {"a": {"team": "T1", "depth_chart_order": 1, "depth_chart_position": "RB", "active": True},
          "c": {"team": "T1", "depth_chart_order": 3, "depth_chart_position": "RB", "active": True},
          "d": {"team": "T1", "depth_chart_order": 4, "depth_chart_position": "RB", "active": False}}
    ctx = ValuationContext(NoHistory(), FlatProjections({}), {}, {}, {"by_id": {}, "by_name": {}}, WEEKS,
                           players_db=db)
    assert ctx.promotion_note(_p("c", "RB")) is None  # loop stops at his own spot
    assert ctx.promotion_note(_p("d", "RB")) is None  # inactive, so not on the chart: loop runs out


def test_backtest_skips_unmatched_players_unplayed_weeks_and_thin_history(monkeypatch):
    import src.backtest as backtest
    rows = [("g1", "Gap Guy", "WR", w, 0, 0, 5, 50, 0) for w in (1, 2, 3, 6, 7)]   # missed weeks 4-5
    rows += [("g2", "Late Start", "WR", w, 0, 0, 3, 30, 0) for w in (5, 6, 7)]     # <2 prior games at week 5-6
    df = pd.DataFrame(rows, columns=["player_id", "player_display_name", "position", "week", "rushing_yards",
                                     "rushing_tds", "receptions", "receiving_yards", "receiving_tds"])
    df["season"] = 2025
    monkeypatch.setattr(backtest, "fetch_weekly_stats", lambda season: df)
    monkeypatch.setattr(backtest, "fetch_all_players", lambda: {
        "s9": {"full_name": "Unmatched Person", "position": "WR", "team": "NYJ", "active": True}})

    class NoProj:
        def __init__(self, *a, **k):
            pass

        def get(self, sid, week):
            return None
    monkeypatch.setattr(backtest, "Projections", NoProj)
    results = backtest.run_backtest(2025, first_week=4, last_week=7)
    n = next(cnt for (kind, params), _, cnt in results["next_week"] if params == backtest.BLEND_HISTORY)
    assert n == 2 + 1  # g1: weeks 6-7 (4-5 unplayed); g2: week 7 only (weeks 5-6 lack 2 prior games)
    assert {k[0] for k, _, _ in results["next_week"]} == {"history"}  # no projections → no blends
