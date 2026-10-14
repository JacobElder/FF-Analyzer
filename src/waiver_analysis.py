"""
Full waiver-pool analysis for one league: every top candidate's rest-of-season
lineup gain, best drop, value to you, rival interest, win curve and bid.
"""

from src.evaluator import DEFAULT_ROSTER_POSITIONS
from src.market import (
    bid_profile, market_price_pct, max_bid, recommend_bid, rival_interest, typical_gain, win_curve,
)
from src.valuation import evaluate_add, lineup_totals, roster_limit

TOP_CANDIDATES = 60      # by raw rest-of-season projection, before lineup math
RIVAL_CANDIDATES = 35    # candidates that get the (slower) rival / win-curve analysis


def analyze_waiver_pool(roster: list, waivers: list, ctx, league: dict, players_db: dict,
                        budget: int, roster_positions: list | None = None) -> dict:
    positions = roster_positions or DEFAULT_ROSTER_POSITIONS
    weeks_left = len(ctx.weeks)
    profile = bid_profile(league.get("transactions") or [], league.get("waiver_budget_total") or 100)
    min_bid = int((league.get("waiver_settings") or {}).get("waiver_bid_min") or 0)
    my_pos = (league.get("record") or {}).get("waiver_position")
    rivals = [r for r in league.get("league_rosters") or [] if r["roster_id"] != league.get("roster_id")]

    base = lineup_totals(roster, ctx, positions)
    pool = sorted(waivers, key=ctx.ros_total, reverse=True)[:TOP_CANDIDATES]
    evals = sorted(((w, evaluate_add(w, roster, ctx, positions, base)) for w in pool),
                   key=lambda we: -we[1]["gain_total"])
    typical = typical_gain([e["gain_total"] for _, e in evals], weeks_left)

    candidates = []
    for i, (w, e) in enumerate(evals):
        max_pay = max_bid(e["gain_total"], typical, budget, weeks_left)
        rivals_view, curve = [], []
        if i < RIVAL_CANDIDATES and e["gain_total"] > 0 and rivals:
            rivals_view = rival_interest(w, rivals, ctx, players_db, positions, typical, profile,
                                         w.get("trend_count", 0))
            top_rival = max((r["est_bid"] for r in rivals_view), default=0)
            ceiling = min(budget, max(3 * max_pay, 3 * top_rival, 10))
            market_pct = market_price_pct(profile, e["gain_total"] / typical if typical > 0 else 1.0)
            market_price = None
            if market_pct is not None:
                market_price = market_pct * profile["budget_total"] * (1 + min(0.5, (w.get("trend_count") or 0) / 4000))
            curve = win_curve(rivals_view, my_pos, ceiling, seed_key=str(w.get("player_id") or w["name"]),
                              market_price=market_price)
            rec = recommend_bid(max_pay, curve, min_bid)
        elif e["gain_total"] > 0:
            rec = {"suggested": min(max(min_bid, max_pay // 2), budget), "p_win": None, "likely_needed": None,
                   "contested": None, "note": "Rival interest not modeled for lower-ranked candidates."}
        else:
            rec = {"suggested": 0, "p_win": None, "likely_needed": None, "contested": None,
                   "note": "Wouldn't improve your lineup."}

        candidates.append({
            **w,
            "ros_proj": ctx.ros_total(w),
            "week_values": ctx.week_values(w),
            "gain_total": e["gain_total"],
            "gain_per_week": e["gain_per_week"],
            "weekly_gain": e["weekly_gain"],
            "drop": e["drop"],
            "drop_name": e["drop"]["name"] if e["drop"] else None,
            "drop_options": e["drop_options"],
            "streamer": e["streamer"],
            "market_pct": market_price_pct(profile, e["gain_total"] / typical if typical > 0 else 1.0),
            "handcuff_for": e["handcuff_for"],
            "handcuff_bonus": e["handcuff_bonus"],
            "promotion": ctx.promotion_note(w),
            "return_week": ctx.return_week(w),
            "max_pay": max_pay,
            "suggested_bid": rec["suggested"],
            "p_win": rec["p_win"] if rec["p_win"] is not None else 0.5,
            "p_win_known": rec["p_win"] is not None,
            "likely_needed": rec["likely_needed"],
            "contested": rec["contested"],
            "bid_note": rec["note"],
            "rivals": rivals_view,
            "curve": curve,
            "interested_rivals": sum(1 for r in rivals_view if r["interested"]),
        })

    eligible = [p for p in roster if p.get("slot") not in ("IR", "TAXI")]
    return {
        "candidates": candidates,
        "profile": profile,
        "typical_gain": typical,
        "base_totals": base,
        "open_spots": max(0, roster_limit(positions) - len(eligible)),
        "weeks_left": weeks_left,
    }
