"""
Full waiver-pool analysis for one league: every top candidate's rest-of-season
lineup gain, best drop, value to you, rival interest, win curve and bid.
"""

from src.evaluator import DEFAULT_ROSTER_POSITIONS
from src.market import (
    bid_profile,
    market_price_pct,
    max_bid,
    plan_claims,
    recommend_bid,
    rival_interest,
    rivals_ahead,
    typical_gain,
    win_curve,
)
from src.valuation import evaluate_add, lineup_totals, roster_limit

TOP_CANDIDATES = 60      # by raw rest-of-season projection, before lineup math
RIVAL_CANDIDATES = 35    # candidates that get the (slower) rival / win-curve analysis


def faab_enabled(league: dict) -> bool:
    """FAAB bidding is on: Sleeper's waiver_type is FAAB and the season budget isn't $0."""
    return bool(league.get("faab_enabled", True)) and league.get("waiver_budget_total") != 0


def analyze_waiver_pool(roster: list, waivers: list, ctx, league: dict, players_db: dict,
                        budget: int, roster_positions: list | None = None, weights: dict | None = None,
                        rival_weights: dict | None = None) -> dict:
    """
    `weights` ({week: multiplier}) values your gains by how much each week matters to you
    (see season.week_weights); `rival_weights` does the same per rival roster_id.

    Without FAAB (league["faab_enabled"] False or a $0 season budget) claims go by
    waiver priority: every bid is $0 and P(win) is the chance no interested rival
    ahead of you in the order claims him. The result's top-level "faab" (and each
    candidate's "faab") is then False so the UI can hide the $ columns.
    """
    positions = roster_positions or DEFAULT_ROSTER_POSITIONS
    weeks_left = len(ctx.weeks)
    faab = faab_enabled(league)
    profile = bid_profile(league.get("transactions") or [], league.get("waiver_budget_total") or 100)
    min_bid = int((league.get("waiver_settings") or {}).get("waiver_bid_min") or 0) if faab else 0
    my_pos = (league.get("record") or {}).get("waiver_position")
    rivals = [r for r in league.get("league_rosters") or [] if r["roster_id"] != league.get("roster_id")]

    def weight_sum(w):
        return sum((w or {}).get(wk, 1.0) for wk in ctx.weeks)

    base = lineup_totals(roster, ctx, positions)
    pool = sorted(waivers, key=ctx.ros_total, reverse=True)[:TOP_CANDIDATES]
    evals = sorted(((w, evaluate_add(w, roster, ctx, positions, base, weights=weights)) for w in pool),
                   key=lambda we: -we[1]["gain_total"])
    my_weight_sum = weight_sum(weights)
    typical = typical_gain([e["gain_total"] for _, e in evals], weeks_left, my_weight_sum)
    # Each rival's typical add, valued with its own week weights (same scale as its gains)
    rival_typicals = ({rid: typical * weight_sum(rw) / my_weight_sum for rid, rw in rival_weights.items()}
                      if rival_weights and my_weight_sum > 0 else None)

    candidates = []
    for i, (w, e) in enumerate(evals):
        max_pay = max_bid(e["gain_total"], typical, budget, weeks_left) if faab else 0
        rivals_view, curve = [], []
        modeled = i < RIVAL_CANDIDATES and rivals
        if modeled and e["gain_total"] > 0:
            rivals_view = rival_interest(w, rivals, ctx, players_db, positions, typical, profile,
                                         w.get("trend_count", 0), rival_weights=rival_weights,
                                         rival_typicals=rival_typicals, faab=faab)
        if e["gain_total"] <= 0:
            rec = {"suggested": 0, "p_win": None, "likely_needed": None, "contested": None,
                   "note": "Wouldn't improve your lineup."}
        elif not faab and (modeled or not rivals):
            curve = win_curve(rivals_view, my_pos, 0, faab=False)
            p = curve[0][1]
            ahead = rivals_ahead(rivals_view, my_pos)
            rec = {"suggested": 0, "p_win": p, "likely_needed": None, "contested": p < 1,
                   "note": (f"No FAAB: claims go by waiver priority — {ahead} interested rival(s) could claim first."
                            if p < 1 else "No FAAB, and no interested rival ahead of you in waiver priority.")}
        elif not faab:
            rec = {"suggested": 0, "p_win": None, "likely_needed": None, "contested": None,
                   "note": "No FAAB; rival interest not modeled for lower-ranked candidates."}
        elif not rivals:
            rec = recommend_bid(max_pay, [(0, 1.0)], min_bid, budget)  # nobody to compete with
        elif modeled:
            top_rival = max((r["est_bid"] for r in rivals_view), default=0)
            ceiling = min(budget, max(3 * max_pay, 3 * top_rival, 10))
            market_pct = market_price_pct(profile, e["gain_total"] / typical if typical > 0 else 1.0)
            market_price = None
            if market_pct is not None:
                market_price = market_pct * profile["budget_total"] * (1 + min(0.5, (w.get("trend_count") or 0) / 4000))
            curve = win_curve(rivals_view, my_pos, ceiling, seed_key=str(w.get("player_id") or w["name"]),
                              market_price=market_price)
            rec = recommend_bid(max_pay, curve, min_bid, budget)
        elif budget < min_bid:
            rec = {"suggested": 0, "p_win": None, "likely_needed": None, "contested": None,
                   "note": "Not enough FAAB left for the league's minimum bid.", "excluded": True}
        else:
            rec = {"suggested": min(max(min_bid, max_pay // 2), budget), "p_win": None, "likely_needed": None,
                   "contested": None, "note": "Rival interest not modeled for lower-ranked candidates."}

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
            "min_bid": min_bid,
            "suggested_bid": rec["suggested"],
            "p_win": rec["p_win"] if rec["p_win"] is not None else 0.5,
            "p_win_known": rec["p_win"] is not None,
            "likely_needed": rec["likely_needed"],
            "contested": rec["contested"],
            "bid_note": rec["note"],
            "rivals": rivals_view,
            "curve": curve,
            "interested_rivals": sum(1 for r in rivals_view if r["interested"]),
            "below_value": rec.get("below_value", rec["suggested"] > max_pay),
            "excluded": rec.get("excluded", False),
            "faab": faab,
        })

    eligible = [p for p in roster if p.get("slot") not in ("IR", "TAXI")]
    return {
        "candidates": candidates,
        "profile": profile,
        "typical_gain": typical,
        "base_totals": base,
        "open_spots": max(0, roster_limit(positions) - len(eligible)),
        "weeks_left": weeks_left,
        "faab": faab,
    }


def build_claim_plan(analysis: dict, roster: list, ctx, budget: int, roster_positions: list | None = None,
                     max_claims: int = 5, weights: dict | None = None) -> list:
    """
    This week's claims in priority order. Each later claim is re-valued against
    the roster as it would be if the earlier claims (and their drops) went through.
    """
    positions = roster_positions or DEFAULT_ROSTER_POSITIONS

    def reevaluate(c, plan_so_far):
        dropped = {p["drop_name"] for p in plan_so_far if p["drop_name"]}
        added = [{**p, "slot": "BN"} for p in plan_so_far]
        new_roster = [p for p in roster if p["name"] not in dropped] + added
        return evaluate_add(c, new_roster, ctx, positions, weights=weights)["drop_options"]

    return plan_claims(analysis["candidates"], budget, analysis["open_spots"], max_claims, reevaluate=reevaluate)
