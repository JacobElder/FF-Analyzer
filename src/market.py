"""
FAAB market: what your league pays, who you're bidding against, and what to bid.

Bid logic for a waiver target:
  max_pay       — what the player is worth to YOU: his rest-of-season lineup gain
                  relative to a typical add, scaled so your budget lasts the
                  season (FAAB is worthless after the last waiver run).
  rival bids    — every other team's lineup gain from the same player, priced
                  with the same formula on THEIR budget, scaled by how
                  aggressively that manager has bid this season.
  win curve     — Monte Carlo over rival bids (each interested rival may or may
                  not bid, with noise) → P(win) at each bid you could make.
  suggestion    — the cheapest bid with P(win) ≥ TARGET_WIN_PROB, never above
                  max_pay; $min when nobody else wants him.

The constants below are judgment calls, not fitted values — the app shows
the inputs so you can override the suggestion.
"""

import math
import zlib
from collections import defaultdict
from statistics import median

import numpy as np

from src.players import sleeper_to_player
from src.valuation import evaluate_add, lineup_totals

TARGET_WIN_PROB = 0.75
RIVAL_BID_PROB = 0.65        # chance an interested rival actually submits a claim
RIVAL_BID_NOISE = 0.5        # lognormal sigma around the estimated competing bid
AGGRESSION_PSEUDO_CLAIMS = 4 # shrink each manager's bidding style toward the league average
AGGRESSION_WEIGHT = 0.5      # how much the top interested rival's aggression moves the market price
MIN_INTEREST_PER_WEEK = 0.5  # rival must gain ≥ this many lineup pts/week to care
CLAIMS_PER_TWO_WEEKS = 1     # pacing: assume one meaningful add every two weeks
CLAIMED_NOTE = "claimed by another owner"


# ── League bidding history ───────────────────────────────────────────────────

def bid_profile(transactions: list, budget_total: float) -> dict:
    """Summary of this season's FAAB market from Sleeper's transaction log."""
    budget_total = budget_total or 100
    wins = [t for t in transactions if t["type"] == "waiver" and t["status"] == "complete" and t.get("bid") is not None]
    outbid = [t for t in transactions if t["type"] == "waiver" and t["status"] == "failed"
              and CLAIMED_NOTE in (t.get("note") or "")]

    win_pcts = [t["bid"] / budget_total for t in wins]
    paid = [t["bid"] for t in wins if t["bid"] > 0]

    # Contested claims: winner vs. best losing bid for the same player and week
    losing = defaultdict(list)
    for t in outbid:
        for pid in t["adds"]:
            losing[(t["week"], pid)].append(t.get("bid") or 0)
    margins, contested = [], 0
    for t in wins:
        for pid in t["adds"]:
            if (t["week"], pid) in losing:
                contested += 1
                margins.append(t["bid"] - max(losing[(t["week"], pid)]))

    league_avg = float(np.mean(win_pcts)) if win_pcts else 0.0
    by_roster = {}
    grouped = defaultdict(list)
    for t in wins:
        grouped[t["roster_id"]].append(t["bid"] / budget_total)
    for rid, pcts in grouped.items():
        n = len(pcts)
        # Shrink toward the league average so one big bid doesn't define a manager
        avg = (sum(pcts) + AGGRESSION_PSEUDO_CLAIMS * league_avg) / (n + AGGRESSION_PSEUDO_CLAIMS)
        by_roster[rid] = {"wins": n, "avg_pct": round(avg, 4), "spent": round(sum(pcts) * budget_total),
                          "aggression": round(avg / league_avg, 2) if league_avg > 0 else 1.0}

    def pct(q):
        return round(float(np.percentile(win_pcts, q)), 4) if win_pcts else None

    return {
        "budget_total": budget_total,
        "win_pct_p25": pct(25),
        "claims_won": len(wins),
        "claims_outbid": len(outbid),
        "contested": contested,
        "zero_dollar_share": round(sum(1 for t in wins if t["bid"] == 0) / len(wins), 2) if wins else None,
        "median_paid": median(paid) if paid else 0,
        "win_pct_p50": pct(50), "win_pct_p75": pct(75), "win_pct_p90": pct(90),
        "median_overbid": median(margins) if margins else None,
        "league_avg_pct": round(league_avg, 4),
        "by_roster": by_roster,
    }


# ── Valuing a bid ────────────────────────────────────────────────────────────

def typical_gain(gains: list, n_weeks: int) -> float:
    """Rest-of-season gain of a 'typical good add': median of the 10 best positive gains."""
    top = sorted((g for g in gains if g > 0), reverse=True)[:10]
    floor = 0.5 * max(n_weeks, 1)
    return max(median(top), floor) if top else floor


def max_bid(gain_total: float, typical: float, budget: float, weeks_left: int) -> int:
    """
    Most a player is worth to a team: its budget spread over the claims it can
    still make (one every two weeks), scaled by how this add compares with a
    typical one. Late in the season fewer claims remain, so each is worth more.
    """
    if gain_total <= 0 or budget <= 0:
        return 0
    future_claims = max(1.0, weeks_left / 2 * CLAIMS_PER_TWO_WEEKS)
    share = gain_total / (typical * future_claims)
    return int(min(budget, round(budget * share)))


MIN_HISTORY_CLAIMS = 5  # below this, fall back to the formula instead of the league's market


def market_price_pct(profile: dict, ratio: float) -> float | None:
    """
    Share of the season budget this league has historically paid for an add
    `ratio` times as valuable as a typical one: ≤0.5× → 25th percentile of
    winning bids, 1× → median, 2× → 75th, ≥3× → 90th (linear in between).
    """
    if profile.get("claims_won", 0) < MIN_HISTORY_CLAIMS or profile.get("win_pct_p50") is None:
        return None
    knots = [(0.5, profile["win_pct_p25"]), (1.0, profile["win_pct_p50"]),
             (2.0, profile["win_pct_p75"]), (3.0, profile["win_pct_p90"])]
    if ratio <= knots[0][0]:
        return knots[0][1]
    for (r0, p0), (r1, p1) in zip(knots, knots[1:]):
        if ratio <= r1:
            return p0 + (p1 - p0) * (ratio - r0) / (r1 - r0)
    return knots[-1][1]


def rival_interest(add: dict, rivals: list, ctx, players_db: dict, roster_positions: list,
                   typical: float, profile: dict, trend_count: int = 0) -> list:
    """Each rival's lineup gain from `add` and an estimated bid."""
    weeks_left = len(ctx.weeks)
    trend_mult = 1 + min(0.5, (trend_count or 0) / 4000)  # league-wide add frenzy → bids run hotter
    out = []
    for r in rivals:
        roster = r.get("_roster")
        if roster is None:
            reserve = set(r.get("reserve") or [])
            roster = [{**sleeper_to_player(pid, players_db.get(pid, {})), "slot": "IR" if pid in reserve else "BN"}
                      for pid in r.get("players") or []]
            r["_roster"] = roster
            r["_base"] = lineup_totals(roster, ctx, roster_positions)
        ev = evaluate_add(add, roster, ctx, roster_positions, base_totals=r["_base"], max_drop_candidates=4)
        per_week = ev["gain_total"] / max(weeks_left, 1)
        aggression = profile["by_roster"].get(r["roster_id"], {}).get("aggression", 1.0)
        est = 0
        if per_week >= MIN_INTEREST_PER_WEEK:
            market = market_price_pct(profile, ev["gain_total"] / typical if typical > 0 else 1.0)
            if market is not None:
                # Anchor to what this league actually pays for an add of this size
                raw = market * profile["budget_total"]
            else:
                raw = max_bid(ev["gain_total"], typical, r["faab_remaining"], weeks_left)
            est = min(r["faab_remaining"], round(raw * aggression * trend_mult))
        out.append({
            "roster_id": r["roster_id"], "team": r["team_name"],
            "gain_per_week": round(per_week, 2), "budget": r["faab_remaining"],
            "waiver_position": (r.get("record") or {}).get("waiver_position"),
            "aggression": aggression, "interested": per_week >= MIN_INTEREST_PER_WEEK, "est_bid": int(est),
        })
    return sorted(out, key=lambda x: -x["est_bid"])


def win_curve(rivals: list, my_waiver_position, max_level: int, seed_key: str = "",
              market_price: float | None = None, n_sims: int = 3000) -> list:
    """
    [(bid, P(win))] for bids 0..max_level. FAAB ties go to the better waiver position.

    With league history (market_price = the league's typical WINNING bid for an
    add of this size), the top competing bid is simulated directly:
    contested with probability 1 − (1 − RIVAL_BID_PROB)^interested, and then
    market_price × (partly) the most aggressive interested rival's aggression × noise,
    capped at the richest interested rival's budget. Without history, each
    interested rival's own estimated bid is simulated and the highest one wins.
    """
    max_level = max(0, int(max_level))
    bidders = [r for r in rivals if r["interested"] and r["budget"] > 0]
    if not bidders:
        return [(b, 1.0) for b in range(max_level + 1)]
    rng = np.random.default_rng(zlib.crc32(seed_key.encode()))

    if market_price is not None:
        p_contested = 1 - (1 - RIVAL_BID_PROB) ** len(bidders)
        aggression = 1 + (max(r["aggression"] for r in bidders) - 1) * AGGRESSION_WEIGHT
        cap = max(r["budget"] for r in bidders)
        amount = np.round(max(market_price * aggression, 0.5) * rng.lognormal(0, RIVAL_BID_NOISE, n_sims))
        best_bid = np.where(rng.random(n_sims) < p_contested, np.minimum(amount, cap), -1.0)
        best_pos = np.full(n_sims, float(min((r.get("waiver_position") or 99) for r in bidders)))
    else:
        best_bid = np.full(n_sims, -1.0)
        best_pos = np.full(n_sims, 99.0)
        for r in bidders:
            bids_now = rng.random(n_sims) < RIVAL_BID_PROB
            amount = np.round(np.maximum(r["est_bid"], 0.5) * rng.lognormal(0, RIVAL_BID_NOISE, n_sims))
            amount = np.where(bids_now, np.minimum(amount, r["budget"]), -1)
            pos = r.get("waiver_position") or 99
            better = (amount > best_bid) | ((amount == best_bid) & (pos < best_pos))
            best_bid = np.where(better, amount, best_bid)
            best_pos = np.where(better, pos, best_pos)

    me = my_waiver_position or 99
    curve = []
    for b in range(max_level + 1):
        win = (b > best_bid) | ((b == best_bid) & (me < best_pos))
        curve.append((b, round(float(np.mean(win)), 3)))
    return curve


def recommend_bid(max_pay: int, curve: list, min_bid: int = 0) -> dict:
    """Cheapest bid reaching TARGET_WIN_PROB, capped at max_pay."""
    p_at = dict(curve)
    contested = any(p < 1.0 for _, p in curve)
    if not contested:
        return {"suggested": min_bid, "p_win": 1.0, "likely_needed": min_bid, "contested": False,
                "note": "No other team gains much from him — a minimum bid should do."}
    likely = next((b for b, p in curve if p >= 0.5), None)
    target = next((b for b, p in curve if p >= TARGET_WIN_PROB), None)
    if target is not None and target <= max_pay:
        bid, note = max(target, min_bid), f"Cheapest bid with a {TARGET_WIN_PROB:.0%}+ chance to win."
    else:
        bid = max(min(max_pay, curve[-1][0]), min_bid)
        note = ("Winning likely costs more than he's worth to you — bid your max only if you want a shot."
                if max_pay > 0 else "Not worth a bid for your roster.")
    return {"suggested": int(bid), "p_win": p_at.get(bid, 0.0), "likely_needed": likely,
            "contested": True, "note": note}


# ── Claims planner ───────────────────────────────────────────────────────────

def plan_claims(candidates: list, budget: int, open_spots: int = 0, max_claims: int = 5,
                reevaluate=None, pool_size: int = 25) -> list:
    """
    Choose this week's claims greedily by expected gain (P(win) × lineup gain),
    each with its own drop (or an open roster spot), total bids within budget.

    Sleeper processes every claim you submit, so later claims are valued
    assuming the earlier ones succeed: `reevaluate(candidate, plan_so_far)`
    returns the candidate's drop options against the roster after those
    claims. Without it, each claim's standalone gain is used.
    """
    pool = sorted((c for c in candidates if c["gain_total"] > 0 and c["suggested_bid"] <= budget),
                  key=lambda c: -c["p_win"] * c["gain_total"])[:pool_size]
    plan, used_drops, spent = [], set(), 0

    def best_option(c):
        options = (reevaluate(c, plan) if (reevaluate and plan) else None) or c.get("drop_options") \
            or [{"name": c.get("drop_name"), "gain_total": c["gain_total"]}]
        planned_adds = {p["name"] for p in plan}
        return next((o for o in options
                     if (o["name"] is None and open_spots > 0)
                     or (o["name"] and o["name"] not in used_drops and o["name"] not in planned_adds)), None)

    while pool and len(plan) < max_claims:
        scored = []
        for c in pool:
            if spent + c["suggested_bid"] > budget:
                continue
            o = best_option(c)
            if o and o["gain_total"] > 0:
                scored.append((c["p_win"] * o["gain_total"], c, o))
        if not scored:
            break
        ev, c, o = max(scored, key=lambda x: x[0])
        pool.remove(c)
        plan.append({**c, "drop_name": o["name"], "gain_total": o["gain_total"],
                     "priority": len(plan) + 1, "expected_gain": round(ev, 1)})
        spent += c["suggested_bid"]
        if o["name"] is None:
            open_spots -= 1
        else:
            used_drops.add(o["name"])
    return plan


WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def waiver_day(waiver_settings: dict) -> str | None:
    """Day waivers process (Sleeper's waiver_day_of_week, 0 = Monday)."""
    d = (waiver_settings or {}).get("waiver_day_of_week")
    return WEEKDAYS[d] if isinstance(d, int) and 0 <= d < 7 else None


# ── Your FAAB log ────────────────────────────────────────────────────────────

def faab_log(transactions: list, my_roster_id, index, players_db: dict) -> list:
    """Your completed adds this season with cost and points scored since."""
    rows = []
    for t in transactions:
        if t["roster_id"] != my_roster_id or t["status"] != "complete":
            continue
        for pid in t["adds"]:
            p = sleeper_to_player(pid, players_db.get(pid, {}))
            s = index.get_player_stats(p)
            pts = (round(sum(pt for pt, w in zip(s.get("weekly", []), s.get("weeks", [])) if w >= t["week"]), 1)
                   if s["found"] else None)  # None: no stats source (team DEF) or no games
            cost = t.get("bid") or 0
            rows.append({"Week": t["week"], "Player": p["name"], "Pos": p["position"],
                         "Type": "Waiver" if t["type"] == "waiver" else "Free agent",
                         "Cost": cost, "Pts since added": pts,
                         "$ per pt": round(cost / pts, 2) if pts and cost else None})
    return sorted(rows, key=lambda r: r["Week"])


# ── Standings and this week's matchup ────────────────────────────────────────

def standings(league_rosters: list) -> list:
    rows = sorted(league_rosters, key=lambda r: (-(r["record"]["wins"] + 0.5 * r["record"].get("ties", 0)),
                                                 -r["record"]["points_for"]))
    return [{"rank": i + 1, **r} for i, r in enumerate(rows)]


def playoff_position(league_rosters: list, my_roster_id, playoff_teams: int | None) -> dict | None:
    if not league_rosters or not playoff_teams:
        return None
    table = standings(league_rosters)
    me = next((r for r in table if r["roster_id"] == my_roster_id), None)
    if not me:
        return None
    wins = lambda r: r["record"]["wins"] + 0.5 * r["record"].get("ties", 0)
    cutoff = table[playoff_teams - 1] if len(table) >= playoff_teams else table[-1]
    first_out = table[playoff_teams] if len(table) > playoff_teams else None
    if me["rank"] <= playoff_teams:
        margin = wins(me) - wins(first_out) if first_out else None
    else:
        margin = wins(me) - wins(cutoff)
    return {"rank": me["rank"], "teams": len(table), "playoff_teams": playoff_teams,
            "in_position": me["rank"] <= playoff_teams, "games_margin": margin}


def matchup_projection(my_starters: list, opp_starters: list, ctx, players_db: dict) -> dict | None:
    """Projected points for both lineups this week and a normal-approximation win probability."""
    week = ctx.current_week
    if week is None or not my_starters or not opp_starters:
        return None

    def total(pids):
        vals = [ctx.week_values(sleeper_to_player(pid, players_db.get(pid, {})))[week] for pid in pids]
        # Weekly fantasy scores have a coefficient of variation around 0.6
        return sum(vals), sum((0.6 * v) ** 2 for v in vals)

    mine, var_m = total(my_starters)
    theirs, var_t = total(opp_starters)
    sd = math.sqrt(var_m + var_t) or 1.0
    p_win = 0.5 * (1 + math.erf((mine - theirs) / (sd * math.sqrt(2))))
    return {"mine": round(mine, 1), "theirs": round(theirs, 1), "p_win": round(p_win, 3)}


def risk_mode(p_win: float | None) -> str:
    """Favorites should cut variance, underdogs should chase it."""
    if p_win is None:
        return "Balanced"
    if p_win >= 0.65:
        return "Safe"
    if p_win <= 0.35:
        return "Upside"
    return "Balanced"
