import statistics
from collections import Counter

# Static fallback replacement levels (PPG)
POSITION_REPLACEMENT = {
    "QB": 18.0, "RB": 10.0, "WR": 10.0,
    "TE": 8.0,  "K": 7.0,   "DEF": 7.0,
}

DEFAULT_ROSTER_POSITIONS = ["QB", "RB", "RB", "WR", "WR", "TE", "FLEX", "K", "DEF"]

# Which player positions may fill each Sleeper lineup slot
SLOT_ELIGIBILITY = {
    "QB": {"QB"}, "RB": {"RB"}, "WR": {"WR"}, "TE": {"TE"}, "K": {"K"}, "DEF": {"DEF"},
    "FLEX": {"RB", "WR", "TE"},
    "WRRB_FLEX": {"RB", "WR"},
    "REC_FLEX": {"WR", "TE"},
    "SUPER_FLEX": {"QB", "RB", "WR", "TE"},
}

# How flex slots are typically filled league-wide (for replacement levels)
_FLEX_SHARES = {
    "FLEX": {"RB": 0.45, "WR": 0.45, "TE": 0.10},
    "WRRB_FLEX": {"RB": 0.5, "WR": 0.5},
    "REC_FLEX": {"WR": 0.8, "TE": 0.2},
    "SUPER_FLEX": {"QB": 0.8, "RB": 0.1, "WR": 0.1},
}

# Rostered depth beyond starters before a player is "freely available"
_DEPTH_FACTOR = 1.25

# Multiplier on projected points by injury designation
INJURY_FACTORS = {
    "Questionable": 0.85, "Doubtful": 0.25,
    "Out": 0.0, "IR": 0.0, "PUP": 0.0, "Sus": 0.0, "COV": 0.0, "DNR": 0.0, "NFI": 0.0,
}


def injury_factor(status: str | None) -> float:
    return INJURY_FACTORS.get(status, 1.0) if status else 1.0


def starter_counts(num_teams: int = 12, roster_positions: list | None = None) -> dict:
    """
    League-wide number of rostered players per position, derived from the
    league's lineup slots and team count. The replacement player at a position
    is the next one after this many.
    """
    slots = Counter(s for s in (roster_positions or DEFAULT_ROSTER_POSITIONS) if s in SLOT_ELIGIBILITY)
    per_team = {pos: float(slots.get(pos, 0)) for pos in POSITION_REPLACEMENT}
    for slot, shares in _FLEX_SHARES.items():
        for pos, share in shares.items():
            per_team[pos] += slots.get(slot, 0) * share
    return {pos: max(1, round(num_teams * n * _DEPTH_FACTOR)) for pos, n in per_team.items()}


def compute_positional_scarcity(stats_index, counts: dict | None = None) -> dict:
    """
    Dynamic replacement-level PPG per position: the PPG of the (N+1)th-ranked
    player, where N comes from starter_counts(). Players need a minimum number
    of games so one-week flukes don't set the bar. Falls back to
    POSITION_REPLACEMENT where data is insufficient (e.g. DEF).
    """
    counts = counts or starter_counts()
    min_games = min(3, max(1, (stats_index.last_week or 0) // 2))
    table = stats_index.season_ppg_table(min_games=min_games)

    result = {}
    for pos, n in counts.items():
        ppg = table.loc[table["position"] == pos, "ppg"].sort_values(ascending=False) if not table.empty else []
        result[pos] = round(float(ppg.iloc[n]), 2) if len(ppg) > n else POSITION_REPLACEMENT[pos]
    return result


# ── Core value model ─────────────────────────────────────────────────────────

# Backtested on the 2025 season (python -m src.backtest): 35% weight on the last
# 6 weeks beat the previous 65% / last-4-weeks blend at predicting the next 4 weeks.
RECENT_WEIGHT = 0.35


def _expected_pts(season_ppg: float, recent_ppg: float | None) -> float:
    """Expected points: RECENT_WEIGHT × recent form + the rest season baseline."""
    if season_ppg and recent_ppg is not None:
        return RECENT_WEIGHT * recent_ppg + (1 - RECENT_WEIGHT) * season_ppg
    return (recent_ppg if recent_ppg is not None else season_ppg) or 0.0


def _consistency(weekly_pts: list, expected: float) -> float:
    """
    0.0–1.0 consistency from the coefficient of variation over the last 8
    games (duds included). High consistency = low CV = closer to 1.0.
    """
    pts = list(weekly_pts[-8:])
    if len(pts) < 3 or expected <= 0:
        return 0.5  # neutral when not enough data
    cv = statistics.stdev(pts) / expected
    return max(0.0, min(1.0, 1.0 - cv * 0.8))


def player_value(stats: dict) -> dict:
    """
    expected_pts, consistency (0-1), and adj_value for a player's stats dict.

    adj_value = expected_pts * (0.70 + 0.30 * consistency)
    """
    season_ppg = (stats.get("season") or {}).get("ppg") or 0.0
    recent_ppg = (stats.get("recent") or {}).get("ppg")
    weekly = stats.get("weekly") or []

    exp = _expected_pts(season_ppg, recent_ppg)
    cons = _consistency(weekly, exp)
    adj = exp * (0.70 + 0.30 * cons)
    return {"expected_pts": round(exp, 2), "consistency": round(cons, 2), "adj_value": round(adj, 2)}


def _replacement(position: str, replacement_levels: dict | None) -> float:
    repl_map = replacement_levels or POSITION_REPLACEMENT
    pos = (position or "").upper()
    return repl_map.get(pos, POSITION_REPLACEMENT.get(pos, 10.0))


# ── Decision scoring ─────────────────────────────────────────────────────────

def score_waiver(player_stats: dict, position: str, trend_count: int = 0,
                 defense_multiplier: float = 1.0,
                 replacement_levels: dict | None = None) -> dict:
    """
    Score a waiver pickup 0–100 relative to positional replacement level.

    50 = replacement level player; >72 strong add; <40 avoid.
    FAAB bid % = max(0, surplus/replacement) * 0.40, capped at 50%.
    """
    v = player_value(player_stats)
    repl = _replacement(position, replacement_levels)

    effective_value = v["adj_value"] * defense_multiplier
    surplus = effective_value - repl

    raw_score = 50.0 + (surplus / max(repl, 0.1)) * 50.0
    trend_bonus = min(5.0, trend_count / 40.0)
    score = min(100.0, max(0.0, raw_score + trend_bonus))
    bid_pct = max(0.0, surplus / max(repl, 0.1)) * 0.40

    if score >= 72:
        verdict = "STRONG ADD"
    elif score >= 55:
        verdict = "ADD"
    elif score >= 40:
        verdict = "STREAM"
    else:
        verdict = "AVOID"

    return {
        "score": round(score, 1),
        "expected_pts": v["expected_pts"],
        "consistency_pct": round(v["consistency"] * 100, 0),
        "adj_value": v["adj_value"],
        "surplus_vs_replacement": round(surplus, 2),
        "verdict": verdict,
        "faab_bid_pct": round(min(bid_pct, 0.50), 3),
    }


def rank_players(named_stats: dict, adjustments: dict | None = None) -> list:
    """
    Rank 2–4 players for start/sit.

    adjustments: {name: {"matchup": multiplier, "injury": status}} — this week's
    opponent defense and injury designation. Ranking uses
    start_value = adj_value × matchup × injury_factor, so an Out player is never
    recommended. Confidence between top two: margin / top start_value.
    """
    adjustments = adjustments or {}
    results = []
    for name, stats in named_stats.items():
        v = player_value(stats)
        adj = adjustments.get(name, {})
        matchup = adj.get("matchup", 1.0) or 1.0
        inj = adj.get("injury")
        factor = matchup * injury_factor(inj)
        results.append({"name": name, **v, "matchup_mult": round(matchup, 3), "injury": inj,
                        "injury_factor": injury_factor(inj),
                        "start_value": round(v["adj_value"] * factor, 2)})
    results.sort(key=lambda x: x["start_value"], reverse=True)

    for i, r in enumerate(results):
        r["rank"] = i + 1
        r["recommendation"] = "START" if i == 0 else "BENCH"
        r["margin_to_next"] = (round(r["start_value"] - results[i + 1]["start_value"], 2)
                               if i < len(results) - 1 else 0.0)

    if len(results) >= 2:
        top, second = results[0]["start_value"], results[1]["start_value"]
        conf = min(1.0, (top - second) / max(top, 0.1))
        results[0]["confidence_pct"] = round(conf * 100, 1)
        results[0]["confidence_label"] = "Clear" if conf >= 0.25 else ("Lean" if conf >= 0.10 else "Toss-up")

    return results


# ── Lineups & trades ─────────────────────────────────────────────────────────

def optimal_lineup(players: list, roster_positions: list | None = None) -> dict:
    """
    Best starting lineup by value. players: [{name, position, value}].
    Fills the most restrictive slots first (QB before FLEX before SUPER_FLEX).

    Returns {"starters": [{slot, name, position, value}], "total": float}
    """
    slots = [s for s in (roster_positions or DEFAULT_ROSTER_POSITIONS) if s in SLOT_ELIGIBILITY]
    slots.sort(key=lambda s: len(SLOT_ELIGIBILITY[s]))
    pool = sorted(players, key=lambda p: p["value"], reverse=True)
    used, starters = set(), []

    for slot in slots:
        eligible = SLOT_ELIGIBILITY[slot]
        pick = next((i for i, p in enumerate(pool) if i not in used and p["position"] in eligible), None)
        if pick is None:
            starters.append({"slot": slot, "name": None, "position": None, "value": 0.0})
            continue
        used.add(pick)
        p = pool[pick]
        starters.append({"slot": slot, "name": p["name"], "position": p["position"], "value": p["value"]})

    return {"starters": starters, "total": round(sum(s["value"] for s in starters), 2)}


def score_trade(
    roster: list,
    giving: list,
    receiving: list,
    stats_by_name: dict,
    replacement_levels: dict | None = None,
    roster_positions: list | None = None,
) -> dict:
    """
    Evaluate a trade by its effect on your best starting lineup.

    roster:    your current roster [{name, position}]
    giving:    names from your roster you send away
    receiving: [{name, position}] players you get
    stats_by_name: {name: stats} for everyone involved

    Primary metric: change in optimal-lineup adj value (so a 2-for-1 is judged
    by what actually starts, not by summed points). Also reports value over
    replacement (VORP) per player. Without a roster, falls back to comparing
    VORP totals.

    Verdict: lineup change ≥ +2% ACCEPT, ≤ −2% DECLINE, else NEGOTIATE.
    """
    def valued(p):
        v = player_value(stats_by_name.get(p["name"], {}))["adj_value"]
        return {"name": p["name"], "position": p.get("position", ""), "value": v,
                "surplus": round(max(0.0, v - _replacement(p.get("position", ""), replacement_levels)), 2)}

    give_set = set(giving)
    roster_vals = [valued(p) for p in roster]
    give_vals = [v for v in roster_vals if v["name"] in give_set]
    recv_vals = [valued(p) for p in receiving]

    give_vorp = round(sum(v["surplus"] for v in give_vals), 2)
    recv_vorp = round(sum(v["surplus"] for v in recv_vals), 2)

    result = {
        "you_give": {v["name"]: {"adj_value": v["value"], "surplus": v["surplus"], "position": v["position"]}
                     for v in give_vals},
        "you_receive": {v["name"]: {"adj_value": v["value"], "surplus": v["surplus"], "position": v["position"]}
                        for v in recv_vals},
        "give_vorp": give_vorp,
        "recv_vorp": recv_vorp,
    }

    if roster_vals:
        before = optimal_lineup(roster_vals, roster_positions)
        after_pool = [v for v in roster_vals if v["name"] not in give_set] + recv_vals
        after = optimal_lineup(after_pool, roster_positions)
        delta = after["total"] - before["total"]
        delta_pct = delta / before["total"] if before["total"] > 0 else 0.0
        result.update({
            "method": "lineup",
            "before": before, "after": after,
            "delta": round(delta, 2), "delta_pct": round(delta_pct, 4),
        })
        change = delta_pct
    else:
        diff = recv_vorp - give_vorp
        base = max(give_vorp, recv_vorp, 1.0)
        result.update({"method": "vorp", "delta": round(diff, 2), "delta_pct": round(diff / base, 4)})
        change = diff / base

    result["verdict"] = "ACCEPT" if change >= 0.02 else ("DECLINE" if change <= -0.02 else "NEGOTIATE")
    return result


# ── Roster construction analysis ─────────────────────────────────────────────

# Positions graded when the league's lineup has a dedicated slot for them.
# DEF is never graded: nflverse has no team-defense stats to value it with.
_GRADED_POSITIONS = ("QB", "RB", "WR", "TE", "K")


def roster_needs(roster_positions: list | None = None) -> dict:
    """{pos: {"min_starters", "depth"}} from the league's dedicated lineup slots."""
    slots = Counter(roster_positions or DEFAULT_ROSTER_POSITIONS)
    return {pos: {"min_starters": slots[pos], "depth": slots[pos] if pos == "K" else slots[pos] * 2}
            for pos in _GRADED_POSITIONS if slots.get(pos)}


def analyze_roster_construction(
    roster: list,
    available_players: list,
    roster_stats: dict,
    available_stats: dict,
    replacement_levels: dict | None = None,
    matchup_mults: dict | None = None,
    roster_positions: list | None = None,
) -> dict:
    """
    Grade each position on the roster and rank available players by how much
    they address genuine needs.

    matchup_mults: {name: next-game defense multiplier} so scores match the
    Waivers tab.
    roster_positions: the league's lineup slots; only positions with a
    dedicated slot are graded.

    Returns {"position_grades": {pos: grade_dict}, "weaknesses": [pos],
             "targeted_adds": [player_dict]}
    """
    repl = replacement_levels or POSITION_REPLACEMENT
    matchup_mults = matchup_mults or {}
    needs = roster_needs(roster_positions)

    pos_players = {}
    for p in roster:
        pos_players.setdefault(p.get("position", ""), []).append(p["name"])

    position_grades = {}
    for pos, config in needs.items():
        repl_ppg = repl.get(pos, POSITION_REPLACEMENT.get(pos, 10.0))
        values = sorted(
            ({"name": n, "adj_value": player_value(roster_stats.get(n, {}))["adj_value"]}
             for n in pos_players.get(pos, [])),
            key=lambda x: x["adj_value"], reverse=True,
        )
        starter_values = [v["adj_value"] for v in values[:config["min_starters"]]]
        avg_starter_val = sum(starter_values) / max(len(starter_values), 1)
        strength_ratio = avg_starter_val / max(repl_ppg, 0.1)

        if strength_ratio >= 1.4:
            grade = "A"
        elif strength_ratio >= 1.15:
            grade = "B"
        elif strength_ratio >= 0.90:
            grade = "C"
        elif strength_ratio >= 0.70:
            grade = "D"
        else:
            grade = "F"

        position_grades[pos] = {
            "grade":           grade,
            "strength_ratio":  round(strength_ratio, 2),
            "avg_starter_val": round(avg_starter_val, 2),
            "depth_count":     len(values),
            "needed_depth":    config["depth"],
            "players":         values,
        }

    weaknesses = [
        pos for pos, g in position_grades.items()
        if g["grade"] in ("D", "F") or g["depth_count"] < needs[pos]["min_starters"]
    ]

    targeted_adds = []
    for p in available_players:
        pos = p.get("position", "")
        if pos not in needs:
            continue
        stats = available_stats.get(p["name"], {})
        base = score_waiver(stats, pos, p.get("trend_count", 0),
                            defense_multiplier=matchup_mults.get(p["name"], 1.0),
                            replacement_levels=repl)
        need_bonus = {"F": 20, "D": 12}.get(position_grades[pos]["grade"], 0) if pos in weaknesses else 0
        targeted_adds.append({
            **p,
            "adj_value":           base["adj_value"],
            "base_score":          base["score"],
            "need_adjusted_score": round(min(100.0, base["score"] + need_bonus), 1),
            "fills_need":          pos in weaknesses,
            "verdict":             base["verdict"],
        })

    targeted_adds.sort(key=lambda x: x["need_adjusted_score"], reverse=True)
    return {"position_grades": position_grades, "weaknesses": weaknesses, "targeted_adds": targeted_adds[:20]}


def recommend_drop(
    add_position: str,
    roster: list,
    roster_stats: dict,
    replacement_levels: dict | None = None,
) -> dict | None:
    """
    Best drop candidate when adding a player at add_position.

    Priority:
      1. Weakest same-position player below replacement level
      2. Weakest any-position player below replacement level
      3. Weakest same-position player (all above replacement)
      4. Weakest player overall (last resort)
    """
    if not roster:
        return None

    scored = []
    for p in roster:
        pos = p.get("position", "")
        v = player_value(roster_stats.get(p["name"], {}))
        scored.append({"name": p["name"], "position": pos, "adj_value": v["adj_value"],
                       "surplus": round(v["adj_value"] - _replacement(pos, replacement_levels), 2)})

    same_pos = [p for p in scored if p["position"] == add_position]
    below_repl = sorted([p for p in scored if p["surplus"] < 0], key=lambda x: x["adj_value"])
    same_pos_below = [p for p in below_repl if p["position"] == add_position]

    if same_pos_below:
        return {**same_pos_below[0], "reason": f"Weakest {add_position} — below replacement level"}
    if below_repl:
        return {**below_repl[0], "reason": "Weakest below-replacement player on roster"}
    if same_pos:
        return {**min(same_pos, key=lambda x: x["adj_value"]), "reason": f"Weakest {add_position} (all above replacement)"}
    return {**min(scored, key=lambda x: x["adj_value"]), "reason": "Weakest player on roster"}
