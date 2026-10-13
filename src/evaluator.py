import statistics

# Static fallback replacement levels (PPG)
POSITION_REPLACEMENT = {
    "QB": 18.0, "RB": 10.0, "WR": 10.0,
    "TE": 8.0,  "K": 7.0,   "DEF": 7.0,
}

# Typical starter counts in a 12-team league
_STARTER_COUNTS = {
    "QB": 14, "RB": 40, "WR": 40, "TE": 14, "K": 14, "DEF": 14,
}


def compute_positional_scarcity(weekly_df, scoring: dict) -> dict:
    """
    Compute dynamic replacement-level PPG for each position from the current
    season's weekly_df. Returns {pos: replacement_ppg}.

    Replacement level = PPG of the (N+1)th-ranked player at each position,
    where N is the typical starter count in a 12-team league.
    Falls back to POSITION_REPLACEMENT if data is insufficient.
    """
    try:
        from src.wrangler import _pick_fp_col, _compute_fp
        fp_col = _pick_fp_col(weekly_df, scoring)
        work = weekly_df.copy()
        if fp_col == "_fp_half":
            work["_fp"] = work.apply(lambda r: _compute_fp(r.to_dict(), scoring), axis=1)
            fp_col = "_fp"

        player_ppg = (
            work[work["position"].isin(_STARTER_COUNTS)]
            .groupby(["player_display_name", "position"])[fp_col]
            .mean()
            .reset_index()
            .rename(columns={fp_col: "ppg"})
        )

        result = {}
        for pos, n in _STARTER_COUNTS.items():
            pos_df = player_ppg[player_ppg["position"] == pos].sort_values("ppg", ascending=False)
            if len(pos_df) > n:
                result[pos] = round(float(pos_df.iloc[n]["ppg"]), 2)
            else:
                result[pos] = POSITION_REPLACEMENT[pos]
        return result
    except Exception:
        return dict(POSITION_REPLACEMENT)


# ── Core value model ─────────────────────────────────────────────────────────

def _expected_pts(season_ppg: float, recent_ppg: float) -> float:
    """Expected points: 65% recent form, 35% season baseline."""
    if season_ppg and recent_ppg:
        return 0.65 * recent_ppg + 0.35 * season_ppg
    return recent_ppg or season_ppg or 0.0


def _consistency(weekly_pts: list, expected: float) -> float:
    """
    0.0–1.0 consistency score.
    Uses coefficient of variation (std/mean) on last 8 weeks.
    High consistency = low CV = closer to 1.0.
    """
    pts = [p for p in weekly_pts[-8:] if p > 0]
    if len(pts) < 2 or expected <= 0:
        return 0.5  # neutral when not enough data
    std = statistics.stdev(pts)
    cv = std / expected
    return max(0.0, min(1.0, 1.0 - cv * 0.8))


def player_value(stats: dict) -> dict:
    """
    Compute adjusted value for a player from their stats dict.
    Returns: expected_pts, consistency (0-1), adj_value.

    adj_value = expected_pts * (0.70 + 0.30 * consistency)
    Consistent players earn up to 30% bonus; volatile ones get 30% penalty.
    """
    season_ppg = (stats.get("season") or {}).get("ppg") or 0.0
    recent_ppg = (stats.get("recent") or {}).get("ppg") or 0.0
    weekly = stats.get("weekly") or []

    exp = _expected_pts(season_ppg, recent_ppg)
    cons = _consistency(weekly, exp)
    adj = exp * (0.70 + 0.30 * cons)

    return {
        "expected_pts": round(exp, 2),
        "consistency": round(cons, 2),
        "adj_value": round(adj, 2),
    }


# ── Decision scoring ─────────────────────────────────────────────────────────

def score_waiver(player_stats: dict, position: str, trend_count: int = 0,
                 defense_multiplier: float = 1.0,
                 replacement_levels: dict | None = None) -> dict:
    """
    Score a waiver pickup 0–100 relative to positional replacement level.

    50 = replacement level player
    >70 = strong add, <40 = avoid

    FAAB bid % = max(0, surplus/replacement) * 0.40
    (caps at 40% of budget for a clearly elite add)
    """
    v = player_value(player_stats)
    repl_map = replacement_levels or POSITION_REPLACEMENT
    repl = repl_map.get(position.upper(), POSITION_REPLACEMENT.get(position.upper(), 10.0))

    # Apply defense matchup multiplier to effective value (next-game context)
    effective_value = v["adj_value"] * defense_multiplier
    surplus = effective_value - repl

    raw_score = 50.0 + (surplus / max(repl, 0.1)) * 50.0
    # Trending bonus: up to +5 points
    trend_bonus = min(5.0, trend_count / 40.0)
    score = min(100.0, max(0.0, raw_score + trend_bonus))

    bid_pct = max(0.0, (surplus / max(repl, 0.1))) * 0.40

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


def rank_players(named_stats: dict) -> list:
    """
    Rank 2–4 players for start/sit by adj_value.

    named_stats: {player_name: stats_dict}
    Returns list sorted best→worst with rank, recommendation, and margin to next.
    Confidence between top two: |margin| / top_adj_value.
    """
    results = []
    for name, stats in named_stats.items():
        v = player_value(stats)
        results.append({"name": name, **v})
    results.sort(key=lambda x: x["adj_value"], reverse=True)

    for i, r in enumerate(results):
        r["rank"] = i + 1
        r["recommendation"] = "START" if i == 0 else "BENCH"
        if i < len(results) - 1:
            r["margin_to_next"] = round(r["adj_value"] - results[i + 1]["adj_value"], 2)
        else:
            r["margin_to_next"] = 0.0

    if len(results) >= 2:
        top, second = results[0]["adj_value"], results[1]["adj_value"]
        conf = min(1.0, (top - second) / max(top, 0.1))
        results[0]["confidence_pct"] = round(conf * 100, 1)
        results[0]["confidence_label"] = (
            "Clear" if conf >= 0.25 else ("Lean" if conf >= 0.10 else "Toss-up")
        )

    return results


def score_start_sit(p1_name: str, p1_stats: dict, p2_name: str, p2_stats: dict) -> dict:
    """
    Compare two players for start/sit.

    Confidence = |margin| / max(adj_value) — scales 0→1.
    <10% = Toss-up, 10-25% = Lean, >25% = Clear.
    """
    v1 = player_value(p1_stats)
    v2 = player_value(p2_stats)

    diff = v1["adj_value"] - v2["adj_value"]
    max_val = max(v1["adj_value"], v2["adj_value"], 0.1)
    confidence = min(1.0, abs(diff) / max_val)

    if confidence >= 0.25:
        conf_label = "Clear"
    elif confidence >= 0.10:
        conf_label = "Lean"
    else:
        conf_label = "Toss-up"

    start = p1_name if diff >= 0 else p2_name
    sit = p2_name if diff >= 0 else p1_name

    return {
        "start": start,
        "sit": sit,
        "margin": round(abs(diff), 2),
        "confidence_pct": round(confidence * 100, 1),
        "confidence_label": conf_label,
        p1_name: v1,
        p2_name: v2,
    }


def score_trade(giving_stats: dict, receiving_stats: dict) -> dict:
    """
    Evaluate trade fairness by comparing adjusted player values.

    ratio = recv_total / give_total
    ACCEPT ≥ 1.10, DECLINE ≤ 0.90, else NEGOTIATE.
    """
    give_vals = {n: player_value(s) for n, s in giving_stats.items()}
    recv_vals = {n: player_value(s) for n, s in receiving_stats.items()}

    give_total = sum(v["adj_value"] for v in give_vals.values())
    recv_total = sum(v["adj_value"] for v in recv_vals.values())
    ratio = recv_total / give_total if give_total > 0 else 1.0

    if ratio >= 1.10:
        verdict = "ACCEPT"
    elif ratio <= 0.90:
        verdict = "DECLINE"
    else:
        verdict = "NEGOTIATE"

    return {
        "verdict": verdict,
        "ratio": round(ratio, 3),
        "give_total": round(give_total, 2),
        "recv_total": round(recv_total, 2),
        "you_give": {n: v["adj_value"] for n, v in give_vals.items()},
        "you_receive": {n: v["adj_value"] for n, v in recv_vals.items()},
    }


# ── Roster construction analysis ─────────────────────────────────────────────

# Typical roster slot requirements (12-team standard)
_ROSTER_NEEDS = {
    "QB":  {"min_starters": 1, "depth": 2},
    "RB":  {"min_starters": 2, "depth": 4},
    "WR":  {"min_starters": 2, "depth": 4},
    "TE":  {"min_starters": 1, "depth": 2},
    "K":   {"min_starters": 1, "depth": 1},
    "DEF": {"min_starters": 1, "depth": 1},
}


def analyze_roster_construction(
    roster: list,
    available_players: list,
    roster_stats: dict,
    available_stats: dict,
    replacement_levels: dict | None = None,
) -> dict:
    """
    Identify positional weaknesses in the roster and rank available waiver
    targets by how much they address genuine needs.

    roster: list of {name, position}
    available_players: list of {name, position, ...}
    roster_stats: {name: stats_dict} for current roster
    available_stats: {name: stats_dict} for available players
    replacement_levels: {pos: ppg} — uses POSITION_REPLACEMENT if None

    Returns:
        {
          "position_grades": {pos: grade_dict},   # strength at each position
          "weaknesses": [pos, ...],                # positions below threshold
          "targeted_adds": [player_dict, ...],     # available players ranked by need-adjusted value
        }
    """
    repl = replacement_levels or POSITION_REPLACEMENT

    # ── Grade each position ───────────────────────────────────────────────────
    pos_players = {}
    for p in roster:
        pos = p.get("position", "")
        pos_players.setdefault(pos, []).append(p["name"])

    position_grades = {}
    for pos, config in _ROSTER_NEEDS.items():
        names = pos_players.get(pos, [])
        repl_ppg = repl.get(pos, POSITION_REPLACEMENT.get(pos, 10.0))

        # Compute adj_value for each roster player at this position
        values = []
        for name in names:
            stats = roster_stats.get(name, {})
            v = player_value(stats)
            values.append({"name": name, "adj_value": v["adj_value"]})

        values.sort(key=lambda x: x["adj_value"], reverse=True)
        n_starters = config["min_starters"]
        starter_values = [v["adj_value"] for v in values[:n_starters]]
        avg_starter_val = sum(starter_values) / max(len(starter_values), 1)

        depth_count = len(values)
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
            "grade":         grade,
            "strength_ratio": round(strength_ratio, 2),
            "avg_starter_val": round(avg_starter_val, 2),
            "depth_count":   depth_count,
            "needed_depth":  config["depth"],
            "players":       values,
        }

    weaknesses = [
        pos for pos, g in position_grades.items()
        if g["grade"] in ("D", "F")
        or g["depth_count"] < _ROSTER_NEEDS[pos]["min_starters"]
    ]

    # ── Rank available players by need-adjusted value ─────────────────────────
    targeted_adds = []
    for p in available_players:
        pos = p.get("position", "")
        if pos not in _ROSTER_NEEDS:
            continue

        stats = available_stats.get(p["name"], {})
        v = player_value(stats)
        base_score_result = score_waiver(stats, pos, p.get("trend_count", 0))
        base_score = base_score_result["score"]

        # Need multiplier: filling a weakness gets a significant boost
        if pos in weaknesses:
            grade_score = position_grades[pos]["grade"]
            need_bonus = {"F": 20, "D": 12}.get(grade_score, 0)
        else:
            need_bonus = 0

        need_adjusted_score = min(100.0, base_score + need_bonus)

        targeted_adds.append({
            **p,
            "adj_value":         v["adj_value"],
            "base_score":        round(base_score, 1),
            "need_adjusted_score": round(need_adjusted_score, 1),
            "fills_need":        pos in weaknesses,
            "verdict":           base_score_result["verdict"],
        })

    targeted_adds.sort(key=lambda x: x["need_adjusted_score"], reverse=True)

    return {
        "position_grades": position_grades,
        "weaknesses":      weaknesses,
        "targeted_adds":   targeted_adds[:20],  # top 20 need-adjusted pickups
    }


def recommend_drop(
    add_position: str,
    roster: list,
    roster_stats: dict,
    replacement_levels: dict | None = None,
) -> dict | None:
    """
    Identify the best drop candidate when adding a player at add_position.

    Priority:
      1. Weakest same-position player below replacement level
      2. Weakest any-position player below replacement level
      3. Weakest same-position player (all above replacement)
      4. Weakest player overall (last resort)

    Returns None if roster is empty.
    """
    repl = replacement_levels or POSITION_REPLACEMENT
    if not roster:
        return None

    scored = []
    for p in roster:
        pos = p.get("position", "")
        v = player_value(roster_stats.get(p["name"], {}))
        repl_ppg = repl.get(pos, POSITION_REPLACEMENT.get(pos, 10.0))
        scored.append({
            "name": p["name"],
            "position": pos,
            "adj_value": v["adj_value"],
            "surplus": round(v["adj_value"] - repl_ppg, 2),
        })

    same_pos = [p for p in scored if p["position"] == add_position]
    below_repl = sorted([p for p in scored if p["surplus"] < 0], key=lambda x: x["adj_value"])
    same_pos_below = [p for p in below_repl if p["position"] == add_position]

    if same_pos_below:
        drop = dict(same_pos_below[0])
        drop["reason"] = f"Weakest {add_position} — below replacement level"
        return drop
    if below_repl:
        drop = dict(below_repl[0])
        drop["reason"] = "Weakest below-replacement player on roster"
        return drop
    if same_pos:
        drop = dict(sorted(same_pos, key=lambda x: x["adj_value"])[0])
        drop["reason"] = f"Weakest {add_position} (all above replacement)"
        return drop
    drop = dict(sorted(scored, key=lambda x: x["adj_value"])[0])
    drop["reason"] = "Weakest player on roster"
    return drop
