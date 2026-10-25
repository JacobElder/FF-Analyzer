"""
Week-by-week player values and lineup-based roster math.

A player's value in week W is, in order of preference:
  1. PROJECTION_WEIGHT × Sleeper projection + (1 − PROJECTION_WEIGHT) × history value
  2. the Sleeper projection alone (no usable history, e.g. rookies)
  3. 0 when Sleeper published projections for W but not for him: backups and injured
     players it leaves out played 7% of the time and averaged 0.09 pts in 2025
     (their history predicted 4.3)
  4. the history value: expected points × that week's opponent-defense multiplier
  5. DEF/K only: a rough Vegas-based estimate for the current week
and is 0 on bye weeks and while injured. Injury timing comes from Sleeper's
projections for the weeks they cover (they stop projecting a player until he's
expected back); INJURY_WEEKS_OUT estimates are used only for weeks they don't.

An add is worth what it does to your *best starting lineup* in each remaining
week through the end of the fantasy playoffs, net of the player you drop.
"""

import math
from dataclasses import dataclass, field

import pandas as pd

from src.defense import get_matchup_info
from src.evaluator import DEFAULT_ROSTER_POSITIONS, _expected_pts, optimal_lineup

# Backtested on 2025 (python -m src.backtest): 80% projection / 20% history had
# the lowest error for both next-week and next-4-weeks points.
PROJECTION_WEIGHT = 0.8

# Estimated games missed by designation (Sleeper gives no return dates).
# IR/PUP/NFI: league minimum of 4 games. These are estimates — shown as such in the UI.
INJURY_WEEKS_OUT = {"Out": 1, "COV": 1, "Sus": 2, "IR": 4, "PUP": 4, "NFI": 4, "DNR": 99}
# Applied to the current week only
CURRENT_WEEK_FACTORS = {"Questionable": 0.85, "Doubtful": 0.25}

NON_ROSTER_SLOTS = {"IR", "TAXI"}
NON_LINEUP_SLOTS = {"BN", "IR", "TAXI"}

# DEF and K can be re-streamed from free agency most weeks for $0, so adding one
# is only worth its edge over the next STREAM_HORIZON weeks, not the whole season.
STREAM_POSITIONS = {"DEF", "K"}
STREAM_HORIZON = 2

# Handcuffs: chance per week that a starting RB misses the game, and how much of
# the starter's production the backup inherits when he does.
STARTER_MISS_RATE = 0.06
HANDCUFF_SHARE = 0.65


def _finite(x):
    """x as a float, or None when missing or NaN/inf (bad upstream data)."""
    if x is None:
        return None
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def team_week_opponents(schedules_df: pd.DataFrame) -> dict:
    """{team: {week: opponent}} for the season schedule."""
    out: dict[str, dict[int, str]] = {}
    if schedules_df is None or schedules_df.empty:
        return out
    for wk, home, away in schedules_df[["week", "home_team", "away_team"]].itertuples(index=False):
        out.setdefault(home, {})[int(wk)] = away
        out.setdefault(away, {})[int(wk)] = home
    return out


def fantasy_weeks(current_week: int | None, playoff_week_start: int | None, last_nfl_week: int = 18) -> list:
    """Weeks still to be played through the end of the fantasy playoffs."""
    if not current_week:
        return []
    last = min(last_nfl_week, (playoff_week_start or 15) + 2)
    return list(range(int(current_week), last + 1))


@dataclass
class ValuationContext:
    index: object                        # StatsIndex
    projections: object | None           # Projections or None
    opponents: dict                      # team → {week: opponent}
    matchup_table: dict
    injury_map: dict
    weeks: list
    vegas: dict = field(default_factory=dict)
    players_db: dict = field(default_factory=dict)
    _cache: dict = field(default_factory=dict, repr=False)

    @property
    def current_week(self) -> int | None:
        return self.weeks[0] if self.weeks else None

    # -- single player -----------------------------------------------------------

    def injury(self, p: dict) -> str | None:
        from src.injuries import injury_status
        return injury_status(p, self.injury_map)

    def weeks_out(self, p: dict) -> int:
        return INJURY_WEEKS_OUT.get(self.injury(p), 0)

    def covers(self, week: int) -> bool:
        """Sleeper projections exist for `week` (stand-ins without `covers` never do)."""
        return bool(self.projections is not None and hasattr(self.projections, "covers")
                    and self.projections.covers(week))

    def return_week(self, p: dict) -> int | None:
        """Estimated first week back for an injured player, or None."""
        out = self.weeks_out(p)
        if not out or not self.current_week:
            return None
        pid = p.get("player_id")
        if pid:
            for w in self.weeks[1:]:
                if not self.covers(w):
                    break  # past Sleeper's projections: fall back to the estimate
                if self.projections.get(pid, w) is not None:
                    return w
            else:
                if any(self.covers(w) for w in self.weeks[1:]):
                    return None  # Sleeper doesn't project him back this season
        return self.current_week + out

    def _history_value(self, p: dict, stats: dict, week: int) -> float | None:
        if not stats.get("found") or stats["season"]["games"] < 2:
            return None
        base = _expected_pts(stats["season"]["ppg"], stats["recent"]["ppg"])
        opp = self.opponents.get(p.get("team"), {}).get(week)
        mult = get_matchup_info(opp, p.get("position", ""), self.matchup_table)["multiplier"] if opp else 1.0
        return base * mult

    def _team_defense_fallback(self, p: dict, week: int) -> float | None:
        """Rough DEF/K value for the current week from Vegas lines, when nothing better exists."""
        if week != self.current_week or not self.vegas:
            return None
        team = p.get("team")
        if p.get("position") == "DEF":
            opp = self.opponents.get(team, {}).get(week)
            opp_total = self.vegas.get(opp)
            return max(0.0, 7.0 + (21.0 - opp_total) * 0.45) if opp_total else None
        if p.get("position") == "K":
            total = self.vegas.get(team)
            return max(0.0, 8.0 + (total - 22.0) * 0.2) if total else None
        return None

    def week_values(self, p: dict) -> dict:
        """{week: projected points} for every valuation week."""
        key = p.get("player_id") or (p.get("name"), p.get("position"))
        if key in self._cache:
            return self._cache[key]

        stats = self.index.get_player_stats(p)
        team_sched = self.opponents.get(p.get("team"), {})
        status = self.injury(p)
        out_weeks = INJURY_WEEKS_OUT.get(status, 0)
        values = {}
        for w in self.weeks:
            if not p.get("team") or (team_sched and w not in team_sched):
                values[w] = 0.0  # bye week or no NFL team
                continue
            covered = self.covers(w)
            # Out this week by designation; later weeks: Sleeper's projections decide, else the estimate
            if out_weeks and (w == self.current_week or (not covered and w < self.current_week + out_weeks)):
                values[w] = 0.0
                continue
            proj = _finite(self.projections.get(p.get("player_id"), w) if self.projections else None)
            if covered and proj is None and p.get("player_id"):
                values[w] = 0.0  # Sleeper projected this week but left him out: not expected to play
                continue
            hist = _finite(self._history_value(p, stats, w))
            if proj is not None and hist is not None:
                v = PROJECTION_WEIGHT * proj + (1 - PROJECTION_WEIGHT) * hist
            elif proj is not None:
                v = proj
            elif hist is not None:
                v = hist
            else:
                v = self._team_defense_fallback(p, w) or 0.0
            if w == self.current_week:
                v *= CURRENT_WEEK_FACTORS.get(status, 1.0)
            values[w] = round(v, 2)
        self._cache[key] = values
        return values

    def ros_total(self, p: dict) -> float:
        return round(sum(self.week_values(p).values()), 2)

    # -- depth chart context ----------------------------------------------------

    def depth_info(self, p: dict) -> dict:
        info = self.players_db.get(p.get("player_id") or "", {})
        return {"order": info.get("depth_chart_order"), "position": info.get("depth_chart_position")}

    def _depth_chart(self) -> dict:
        """(team, depth_chart_position) → [(order, player_id)], built once."""
        if "_depth" not in self._cache:
            chart: dict = {}
            for pid, info in self.players_db.items():
                if info.get("team") and info.get("depth_chart_order") and info.get("active"):
                    chart.setdefault((info["team"], info.get("depth_chart_position")), []).append(
                        (info["depth_chart_order"], pid))
            self._cache["_depth"] = chart
        return self._cache["_depth"]

    def promotion_note(self, p: dict) -> str | None:
        """'Promoted' when a teammate ahead of him on the depth chart is out."""
        me = self.depth_info(p)
        if not me["order"] or me["order"] < 2 or not p.get("team"):
            return None
        info_team = (self.players_db.get(p.get("player_id") or "", {}).get("team"), me["position"])
        for order, pid in sorted(self._depth_chart().get(info_team, [])):
            if order >= me["order"]:
                break
            status = self.injury({"player_id": pid})
            if status in INJURY_WEEKS_OUT:
                name = self.players_db[pid].get("full_name") or pid
                return f"{name} (ahead on depth chart) is {status}"
        return None


# ── Lineups over weeks ───────────────────────────────────────────────────────

def _eligible(roster: list) -> list:
    return [p for p in roster if p.get("slot") not in NON_ROSTER_SLOTS]


def lineup_totals(roster: list, ctx: ValuationContext, roster_positions: list | None = None) -> dict:
    """{week: best starting-lineup points} for a roster."""
    positions = roster_positions or DEFAULT_ROSTER_POSITIONS
    players = _eligible(roster)
    vals = [ctx.week_values(p) for p in players]
    return {w: optimal_lineup([{"name": p["name"], "position": p.get("position", ""), "value": v[w]}
                               for p, v in zip(players, vals)], positions)["total"]
            for w in ctx.weeks}


def roster_limit(roster_positions: list | None) -> int:
    return len([s for s in (roster_positions or DEFAULT_ROSTER_POSITIONS) if s not in NON_ROSTER_SLOTS])


def handcuff_bonus(add: dict, roster: list, ctx: ValuationContext) -> tuple[float, str | None]:
    """
    Insurance value of a backup RB whose starter is on this roster:
    STARTER_MISS_RATE × HANDCUFF_SHARE × starter's points, summed over weeks.
    """
    if add.get("position") != "RB" or (ctx.depth_info(add)["order"] or 0) < 2:
        return 0.0, None
    for p in roster:
        if (p.get("position") == "RB" and p.get("team") == add.get("team") and p.get("slot") not in NON_LINEUP_SLOTS
                and ctx.depth_info(p)["order"] == 1):
            bonus = sum(STARTER_MISS_RATE * HANDCUFF_SHARE * v for v in ctx.week_values(p).values())
            return round(bonus, 2), p["name"]
    return 0.0, None


def evaluate_add(add: dict, roster: list, ctx: ValuationContext, roster_positions: list | None = None,
                 base_totals: dict | None = None, max_drop_candidates: int = 6, weights: dict | None = None) -> dict:
    """
    Rest-of-season lineup gain from adding `add`, dropping whichever player
    costs the least (no drop if there's an open roster spot).

    `weights` ({week: multiplier}, default 1) scales each week's gain in the total,
    e.g. playoff weeks by the chance of being there; `weekly_gain` stays unweighted.

    Returns {"gain_total", "gain_per_week", "weekly_gain", "drop", "handcuff_for", "handcuff_bonus"}
    """
    positions = roster_positions or DEFAULT_ROSTER_POSITIONS
    base = base_totals or lineup_totals(roster, ctx, positions)
    eligible = _eligible(roster)

    if len(eligible) < roster_limit(positions):
        drop_options = [None]
    else:
        # Cheapest drops: lowest rest-of-season value first
        drop_options = sorted(eligible, key=ctx.ros_total)[:max_drop_candidates]

    horizon = set(ctx.weeks[:STREAM_HORIZON]) if add.get("position") in STREAM_POSITIONS else set(ctx.weeks)
    bonus, cuff_for = handcuff_bonus(add, roster, ctx)
    options = []
    for drop in drop_options:
        new_roster = [p for p in roster if p is not drop] + [{**add, "slot": "BN"}]
        totals = lineup_totals(new_roster, ctx, positions)
        weekly = {w: round(totals[w] - base[w], 2) if w in horizon else 0.0 for w in ctx.weeks}
        total = sum(g * (weights or {}).get(w, 1.0) for w, g in weekly.items())
        options.append({"drop": drop, "gain_total": round(total + bonus, 2), "weekly_gain": weekly})
    options.sort(key=lambda o: -o["gain_total"])

    best = dict(options[0])
    best["drop_options"] = [{"name": o["drop"]["name"] if o["drop"] else None, "gain_total": o["gain_total"]}
                            for o in options]
    best["handcuff_bonus"] = bonus
    best["handcuff_for"] = cuff_for
    best["streamer"] = add.get("position") in STREAM_POSITIONS
    best["gain_per_week"] = round(best["gain_total"] / max(len(ctx.weeks), 1), 2)
    return best


def lineup_suggestions(roster: list, ctx: ValuationContext, roster_positions: list | None = None) -> dict:
    """
    Compare your current Sleeper lineup with the best lineup for this week.
    Returns {"optimal": lineup, "current_total", "optimal_total", "swaps": [...]}.
    Each swap: {"slot", "out", "in", "gain", "reason"}.
    """
    positions = roster_positions or DEFAULT_ROSTER_POSITIONS
    week = ctx.current_week
    if week is None:
        return {"swaps": []}
    players = _eligible(roster)
    val = {p["name"]: ctx.week_values(p)[week] for p in players}
    optimal = optimal_lineup([{"name": p["name"], "position": p.get("position", ""), "value": val[p["name"]]}
                              for p in players], positions)

    starters = {p["name"]: p for p in roster if p.get("slot") and p.get("slot") not in NON_LINEUP_SLOTS}
    current_total = round(sum(val.get(n, 0.0) for n in starters), 2)
    optimal_names = {s["name"] for s in optimal["starters"] if s["name"]}

    from src.evaluator import SLOT_ELIGIBILITY
    to_start = sorted((n for n in optimal_names if n not in starters), key=lambda n: -val[n])
    to_bench = sorted((p for n, p in starters.items() if n not in optimal_names), key=lambda p: val.get(p["name"], 0))
    by_name = {p["name"]: p for p in players}
    swaps = []
    for out in to_bench:
        eligible = SLOT_ELIGIBILITY.get(out.get("slot"), set())
        inn = next((n for n in to_start if by_name[n].get("position") in eligible), None)
        if inn is None:
            continue
        to_start.remove(inn)
        status = ctx.injury(out)
        if status in INJURY_WEEKS_OUT or status == "Doubtful":
            reason = f"{out['name']} is {status}"
        elif val.get(out["name"], 0) == 0 and out.get("team"):
            reason = f"{out['name']} is on bye"
        else:
            reason = "Higher projection"
        swaps.append({"slot": out.get("slot"), "out": out["name"], "in": inn,
                      "gain": round(val[inn] - val.get(out["name"], 0), 2), "reason": reason})

    starting_slots = [s for s in positions if s not in NON_LINEUP_SLOTS]
    for inn in to_start[:max(len(starting_slots) - len(starters), 0)]:
        swaps.append({"slot": "(empty)", "out": None, "in": inn, "gain": val[inn], "reason": "Empty starting slot"})

    return {"optimal": optimal, "current_total": current_total, "optimal_total": optimal["total"],
            "swaps": [s for s in swaps if s["gain"] > 0.05]}
