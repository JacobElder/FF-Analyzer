"""
Rest-of-season view of one league, built on the same weekly projections as
everything else:

  - playoff odds for every team, from simulating the remaining regular-season
    schedule (head-to-head games, plus the league-median game when the league
    uses one; ties in wins broken by points for, as Sleeper does)
  - this week's chance of beating the league median
  - week weights: how much a point in each remaining week is worth to a team.
    Regular-season weeks count fully; a playoff week counts by the chance the
    team is still alive and playing then (playoff odds, halved for each earlier
    round; a first-round bye seed doesn't play round 1). The bracket's weeks come
    from schedule.playoff_rounds
  - a trade finder: 1-for-1 swaps that raise both teams' weighted lineup totals

Team scores are normal around the best lineup's projection, with each starter's
standard deviation at SCORE_CV × his projection (as in market.matchup_projection).
"""

import zlib

import numpy as np

from src.evaluator import optimal_lineup
from src.players import sleeper_to_player
from src.schedule import DEFAULT_PLAYOFF_START, playoff_byes, playoff_rounds
from src.simulator import SCORE_CV  # noqa: F401  (re-exported: weekly scores vary by ~60% of their mean)
from src.valuation import NON_ROSTER_SLOTS, lineup_totals

PLAYOFF_ROUND_SURVIVAL = 0.5  # chance of winning each playoff round, for weighting later rounds
N_SIMS = 4000


def _rng(seed: str):
    return np.random.default_rng(zlib.crc32(seed.encode()))


def team_roster(r: dict, players_db: dict) -> list:
    """A league_rosters entry as roster dicts (reserve players in the IR slot, taxi squad in TAXI)."""
    reserve = set(r.get("reserve") or [])
    taxi = set(r.get("taxi") or [])
    return [{**sleeper_to_player(pid, players_db.get(pid, {})),
             "slot": "IR" if pid in reserve else "TAXI" if pid in taxi else "BN"}
            for pid in r.get("players") or []]


def lineup_dists(roster: list, ctx, positions: list) -> dict:
    """{week: (mean, variance)} of the best starting lineup's score."""
    players = [p for p in roster if p.get("slot") not in NON_ROSTER_SLOTS]
    vals = [ctx.week_values(p) for p in players]
    out = {}
    for w in ctx.weeks:
        lineup = optimal_lineup([{"name": p["name"], "position": p.get("position", ""), "value": v[w]}
                                 for p, v in zip(players, vals)], positions)
        starters = [s.get("value") or 0.0 for s in lineup["starters"] if s.get("name")]
        out[w] = (lineup["total"], sum((SCORE_CV * v) ** 2 for v in starters))
    return out


def _draw(rng, mus: list, variances: list, n_sims: int):
    """(n_sims, teams) simulated scores, floored at 0."""
    return np.maximum(rng.normal(np.array(mus), np.sqrt(np.array(variances)), (n_sims, len(mus))), 0.0)


def median_win_prob(mine: tuple, others: list, n_sims: int = N_SIMS, seed: str = "") -> float | None:
    """
    Chance of winning the league-median game: finishing in the top half of all
    scores this week. `mine` and each of `others` are (mean, variance).
    """
    if not others:
        return None
    scores = _draw(_rng(f"median|{seed}"), [mine[0]] + [o[0] for o in others],
                   [mine[1]] + [o[1] for o in others], n_sims)
    return round(float(np.mean(scores[:, 0] > np.median(scores, axis=1))), 3)


def _wins(record: dict) -> float:
    return (record or {}).get("wins", 0) + 0.5 * (record or {}).get("ties", 0)


def _games(record: dict) -> int:
    r = record or {}
    return (r.get("wins") or 0) + (r.get("losses") or 0) + (r.get("ties") or 0)


def league_dists(league: dict, ctx, players_db: dict, positions: list) -> dict:
    """{roster_id: {week: (mean, variance)}} of every team's best lineup."""
    return {r["roster_id"]: lineup_dists(team_roster(r, players_db), ctx, positions)
            for r in league.get("league_rosters") or []}


def simulate_season(league: dict, ctx, players_db: dict, positions: list, n_sims: int = N_SIMS,
                    dists: dict | None = None) -> dict | None:
    """
    Playoff odds for every team: {"p_playoffs": {roster_id: p}, "p_bye": {roster_id: p},
    "exp_wins": {roster_id: wins}, "weeks": [simulated weeks]}. `p_bye` is the chance
    of a first-round bye seed (all 0 when the bracket has none, see schedule.playoff_byes).
    `dists` (from league_dists) is computed when not given. None when the league has
    no synced schedule or playoff size.

    Sleeper can post this week's results to the records before it advances the
    current week; when every week through the current one is already in the
    records, the current week isn't simulated again.
    """
    rosters = league.get("league_rosters") or []
    schedule = league.get("schedule") or {}
    n_playoff = league.get("playoff_teams")
    if len(rosters) < 2 or not n_playoff or not schedule:
        return None
    playoff_start = league.get("playoff_week_start") or DEFAULT_PLAYOFF_START
    ids = [r["roster_id"] for r in rosters]
    idx = {rid: i for i, rid in enumerate(ids)}
    dists = dists or league_dists(league, ctx, players_db, positions)
    weeks = [w for w in ctx.weeks if w < playoff_start and str(w) in schedule]
    games_per_week = 2 if league.get("median_game") else 1
    if ctx.current_week and max(_games(r.get("record")) for r in rosters) >= ctx.current_week * games_per_week:
        weeks = [w for w in weeks if w != ctx.current_week]  # already final in the records

    rng = _rng(f"season|{league.get('league_id')}")
    wins = np.tile([_wins(r.get("record")) for r in rosters], (n_sims, 1)).astype(float)
    pf = np.tile([(r.get("record") or {}).get("points_for") or 0.0 for r in rosters], (n_sims, 1)).astype(float)
    for w in weeks:
        scores = _draw(rng, [dists[rid][w][0] for rid in ids], [dists[rid][w][1] for rid in ids], n_sims)
        pf += scores
        for a, b in schedule[str(w)]:
            if a in idx and b in idx:
                i, j = idx[a], idx[b]
                wins[:, i] += (scores[:, i] > scores[:, j]) + 0.5 * (scores[:, i] == scores[:, j])
                wins[:, j] += (scores[:, j] > scores[:, i]) + 0.5 * (scores[:, i] == scores[:, j])
        if league.get("median_game"):
            wins += scores > np.median(scores, axis=1, keepdims=True)

    # Rank by wins, then points for (points never reach 1e6, so wins always dominate)
    order = np.argsort(-(wins * 1e6 + pf), axis=1)
    made = np.zeros_like(wins, dtype=bool)
    made[np.arange(n_sims)[:, None], order[:, :min(n_playoff, len(ids))]] = True
    bye = np.zeros_like(wins, dtype=bool)
    bye[np.arange(n_sims)[:, None], order[:, :min(playoff_byes(league), n_playoff, len(ids))]] = True
    return {
        "p_playoffs": {rid: round(float(made[:, idx[rid]].mean()), 3) for rid in ids},
        "p_bye": {rid: round(float(bye[:, idx[rid]].mean()), 3) for rid in ids},
        "exp_wins": {rid: round(float(wins[:, idx[rid]].mean()), 1) for rid in ids},
        "weeks": weeks,
    }


def round_alive(p_playoffs: float, n_rounds: int, byes: int = 0, playoff_teams: int = 0,
                p_bye: float | None = None) -> list:
    """
    Chance of playing in each playoff round. Without byes: p_playoffs, then halved
    each round. With byes, round 1 is played only by seeds without one
    (p_playoffs − p_bye), and the bye seeds rejoin in round 2. An unknown `p_bye`
    is approximated as p_playoffs × byes / playoff_teams.
    """
    if not byes:
        return [p_playoffs * PLAYOFF_ROUND_SURVIVAL ** k for k in range(n_rounds)]
    if p_bye is None:
        p_bye = p_playoffs * byes / playoff_teams
    p_bye = min(max(p_bye, 0.0), p_playoffs)
    second = p_bye + (p_playoffs - p_bye) * PLAYOFF_ROUND_SURVIVAL
    return [p_playoffs - p_bye] + [second * PLAYOFF_ROUND_SURVIVAL ** k for k in range(n_rounds - 1)]


def week_weights(weeks: list, playoff_week_start: int | None, p_playoffs: float | None,
                 settings: dict | None = None, p_bye: float | None = None) -> dict:
    """
    {week: weight}: 1 in the regular season; each playoff week by the chance of
    playing in that round (round_alive); 0 after the championship. `settings`
    (league settings / synced league dict: playoff_teams, playoff_round_type) sets
    the bracket; `playoff_week_start` overrides its start. Without settings: three
    one-week rounds, no byes.
    """
    if p_playoffs is None:
        return {w: 1.0 for w in weeks}
    s = dict(settings or {})
    if playoff_week_start:
        s["playoff_week_start"] = playoff_week_start
    start = int(s.get("playoff_week_start") or DEFAULT_PLAYOFF_START)
    rounds = playoff_rounds(s)
    alive = round_alive(p_playoffs, len(rounds), playoff_byes(s), int(s.get("playoff_teams") or 0), p_bye)
    by_week = {w: a for r, a in zip(rounds, alive) for w in r}
    return {w: 1.0 if w < start else round(by_week.get(w, 0.0), 4) for w in weeks}


def all_week_weights(sim: dict | None, weeks: list, playoff_week_start: int | None,
                     settings: dict | None = None) -> dict:
    """{roster_id: week weights} for every team in the simulation ({} without one)."""
    if not sim:
        return {}
    p_bye = sim.get("p_bye") or {}
    return {rid: week_weights(weeks, playoff_week_start, p, settings, p_bye.get(rid))
            for rid, p in sim["p_playoffs"].items()}


def weeks_until_deadline(trade_deadline, current_week) -> int | None:
    """Weeks left to trade (0 = this is the deadline week), negative once it has passed; None if unknown."""
    if not trade_deadline or not current_week:
        return None
    return int(trade_deadline) - int(current_week)


def find_trades(my_roster: list, rivals: list, ctx, players_db: dict, positions: list,
                my_weights: dict | None = None, rival_weights: dict | None = None,
                min_gain: float = 1.0, per_side: int = 12, max_results: int = 15) -> list:
    """
    1-for-1 trades with each rival that raise BOTH teams' weighted rest-of-season
    lineup totals (you by at least `min_gain` points). Only each side's top
    `per_side` players by rest-of-season projection are considered. Sorted by the
    smaller of the two gains, since trades that help both sides get accepted.
    """
    def gain(new_roster, base, weights):
        totals = lineup_totals(new_roster, ctx, positions)
        return sum((totals[w] - base[w]) * (weights or {}).get(w, 1.0) for w in ctx.weeks)

    def tradeable(roster):
        return sorted((p for p in roster if p.get("slot") not in NON_ROSTER_SLOTS),
                      key=ctx.ros_total, reverse=True)[:per_side]

    my_base = lineup_totals(my_roster, ctx, positions)
    mine = tradeable(my_roster)
    out = []
    for r in rivals:
        theirs_full = team_roster(r, players_db)
        theirs = tradeable(theirs_full)
        their_base = lineup_totals(theirs_full, ctx, positions)
        their_weights = (rival_weights or {}).get(r["roster_id"])
        for give in mine:
            for get in theirs:
                g_me = gain([p for p in my_roster if p is not give] + [{**get, "slot": "BN"}], my_base, my_weights)
                if g_me < min_gain:
                    continue
                g_them = gain([p for p in theirs_full if p is not get] + [{**give, "slot": "BN"}],
                              their_base, their_weights)
                if g_them <= 0:
                    continue
                out.append({"team": r.get("team_name"), "roster_id": r["roster_id"],
                            "give": give["name"], "give_pos": give.get("position"),
                            "get": get["name"], "get_pos": get.get("position"),
                            "gain_me": round(g_me, 1), "gain_them": round(g_them, 1)})
    out.sort(key=lambda t: (-min(t["gain_me"], t["gain_them"]), -t["gain_me"]))
    return out[:max_results]


def team_week_dist(player_ids: list, ctx, players_db: dict, week: int) -> tuple:
    """(mean, variance) of a set lineup's score this week."""
    vals = [ctx.week_values(sleeper_to_player(pid, players_db.get(pid, {})))[week] for pid in player_ids]
    return sum(vals), sum((SCORE_CV * v) ** 2 for v in vals)


def combined_risk_prob(p_win: float | None, p_median: float | None) -> float | None:
    """The win chance that should drive start/sit risk: the average of both games when there's a median game."""
    if p_win is None or p_median is None:
        return p_win
    return (p_win + p_median) / 2
