"""
Backtest the projection inputs against a completed season.

For every week W, each model predicts a player's points using only data
available before W, then is scored against what actually happened:
  - "next week"   : points in week W
  - "next 4 weeks": average points over weeks W..W+3 (games played)

Models compared:
  - history(recent_weight, window): (recent_weight × last-`window`-weeks PPG
                                     + (1 − recent_weight) × season-to-date PPG)
                                    × opponent-defense multiplier (as in production)
  - Sleeper's weekly projection
  - blends of projection and history

Rows are player-weeks the player played or Sleeper projected him; a projected
week he didn't play counts as 0 actual points, and a week Sleeper projected but
left him out is predicted as 0 (production's "missing = 0" rule). When any
projections exist, every model is scored on the same rows (those with both a
projection and a history value), so history and blend MAEs are comparable.

Run:  python -m src.backtest [season]
"""

import sys
from collections import defaultdict

import numpy as np

from src.defense import build_matchup_table, compute_defense_rankings, get_matchup_info
from src.players import build_sleeper_name_index, enrich_with_ids, sleeper_to_player
from src.projections import Projections
from src.scoring import compute_fantasy_points, effective_scoring
from src.sleeper_integration import fetch_all_players
from src.wrangler import fetch_weekly_stats

SKILL = {"QB", "RB", "WR", "TE"}
BLEND_HISTORY = (0.35, 6)  # history settings the projection is blended with


def _history_pred(games: list, week: int, recent_weight: float, window: int) -> float | None:
    """games: [(week, pts)] sorted. Uses only games before `week`."""
    prior = [p for w, p in games if w < week]
    if len(prior) < 2:
        return None
    season = np.mean(prior)
    recent = [p for w, p in games if week - window <= w < week]
    if not recent:
        return float(season)
    return float(recent_weight * np.mean(recent) + (1 - recent_weight) * season)


def _team_opponents(df) -> dict:
    """{(team, week): opponent} from the stats frame (empty when the columns are missing)."""
    if not {"team", "opponent_team"} <= set(df.columns):
        return {}
    return {(t, int(w)): o for t, w, o in df[["team", "week", "opponent_team"]].itertuples(index=False)
            if isinstance(t, str) and isinstance(o, str) and t and o}


def _matchup_tables(df, weeks) -> dict:
    """{week: matchup table built only from games before that week} (no look-ahead)."""
    if "opponent_team" not in df.columns:
        return {w: {} for w in weeks}
    return {w: build_matchup_table(compute_defense_rankings(df[df["week"] < w])) for w in weeks}


def run_backtest(season: int = 2025, first_week: int = 4, last_week: int = 17,
                 scoring: dict | None = None) -> dict:
    scoring = effective_scoring(scoring)
    df = fetch_weekly_stats(season)
    df = df[df["season"] == season]
    df = df.assign(fp=compute_fantasy_points(df, scoring))
    games = defaultdict(list)
    pos_of, team_of = {}, defaultdict(dict)
    teams = df["team"] if "team" in df.columns else [None] * len(df)
    for pid, w, p, pos, team in zip(df["player_id"], df["week"], df["fp"], df["position"], teams):
        if pos in SKILL:
            games[pid].append((int(w), float(p)))
            pos_of[pid] = pos
            team_of[pid][int(w)] = team
    for g in games.values():
        g.sort()

    # Sleeper player_id → nflverse player_id
    db = fetch_all_players()
    name_index = build_sleeper_name_index(db)
    sleeper_players = enrich_with_ids(
        [sleeper_to_player(pid, info) for pid, info in db.items() if info.get("position") in SKILL], db, name_index)
    names = df.drop_duplicates("player_id").set_index("player_id")[["player_display_name", "position"]]
    from src.players import normalize_name
    by_name = defaultdict(list)
    for gid, row in names.iterrows():
        by_name[(normalize_name(row["player_display_name"]), row["position"])].append(gid)
    sleeper_to_gsis = {}
    for p in sleeper_players:
        gid = p.get("gsis_id") if p.get("gsis_id") in games else None
        if not gid:
            cands = by_name.get((normalize_name(p["name"]), p["position"]), [])
            gid = cands[0] if len(cands) == 1 else None
        if gid:
            sleeper_to_gsis[p["player_id"]] = gid
    gsis_to_sleeper = {g: s for s, g in sleeper_to_gsis.items()}

    proj = Projections(season, range(first_week, last_week + 1), scoring, completed_through=99)

    weeks = range(first_week, last_week + 1)
    opponents = _team_opponents(df)
    tables = _matchup_tables(df, weeks)
    covers = getattr(proj, "covers", lambda week: False)

    hist_params = [(rw, win) for rw in (0.0, 0.35, 0.5, 0.65, 0.8) for win in (3, 4, 6)]
    alphas = (0.0, 0.25, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)

    # Every candidate row: (actuals, history preds, projection or None)
    rows = []
    for gid, g in games.items():
        played = dict(g)
        sid = gsis_to_sleeper.get(gid)
        projected = {}
        for week in weeks:
            p = proj.get(sid, week) if sid else None
            if p is None and sid and covers(week):
                p = 0.0  # Sleeper projected the week but left him out: production values him at 0
            projected[week] = p
        # Weeks he played, plus weeks Sleeper actually projected him (a DNP there scores 0)
        active = [w for w in weeks if w in played or (sid and proj.get(sid, w) is not None)]
        for week in active:
            future = [played.get(w, 0.0) for w in active if week <= w < week + 4]
            actual = {"next_week": played.get(week, 0.0), "next_4": float(np.mean(future))}
            hist = {hp: _history_pred(g, week, *hp) for hp in hist_params}
            if hist[BLEND_HISTORY] is None:
                continue
            team = team_of[gid][max(w for w, _ in g if w <= week)]  # that week's team, else his latest
            opp = opponents.get((team, week))
            mult = get_matchup_info(opp, pos_of[gid], tables[week])["multiplier"] if opp else 1.0
            hist = {hp: h * mult for hp, h in hist.items()}
            rows.append((actual, hist, projected[week]))

    # Compare every model on the same rows: those with a projection, when there are any
    if any(p is not None for _, _, p in rows):
        rows = [r for r in rows if r[2] is not None]

    errs = {"next_week": defaultdict(list), "next_4": defaultdict(list)}
    for actual, hist, p in rows:
        for target, y in actual.items():
            for hp, h in hist.items():
                errs[target][("history", hp)].append(abs(h - y))
            if p is not None:
                for a in alphas:
                    errs[target][("blend", a)].append(abs(a * p + (1 - a) * hist[BLEND_HISTORY] - y))

    results = {}
    for target, table in errs.items():
        results[target] = sorted(((k, float(np.mean(v)), len(v)) for k, v in table.items()), key=lambda r: r[1])
    return results


def main():
    season = int(sys.argv[1]) if len(sys.argv) > 1 else 2025
    results = run_backtest(season)
    for target, rows in results.items():
        print(f"\n== {target}: mean absolute error (lower is better) ==")
        for (kind, params), mae, n in rows[:12]:
            label = (f"history recent_weight={params[0]} window={params[1]}" if kind == "history"
                     else f"blend α={params} (α × projection + (1−α) × history)")
            print(f"  {mae:6.3f}  n={n:5d}  {label}")


if __name__ == "__main__":
    main()
