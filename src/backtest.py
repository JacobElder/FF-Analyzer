"""
Backtest the projection inputs against a completed season.

For every week W, each model predicts a player's points using only data
available before W, then is scored against what actually happened:
  - "next week"   : points in week W
  - "next 4 weeks": average points over weeks W..W+3 (games played)

Models compared:
  - history(recent_weight, window): recent_weight × last-`window`-weeks PPG
                                     + (1 − recent_weight) × season-to-date PPG
  - Sleeper's weekly projection
  - blends of projection and history

Run:  python -m src.backtest [season]
"""

import sys
from collections import defaultdict

import numpy as np

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


def run_backtest(season: int = 2025, first_week: int = 4, last_week: int = 17,
                 scoring: dict | None = None) -> dict:
    scoring = effective_scoring(scoring)
    df = fetch_weekly_stats(season)
    df = df[df["season"] == season]
    df = df.assign(fp=compute_fantasy_points(df, scoring))
    games = defaultdict(list)
    for pid, w, p, pos in df[["player_id", "week", "fp", "position"]].itertuples(index=False):
        if pos in SKILL:
            games[pid].append((int(w), float(p)))
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

    hist_params = [(rw, win) for rw in (0.0, 0.35, 0.5, 0.65, 0.8) for win in (3, 4, 6)]
    alphas = (0.0, 0.25, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
    errs = {"next_week": defaultdict(list), "next_4": defaultdict(list)}

    for gid, g in games.items():
        played = dict(g)
        sid = gsis_to_sleeper.get(gid)
        for week in range(first_week, last_week + 1):
            if week not in played:
                continue  # only score weeks the player actually played
            future = [played[w] for w in range(week, week + 4) if w in played]
            actual = {"next_week": played[week], "next_4": float(np.mean(future))}
            hist = {hp: _history_pred(g, week, *hp) for hp in hist_params}
            if hist[BLEND_HISTORY] is None:
                continue
            p = proj.get(sid, week) if sid else None
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
