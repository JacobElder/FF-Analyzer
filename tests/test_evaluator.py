from src.evaluator import optimal_lineup, rank_players, score_trade, starter_counts


def _stats(ppg, games=8):
    return {"season": {"ppg": ppg, "games": games}, "recent": {"ppg": ppg}, "weekly": [ppg] * games, "found": True}


POSITIONS = ["QB", "RB", "RB", "WR", "WR", "TE", "FLEX"]
ROSTER = [
    {"name": "QB1", "position": "QB"},
    {"name": "RB1", "position": "RB"}, {"name": "RB2", "position": "RB"}, {"name": "RB3", "position": "RB"},
    {"name": "WR1", "position": "WR"}, {"name": "WR2", "position": "WR"}, {"name": "WR3", "position": "WR"},
    {"name": "TE1", "position": "TE"},
]
STATS = {"QB1": _stats(20), "RB1": _stats(18), "RB2": _stats(12), "RB3": _stats(9),
         "WR1": _stats(17), "WR2": _stats(13), "WR3": _stats(10), "TE1": _stats(9)}


def test_two_bench_players_for_one_star_is_not_a_loss():
    # Giving two bench-level players (who barely start) for a star should be an upgrade,
    # even though summed points favor the 2-player side.
    star = {"name": "Star", "position": "WR"}
    stats = {**STATS, "Star": _stats(19)}
    r = score_trade(ROSTER, ["RB3", "WR3"], [star], stats, roster_positions=POSITIONS)
    assert r["method"] == "lineup"
    assert r["delta"] > 0 and r["verdict"] == "ACCEPT"


def test_summed_points_two_for_one_is_declined_when_it_hurts_the_lineup():
    # Two mediocre players (combined 22) for my 18-pt RB1 — lineup gets worse.
    recv = [{"name": "A", "position": "RB"}, {"name": "B", "position": "RB"}]
    stats = {**STATS, "A": _stats(11), "B": _stats(11)}
    r = score_trade(ROSTER, ["RB1"], recv, stats, roster_positions=POSITIONS)
    assert r["delta"] < 0 and r["verdict"] == "DECLINE"


def test_one_sided_trade_works():
    r = score_trade(ROSTER, [], [{"name": "Star", "position": "WR"}], {**STATS, "Star": _stats(19)},
                    roster_positions=POSITIONS)
    assert r["verdict"] == "ACCEPT"
    r = score_trade([], [], [{"name": "Star", "position": "WR"}], {"Star": _stats(19)})
    assert r["method"] == "vorp"


def test_optimal_lineup_fills_flex_with_best_remaining():
    players = [{"name": n, "position": p["position"], "value": STATS[n]["season"]["ppg"]}
               for n, p in ((p["name"], p) for p in ROSTER)]
    lineup = optimal_lineup(players, POSITIONS)
    flex = next(s for s in lineup["starters"] if s["slot"] == "FLEX")
    assert flex["name"] == "WR3"  # 10 beats RB3's 9
    assert lineup["total"] == 20 + 18 + 12 + 17 + 13 + 9 + 10


def test_out_player_never_recommended_to_start():
    ranked = rank_players({"Hurt": _stats(25), "Healthy": _stats(10)},
                          {"Hurt": {"matchup": 1.0, "injury": "Out"}, "Healthy": {"matchup": 1.0}})
    assert ranked[0]["name"] == "Healthy"


def test_matchup_moves_close_call():
    ranked = rank_players({"A": _stats(12), "B": _stats(12.5)},
                          {"A": {"matchup": 1.2}, "B": {"matchup": 0.85}})
    assert ranked[0]["name"] == "A"


def test_starter_counts_scale_with_league():
    assert starter_counts(10)["QB"] < starter_counts(14)["QB"]
    sf = starter_counts(12, ["QB", "SUPER_FLEX", "RB", "RB", "WR", "WR", "TE"])
    assert sf["QB"] > starter_counts(12)["QB"]
