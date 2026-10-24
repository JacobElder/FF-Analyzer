import numpy as np

from src.simulator import N_SAMPLES, simulate_lineup_change, simulate_player, simulate_trade

STATS = {"weekly": [12, 0, 18, 9, 22, 14, 7, 16], "season": {"ppg": 12.25}, "recent": {"ppg": 14.0}}


def test_samples_are_deterministic_per_player():
    assert np.array_equal(simulate_player(STATS, "A"), simulate_player(STATS, "A"))
    assert not np.array_equal(simulate_player(STATS, "A"), simulate_player(STATS, "B"))


def test_duds_are_simulated():
    s = simulate_player(STATS, "A")
    assert 0.05 < np.mean(s == 0) < 0.25  # 1 dud in 8 games


def test_out_player_scores_zero():
    assert np.all(simulate_player(STATS, "A", multiplier=0.0) == 0)


def test_empty_trade_side_does_not_crash():
    r = simulate_trade({}, {"A": STATS})
    assert r["give_samples"].shape == (N_SAMPLES,)
    assert r["verdict"] == "ACCEPT"


def test_identical_lineups_are_a_coin_flip():
    stats = {"A": STATS, "B": STATS}
    r = simulate_lineup_change(["A", "B"], ["A", "B"], stats)
    assert r["p_better"] == 0.5 and r["verdict"] == "NEGOTIATE"


# ── added: fallbacks, multi-start ties, start/sit ────────────────────────────

from src.simulator import simulate_multi_start, simulate_start_sit  # noqa: E402

FEW_GAMES = {"weekly": [10, 14], "season": {"ppg": 12.0}, "recent": {"ppg": 12.0}}


def test_sparse_data_uses_normal_fallback_clipped_at_zero():
    s = simulate_player(FEW_GAMES, "rookie")
    assert s.min() >= 0 and abs(s.mean() - 12) < 1.0


def test_player_with_no_data_is_valued_near_zero():
    # The evaluator values an unknown player at 0; the simulator must agree (it used to assume 10 ppg)
    assert simulate_player({}, "nobody").mean() < 1.5
    assert simulate_player({"recent": {"ppg": 0.0}}, "benched").mean() < 1.5


def test_sample_mean_tracks_history():
    s = simulate_player(STATS, "A")
    assert abs(s.mean() - np.mean(STATS["weekly"])) < 1.5


def test_nan_weeks_are_ignored():
    stats = {**STATS, "weekly": STATS["weekly"] + [float("nan")]}
    assert np.isfinite(simulate_player(stats, "A")).all()


def test_multi_start_probabilities_sum_to_one_and_ties_split():
    r = simulate_multi_start({"A": STATS, "B": FEW_GAMES, "C": {}})
    assert abs(sum(r["win_probs"].values()) - 1) < 0.005  # rounding to 3 places only
    out = simulate_multi_start({"X": STATS, "Y": STATS}, multipliers={"X": 0.0, "Y": 0.0})
    assert out["win_probs"] == {"X": 0.5, "Y": 0.5}  # both Out: a tie, not a win for whoever is listed first


def test_start_sit_out_player_loses():
    r = simulate_start_sit("A", STATS, "B", STATS, multipliers={"A": 0.0})
    assert r["start"] == "B" and r["p2_win_prob"] > 0.5
    assert r["p1_win_prob"] + r["p2_win_prob"] == 1.0


def test_better_lineup_is_accepted():
    stats = {"A": STATS, "weak": {"weekly": [2, 3, 1, 2, 3, 2], "season": {"ppg": 2}, "recent": {"ppg": 2}}}
    r = simulate_lineup_change(["weak"], ["A"], stats)
    assert r["verdict"] == "ACCEPT" and r["p_better"] > 0.8  # A duds (0 pts) about 1 week in 8
