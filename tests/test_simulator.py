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
