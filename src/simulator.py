"""
Monte Carlo simulation for fantasy football decisions.
Uses Gamma distribution (non-negative, right-skewed) fitted to historical weekly scores.
Falls back to Normal when data is sparse (<4 data points).
"""

import numpy as np
from scipy import stats as sp_stats

N_SAMPLES = 10_000
RNG = np.random.default_rng(42)


# ── Distribution fitting ──────────────────────────────────────────────────────

def _fit_samples(weekly_pts: list, fallback_mean: float) -> np.ndarray:
    """Fit a Gamma to non-zero scores; fall back to Normal if too few points."""
    pts = np.array([p for p in weekly_pts if p > 0], dtype=float)

    if len(pts) >= 4:
        try:
            shape, _, scale = sp_stats.gamma.fit(pts, floc=0)
            if shape > 0 and scale > 0:
                samples = RNG.gamma(shape, scale, N_SAMPLES)
                return np.clip(samples, 0, None)
        except Exception:
            pass

    # Fallback: Normal parameterised from available data or expected value
    mean = float(np.mean(pts)) if len(pts) >= 2 else (fallback_mean or 10.0)
    std = float(np.std(pts)) if len(pts) >= 2 else mean * 0.5
    std = max(std, 1.0)  # prevent degenerate distribution
    return np.clip(RNG.normal(mean, std, N_SAMPLES), 0, None)


def _summary(samples: np.ndarray) -> dict:
    return {
        "median":   round(float(np.median(samples)), 1),
        "mean":     round(float(np.mean(samples)), 1),
        "floor":    round(float(np.percentile(samples, 10)), 1),
        "ceiling":  round(float(np.percentile(samples, 90)), 1),
        "std":      round(float(np.std(samples)), 1),
        "ci_80":    [round(float(np.percentile(samples, 10)), 1),
                     round(float(np.percentile(samples, 90)), 1)],
    }


def simulate_player(stats: dict) -> np.ndarray:
    """Return N_SAMPLES simulated weekly fantasy point outcomes for a player."""
    weekly = stats.get("weekly") or []
    fallback = (
        (stats.get("recent") or {}).get("ppg")
        or (stats.get("season") or {}).get("ppg")
        or 10.0
    )
    return _fit_samples(weekly, fallback)


# ── Decision simulations ──────────────────────────────────────────────────────

def simulate_start_sit(p1_name: str, p1_stats: dict,
                       p2_name: str, p2_stats: dict) -> dict:
    """
    Head-to-head Monte Carlo.
    Returns P(p1 > p2), P(p2 > p1), and 80% credible intervals for each.
    """
    s1 = simulate_player(p1_stats)
    s2 = simulate_player(p2_stats)
    p1_win = float(np.mean(s1 > s2))

    return {
        "start":        p1_name if p1_win >= 0.5 else p2_name,
        "p1_win_prob":  round(p1_win, 3),
        "p2_win_prob":  round(1 - p1_win, 3),
        p1_name:        _summary(s1),
        p2_name:        _summary(s2),
    }


def simulate_multi_start(named_stats: dict) -> dict:
    """
    Multi-player Monte Carlo (2–4 players).
    Returns win probability and credible interval for each player.
    """
    names = list(named_stats.keys())
    samples = {n: simulate_player(s) for n, s in named_stats.items()}
    arr = np.array([samples[n] for n in names])   # (n_players, N_SAMPLES)
    winners = np.argmax(arr, axis=0)

    win_probs = {names[i]: round(float(np.mean(winners == i)), 3) for i in range(len(names))}
    summaries = {n: _summary(samples[n]) for n in names}
    best = max(win_probs, key=win_probs.get)

    return {"best": best, "win_probs": win_probs, "summaries": summaries}


def simulate_trade(giving_stats: dict, receiving_stats: dict) -> dict:
    """
    Simulate total weekly output for both sides of a trade.
    Returns P(receiving side outscores giving side) and credible intervals.
    """
    def total(stats_dict):
        t = np.zeros(N_SAMPLES)
        for s in stats_dict.values():
            t += simulate_player(s)
        return t

    give_s = total(giving_stats)
    recv_s = total(receiving_stats)
    p_better = float(np.mean(recv_s > give_s))

    if p_better >= 0.55:
        verdict = "ACCEPT"
    elif p_better <= 0.45:
        verdict = "DECLINE"
    else:
        verdict = "NEGOTIATE"

    return {
        "verdict":           verdict,
        "p_receive_better":  round(p_better, 3),
        "give":              _summary(give_s),
        "receive":           _summary(recv_s),
    }
