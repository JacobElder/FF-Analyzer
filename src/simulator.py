"""
Monte Carlo simulation for fantasy football decisions.

Each player's weekly score is a zero-inflated Gamma: a "dud" probability equal
to their share of ≤0-point games, otherwise Gamma fitted to positive scores.
Falls back to Normal when data is sparse.

Samples are seeded per player (name + scoring history), so the same player
always gets the same samples. Every chart and probability for that player is
then computed from identical draws, and lineup comparisons use common random
numbers for players on both sides.
"""

import zlib

import numpy as np
from scipy import stats as sp_stats

N_SAMPLES = 10_000


# ── Distribution fitting ──────────────────────────────────────────────────────

def _seed(key: str, weekly: list) -> int:
    return zlib.crc32(f"{key}|{weekly}".encode())


def _fit_samples(weekly_pts: list, fallback_mean: float, rng: np.random.Generator) -> np.ndarray:
    pts = np.asarray(weekly_pts, dtype=float)
    pts = pts[np.isfinite(pts)]
    positive = pts[pts > 0]

    if len(pts) >= 4 and len(positive) >= 4:
        try:
            shape, _, scale = sp_stats.gamma.fit(positive, floc=0)
            if shape > 0 and scale > 0:
                samples = rng.gamma(shape, scale, N_SAMPLES)
                p_dud = float(np.mean(pts <= 0))
                if p_dud > 0:
                    samples[rng.random(N_SAMPLES) < p_dud] = 0.0
                return samples
        except Exception:
            pass

    # Fallback: Normal from available data, or from the expected value.
    # No data at all means 0 — the same value the evaluator gives an unknown player.
    mean = float(np.mean(pts)) if len(pts) >= 2 else float(fallback_mean or 0.0)
    std = float(np.std(pts)) if len(pts) >= 2 else mean * 0.5
    std = max(std, 1.0)  # prevent degenerate distribution
    return np.clip(rng.normal(mean, std, N_SAMPLES), 0, None)


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


def simulate_player(stats: dict, key: str = "", multiplier: float = 1.0) -> np.ndarray:
    """
    N_SAMPLES simulated weekly outcomes. `multiplier` scales for this week's
    matchup / injury (0 for an inactive player).
    """
    weekly = stats.get("weekly") or []
    recent = (stats.get("recent") or {}).get("ppg")
    fallback = recent if recent is not None else ((stats.get("season") or {}).get("ppg") or 0.0)
    rng = np.random.default_rng(_seed(key, weekly))
    return _fit_samples(weekly, fallback, rng) * multiplier


def _p_greater(a: np.ndarray, b: np.ndarray) -> float:
    """P(a > b), counting exact ties as half."""
    return float(np.mean(a > b) + 0.5 * np.mean(a == b))


# ── Decision simulations ──────────────────────────────────────────────────────

def simulate_start_sit(p1_name: str, p1_stats: dict, p2_name: str, p2_stats: dict,
                       multipliers: dict | None = None) -> dict:
    """Head-to-head: P(p1 > p2), P(p2 > p1), and 80% intervals."""
    m = multipliers or {}
    s1 = simulate_player(p1_stats, p1_name, m.get(p1_name, 1.0))
    s2 = simulate_player(p2_stats, p2_name, m.get(p2_name, 1.0))
    p1_win = _p_greater(s1, s2)
    return {
        "start":       p1_name if p1_win >= 0.5 else p2_name,
        "p1_win_prob": round(p1_win, 3),
        "p2_win_prob": round(1 - p1_win, 3),
        p1_name:       _summary(s1),
        p2_name:       _summary(s2),
    }


def simulate_multi_start(named_stats: dict, multipliers: dict | None = None) -> dict:
    """Multi-player (2–4): probability each player scores the most, plus intervals."""
    m = multipliers or {}
    names = list(named_stats.keys())
    samples = {n: simulate_player(s, n, m.get(n, 1.0)) for n, s in named_stats.items()}
    arr = np.array([samples[n] for n in names])
    # Each sample's win is shared equally among everyone tied for the top score
    top = arr == arr.max(axis=0)
    share = top / top.sum(axis=0)

    win_probs = {names[i]: round(float(share[i].mean()), 3) for i in range(len(names))}
    return {
        "best": max(win_probs, key=win_probs.get),
        "win_probs": win_probs,
        "summaries": {n: _summary(samples[n]) for n in names},
        "samples": samples,
    }


def _verdict(p_better: float) -> str:
    if p_better >= 0.55:
        return "ACCEPT"
    if p_better <= 0.45:
        return "DECLINE"
    return "NEGOTIATE"


def simulate_lineup_change(before_names: list, after_names: list, stats_by_name: dict) -> dict:
    """
    Weekly starting-lineup totals before vs. after a trade. Players on both
    lineups share the same samples, so only the changed starters drive the
    difference.
    """
    cache = {}

    def samples(name):
        if name not in cache:
            cache[name] = simulate_player(stats_by_name.get(name, {}), name)
        return cache[name]

    before = sum((samples(n) for n in before_names if n), np.zeros(N_SAMPLES))
    after = sum((samples(n) for n in after_names if n), np.zeros(N_SAMPLES))
    p_better = _p_greater(after, before)
    return {
        "verdict": _verdict(p_better),
        "p_better": round(p_better, 3),
        "before": _summary(before), "after": _summary(after),
        "before_samples": before, "after_samples": after,
    }


def simulate_trade(giving_stats: dict, receiving_stats: dict) -> dict:
    """
    Total weekly output of each side (used when no roster is loaded).
    Either side may be empty.
    """
    give = sum((simulate_player(s, n) for n, s in giving_stats.items()), np.zeros(N_SAMPLES))
    recv = sum((simulate_player(s, n) for n, s in receiving_stats.items()), np.zeros(N_SAMPLES))
    p_better = _p_greater(recv, give)
    return {
        "verdict":          _verdict(p_better),
        "p_receive_better": round(p_better, 3),
        "give":             _summary(give),
        "receive":          _summary(recv),
        "give_samples":     give,
        "receive_samples":  recv,
    }
