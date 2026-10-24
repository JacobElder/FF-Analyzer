"""Small display helpers used by the Streamlit UI (kept here so they're testable)."""


def fmt_age(seconds: float) -> str:
    """'5 min ago', '3h ago', '2 days ago'. Rounds before choosing the unit (no '60 min ago')."""
    minutes = max(1, round(seconds / 60))
    if minutes < 60:
        return f"{minutes} min ago"
    hours = round(seconds / 3600)
    if hours < 24:
        return f"{hours}h ago"
    return f"{round(seconds / 86400)} days ago"


def lineup_summary(positions: list) -> str:
    """'QB, 2 RB, FLEX (+ 6 bench, 1 IR)' from Sleeper roster_positions."""
    counts = {}
    for pos in positions:
        counts[pos] = counts.get(pos, 0) + 1
    starting = [f"{n} {pos}" if n > 1 else pos for pos, n in counts.items() if pos not in ("BN", "IR", "TAXI")]
    extras = [f"{counts[k]} {label}" for k, label in (("BN", "bench"), ("IR", "IR"), ("TAXI", "taxi")) if k in counts]
    return ", ".join(starting) + (f" (+ {', '.join(extras)})" if extras else "")


def verdict_color(v: str) -> str:
    return {"STRONG ADD": "green", "ADD": "green", "STREAM": "orange", "AVOID": "red",
            "ACCEPT": "green", "NEGOTIATE": "orange", "DECLINE": "red"}.get(v, "gray")


def grade_color(grade: str) -> str:
    return {"A": "green", "B": "blue", "C": "orange", "D": "red", "F": "red"}.get(grade, "gray")
