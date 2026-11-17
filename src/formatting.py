"""Small display helpers used by the Streamlit UI (kept here so they're testable)."""


def _plural(n: int, unit: str) -> str:
    return f"{n} {unit}{'' if n == 1 else 's'} ago"


def fmt_age(seconds: float | None) -> str:
    """'just now', '1 minute ago', '5 minutes ago', '1 hour ago', '2 days ago'.

    Under 30 seconds (or a missing/negative age) reads 'just now'. Rounds before choosing
    the unit, so there is never a '60 minutes ago' or '24 hours ago'.
    """
    if seconds is None or seconds < 30:
        return "just now"
    minutes = max(1, round(seconds / 60))
    if minutes < 60:
        return _plural(minutes, "minute")
    hours = round(seconds / 3600)
    if hours < 24:
        return _plural(hours, "hour")
    return _plural(round(seconds / 86400), "day")


def fmt_as_of(label: str, seconds: float | None) -> str:
    """'Injuries as of 3 hours ago' — a freshness caption built on fmt_age."""
    return f"{label} as of {fmt_age(seconds)}"


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
