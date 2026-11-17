"""
Injury status from the shared Sleeper player database (no separate download).

Keyed by Sleeper player_id; the name fallback only covers names that belong to
exactly one relevant player, so namesakes can't swap statuses.
"""

from src.players import RELEVANT_POSITIONS, normalize_name, sleeper_display_name

# Statuses that mean the player won't play this week
INACTIVE_STATUSES = {"Out", "IR", "PUP", "Sus", "COV", "DNR", "NFI"}

GAME_STATUS_ORDER = {"Out": 4, "Doubtful": 3, "Questionable": 2, "Probable": 1}

# Sleeper's roster `status` → injury code, used when `injury_status` is empty
# (players placed on a reserve list often carry only this)
ROSTER_STATUS_MAP = {
    "Injured Reserve": "IR", "IR": "IR",
    "Physically Unable to Perform": "PUP", "PUP": "PUP",
    "Suspended": "Sus", "Sus": "Sus",
    "Non Football Injury": "NFI", "Non-Football Injury": "NFI", "NFI": "NFI",
    "Reserve/COVID-19": "COV", "COV": "COV",
}


def player_injury_status(p_info: dict) -> str | None:
    """Sleeper's injury_status, else a reserve-list `status` (IR, PUP, suspended…); None if healthy."""
    status = p_info.get("injury_status")
    if status and status != "NA":
        return status
    return ROSTER_STATUS_MAP.get(p_info.get("status") or "")


def build_injury_map(players_db: dict) -> dict:
    by_id, by_name, name_counts = {}, {}, {}
    for p_id, p_info in players_db.items():
        if p_info.get("position") not in RELEVANT_POSITIONS:
            continue
        key = normalize_name(sleeper_display_name(p_info))
        status = player_injury_status(p_info)
        if p_info.get("active") is False and not status:
            continue  # a retired namesake shouldn't make an active player's name ambiguous
        name_counts[key] = name_counts.get(key, 0) + 1
        if status:
            by_id[p_id] = status
            by_name[key] = status
    # Drop ambiguous names entirely rather than guess
    by_name = {k: v for k, v in by_name.items() if name_counts.get(k) == 1}
    return {"by_id": by_id, "by_name": by_name}


def injury_status(player, injury_map: dict) -> str | None:
    """Status string ('Out', 'Questionable', ...) or None if healthy/unknown."""
    if not injury_map:
        return None
    if isinstance(player, str):
        player = {"name": player}
    p_id = player.get("player_id")
    if p_id:
        return injury_map["by_id"].get(p_id)
    return injury_map["by_name"].get(normalize_name(player.get("name", "")))


def is_inactive(status: str | None) -> bool:
    return status in INACTIVE_STATUSES or status == "Doubtful"
