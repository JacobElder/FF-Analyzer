"""
Player identity: name normalization, team-abbreviation normalization, and
linking Sleeper players (player_id / gsis_id) across data sources.

Matching rules are deliberately strict: an ID match, or an exact normalized
name match disambiguated by position/team. There is no last-name-only or
substring fallback — a missing match is reported as missing, never silently
attached to a different player.
"""

import re
import unicodedata

RELEVANT_POSITIONS = {"QB", "RB", "WR", "TE", "K", "DEF"}

_SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}

# nflverse / odds feeds → Sleeper abbreviations
TEAM_ALIASES = {"LA": "LAR", "JAC": "JAX", "WSH": "WAS", "LVR": "LV", "OAK": "LV", "SD": "LAC", "STL": "LAR"}


def normalize_name(name: str) -> str:
    """'Calvin Austin III' → 'calvin austin'; "Ja'Marr Chase" → 'jamarr chase'."""
    if not name:
        return ""
    s = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    s = s.lower().replace("'", "").replace("’", "").replace(".", "")
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    tokens = [t for t in s.split() if t not in _SUFFIXES]
    return " ".join(tokens)


def norm_team(team) -> str:
    if not team or not isinstance(team, str):
        return ""
    team = team.strip().upper()
    return TEAM_ALIASES.get(team, team)


def clean_gsis_id(gsis_id) -> str | None:
    if not gsis_id or not isinstance(gsis_id, str):
        return None
    gsis_id = gsis_id.strip()
    return gsis_id or None


def sleeper_display_name(p_info: dict, fallback: str = "") -> str:
    full = p_info.get("full_name")
    if full:
        return full
    # Team defenses have no full_name: "Minnesota" + "Vikings"
    name = f"{p_info.get('first_name') or ''} {p_info.get('last_name') or ''}".strip()
    return name or fallback


def sleeper_to_player(p_id: str, p_info: dict) -> dict:
    return {
        "name": sleeper_display_name(p_info, fallback=p_id),
        "position": p_info.get("position") or "UNK",
        "team": norm_team(p_info.get("team")),
        "player_id": p_id,
        "gsis_id": clean_gsis_id(p_info.get("gsis_id")),
    }


def build_sleeper_name_index(players_db: dict) -> dict:
    """normalized name → [sleeper player_id, ...] (active players first)."""
    index: dict[str, list[str]] = {}
    for p_id, p_info in players_db.items():
        if p_info.get("position") not in RELEVANT_POSITIONS:
            continue
        key = normalize_name(sleeper_display_name(p_info))
        if key:
            index.setdefault(key, []).append(p_id)
    for ids in index.values():
        ids.sort(key=lambda i: not players_db[i].get("active"))
    return index


def _pick_candidate(ids: list[str], players_db: dict, position: str, team: str) -> str | None:
    if len(ids) == 1:
        return ids[0]
    if position:
        ids = [i for i in ids if players_db[i].get("position") == position] or ids
    if len(ids) > 1 and team:
        ids = [i for i in ids if norm_team(players_db[i].get("team")) == team] or ids
    if len(ids) > 1:
        active = [i for i in ids if players_db[i].get("active")]
        ids = active if len(active) == 1 else ids
    return ids[0] if len(ids) == 1 else None


def enrich_with_ids(players: list, players_db: dict, name_index: dict) -> list:
    """
    Fill in Sleeper player_id / gsis_id / team for player dicts that lack them
    (e.g. league files saved by older versions). Never overwrites existing IDs.
    """
    out = []
    for p in players:
        p = dict(p)
        p_id = p.get("player_id")
        if not p_id or p_id not in players_db:
            ids = name_index.get(normalize_name(p.get("name", "")), [])
            p_id = _pick_candidate(ids, players_db, p.get("position", ""), norm_team(p.get("team")))
            if p_id:
                p["player_id"] = p_id
        info = players_db.get(p_id or "", {})
        if info:
            if not p.get("gsis_id"):
                p["gsis_id"] = clean_gsis_id(info.get("gsis_id"))
            if not p.get("team"):
                p["team"] = norm_team(info.get("team"))
        p["team"] = norm_team(p.get("team"))
        out.append(p)
    return out


def player_label(p: dict) -> str:
    parts = [x for x in (p.get("position"), p.get("team")) if x]
    return f"{p.get('name', '?')} ({', '.join(parts)})" if parts else p.get("name", "?")
