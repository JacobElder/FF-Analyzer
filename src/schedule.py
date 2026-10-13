"""
Schedule analysis: upcoming matchups, playoff schedule strength, and weather.

Weather via Open-Meteo (free, no API key required).
Schedule via the nflverse games file.
"""

import logging
from datetime import date, datetime

import pandas as pd
import requests

from src.defense import get_matchup_info
from src.players import norm_team
from src.storage import data_path, is_fresh, write_parquet_atomic

log = logging.getLogger(__name__)

SCHEDULE_TTL = 86400  # 24 hours
NFLVERSE_GAMES_URL = "https://github.com/nflverse/nflverse-data/releases/download/schedules/games.parquet"

# Covered/dome stadiums — weather-immune (Sleeper abbreviations)
COVERED = {"ARI", "ATL", "DAL", "DET", "HOU", "IND", "LAC", "LAR", "LV", "MIN", "NO"}

# Stadium coordinates (lat, lon) keyed by home team abbreviation
STADIUM_COORDS = {
    "ARI": (33.5277, -112.2626), "ATL": (33.7554, -84.4009),
    "BAL": (39.2780, -76.6228),  "BUF": (42.7738, -78.7870),
    "CAR": (35.2258, -80.8528),  "CHI": (41.8623, -87.6167),
    "CIN": (39.0954, -84.5160),  "CLE": (41.5061, -81.6995),
    "DAL": (32.7473, -97.0945),  "DEN": (39.7439, -105.0200),
    "DET": (42.3400, -83.0456),  "GB":  (44.5013, -88.0622),
    "HOU": (29.6847, -95.4107),  "IND": (39.7601, -86.1639),
    "JAX": (30.3239, -81.6373),  "KC":  (39.0489, -94.4839),
    "LAC": (33.8644, -118.2611), "LAR": (33.8644, -118.2611),
    "LV":  (36.0908, -115.1836), "MIA": (25.9580, -80.2389),
    "MIN": (44.9737, -93.2572),  "NE":  (42.0909, -71.2643),
    "NO":  (29.9511, -90.0812),  "NYG": (40.8128, -74.0742),
    "NYJ": (40.8128, -74.0742),  "PHI": (39.9008, -75.1675),
    "PIT": (40.4468, -80.0158),  "SEA": (47.5952, -122.3316),
    "SF":  (37.4032, -121.9698), "TB":  (27.9759, -82.5033),
    "TEN": (36.1665, -86.7713),  "WAS": (38.9076, -76.8645),
}

PLAYOFF_WEEKS = (15, 16, 17)  # Sleeper's default fantasy playoffs; leagues override via playoff_week_start


def playoff_weeks_for(playoff_week_start: int | None) -> tuple:
    if playoff_week_start:
        start = int(playoff_week_start)
        return (start, start + 1, start + 2)
    return PLAYOFF_WEEKS


# ── Schedule data ─────────────────────────────────────────────────────────────

def fetch_schedules(year: int | None = None, force: bool = False) -> pd.DataFrame:
    """Regular-season schedule for `year` with a 24h file cache."""
    from src.wrangler import current_nfl_season
    if year is None:
        year = current_nfl_season()
    path = data_path(f"schedules_{year}.parquet")

    if not force and is_fresh(path, SCHEDULE_TTL):
        return pd.read_parquet(path)

    try:
        df = pd.read_parquet(NFLVERSE_GAMES_URL)
        df = df[df["season"] == year]
        if "game_type" in df.columns:
            df = df[df["game_type"] == "REG"]
        df = df.copy()
        for col in ("home_team", "away_team"):
            df[col] = df[col].map(norm_team)
        write_parquet_atomic(df, path)
        return df
    except Exception as e:
        log.warning("Schedule fetch failed: %s", e)
        return pd.read_parquet(path) if path.exists() else pd.DataFrame()


# ── Next game lookup ──────────────────────────────────────────────────────────

def get_next_game(team: str, schedules_df: pd.DataFrame) -> dict | None:
    """Next unplayed game for a team, or None (offseason / schedule unavailable)."""
    if schedules_df is None or schedules_df.empty or not team:
        return None

    team_games = schedules_df[(schedules_df["home_team"] == team) | (schedules_df["away_team"] == team)]
    if "home_score" in team_games.columns:
        team_games = team_games[team_games["home_score"].isna()]
    if team_games.empty:
        return None

    row = team_games.sort_values("week").iloc[0]
    is_home = row["home_team"] == team
    return {
        "week":      int(row["week"]),
        "opponent":  row["away_team"] if is_home else row["home_team"],
        "is_home":   bool(is_home),
        "gameday":   str(row.get("gameday", "")),
        "gametime":  str(row.get("gametime", "")),
        "home_team": row["home_team"],
    }


def next_game_map(schedules_df: pd.DataFrame) -> dict:
    """team → next game dict, for every team on the schedule."""
    if schedules_df is None or schedules_df.empty:
        return {}
    teams = set(schedules_df["home_team"]) | set(schedules_df["away_team"])
    return {t: get_next_game(t, schedules_df) for t in teams}


# ── Weather ───────────────────────────────────────────────────────────────────

def get_weather_for_game(home_team: str, gameday: str) -> dict | None:
    """
    Daily forecast for a game via Open-Meteo. Returns {"dome": True} for covered
    stadiums and None when unavailable (no coords, past game, beyond forecast range).

    Flags: wind ≥ 15 mph, precipitation ≥ 0.5mm, max temp < 35°F.
    """
    if home_team in COVERED:
        return {"dome": True, "note": "Indoor/dome — weather irrelevant"}

    coords = STADIUM_COORDS.get(home_team)
    if not coords or not gameday or gameday == "None":
        return None

    try:
        game_date = datetime.strptime(gameday, "%Y-%m-%d").date()
        if game_date < date.today():
            return None

        lat, lon = coords
        resp = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": lat, "longitude": lon,
                "daily": "temperature_2m_max,precipitation_sum,windspeed_10m_max",
                "temperature_unit": "fahrenheit",
                "windspeed_unit": "mph",
                "precipitation_unit": "mm",
                "start_date": str(game_date),
                "end_date": str(game_date),
                "timezone": "America/New_York",
            },
            timeout=8,
        )
        if resp.status_code != 200:
            return None

        data = resp.json().get("daily", {})
        temp_f = (data.get("temperature_2m_max") or [None])[0]
        precip = (data.get("precipitation_sum") or [None])[0]
        wind = (data.get("windspeed_10m_max") or [None])[0]

        flags = []
        if wind is not None and wind >= 15:
            flags.append(f"⚠️ Wind {wind:.0f} mph — suppresses passing")
        if precip is not None and precip >= 0.5:
            flags.append(f"🌧️ Precipitation {precip:.1f}mm expected")
        if temp_f is not None and temp_f < 35:
            flags.append(f"🥶 Cold ({temp_f:.0f}°F) — outdoor cold-weather game")

        return {
            "dome":      False,
            "temp_f":    round(temp_f) if temp_f is not None else None,
            "wind_mph":  round(wind) if wind is not None else None,
            "precip_mm": round(precip, 1) if precip is not None else None,
            "flags":     flags,
            "note":      "; ".join(flags) if flags else "No significant weather concerns",
        }
    except Exception as e:
        log.warning("Weather fetch failed: %s", e)
        return None


# ── Playoff schedule strength ─────────────────────────────────────────────────

def get_playoff_schedule_strength(
    team: str,
    schedules_df: pd.DataFrame,
    matchup_table: dict,
    position: str,
    playoff_weeks: tuple = PLAYOFF_WEEKS,
) -> dict:
    """
    Opponent defense quality for each fantasy playoff week.
    Grade: A (easiest) → F (hardest) from the average matchup multiplier.
    """
    if not team or schedules_df is None or schedules_df.empty or not matchup_table:
        return {"grade": "N/A", "weeks": [], "avg_multiplier": 1.0}

    team_games = schedules_df[(schedules_df["home_team"] == team) | (schedules_df["away_team"] == team)]

    weeks_info, multipliers = [], []
    for wk in playoff_weeks:
        game = team_games[team_games["week"] == wk]
        if game.empty:
            weeks_info.append({"week": wk, "opponent": "BYE", "matchup": "—", "multiplier": None})
            continue
        row = game.iloc[0]
        opponent = row["away_team"] if row["home_team"] == team else row["home_team"]
        matchup = get_matchup_info(opponent, position, matchup_table)
        multipliers.append(matchup["multiplier"])
        weeks_info.append({"week": wk, "opponent": opponent, "matchup": matchup["rank_label"],
                           "multiplier": matchup["multiplier"]})

    if not multipliers:
        return {"grade": "N/A", "weeks": weeks_info, "avg_multiplier": 1.0}

    avg = sum(multipliers) / len(multipliers)
    if avg >= 1.20:
        grade = "A"
    elif avg >= 1.08:
        grade = "B"
    elif avg >= 0.92:
        grade = "C"
    elif avg >= 0.80:
        grade = "D"
    else:
        grade = "F"
    return {"grade": grade, "weeks": weeks_info, "avg_multiplier": round(avg, 3)}
