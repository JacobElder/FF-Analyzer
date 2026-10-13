"""
Schedule analysis: upcoming matchups, playoff schedule strength, and weather.

Weather via Open-Meteo (free, no API key required).
Schedule via nfl_data_py.import_schedules().
"""

import os
import time
import pickle
import requests
import pandas as pd
from datetime import date, datetime

SCHEDULE_CACHE = "data/schedules_{year}.pkl"
SCHEDULE_TTL = 86400  # 24 hours

NFLVERSE_GAMES_URL = "https://github.com/nflverse/nflverse-data/releases/download/schedules/games.parquet"

# Covered/dome stadiums — weather-immune
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

PLAYOFF_WEEKS = (14, 15, 16)  # typical fantasy playoff weeks


# ── Schedule data ─────────────────────────────────────────────────────────────

def fetch_schedules(year: int | None = None) -> pd.DataFrame:
    """Fetch season schedule with 24h file cache."""
    from src.wrangler import current_nfl_season
    if year is None:
        year = current_nfl_season()

    os.makedirs("data", exist_ok=True)
    path = SCHEDULE_CACHE.format(year=year)

    if os.path.exists(path):
        if time.time() - os.path.getmtime(path) < SCHEDULE_TTL:
            with open(path, "rb") as f:
                return pickle.load(f)

    try:
        df = pd.read_parquet(NFLVERSE_GAMES_URL, engine="auto")
        df = df[df["season"] == year].copy()
        reg = df[df["game_type"] == "REG"].copy() if "game_type" in df.columns else df.copy()
        with open(path, "wb") as f:
            pickle.dump(reg, f)
        return reg
    except Exception as e:
        print(f"Schedule fetch failed: {e}")
        return pd.DataFrame()


# ── Next game lookup ──────────────────────────────────────────────────────────

def get_next_game(team: str, schedules_df: pd.DataFrame) -> dict | None:
    """
    Find the next unplayed game for a team.
    Returns None during offseason or if schedule unavailable.
    """
    if schedules_df is None or schedules_df.empty or not team:
        return None

    team_games = schedules_df[
        (schedules_df["home_team"] == team) | (schedules_df["away_team"] == team)
    ].copy()

    # Unplayed = no home_score (null/NaN)
    if "home_score" in team_games.columns:
        upcoming = team_games[team_games["home_score"].isna()]
    else:
        upcoming = team_games

    if upcoming.empty:
        return None

    upcoming = upcoming.sort_values("week")
    row = upcoming.iloc[0]

    is_home = row["home_team"] == team
    opponent = row["away_team"] if is_home else row["home_team"]

    return {
        "week":      int(row["week"]),
        "opponent":  opponent,
        "is_home":   is_home,
        "gameday":   str(row.get("gameday", "")),
        "gametime":  str(row.get("gametime", "")),
        "home_team": row["home_team"],
    }


# ── Weather ───────────────────────────────────────────────────────────────────

def get_weather_for_game(next_game: dict | None) -> dict | None:
    """
    Fetch weather forecast for the next game via Open-Meteo (free, no key).
    Returns None for dome/covered stadiums or during offseason.

    Flags: wind_mph >= 15 (affects passing), precipitation > 0.5mm, temp_f < 35.
    """
    if not next_game:
        return None

    home_team = next_game.get("home_team", "")
    if home_team in COVERED:
        return {"dome": True, "note": "Indoor/dome — weather irrelevant"}

    coords = STADIUM_COORDS.get(home_team)
    if not coords:
        return None

    gameday = next_game.get("gameday", "")
    if not gameday or gameday == "None":
        return None

    try:
        # Parse game date
        game_date = datetime.strptime(gameday, "%Y-%m-%d").date()
        if game_date < date.today():
            return None  # game already played

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
        temp_f = data.get("temperature_2m_max", [None])[0]
        precip = data.get("precipitation_sum", [None])[0]
        wind = data.get("windspeed_10m_max", [None])[0]

        flags = []
        if wind and wind >= 15:
            flags.append(f"⚠️ Wind {wind:.0f} mph — suppresses passing")
        if precip and precip >= 0.5:
            flags.append(f"🌧️ Precipitation {precip:.1f}mm expected")
        if temp_f and temp_f < 35:
            flags.append(f"🥶 Cold ({temp_f:.0f}°F) — outdoor cold-weather game")

        return {
            "dome":    False,
            "temp_f":  round(temp_f, 0) if temp_f else None,
            "wind_mph": round(wind, 0) if wind else None,
            "precip_mm": round(precip, 1) if precip else None,
            "flags":   flags,
            "note":    "; ".join(flags) if flags else "No significant weather concerns",
        }

    except Exception as e:
        print(f"Weather fetch failed: {e}")
        return None


# ── Playoff schedule strength ─────────────────────────────────────────────────

def get_playoff_schedule_strength(
    team: str,
    schedules_df: pd.DataFrame,
    defense_df: pd.DataFrame,
    position: str,
    playoff_weeks: tuple = PLAYOFF_WEEKS,
) -> dict:
    """
    For each fantasy playoff week, find the opponent and their defense quality
    for the given position. Returns aggregate difficulty grade.

    Grade: A (easiest) → F (hardest) based on avg multiplier vs. league average.
    """
    from src.defense import get_matchup_info

    if schedules_df is None or schedules_df.empty or defense_df is None or defense_df.empty:
        return {"grade": "N/A", "weeks": [], "avg_multiplier": 1.0}

    team_games = schedules_df[
        (schedules_df["home_team"] == team) | (schedules_df["away_team"] == team)
    ]

    weeks_info = []
    multipliers = []

    for wk in playoff_weeks:
        game = team_games[team_games["week"] == wk]
        if game.empty:
            continue
        row = game.iloc[0]
        opponent = row["away_team"] if row["home_team"] == team else row["home_team"]
        matchup = get_matchup_info(opponent, position, defense_df)
        mult = matchup["multiplier"]
        multipliers.append(mult)
        weeks_info.append({
            "week":     wk,
            "opponent": opponent,
            "matchup":  matchup["rank_label"],
            "multiplier": mult,
        })

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
