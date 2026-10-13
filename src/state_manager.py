import json
import os

LEAGUES_DIR = "leagues"

def _ensure_leagues_dir():
    if not os.path.exists(LEAGUES_DIR):
        os.makedirs(LEAGUES_DIR)

def get_league_list():
    _ensure_leagues_dir()
    files = [f.replace(".json", "") for f in os.listdir(LEAGUES_DIR) if f.endswith(".json")]
    return files if files else ["default_league"]

def get_roster(league_name="default_league"):
    _ensure_leagues_dir()
    path = os.path.join(LEAGUES_DIR, f"{league_name}.json")
    if not os.path.exists(path):
        default_data = {"team_name": league_name, "budget": 100, "roster": []}
        update_roster(default_data, league_name)
        return default_data
    with open(path, 'r') as f:
        return json.load(f)

def update_roster(data, league_name="default_league"):
    _ensure_leagues_dir()
    path = os.path.join(LEAGUES_DIR, f"{league_name}.json")
    with open(path, 'w') as f:
        json.dump(data, f, indent=4)
