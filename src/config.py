"""
App configuration: config.toml at the project root, overridable by environment
variables (.env).
"""

import os
import tomllib

from src.storage import PROJECT_ROOT

CONFIG_PATH = PROJECT_ROOT / "config.toml"


def load_config(path=CONFIG_PATH) -> dict:
    cfg = {}
    if path.exists():
        with open(path, "rb") as f:
            cfg = tomllib.load(f)
    sleeper = cfg.get("sleeper", {})

    env_ids = os.getenv("SLEEPER_LEAGUE_IDS")
    if env_ids:
        league_ids = [x.strip() for x in env_ids.split(",") if x.strip()]
    else:
        league_ids = [str(x) for x in sleeper.get("league_ids", [])]

    return {
        "username": os.getenv("SLEEPER_USERNAME") or sleeper.get("username", ""),
        "league_ids": league_ids,
        "auto_sync_minutes": float(os.getenv("SLEEPER_AUTO_SYNC_MINUTES") or sleeper.get("auto_sync_minutes", 30)),
    }
