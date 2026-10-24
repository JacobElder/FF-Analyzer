import socket
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # shared helpers in tests/

import src.state_manager as sm  # noqa: E402
import src.storage as storage  # noqa: E402

_ENV_VARS = ["SLEEPER_USERNAME", "SLEEPER_LEAGUE_IDS", "SLEEPER_AUTO_SYNC_MINUTES",
             "ANTHROPIC_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY", "ODDS_API_KEY",
             "CLAUDE_MODEL", "GEMINI_MODEL"]


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    """Every test gets temp data/leagues dirs, a clean env, and no network."""
    data = tmp_path / "data"
    monkeypatch.setattr(storage, "DATA_DIR", data)
    monkeypatch.setattr(sm, "LEAGUES_DIR", tmp_path / "leagues")
    for var in _ENV_VARS:
        monkeypatch.delenv(var, raising=False)

    def _blocked(*a, **k):
        raise RuntimeError("Network access is blocked in tests — mock the request.")
    monkeypatch.setattr(socket.socket, "connect", _blocked)
    monkeypatch.setattr(socket, "create_connection", _blocked)
    return tmp_path


@pytest.fixture
def leagues_dir(isolated):
    path = isolated / "leagues"
    path.mkdir(exist_ok=True)
    return path
