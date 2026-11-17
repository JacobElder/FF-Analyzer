"""
File-cache helpers shared by the data modules.

All paths are anchored at the project root (not the current working
directory), and every write goes to a temp file first and is then renamed
into place, so a crash or concurrent rerun never leaves a half-written cache.
"""

import contextlib
import json
import logging
import os
import time
import uuid
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
LEAGUES_DIR = PROJECT_ROOT / "leagues"

log = logging.getLogger(__name__)


class DataUnavailable(RuntimeError):
    """A download failed and there is no cached copy to fall back on."""


def data_path(name: str) -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    return DATA_DIR / name


def is_fresh(path: Path, ttl_seconds: float) -> bool:
    return path.exists() and (time.time() - path.stat().st_mtime) < ttl_seconds


def file_age_seconds(path: Path) -> float | None:
    return time.time() - path.stat().st_mtime if path.exists() else None


def _atomic_write(path: Path, write_fn, mode: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    # O_EXCL + mode 0o666: a unique new file that gets normal (umask-filtered) permissions,
    # without touching the process-wide umask
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o666)
    try:
        with os.fdopen(fd, mode) as f:
            write_fn(f)
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(tmp)
        raise


def write_json_atomic(obj, path: Path, indent: int | None = None) -> None:
    # dumps + one write is ~5x faster than json.dump's many small writes on the 16 MB player file
    _atomic_write(path, lambda f: f.write(json.dumps(obj, indent=indent)), "w")


def read_json(path: Path, default=None):
    """Parsed JSON, or `default` when the file is truncated or corrupt (missing files still raise)."""
    with open(path, "r") as f:
        try:
            return json.load(f)
        except ValueError as e:  # JSONDecodeError / UnicodeDecodeError
            log.warning("Ignoring unreadable JSON file %s: %s", path.name, e)
            return default


def write_parquet_atomic(df: pd.DataFrame, path: Path) -> None:
    _atomic_write(path, lambda f: df.to_parquet(f, index=False), "wb")
