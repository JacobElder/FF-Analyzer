"""
File-cache helpers shared by the data modules.

All paths are anchored at the project root (not the current working
directory), and every write goes to a temp file first and is then renamed
into place, so a crash or concurrent rerun never leaves a half-written cache.
"""

import contextlib
import json
import os
import tempfile
import time
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
LEAGUES_DIR = PROJECT_ROOT / "leagues"

_UMASK = os.umask(0)  # os has no umask getter: read it by setting, then restore
os.umask(_UMASK)


def data_path(name: str) -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    return DATA_DIR / name


def is_fresh(path: Path, ttl_seconds: float) -> bool:
    return path.exists() and (time.time() - path.stat().st_mtime) < ttl_seconds


def file_age_seconds(path: Path) -> float | None:
    return time.time() - path.stat().st_mtime if path.exists() else None


def _atomic_write(path: Path, write_fn, mode: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, mode) as f:
            write_fn(f)
        os.chmod(tmp, 0o666 & ~_UMASK)  # mkstemp creates 0600; match normal file permissions
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(tmp)
        raise


def write_json_atomic(obj, path: Path, indent: int | None = None) -> None:
    _atomic_write(path, lambda f: json.dump(obj, f, indent=indent), "w")


def read_json(path: Path):
    with open(path, "r") as f:
        return json.load(f)


def write_parquet_atomic(df: pd.DataFrame, path: Path) -> None:
    _atomic_write(path, lambda f: df.to_parquet(f, index=False), "wb")
