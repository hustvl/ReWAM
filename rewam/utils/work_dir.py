from __future__ import annotations

import os

_WORK_DIR: str | None = None
_DEFAULT_WORK_DIR = "./runs/"


def register_work_dir(path: str | os.PathLike | None) -> None:
    global _WORK_DIR
    resolved = str(path) if path is not None else _DEFAULT_WORK_DIR
    os.makedirs(resolved, exist_ok=True)
    _WORK_DIR = resolved


def get_work_dir() -> str:
    return _WORK_DIR if _WORK_DIR is not None else _DEFAULT_WORK_DIR
