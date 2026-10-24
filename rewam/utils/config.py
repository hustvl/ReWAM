from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from omegaconf import DictConfig, OmegaConf


def as_plain_dict(
    cfg: Any,
    *,
    required: bool = False,
) -> dict[str, Any]:
    """Resolve a mapping config, or use an empty dict for an optional section."""
    if cfg is None:
        if required:
            raise ValueError("A non-null dict config is required.")
        return {}
    if isinstance(cfg, DictConfig):
        cfg = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(cfg, Mapping):
        raise ValueError(f"Config must resolve to a dict, got {type(cfg)}")
    return dict(cfg)


def as_thw(value: Any) -> tuple[int, int, int]:
    if len(value) != 3 or any(int(size) <= 0 or int(size) != size for size in value):
        raise ValueError(f"Expected positive integer patch_size [T,H,W], got {value}.")
    return int(value[0]), int(value[1]), int(value[2])
