import logging
import os

import torch.distributed as dist
from rich.logging import RichHandler


def setup_logging(log_level: int = logging.INFO, is_main_process: bool | None = None) -> None:
    """Configure the root logger; worker ranks emit only errors."""
    if is_main_process is None:
        is_main_process = _is_main_process()

    root_logger = logging.getLogger()
    if not is_main_process:
        root_logger.setLevel(logging.ERROR)
        return

    # Hydra owns its file handlers; close only the handlers we replace.
    for handler in root_logger.handlers[:]:
        if not isinstance(handler, logging.FileHandler):
            root_logger.removeHandler(handler)
            handler.close()

    handler = RichHandler(
        markup=False,
        rich_tracebacks=True,
        show_level=True,
        show_path=True,
        show_time=True,
    )
    handler.setFormatter(logging.Formatter(fmt="%(message)s", datefmt="%m/%d [%H:%M:%S]"))
    root_logger.addHandler(handler)
    root_logger.setLevel(log_level)


def _is_main_process() -> bool:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank() == 0
    for key in ("RANK", "SLURM_PROCID", "LOCAL_RANK"):
        if key in os.environ:
            return os.environ[key].strip() in {"", "0"}
    return True
