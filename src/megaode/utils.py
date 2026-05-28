"""Small utility helpers for MEGA-ODE."""

from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch

PathLike = Union[str, Path]


def as_path(path: PathLike) -> Path:
    """Return a normalized Path object."""
    return Path(path).expanduser().resolve()


def get_device(device: Optional[Union[str, torch.device]] = None) -> torch.device:
    """Resolve a torch device, defaulting to CUDA when available."""
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int = 123, deterministic: bool = False) -> None:
    """Set random seeds for reproducible demo runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_logger(name: str = "megaode", level: int = logging.INFO) -> logging.Logger:
    """Create a simple console logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(levelname)s:%(name)s:%(message)s"))
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger
