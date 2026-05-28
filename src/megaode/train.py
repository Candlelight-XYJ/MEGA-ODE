"""Training helpers for MEGA-ODE."""

from __future__ import annotations

from typing import Any, Optional

from .data import MegaODEData
from .model import MEGAODE


def train_model(data: MegaODEData, model: Optional[MEGAODE] = None, **kwargs: Any) -> MEGAODE:
    """Train a MEGAODE model and return it."""
    if model is None:
        model = MEGAODE(
            in_feats=data.num_features,
            out_feats=data.num_features,
            time_tick_num=len(data.input_timepoints) + 1,
        )
    model.fit(data, **kwargs)
    return model

