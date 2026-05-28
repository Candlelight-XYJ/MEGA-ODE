"""Prediction helpers for MEGA-ODE."""

from __future__ import annotations

from typing import Any

from .data import MegaODEData
from .model import MEGAODE


def predict_response(model: MEGAODE, data: MegaODEData, **kwargs: Any):
    """Predict responses for a MEGAODEData object."""
    return model.predict(data, **kwargs)
