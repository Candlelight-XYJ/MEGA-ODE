"""Public API for MEGA-ODE."""

from .data import MegaODEData, MegaODEGraph, load_demo_data, load_megaode_data
from .demo import run_demo
from .metrics import evaluate_prediction
from .model import (
    GATODEFunc,
    GATODEMLP,
    MEGAODE,
    MLPODEFunc,
    ODE_MOE,
    build_ode_function,
)
from .predict import predict_response
from .train import train_model

MegaODE = MEGAODE

__all__ = [
    "GATODEFunc",
    "GATODEMLP",
    "MEGAODE",
    "MLPODEFunc",
    "MegaODE",
    "MegaODEData",
    "MegaODEGraph",
    "ODE_MOE",
    "build_ode_function",
    "evaluate_prediction",
    "load_demo_data",
    "load_megaode_data",
    "predict_response",
    "run_demo",
    "train_model",
]
