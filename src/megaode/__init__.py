"""Public API for MEGA-ODE."""

from .data import MegaODEData, MegaODEGraph, load_demo_data, load_megaode_data
from .demo import run_demo
from .metrics import evaluate_prediction
from .model import GATODEMLP, MEGAODE, ODE_MOE
from .predict import predict_response
from .train import train_model

MegaODE = MEGAODE

__all__ = [
    "GATODEMLP",
    "MEGAODE",
    "MegaODE",
    "MegaODEData",
    "MegaODEGraph",
    "ODE_MOE",
    "evaluate_prediction",
    "load_demo_data",
    "load_megaode_data",
    "predict_response",
    "run_demo",
    "train_model",
]
