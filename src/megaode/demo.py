"""Runnable demo for MEGA-ODE."""

from __future__ import annotations

import argparse
from typing import Dict, Optional

import torch

from .data import load_demo_data
from .metrics import evaluate_prediction
from .model import MEGAODE
from .utils import PathLike, set_seed


def run_demo(
    data_dir: PathLike = "demo_data",
    epochs: int = 5,
    hidden_size: int = 16,
    num_experts: int = 2,
    gate_hidden_dim: int = 16,
    lr: float = 1e-3,
    seed: int = 123,
    max_nodes: Optional[int] = 300,
    ode_function: str = "gat",
) -> Dict[str, object]:
    """Load demo data, train a small MEGA-ODE model, and predict y96."""
    set_seed(seed)
    data = load_demo_data(data_dir, max_nodes=max_nodes)
    model = MEGAODE(
        in_feats=data.num_features,
        hidden_size=hidden_size,
        num_experts=num_experts,
        out_feats=data.num_features,
        time_tick_num=len(data.input_timepoints) + 1,
        gate_hidden_dim=gate_hidden_dim,
        ode_function=ode_function,
        seed=seed,
    )
    model.fit(data, epochs=epochs, lr=lr, patience=None, verbose=True)
    pred = model.predict(data, input_key="x36", output_index=3)
    truth = [graph.ndata["y96"] for graph in data.graphs]
    metrics = [evaluate_prediction(p.squeeze(-1), y.squeeze(-1)) for p, y in zip(pred, truth)]
    mean_mse = float(torch.tensor([m["mse"] for m in metrics]).mean())
    print(f"Demo complete. Mean MSE: {mean_mse:3.4f}")
    return {"data": data, "model": model, "prediction": pred, "metrics": metrics}


def main() -> None:
    """Run the bundled demo from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="demo_data")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--max-nodes", type=int, default=300)
    parser.add_argument(
        "--ode-function",
        "--ode_function",
        dest="ode_function",
        choices=("gat", "mlp"),
        default="gat",
        help="ODE vector field architecture (default: gat).",
    )
    args = parser.parse_args()
    run_demo(
        data_dir=args.data_dir,
        epochs=args.epochs,
        max_nodes=args.max_nodes,
        ode_function=args.ode_function,
    )


if __name__ == "__main__":
    main()
