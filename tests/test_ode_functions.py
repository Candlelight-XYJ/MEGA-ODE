"""Tests for selectable MEGA-ODE vector fields."""

import torch
from torchdyn.core import NeuralODE

from megaode import GATODEFunc, MEGAODE, MLPODEFunc, build_ode_function


def _edge_index() -> torch.Tensor:
    return torch.tensor(
        [[0, 1, 2, 3, 0], [1, 2, 3, 0, 2]],
        dtype=torch.long,
    )


def test_default_and_explicit_ode_function_initialization():
    default_func = build_ode_function(hidden_size=3, edge_index=_edge_index())
    baseline_func = build_ode_function("mlp", hidden_size=3)

    assert isinstance(default_func, GATODEFunc)
    assert isinstance(baseline_func, MLPODEFunc)


def test_ode_function_forward_shapes_and_gradients():
    for name in ("gat", "mlp"):
        x = torch.randn(4, 3, requires_grad=True)
        func = build_ode_function(name, hidden_size=3, edge_index=_edge_index())
        output = func(x)

        assert output.shape == x.shape
        output.square().mean().backward()
        assert x.grad is not None


def test_both_functions_integrate_with_torchdyn():
    for name in ("gat", "mlp"):
        x = torch.randn(4, 3, requires_grad=True)
        func = build_ode_function(name, hidden_size=3, edge_index=_edge_index())
        neural_ode = NeuralODE(func, solver="rk4")
        _, trajectory = neural_ode(x, torch.tensor([0.0, 1.0]))

        assert trajectory.shape == (2, *x.shape)
        trajectory[-1].sum().backward()
        assert x.grad is not None


def test_checkpoint_restores_gat_architecture(tmp_path):
    checkpoint_path = tmp_path / "gat.pt"
    source = MEGAODE(
        in_feats=1,
        hidden_size=2,
        num_experts=1,
        time_tick_num=3,
        gate_hidden_dim=2,
        ode_function="gat",
        device="cpu",
    )
    source.save(checkpoint_path)
    payload = torch.load(checkpoint_path, map_location="cpu")
    assert payload["ode_function"] == "gat"
    assert payload["model_config"]["ode_function"] == "gat"

    restored = MEGAODE(
        in_feats=1,
        hidden_size=2,
        num_experts=1,
        time_tick_num=3,
        gate_hidden_dim=2,
        ode_function="mlp",
        device="cpu",
    ).load(checkpoint_path)

    assert restored.ode_function == "gat"
    assert isinstance(restored.network.experts[0].ode_func, GATODEFunc)


def test_legacy_checkpoint_is_loaded_as_mlp(tmp_path):
    checkpoint_path = tmp_path / "legacy.pt"
    legacy = MEGAODE(
        in_feats=1,
        hidden_size=2,
        num_experts=1,
        time_tick_num=3,
        gate_hidden_dim=2,
        ode_function="mlp",
        device="cpu",
    )
    torch.save(legacy.network.state_dict(), checkpoint_path)

    restored = MEGAODE(
        in_feats=1,
        hidden_size=2,
        num_experts=1,
        time_tick_num=3,
        gate_hidden_dim=2,
        device="cpu",
    ).load(checkpoint_path)

    assert restored.ode_function == "mlp"
    assert isinstance(restored.network.experts[0].ode_func, MLPODEFunc)
