"""MEGA-ODE model definitions."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torchdyn.core import NeuralODE

from .data import MegaODEData, MegaODEGraph
from .metrics import combined_loss
from .utils import get_device, set_seed


class GATODEMLP(nn.Module):
    """GAT + Neural ODE expert module from the original notebook."""

    def __init__(
        self,
        in_feats: int,
        gat_hidden_feats: int,
        out_feats: int,
        time_tick_num: int,
        heads: int = 2,
        dropout: float = 0.0,
        ode_hidden_size: int = 256,
    ) -> None:
        super().__init__()
        self.dropout = dropout
        self.heads = heads
        self.time_tick_num = time_tick_num
        self.gatconv1 = GATConv(in_channels=in_feats, out_channels=gat_hidden_feats, heads=self.heads)
        self.gatconv11 = GATConv(in_channels=gat_hidden_feats * self.heads, out_channels=gat_hidden_feats, heads=1)
        activation = nn.LeakyReLU(0.1)
        func_ode = nn.Sequential(
            nn.Linear(gat_hidden_feats, ode_hidden_size),
            activation,
            nn.Linear(ode_hidden_size, ode_hidden_size),
            activation,
            nn.Linear(ode_hidden_size, gat_hidden_feats),
        )
        self.neuralDE = NeuralODE(func_ode, solver="rk4")
        self.linear1 = nn.Linear((self.time_tick_num - 1) * gat_hidden_feats, (self.time_tick_num - 1) * gat_hidden_feats)
        self.linear11 = nn.Linear((self.time_tick_num - 1) * gat_hidden_feats, (self.time_tick_num - 1) * out_feats)

    def forward(self, h: torch.Tensor, edge_index: torch.Tensor, return_attention_weights: bool = True):
        gath = self.gatconv1(h, edge_index)
        gath = F.leaky_relu(gath, 0.2)
        gath = F.dropout(gath, p=self.dropout, training=self.training)
        gath, attenmat = self.gatconv11(gath, edge_index, return_attention_weights=return_attention_weights)
        t_span = torch.linspace(0, self.time_tick_num - 1, self.time_tick_num, device=h.device)
        _, ode_h = self.neuralDE(gath, t_span)
        ode_h = ode_h[1:]
        h_hidden = torch.transpose(ode_h, 0, 1)
        h_hidden_reshape = h_hidden.reshape(h_hidden.size(0), -1)
        out = self.linear1(h_hidden_reshape)
        out = F.leaky_relu(out, 0.2)
        out = F.dropout(out, p=self.dropout, training=self.training)
        out = self.linear11(out)
        out = out.view(self.time_tick_num - 1, out.size(0), -1)
        return out, attenmat


class ODE_MOE(nn.Module):
    """Mixture-of-experts wrapper over GATODEMLP experts."""

    def __init__(
        self,
        in_feats: int,
        hidden_size: int,
        num_experts: int,
        out_feats: int,
        time_tick_num: int,
        gate_hidden_dim: int,
        heads: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.experts = nn.ModuleList([
            GATODEMLP(in_feats, hidden_size, out_feats, time_tick_num, heads=heads, dropout=dropout)
            for _ in range(num_experts)
        ])
        self.gating = nn.Sequential(
            nn.Linear(in_feats, gate_hidden_dim),
            nn.ReLU(),
            nn.Linear(gate_hidden_dim, gate_hidden_dim),
            nn.ReLU(),
            nn.Linear(gate_hidden_dim, num_experts),
            nn.Softmax(dim=1),
        )
        self.time_tick_num = time_tick_num

    def forward(self, h: torch.Tensor, edge_index: torch.Tensor):
        gating_scores = self.gating(h)
        expert_outputs = []
        expert_attn_weights = []
        for i, expert in enumerate(self.experts):
            expert_output, expert_atten = expert(h, edge_index)
            edgemat, attenmat = expert_atten
            expert_attn_weights.append((edgemat.detach().cpu(), attenmat.detach().cpu()))
            expanded_gates = gating_scores[:, i].view(1, -1, 1).expand(self.time_tick_num - 1, -1, -1)
            expert_outputs.append(expert_output * expanded_gates)
        stacked_outputs = torch.stack(expert_outputs, dim=0)
        return torch.sum(stacked_outputs, dim=0), gating_scores, expert_attn_weights


class MEGAODE(nn.Module):
    """User-facing MEGA-ODE estimator."""

    def __init__(
        self,
        in_feats: int,
        hidden_size: int = 64,
        num_experts: int = 4,
        out_feats: Optional[int] = None,
        time_tick_num: int = 5,
        gate_hidden_dim: int = 32,
        heads: int = 2,
        dropout: float = 0.0,
        device: Optional[Union[str, torch.device]] = None,
        seed: Optional[int] = 123,
    ) -> None:
        super().__init__()
        if seed is not None:
            set_seed(seed)
        self.in_feats = in_feats
        self.out_feats = out_feats or in_feats
        self.time_tick_num = time_tick_num
        self.device = get_device(device)
        self.network = ODE_MOE(
            in_feats=in_feats,
            hidden_size=hidden_size,
            num_experts=num_experts,
            out_feats=self.out_feats,
            time_tick_num=time_tick_num,
            gate_hidden_dim=gate_hidden_dim,
            heads=heads,
            dropout=dropout,
        ).to(self.device)
        self.history_: Dict[str, List[float]] = {"loss": []}

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):
        return self.network(x.to(self.device), edge_index.to(self.device))

    def fit(
        self,
        data: MegaODEData,
        epochs: int = 100,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        input_key: str = "x12",
        target_keys: Sequence[str] = ("x24", "x36", "x72"),
        alpha: float = 0.3,
        patience: Optional[int] = 50,
        verbose: bool = True,
        verbose_step: int = 10,
    ) -> "MEGAODE":
        optimizer = torch.optim.Adam(self.network.parameters(), lr=lr, weight_decay=weight_decay)
        best_loss = float("inf")
        stale_epochs = 0
        self.history_ = {"loss": []}
        for epoch in range(epochs):
            self.network.train()
            total_loss = 0.0
            for graph in data.graphs:
                x = graph.ndata[input_key].to(self.device)
                edge_index = graph.edge_index.to(self.device)
                outputs, _, _ = self.network(x, edge_index)
                y_pred = torch.stack([outputs[i] for i in range(len(target_keys))], dim=0)
                y_true = torch.stack([graph.ndata[key].to(self.device) for key in target_keys], dim=0)
                loss = combined_loss(y_pred, y_true, alpha=alpha)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += float(loss.detach().cpu())
            avg_loss = total_loss / max(len(data), 1)
            self.history_["loss"].append(avg_loss)
            if verbose and epoch % verbose_step == 0:
                print(f"Epoch [{epoch}], Training Loss: {avg_loss:3.3f}")
            if patience is not None:
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    stale_epochs = 0
                else:
                    stale_epochs += 1
                    if stale_epochs > patience:
                        if verbose:
                            print(f"Early stopping with best_train_loss: {best_loss:3.3f} and train_loss: {avg_loss:3.3f}")
                        break
        return self

    @torch.no_grad()
    def predict(self, data: Union[MegaODEData, MegaODEGraph], input_key: str = "x36", output_index: Optional[int] = None):
        self.network.eval()

        def _predict_graph(graph: MegaODEGraph) -> torch.Tensor:
            outputs, _, _ = self.network(graph.ndata[input_key].to(self.device), graph.edge_index.to(self.device))
            if output_index is not None:
                outputs = outputs[output_index]
            return outputs.detach().cpu()

        if isinstance(data, MegaODEGraph):
            return _predict_graph(data)
        return [_predict_graph(graph) for graph in data.graphs]

    def save(self, path: Union[str, Path]) -> None:
        torch.save(self.network.state_dict(), path)

    def load(self, path: Union[str, Path], map_location: Optional[str] = None) -> "MEGAODE":
        self.network.load_state_dict(torch.load(path, map_location=map_location or self.device))
        return self
