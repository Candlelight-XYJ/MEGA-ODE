"""Data loading utilities for MEGA-ODE."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Union

import pandas as pd
import torch

from .utils import PathLike, as_path

DEFAULT_INPUT_TIMEPOINTS = ("12", "24", "36", "72")
DEFAULT_TARGET_TIMEPOINTS = ("96",)


@dataclass
class MegaODEGraph:
    """One perturbation or experiment graph with node time-series features."""

    ndata: Dict[str, torch.Tensor]
    edge_index: torch.Tensor
    edge_weight: Optional[torch.Tensor] = None
    experiment_type: Optional[Union[str, int]] = None

    def edges(self) -> torch.Tensor:
        """Return PyG-style edge_index with shape [2, num_edges]."""
        return self.edge_index

    @property
    def num_nodes(self) -> int:
        return next(iter(self.ndata.values())).shape[0]


@dataclass
class MegaODEData:
    """Container returned by MEGA-ODE data loaders."""

    graphs: List[MegaODEGraph]
    nodes: pd.DataFrame
    expression: pd.DataFrame
    labels: pd.DataFrame
    edge_table: pd.DataFrame
    input_timepoints: Sequence[str] = DEFAULT_INPUT_TIMEPOINTS
    target_timepoints: Sequence[str] = DEFAULT_TARGET_TIMEPOINTS

    def __len__(self) -> int:
        return len(self.graphs)

    def __getitem__(self, index: int) -> MegaODEGraph:
        return self.graphs[index]

    @property
    def num_features(self) -> int:
        if not self.graphs:
            raise ValueError("Dataset contains no graphs.")
        return self.graphs[0].ndata[f"x{self.input_timepoints[0]}"].shape[1]


def _find_file(data_dir: Path, candidates: Iterable[str]) -> Path:
    for name in candidates:
        path = data_dir / name
        if path.exists():
            return path
    raise FileNotFoundError(f"Could not find any of {list(candidates)} in {data_dir}.")


def _build_edge_index(
    edges: pd.DataFrame,
    nodes: pd.DataFrame,
    relation_col_index: int = 2,
) -> tuple[torch.Tensor, torch.Tensor, pd.DataFrame]:
    node_names = nodes.iloc[:, 0].astype(str)
    node_name_to_index = {name: i for i, name in enumerate(node_names)}
    valid_edges = edges[
        edges.iloc[:, 0].astype(str).isin(node_name_to_index)
        & edges.iloc[:, 1].astype(str).isin(node_name_to_index)
    ].copy()

    if valid_edges.empty:
        source = torch.arange(len(node_names), dtype=torch.long)
        edge_index = torch.stack([source, source], dim=0)
        edge_weight = torch.ones(len(node_names), dtype=torch.float32)
        return edge_index, edge_weight, valid_edges

    source = valid_edges.iloc[:, 0].astype(str).map(node_name_to_index).to_numpy()
    target = valid_edges.iloc[:, 1].astype(str).map(node_name_to_index).to_numpy()
    edge_index = torch.tensor([source, target], dtype=torch.long)
    if relation_col_index < valid_edges.shape[1]:
        edge_weight = torch.tensor(
            valid_edges.iloc[:, relation_col_index].astype(float).to_numpy(),
            dtype=torch.float32,
        )
    else:
        edge_weight = torch.ones(edge_index.shape[1], dtype=torch.float32)
    return edge_index, edge_weight, valid_edges


def load_megaode_data(
    data_dir: PathLike,
    expression_file: Optional[PathLike] = None,
    label_file: Optional[PathLike] = None,
    node_file: Optional[PathLike] = None,
    edge_file: Optional[PathLike] = None,
    input_timepoints: Sequence[Union[int, str]] = DEFAULT_INPUT_TIMEPOINTS,
    target_timepoints: Sequence[Union[int, str]] = DEFAULT_TARGET_TIMEPOINTS,
    relation_col_index: int = 2,
    max_nodes: Optional[int] = None,
) -> MegaODEData:
    """Load expression, label, node, and edge files into MEGA-ODE format."""
    root = as_path(data_dir)
    expression_path = as_path(expression_file) if expression_file is not None else _find_file(root, ["GSE75748_log2CPM_HVG5k_exp.csv", "expr.csv"])
    label_path = as_path(label_file) if label_file is not None else _find_file(root, ["GSE75748_log2CPM_HVG5k_loo_label.csv", "labels.csv"])
    node_path = as_path(node_file) if node_file is not None else _find_file(root, ["GSE75748_log2CPM_HVG5k_nodes.csv", "nodes.csv"])
    edge_path = as_path(edge_file) if edge_file is not None else _find_file(root, ["HumanNet_DBpathway.csv", "edges.csv", "edges.tsv"])

    expression = pd.read_csv(expression_path, header=None)
    labels = pd.read_csv(label_path, header=None)
    nodes = pd.read_csv(node_path, header=None)
    if max_nodes is not None:
        nodes = nodes.iloc[:max_nodes].reset_index(drop=True)
        expression = expression.iloc[:, :max_nodes]

    sep = "\t" if edge_path.suffix.lower() == ".tsv" else ","
    edges = pd.read_csv(edge_path, sep=sep, header=None)
    edge_index, edge_weight, valid_edges = _build_edge_index(edges, nodes, relation_col_index)

    input_keys = [str(t) for t in input_timepoints]
    target_keys = [str(t) for t in target_timepoints]
    graphs: List[MegaODEGraph] = []
    experiment_types = sorted(labels.iloc[:, 0].drop_duplicates().tolist())

    for experiment_type in experiment_types:
        ndata: Dict[str, torch.Tensor] = {}
        for prefix, timepoints in (("x", input_keys), ("y", target_keys)):
            for timepoint in timepoints:
                mask = (labels.iloc[:, 0] == experiment_type) & (labels.iloc[:, 1].astype(int) == int(timepoint))
                if not mask.any():
                    raise ValueError(f"Missing timepoint {timepoint} for experiment {experiment_type}.")
                row_index = labels.index[mask][0]
                values = expression.iloc[row_index].to_numpy(dtype="float32")
                ndata[f"{prefix}{timepoint}"] = torch.tensor(values).unsqueeze(1)
        graphs.append(
            MegaODEGraph(
                ndata=ndata,
                edge_index=edge_index.clone(),
                edge_weight=edge_weight.clone(),
                experiment_type=experiment_type,
            )
        )

    return MegaODEData(
        graphs=graphs,
        nodes=nodes,
        expression=expression,
        labels=labels,
        edge_table=valid_edges,
        input_timepoints=input_keys,
        target_timepoints=target_keys,
    )


def load_demo_data(data_dir: PathLike = "demo_data", max_nodes: Optional[int] = None) -> MegaODEData:
    """Load the bundled hESC demo input data."""
    return load_megaode_data(data_dir, max_nodes=max_nodes)
