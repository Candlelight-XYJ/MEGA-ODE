# MEGA-ODE

MEGA-ODE: Mixture-Enhanced Graph Attention ODE for Generalizable and Interpretable Modeling of Omics Dynamics.

MEGA-ODE is a multi-level graph-based framework for generalizable and interpretable modeling of omics dynamics. It embeds prior biological knowledge by defining the ODE state space on a molecular interaction network parameterized by graph attention networks, and uses a mixture-of-experts architecture to capture heterogeneous dynamic behaviors across perturbations, timepoints, and molecular programs.

## Collaborators

- [Candlelight-XYJ](https://github.com/Candlelight-XYJ)
- [yonggeli66](https://github.com/yonggeli66)


## Highlights

- MoE-based GraphODE for modeling heterogeneous perturbation dynamics.
- Graph attention over prior biological networks for interpretable molecular interactions.
- Generalization across unseen perturbations, unseen timepoints, and unseen molecular features.
- Expert-level interpretation through gating scores and attention weights.
- Applications to pathway-level causal structure, drug perturbation response, disease-severity gene prioritization, and stem-cell differentiation programs.

## Installation

Clone the repository and install MEGA-ODE in editable mode:

```bash
git clone https://github.com/Candlelight-XYJ/MEGA-ODE.git
cd MEGA-ODE
pip install -e .
```

A conda environment file is also provided:

```bash
conda env create -f environment.yml
conda activate megaode
pip install -e .
```

## Quick Start

Run the bundled demo with one function:

```python
from megaode import run_demo

result = run_demo("demo_data/")
```

Use the model API directly:

```python
from megaode import MEGAODE, load_demo_data

data = load_demo_data("demo_data/")
model = MEGAODE(
    in_feats=data.num_features,
    hidden_size=64,
    num_experts=4,
    out_feats=data.num_features,
    time_tick_num=5,
    gate_hidden_dim=32,
)

model.fit(data, epochs=100, lr=1e-3)
pred = model.predict(data, input_key="x36", output_index=3)
```

## Demo

The demo follows the original notebook workflow while exposing it through importable package APIs.

```python
from megaode import run_demo

result = run_demo(
    data_dir="demo_data/",
    epochs=5,
    hidden_size=16,
    num_experts=2,
    max_nodes=300,
)

model = result["model"]
prediction = result["prediction"]
metrics = result["metrics"]
```

`run_demo()` returns a dictionary containing:

- `data`: loaded `MegaODEData` object.
- `model`: trained `MEGAODE` model.
- `prediction`: predicted response tensors.
- `metrics`: per-graph evaluation metrics.

`run_demo()` uses a node subset by default so the example can run quickly. Use `load_demo_data("demo_data/", max_nodes=None)` to load all demo nodes.

## Demo Data

The `demo_data/` directory contains the hESC time-series example exported from the original notebook:

- `GSE75748_log2CPM_HVG5k_exp.csv`: expression matrix; rows are samples/timepoints and columns are genes.
- `GSE75748_log2CPM_HVG5k_loo_label.csv`: sample labels containing experiment id and timepoint.
- `GSE75748_log2CPM_HVG5k_nodes.csv`: gene/node names matching expression columns.
- `HumanNet_DBpathway.csv`: HumanNet-derived graph edges with optional edge weights.

## Input Data Format

Custom datasets can be loaded with `load_megaode_data()` using the same four-file layout:

```python
from megaode import load_megaode_data

data = load_megaode_data(
    data_dir="demo_data/",
    expression_file="expr.csv",
    label_file="labels.csv",
    node_file="nodes.csv",
    edge_file="edges.csv",
)
```

Expected input files:

- Expression file: numeric matrix with samples/timepoints as rows and genes as columns.
- Label file: at least two columns, experiment id and integer timepoint.
- Node file: one gene/node name per expression column.
- Edge file: source node, target node, and optional edge weight.

The default training setup follows the original demo: train from `x12` to `x24`, `x36`, and `x72`, then predict `y96` from `x36`.

## Output Explanation

`MEGAODE.predict()` returns one tensor per graph. Without `output_index`, each tensor has shape:

```text
[time_tick_num - 1, num_nodes, out_feats]
```

For the bundled demo, `output_index=3` corresponds to the final predicted step used for `y96` evaluation.

Evaluation utilities are available through:

```python
from megaode import evaluate_prediction
```

The package currently includes MSE, RMSE, R2, Pearson correlation, cosine similarity, Wasserstein distance, and KL divergence helpers.

## Repository Structure

```text
MEGA-ODE/
docs/                 # GitHub Pages website
demo_data/            # Small demo input data
examples/             # User-facing demo notebook
src/megaode/          # Python package source code
tests/                # Import and data-loading smoke tests
pyproject.toml
requirements.txt
environment.yml
README.md
LICENSE
```

## Validation

After installing on a server, run:

```bash
python -c "import megaode; print('MEGA-ODE import successful')"
python -c "import torch, dgl, torchdyn, torch_geometric; print('core dependencies import successful')"
python -m pytest tests
```

## Citation

Citation information will be added after manuscript release.

## License

See `LICENSE` for license information.