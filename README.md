# DNN Compression — Dendritic Network with Int8 Quantization

A research project exploring lossless compression of biologically-inspired dendritic neural networks on tabular classification tasks. The core finding: **per-layer int8 quantization achieves ~4× compression with zero accuracy loss**, preserving the network's unique "snowflake" property where each branch learns distinct feature representations.

---

## Architecture

### Dendritic Network

```
Input
  └── FC1 (input_dim → hidden_neurons1)
        ├── Branch 0 (hidden_neurons1 → hidden_per_branch)
        ├── Branch 1 (hidden_neurons1 → hidden_per_branch)
        ├── ...
        └── Branch N (hidden_neurons1 → hidden_per_branch)
  └── Concat → FC2 (branches × hidden_per_branch → hidden_neurons2)
  └── Output (hidden_neurons2 → 1, sigmoid)
```

Each branch operates in parallel on the same layer-1 activations, learning different feature subspaces (the "snowflake" property). This diversity is what allows quantization to be lossless — topology sharing (forcing all branches to share weights) destroys this and causes accuracy loss.

### Compression Pipeline

1. **Per-layer int8 quantization** — one shared scale per layer group (weight + bias). Weights stored as `int8` (1 byte each) plus one `float32` scale per layer.
2. **Optional fine-tuning** — 3 epochs of post-quantization fine-tuning on training data before re-quantizing.
3. Achieves ~4× size reduction; compression ratio depends on model size.

---

## Project Structure

```
dnn_compression/
├── main.py                          # Entry point — argparse CLI
├── run.sh                           # Shell wrapper: python main.py "$@"
│
├── src/
│   ├── models/
│   │   ├── dendritic_network.py     # DendriticNetwork (main model)
│   │   ├── dendritic_neuron.py      # Single dendritic neuron module
│   │   └── mlp_baseline.py          # MLP baseline for comparison
│   │
│   ├── compression/
│   │   ├── compression_pipeline.py  # compress_model, decompress_model
│   │   └── topology_sharing.py      # Branch weight sharing (not used in pipeline)
│   │
│   ├── training/
│   │   ├── train.py                 # Training loop
│   │   └── evaluate.py              # Accuracy evaluation
│   │
│   ├── data/
│   │   ├── load_wine.py             # Wine dataset (sklearn)
│   │   ├── load_adult.py            # UCI Adult Income (OpenML)
│   │   ├── load_folktables.py       # Folktables ACS Income
│   │   └── load_creditcard.py       # CC Fraud (Kaggle CSV — see below)
│   │
│   ├── experiments/
│   │   ├── wine_experiment.py
│   │   ├── uci_adult_experiment.py
│   │   ├── folktables_experiment.py
│   │   ├── folktables_multistate_experiment.py
│   │   ├── creditcard_experiment.py
│   │   ├── ablation_study.py
│   │   └── scaling_experiment.py
│   │
│   └── plots/
│       ├── plot_accuracy.py
│       ├── plot_compression.py
│       ├── plot_ablation.py
│       ├── plot_scaling.py
│       ├── plot_folktables_multistate.py
│       ├── plot_roc_pr.py
│       ├── plot_training_curves.py
│       ├── folktables_plots.py
│       └── save_utils.py
│
└── figures/                         # Auto-generated plots saved here
```

---

## Setup

```bash
pip install torch scikit-learn pandas numpy matplotlib folktables tqdm
```

### Credit Card Fraud dataset

Download `creditcard.csv` from [kaggle.com/mlg-ulb/creditcardfraud](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and place it at:

```
src/data/creditcard.csv
```

All other datasets are downloaded automatically (sklearn, OpenML, Folktables).

---

## Usage

```bash
# Run all experiments
python main.py

# Run a single experiment
python main.py --exp wine
python main.py --exp fraud

# Run a subset
python main.py --exp wine adult folktables

# Override epochs (default: 50)
python main.py --exp wine --epochs 10
```

Available experiment keys:

| Key | Description |
|---|---|
| `wine` | Wine dataset (sklearn) |
| `adult` | UCI Adult Income |
| `folktables` | Folktables CA 2018 ACS Income |
| `multistate` | Folktables: train CA, test CA/TX/NY/FL/WA |
| `fraud` | Credit Card Fraud Detection (Kaggle) |
| `ablation` | Architecture ablation across 3 configs |
| `component` | Compression component ablation (quant vs topo vs both) |
| `scaling` | Grid search over hidden sizes and branch counts |

---

## Experiments

### Standard (wine, adult, folktables, fraud)

Each runs over multiple seeds, reporting:
- Uncompressed accuracy ± std
- Compressed accuracy ± std
- MLP baseline accuracy ± std
- Model size before and after compression

### Component Ablation

Tests four compression conditions on the same architecture:

| Condition | What it does |
|---|---|
| `none` | No compression |
| `topo_only` | Topology sharing only (copies branch[0] weights to all branches) |
| `quant_only` | Int8 quantization only |
| `both` | Topology sharing + quantization |

Key result: `quant_only` matches `none` in accuracy; `topo_only` causes a drop.

### Scaling Experiment

Grid search over `hidden_neurons1`, `hidden_neurons2`, and `branches`. Produces heatmaps per branch count showing:
- Uncompressed accuracy
- Compressed accuracy
- Compression ratio
- Time per config

### Folktables Multi-State

Trains on California (CA) 2018 ACS data, evaluates on CA, TX, NY, FL, WA. Tests whether quantization hurts cross-state generalisation.

---

## Outputs

All plots are saved to `figures/`. Summary statistics are printed to stdout after all experiments complete.
