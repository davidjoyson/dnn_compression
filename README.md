# DNN Compression — Dendritic Network with Int8 Quantization

A research project exploring lossless compression of biologically-inspired dendritic neural networks on real-world tabular/time-series classification tasks.

**Core finding:** Per-layer int8 quantization (Snowflake) achieves **~4× compression with zero accuracy loss** across 4 datasets, while global and dynamic int8 quantization degrade accuracy — validating that per-layer calibration is essential for small models.

---

## Problem Statement

Neural networks deployed on edge devices (wearables, microcontrollers) are constrained by memory. Standard compression methods (pruning, global quantization) degrade accuracy, especially on small models (<200k params). This project asks:

> Can a biologically-inspired dendritic architecture be compressed 4× with no accuracy loss, outperforming standard quantization baselines?

The dendritic network's "snowflake" property — parallel branches each learning distinct feature subspaces — is hypothesised to be especially compatible with per-layer quantization, since each branch's weight distribution is narrow and independently calibrated.

---

## Architecture

### DendriticNetwork

```
Input
  └── FC1 (input_dim → hidden_neurons1)           [shared trunk]
        ├── Branch 0 (hidden_neurons1 → hidden_per_branch)
        ├── Branch 1 (hidden_neurons1 → hidden_per_branch)   [parallel branches]
        ├── ...
        └── Branch N (hidden_neurons1 → hidden_per_branch)
  └── Soma (branches × hidden_per_branch → branches)        [dendritic integration]
  └── FC2 (branches → hidden_neurons2)
  └── Output (hidden_neurons2 → num_classes)
```

Each branch operates in parallel on the same FC1 activations, learning different feature subspaces. The soma layer integrates each branch's output to a single signal, mimicking biological dendritic integration.

### Compression Pipeline

Three methods evaluated head-to-head:

| Method | Description | Size ratio |
|---|---|---|
| **Snowflake (int8)** | Per-layer int8 — one scale per layer group (weight + bias) | **4×** |
| Global int8 | Single global scale across all parameters | 4× |
| Dynamic int8 | PyTorch `quantize_dynamic` on Linear layers | ~4× |

All methods optionally followed by 3 epochs of post-quantization fine-tuning. Compared against a param-matched MLP baseline.

---

## Results — 50 epochs, 3 seeds (42, 0, 7)

| Dataset | Classes | Uncompressed | Snowflake (4×) | Delta | Significant? |
|---|---|---|---|---|---|
| UCI HAR | 6 | 97.94% ±0.32% | 97.93% ±0.34% | -0.02% | n.s. |
| ECG Heartbeat | 5 | 96.08% ±0.43% | **96.60% ±0.42%** | **+0.53%** | n.s. |
| EEG Brainwave | 3 | 97.66% ±0.23% | 97.58% ±0.14% | -0.08% | n.s. |
| HAPT | 12 | 92.45% ±0.52% | **92.80% ±0.62%** | **+0.35%** | n.s. |

Snowflake matches or beats uncompressed on all 4 datasets. No statistically significant degradation (paired t-test, n=3). Dynamic int8 on EEG is the only significant failure across all methods: -2.42%, p=0.001.

---

## Project Structure

```
dnn_compression/
├── main.py                              # Entry point — argparse CLI
├── docs/
│   └── experiment_log.md               # Full run history and findings
│
├── src/
│   ├── models/
│   │   ├── dendritic_network.py         # DendriticNetwork (main model)
│   │   └── mlp_baseline.py              # Param-matched MLP baseline
│   │
│   ├── compression/
│   │   ├── compression_pipeline.py      # compress_model / decompress_model
│   │   └── topology_sharing.py          # Branch weight sharing (ablation only)
│   │
│   ├── training/
│   │   ├── train.py                     # Training loop
│   │   └── evaluate.py                  # Accuracy, F1, confusion matrix
│   │
│   ├── loaders/
│   │   ├── load_har.py                  # UCI HAR (wearable sensors, 6-class)
│   │   ├── load_ecg.py                  # MIT-BIH ECG heartbeat (5-class)
│   │   ├── load_eeg.py                  # EEG brainwave emotions (3-class)
│   │   └── load_hapt.py                 # UCI HAPT smartphone IMU (12-class)
│   │
│   ├── experiments/
│   │   ├── base_experiment.py           # Shared training + eval loop for all datasets
│   │   ├── har_experiment.py
│   │   ├── ecg_experiment.py
│   │   ├── eeg_experiment.py
│   │   ├── hapt_experiment.py
│   │   └── ablation_study.py            # Architecture + component ablations
│   │
│   ├── reporting/
│   │   ├── summary.py                   # Print summary, significance, edge profile
│   │   ├── plots.py                     # Dispatch all plots per experiment
│   │   └── utils.py                     # CSV export, run dir creation
│   │
│   └── plots/                           # Individual plot modules
│       ├── plot_accuracy.py
│       ├── plot_confusion_matrix.py
│       ├── plot_roc_pr.py
│       ├── plot_compression_delta.py
│       ├── plot_edge_profile.py
│       ├── plot_per_class_f1.py
│       ├── plot_cross_dataset.py
│       ├── plot_pareto.py
│       └── ...
│
└── outputs/                             # Auto-generated per run
    └── run_YYYYMMDD_HHMMSS_<tag>/
        ├── run.log
        ├── metrics.csv
        ├── per_seed_metrics.csv
        ├── summary.txt
        ├── figures/
        └── models/
            ├── har/   (dendritic_uncompressed.pt, dendritic_snowflake.pt, mlp.pt)
            ├── ecg/
            ├── eeg/
            └── hapt/
```

---

## Setup

```bash
pip install torch scikit-learn pandas numpy matplotlib torchinfo tqdm
```

### Datasets

| Dataset | Source | Auto-download? |
|---|---|---|
| UCI HAR | [UCI ML Repository](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones) | Manual — place in `data/har/` |
| ECG Heartbeat | Kaggle `shayanfazeli/heartbeat` | Via Kaggle CLI on first load |
| EEG Brainwave | Kaggle `birdy654/eeg-brainwave-dataset-feeling-emotions` | Via Kaggle CLI on first load |
| UCI HAPT | [UCI ML Repository](https://archive.ics.uci.edu/dataset/341/smartphone+based+recognition+of+human+activities+and+postural+transitions) | Manual — place in `data/hapt/` |

For Kaggle datasets, set up `~/.kaggle/kaggle.json` with your credentials. `.npy` cache files are auto-generated on first load alongside the raw data.

---

## Usage

```bash
# Run all 4 datasets (default)
python main.py

# Run specific experiments
python main.py --exp har ecg

# Override epochs and seeds
python main.py --epochs 50 --seeds 42 0 7

# Run ablation studies (not in default run)
python main.py --exp ablation component

# Print model architecture and parameter counts
python main.py --arch
```

### CLI flags

| Flag | Default | Description |
|---|---|---|
| `--exp` | `har ecg eeg hapt` | Experiments to run |
| `--epochs` | `50` | Training epochs per experiment |
| `--seeds` | `42 0 7` | Random seeds (results averaged) |
| `--fine-tune-epochs` | `3` | Post-quantization fine-tuning epochs |
| `--arch` | — | Print model architectures and exit |

---

## Outputs

Each run creates a timestamped directory under `outputs/`:

- `run.log` — full stdout/stderr mirror
- `metrics.csv` — per-experiment summary stats
- `per_seed_metrics.csv` — per-seed breakdown
- `summary.txt` — human-readable summary with significance and edge profile
- `figures/` — all plots (accuracy, confusion matrix, ROC/PR, compression delta, edge profile, etc.)
- `models/{dataset}/` — best model weights per dataset:
  - `dendritic_uncompressed.pt` — float32 state dict (best seed)
  - `dendritic_snowflake.pt` — compressed quantized dict (best seed)
  - `mlp.pt` — float32 MLP state dict (best seed)
