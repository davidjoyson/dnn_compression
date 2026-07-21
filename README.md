# DNN Compression — Dendritic Network with Int8 Quantization

A research project exploring near-lossless compression of biologically-inspired dendritic neural networks on real-world tabular/time-series classification tasks.

**Core finding:** Per-layer int8 quantization (Snowflake) achieves **~4× compression with no statistically significant accuracy loss** across 3 datasets (HAR, ECG, HAPT), at the model sizes used in the main experiments. TOST equivalence testing (n=10 seeds, ±2% margin) confirms **all 24/24 method–dataset pairs are statistically equivalent**. This is a statistical-equivalence claim, not a guarantee of zero information loss: the architecture-size ablation found a real accuracy drop under compression at the smallest model sizes tested (see `docs/experiment_log.md`, 2026-07-20 entry).

> **Note:** A 4th dataset (EEG brainwave emotions) was dropped 2026-07-21 after investigation confirmed unfixable patient/session-level data leakage in the source data — no subject ID or recoverable session structure exists in the published CSV, and the raw per-subject recordings needed to rebuild the split aren't publicly available for this task. See `docs/experiment_log.md` for the investigation. The loader (`src/loaders/load_eeg.py`) and experiment code are kept in the repo for reference but are no longer wired into the experiment CLI.

---

## Problem Statement

Neural networks deployed on edge devices (wearables, microcontrollers) are constrained by memory. Standard compression methods (pruning, global quantization) degrade accuracy, especially on small models (<200k params). This project asks:

> Can a biologically-inspired dendritic architecture be compressed 4× with no statistically significant accuracy loss, outperforming standard quantization baselines?

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

Seven methods evaluated head-to-head:

| Method | Description | Size ratio |
|---|---|---|
| **Snowflake (int8)** | Per-layer int8 — one scale per layer group (weight + bias) | **4×** |
| Global int8 | Single global scale across all parameters | 4× |
| Dynamic int8 | PyTorch `quantize_dynamic` on Linear layers | ~4× |
| Static (int8) | Per-tensor static calibration via `prepare`/`convert` | ~4× |
| Per-channel (int8) | One scale per output neuron row; biases stay float32 | ~4× |
| QAT (int8) | Quantization-aware training via `prepare_qat_fx`/`convert_fx` | ~4× |
| Mixed precision | Inner layers int8, first and last layers float32 | ~0.9× |

All methods optionally followed by 3 epochs of post-quantization fine-tuning. Compared against a param-matched MLP baseline (2 layers: FC+ReLU → output).

---

## Results — 50 epochs, 10 seeds

| Dataset | Classes | Uncompressed | Snowflake (4×) | Delta | TOST (n=10) |
|---|---|---|---|---|---|
| UCI HAR | 6 | 94.12% ±0.48% | 94.16% ±0.45% | +0.04% | EQUIV |
| ECG Heartbeat | 5 | 96.23% ±0.92% | **96.77% ±0.46%** | **+0.54%** | EQUIV |
| HAPT | 12 | 92.22% ±0.57% | **92.50% ±0.47%** | **+0.28%** | EQUIV |

Snowflake matches or beats uncompressed on all 3 datasets. TOST equivalence testing (±2% margin) confirms all 24/24 method–dataset pairs are equivalent.

---

## Edge Deployment — Raspberry Pi 3

Real single-sample (batch=1) inference latency on a Raspberry Pi 3 Model B (ARM Cortex-A53, `qnnpack` backend), via SSH (`benchmark_pi.py`). Full results in `benchmark_pi_output/`.

| Dataset | Float32 baseline | Snowflake (int8) | Static W+A (int8) | Snowflake+Static (int8) |
|---|---|---|---|---|
| HAR  | 9.01 ms | 9.25 ms (0.97×) | 4.62 ms (**1.95×**) | 4.54 ms (**1.98×**) |
| ECG  | 8.30 ms | 8.33 ms (1.00×) | 4.61 ms (**1.80×**) | 4.70 ms (**1.77×**) |
| HAPT | 8.88 ms | 9.02 ms (0.99×) | 4.52 ms (**1.97×**) | 4.56 ms (**1.95×**) |

**Snowflake gives no real speedup on hardware (~1.0×)** — it's weight-only quantization, so weights are dequantized back to float32 before every matmul; the storage savings (4×) don't translate to compute savings. **Static and Snowflake+Static run true INT8 arithmetic** and deliver a genuine ~1.8–2.2× latency reduction. This is a meaningful distinction the accuracy-only tables above don't capture: for actual edge latency, "true INT8 arithmetic" methods matter, not just int8 *storage*.

**Thermal:** a 15-minute sustained-load test (`thermal_test.py`, Snowflake+Static on ECG, no active cooling) held **229 inf/s with zero throughput degradation**; temperature reached a steady-state **46.2°C by the 5-minute mark** and stayed flat — comfortably below the ~80°C throttle threshold.

**Not yet measured:** real power/energy draw per inference — the Pi 3 has no built-in power ADC (`vcgencmd pmic_read_adc` is Pi 4/5-only), so this needs external hardware (e.g. INA219) not currently available. Temperature was used as a free thermal-risk proxy instead. These numbers also validate an ARM Linux SBC, not bare-metal microcontroller-class hardware (e.g. TFLite Micro on ESP32) — a different deployment target not yet attempted.

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
│   │   ├── load_eeg.py                  # EEG brainwave emotions (3-class) — unused, kept for reference
│   │   └── load_hapt.py                 # UCI HAPT smartphone IMU (12-class)
│   │
│   ├── experiments/
│   │   ├── base_experiment.py           # Shared training + eval loop for all datasets
│   │   ├── har_experiment.py
│   │   ├── ecg_experiment.py
│   │   ├── eeg_experiment.py            # unused, kept for reference (see leakage note above)
│   │   ├── hapt_experiment.py
│   │   └── ablation_study.py            # Architecture + component ablations
│   │
│   ├── analysis/
│   │   └── tost.py                      # TOST equivalence testing + 95% CI helpers
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
| UCI HAPT | [UCI ML Repository](https://archive.ics.uci.edu/dataset/341/smartphone+based+recognition+of+human+activities+and+postural+transitions) | Manual — place in `data/hapt/` |
| EEG Brainwave *(unused, kept for reference)* | Kaggle `birdy654/eeg-brainwave-dataset-feeling-emotions` | Not wired into the experiment CLI — see leakage note above |

For Kaggle datasets, set up `~/.kaggle/kaggle.json` with your credentials. `.npy` cache files are auto-generated on first load alongside the raw data.

---

## Usage

```bash
# Run all 3 datasets (default)
python main.py

# Run specific experiments
python main.py --exp har ecg

# Override epochs and seeds
python main.py --epochs 50 --seeds 42 0 7

# Run ablation studies (not in default run)
python main.py --exp ablation component

# Print model architecture and parameter counts
python main.py --arch

# Regenerate plots from a previous run without re-training
python main.py --replot outputs/run_20260708_182443_har_ecg_eeg_hapt_ablation_component_epo50

# Merge results from multiple runs and replot together
python main.py --replot outputs/run_A outputs/run_B
```

### CLI flags

| Flag | Default | Description |
|---|---|---|
| `--exp` | `har ecg hapt` | Experiments to run |
| `--epochs` | `50` | Training epochs per experiment |
| `--seeds` | `42 0 7 1 2 3 4 5 6 8` | Random seeds (results averaged) |
| `--fine-tune-epochs` | `3` | Post-quantization fine-tuning epochs |
| `--arch` | — | Print model architectures and exit |
| `--replot` | — | Load `results.pkl` from one or more run dirs and regenerate plots without re-training |

---

## Outputs

Each run creates a timestamped directory under `outputs/`:

- `run.log` — full stdout/stderr mirror
- `metrics.csv` — per-experiment summary stats
- `per_seed_metrics.csv` — per-seed breakdown for all 8 quantization methods
- `summary.txt` — human-readable summary with TOST equivalence table and edge profile
- `results.pkl` — pickled `{results, timings}` dict for use with `--replot`
- `figures/` — all plots (accuracy, confusion matrix, ROC/PR, compression delta, Pareto frontier, cross-dataset summary, edge profile, per-class F1, etc.)
- `models/{dataset}/` — best model weights per dataset:
  - `dendritic_uncompressed.pt` — float32 state dict (best seed)
  - `dendritic_snowflake.pt` — compressed quantized dict (best seed)
  - `mlp.pt` — float32 MLP state dict (best seed)
