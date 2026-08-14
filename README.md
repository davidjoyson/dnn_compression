# DNN Compression — Dendritic Network with Int8 Quantization

A research project exploring near-lossless compression of biologically-inspired dendritic neural networks on real-world tabular/time-series classification tasks.

**Core finding:** Per-layer int8 quantization (Snowflake) achieves **~4× compression with no statistically significant accuracy loss** on HAPT, and beats its ±2% equivalence margin on ECG in the *positive* direction (compression improves accuracy more than the margin allows, not degrades it). TOST equivalence testing (n=10 seeds, ±2% margin) confirms **13/16 method–dataset pairs are statistically equivalent** (all 8/8 on HAPT; 5/8 on ECG — the 3 failures are Snowflake, Global, and QAT, all over-shooting the margin on the positive side). This is a statistical-equivalence claim, not a guarantee of zero information loss: the architecture-size ablation found a real accuracy drop under compression at the smallest model sizes tested (see `docs/experiment_log.md`, 2026-07-20 entry).

The paper-ready inventory of datasets, models, compression methods, metrics,
ablations, statistical tests, and hardware evaluation is maintained in
[`docs/experimental_matrix.md`](docs/experimental_matrix.md).

**The motivating hypothesis — that the dendritic branching architecture is more robust to quantization than a conventional MLP — is partially supported.** A 10-seed, 3-model (Dendritic vs. total-param-matched MLP vs. per-layer-matched MLP), 8-method comparison shows Dendritic leading both baselines' compression-delta specifically on **Snowflake, Global, and QAT — consistently on both datasets** (e.g. ECG Snowflake: Dendritic +2.5pp vs. MLP +0.09pp vs. LayerMatchedMLP +1.8pp; HAPT QAT: Dendritic +0.29pp vs. MLP -0.10pp vs. LayerMatchedMLP -0.15pp — both baselines actually lose accuracy under QAT while Dendritic gains). It does **not** hold on Static, Snowflake+Static, Dynamic, Per-channel, or Mixed precision, where a baseline matches or beats Dendritic — so "consistently more robust across every method" is not supported, but "more robust on the project's primary method" is (`docs/experiment_log.md`, 2026-07-27 entry). Dendritic's branches are also measurably narrower and more weight-distinct than a structurally-equivalent control layer (2026-07-21 entry) — a real structural property, plausibly related to why Snowflake specifically (which calibrates per layer/branch) is where the advantage shows up.

> **Note:** A 4th dataset (EEG brainwave emotions) was dropped 2026-07-21 after investigation confirmed unfixable patient/session-level data leakage in the source data — no subject ID or recoverable session structure exists in the published CSV, and the raw per-subject recordings needed to rebuild the split aren't publicly available for this task. See `docs/experiment_log.md` for the investigation. The loader (`src/loaders/load_eeg.py`) and experiment code are kept in the repo for reference but are no longer wired into the experiment CLI.

---

## Problem Statement

Neural networks deployed on edge devices (wearables, microcontrollers) are constrained by memory. Standard compression methods (pruning, global quantization) degrade accuracy, especially on small models (<200k params). This project asks:

> Can a biologically-inspired dendritic architecture be compressed 4× with no statistically significant accuracy loss, outperforming standard quantization baselines?

The dendritic network's "snowflake" property — parallel branches each learning distinct feature subspaces — was the motivating hypothesis for compatibility with per-layer quantization, since each branch's weight distribution was expected to be narrow and independently calibrated.

**This part of the claim is now quantitatively confirmed** (see `docs/experiment_log.md`, 2026-07-21 entry): across all 3 datasets, Dendritic's branches have consistently lower weight std (17–27%) than the equivalent rows of a structurally-matched non-branching control layer, and are weight-distinct from each other (near-zero inter-branch cosine similarity). This narrower/distinct branch structure **does** translate into a measurable quantization-robustness advantage over parameter- and layer-matched MLPs, but only on 3 of the 8 compression methods tested (Snowflake, Global int8, QAT — see Results below and `docs/experiment_log.md`, 2026-07-27 entries); on the other 5 methods a baseline matches or beats Dendritic.

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

Eight methods evaluated head-to-head. **"Size ratio" and "real hardware speedup" are different axes and don't move together** — a method can shrink the checkpoint 4× and still run at 1.0× on-device, because storage format and compute format are independent choices. The `Compute` column says which regime each method is actually in (measured on real Pi 3 hardware, not inferred — see Edge Deployment below):

| Method | Description | Size ratio | Compute | Real Pi speedup (batch=1) |
|---|---|---|---|---|
| **Snowflake (int8)** | Per-layer int8 — one scale per layer group (weight + bias) | **4×** | Storage-only — dequantized to float32 before every matmul | ~1.0× |
| Global int8 | Single global scale across all parameters | 4× | Storage-only | ~1.0× |
| Per-channel (int8) | One scale per output neuron row; biases stay float32 | ~4× | Storage-only | ~1.0× |
| Snowflake (int4) | Per-layer int4, ~8× compression | ~8× | Storage-only | ~1.0× |
| Snowflake (int6) | Per-layer signed int6, bit-packed size estimate | ~5.3× | **Storage-only** — dequantized to float32 for evaluation | No native INT6 speedup claimed |
| Dynamic int8 | PyTorch `quantize_dynamic` on Linear layers | ~4× | **True INT8** — activations quantized per-call at runtime | 0.49–0.58× at batch=-1 (overhead dominates), **1.36–1.40× at batch=1** |
| Static (int8) | Per-tensor static calibration via `prepare_fx`/`convert_fx` | ~4× | **True INT8** — weights + activations both int8, real INT8 GEMM | **~1.9×** |
| Snowflake+Static | Snowflake weight scales + FX-calibrated int8 activations | ~4× | **True INT8** | **~1.9×** |
| QAT (int8) | Quantization-aware training via `prepare_qat_fx`/`convert_fx` | ~4× | **True INT8** | **~1.9×** (best accuracy of the true-INT8 group) |
| Mixed precision | Inner layers int8, first and last layers float32 | ~0.9× | **True INT8** on the quantized inner layers only | ~1.2–1.4× (partial — two layers still float32) |

The weight-only rows (including Snowflake int8/int6/int4) cast stored values back to float32 before matmul, so their storage win does not reach the compute path. The activation-quantized methods can dispatch true INT8 kernels. INT6 is included only as an intermediate precision/accuracy and packed-storage comparison; no native INT6 latency claim is made.

The INT4/INT6/INT8 precision comparison is isolated from the main method sweep:
`python run_precision_comparison.py`. The script uses the same trained float
model, seeds, fine-tuning budget, accuracy, macro-F1, confidence intervals,
TOST, and packed-size accounting for all three precisions. Standard Snowflake
INT8 remains in the main experiment as the project's primary method, but INT6
and INT4 do not add workload or columns to normal `main.py` runs.

All methods optionally followed by 3 epochs of post-quantization fine-tuning. Compared against a param-matched MLP baseline (2 layers: FC+ReLU → output).

---

## Results — 50 epochs, 10 seeds

| Dataset | Classes | Uncompressed | Snowflake (4×) | Delta | TOST (n=10) |
|---|---|---|---|---|---|
| UCI HAR | 6 | 94.12% ±0.48% | 94.16% ±0.45% | +0.04% | EQUIV |
| ECG Heartbeat (patient-split) | 5 | 83.71% ±2.21% | **86.21% ±1.04%** | **+2.50%** | **NOT EQUIV** (compression over-improves) |
| HAPT | 12 | 92.51% ±0.74% | **92.77% ±0.56%** | **+0.26%** | EQUIV |

*(Confirmed 2026-07-27 against the current codebase. ECG runs `balance=True` (10 seeds, ~3h08m); HAPT runs `balance=False` (10 seeds, ~13min) — see the paragraph below on why the two datasets now use different settings. Full detail in the 2026-07-27 entries of `docs/experiment_log.md`.)*

Snowflake matches or beats uncompressed on both current default datasets. TOST equivalence testing (±2% margin) confirms **13/16 method–dataset pairs are equivalent** (all 8/8 on HAPT; ECG fails on Snowflake/Global/QAT, all in the direction of compression *improving* accuracy past the margin, not degrading it).

**HAR is no longer part of the default `python main.py` run.** HAR and HAPT were verified to use the *exact same* 21/9 subject-independent train/test split (independently confirmed — zero subject overlap in either), and HAPT's first 6 classes already cover HAR's task, so running both added little beyond what HAPT's larger, imbalanced class set already tests. The default runs `ecg hapt`; INCART remains available as an independent opt-in ECG dataset.

**ECG uses the patient-independent (DS1/DS2) split**, not the original Kaggle random split. The original split was found to leak patient data between train/test (no patient ID, known to split by individual beat), inflating accuracy by ~13 percentage points relative to the (still-leaky) balanced-training version. See `docs/experiment_log.md`, 2026-07-20 and 2026-07-21 entries, for the full investigation. The old `load_ecg.py` (leaky) loader is kept in the repo for reference but is no longer used by the default pipeline.

**ECG and HAPT now use different `balance` settings, deliberately.** ECG trains on oversampled, class-balanced data (`balance=True`) — its dominant class is severe enough (~89% of beats) that without oversampling, Dendritic loses its uncompressed-accuracy edge and the one clean Snowflake/Global/QAT robustness result disappears (see Core Finding above). HAPT trains on its natural, unbalanced distribution (`balance=False`) — the opposite conclusion, reached the same way: oversampling HAPT's rare transition classes (23–90 train examples each) craters Dendritic's accuracy by ~20pp (92.5%→72.5%) and inflates its seed-to-seed variance ~8× (±0.7%→±5.7%), while `balance=False` keeps accuracy/variance normal *and* the Snowflake/Global/QAT robustness edge intact — in fact cleaner, since both MLP baselines go slightly *negative* under compression while Dendritic stays positive. In short: oversampling helps on the one dataset where the majority class is overwhelming, and actively hurts on the one where it isn't. See `docs/experiment_log.md`, 2026-07-27 entries, for the full investigation of both directions. Macro F1 and balanced accuracy remain low on both datasets' rare classes regardless of this choice — a separate, still-open problem, also covered in `docs/experiment_log.md`.

---

## Edge Deployment — Raspberry Pi 3

Real single-sample (batch=1) inference latency on a Raspberry Pi 3 Model B (ARM Cortex-A53, `qnnpack` backend), via SSH (`benchmark_pi.py`). Full results in `benchmark_pi_output/`.

| Dataset | Float32 baseline | Snowflake (int8) | Static W+A (int8) | Snowflake+Static (int8) |
|---|---|---|---|---|
| ECG  | 8.00 ms | 8.06 ms (0.99×) | 4.23 ms (**1.89×**) | 4.21 ms (**1.90×**) |
| HAPT | 8.38 ms | 8.43 ms (0.99×) | 4.38 ms (**1.91×**) | 4.32 ms (**1.94×**) |

HAR is excluded from this table (and `benchmark_pi_output/`) since it's no longer part of the default pipeline — its last real-hardware numbers are archived in `benchmark_pi_output/archive/`, run against a now-superseded checkpoint/data format, not re-verified against the current codebase.

**Snowflake gives no real speedup on hardware (~1.0×)** — it's weight-only quantization, so weights are dequantized back to float32 before every matmul; the storage savings (4×) don't translate to compute savings. **Static and Snowflake+Static run true INT8 arithmetic** and deliver a genuine ~1.9× latency reduction. This is a meaningful distinction the accuracy-only tables above don't capture: for actual edge latency, "true INT8 arithmetic" methods matter, not just int8 *storage*.

Plots for all 10 benchmarked methods (not just the 4 above) are generated by `python plot_pi_benchmark.py` into `benchmark_pi_output/` — latency, memory, per-method speedup, a batch=-1-vs-batch=1 comparison, and a real-hardware compression-vs-speedup Pareto chart. Two things only visible there: **memory usage doesn't trade off consistently against speed** (true-int8 methods use more RAM than weight-only methods on HAPT, but *less* on ECG — dataset-dependent, not a universal rule), and **Dynamic quantization's speedup verdict flips with batch size** — it's slower than Float32 at batch=-1 (0.49–0.58×) but faster at batch=1 (1.36–1.40×), so which one matters depends entirely on your deployment's batch size.

**Cross-dataset plots** (accuracy, F1, model size, and all 3 ablation studies compared across HAR/ECG/HAPT in a single chart each) are generated into `figures/combined/` on every run — see `docs/experiment_log.md`, 2026-07-22 and 2026-07-28 entries. The architecture-size ablation's tiny-model (`h1=16`) collapse is stark on HAR and HAPT (accuracy craters with huge variance). On ECG it used to be nearly invisible under the old `balance=False` default (raw accuracy floors at its ~89% majority-class rate regardless of whether the model learned anything — the same masking effect as the F1 investigation above) — but under the current `balance=True` default it's now the *most* dramatic collapse of either dataset (89.7%→66.6%, a 23pp drop vs. 6pp at the largest config), since oversampling removes the majority-class floor that was hiding it.

**Thermal:** a 15-minute sustained-load test (`thermal_test.py`, Snowflake+Static on ECG, no active cooling) held **225.9 inf/s with zero throughput degradation**; temperature plateaued around **~48°C (mean 47.97°C, range 46.2–48.9°C) from the 5-minute mark onward** — comfortably below the ~80°C throttle threshold.

**Not yet measured:** real power/energy draw per inference — the Pi 3 has no built-in power ADC (`vcgencmd pmic_read_adc` is Pi 4/5-only), so this needs external hardware (e.g. INA219) not currently available. Temperature was used as a free thermal-risk proxy instead. These numbers also validate an ARM Linux SBC, not bare-metal microcontroller-class hardware (e.g. TFLite Micro on ESP32) — a different deployment target not yet attempted.

*(ECG latency was measured pre-leakage-fix, but remains valid — inference latency depends only on model architecture/shape, which is unchanged by the patient-split fix, not on the training data itself.)*

### Benchmarking Protocol

| | |
|---|---|
| **Hardware** | Raspberry Pi 3 Model B, ARM Cortex-A53 (`aarch64`), 4 cores, no active cooling. Accessed headless over SSH. |
| **Power** | A prior insufficient-PSU run silently throttled the SoC ~2× (2026-07-16 entry, `docs/experiment_log.md`) — all currently-reported numbers are post-fix, correct PSU. Power supply quality materially affects latency; not something a software protocol alone controls for. |
| **Software** | PyTorch 2.13.0+cpu, `torch.backends.quantized.engine = "qnnpack"` (ARM NEON int8 kernels; `fbgemm` is the x86 fallback used only for local dev-machine testing, never for reported numbers). |
| **Threading / CPU governor** | **Not pinned.** PyTorch runs with its default intra-op thread pool across all 4 cores; no `taskset`/CPU-affinity, no CPU-frequency-governor lock (e.g. forcing `performance` mode). `benchmark_pi.py`/`thermal_test.py` contain no thread- or governor-control code at all. This is a real gap: unpinned threading + a `powersave`/`ondemand` governor can add run-to-run latency variance from frequency scaling that a stricter protocol would control for. Not yet fixed. |
| **Warm-up** | 50 forward passes per method (`--warmup`, default 50) before timing starts, discarded. |
| **Repetitions** | 500 timed forward passes per method (`--runs`, default 500); reported latency/std are the mean/std **across those 500 calls within one process**, not across independently-restarted runs. |
| **Batch modes** | `--batch-size -1` (full test set per call) and `--batch-size 1` (single sample) are **separate process invocations**, run independently — not two measurements sharing one process's warm state. |
| **Memory (RSS)** | `/proc/self/status` VmRSS, read once per method immediately after that method's timed loop + eval, after forcing `gc.collect()` + `malloc_trim(0)` (added 2026-07-30 to stop stale allocator garbage from a *previous* method being misattributed to the current one). This is a **point-in-time reading, not a continuously-sampled peak** — and even post-fix, cross-method comparisons within one process aren't fully clean (see `docs/experiment_log.md`, 2026-07-30 entries, for a worked example on why the first FX/static-quantized method in a run absorbs a one-time ~100MB QNNPACK initialization cost that every later method then "inherits" for free). Fully isolated per-method numbers would need one fresh process per method — not done. |
| **Temperature** | Separate script (`thermal_test.py`), not part of the main latency sweep. 15-minute (900s) sustained single-sample inference loop, one representative method (Snowflake+Static, ECG only — not run per-method), `vcgencmd measure_temp` sampled by a background thread every 2s (`--interval`, default 2.0), 60s cooldown between methods when run with `--method all`. |
| **Energy / power draw** | **Not measured.** Pi 3 has no built-in power ADC (`vcgencmd pmic_read_adc` is Pi 4/5-only); would need external hardware (e.g. INA219), not currently available. Temperature is used as a free thermal-risk proxy instead — it is not a substitute for a real energy-per-inference number. |

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
│   │   ├── load_har.py                  # UCI HAR (wearable sensors, 6-class) — opt-in, not in default run
│   │   ├── load_ecg.py                  # MIT-BIH ECG, Kaggle random split — unused, kept for reference (leaky, see note above)
│   │   ├── load_ecg_patient_split.py    # MIT-BIH ECG, patient-independent DS1/DS2 split (5-class) — active loader
│   │   ├── load_eeg.py                  # EEG brainwave emotions (3-class) — unused, kept for reference
│   │   └── load_hapt.py                 # UCI HAPT smartphone IMU (12-class)
│   │
│   ├── experiments/
│   │   ├── base_experiment.py           # Shared training + eval loop for all datasets
│   │   ├── har_experiment.py            # opt-in, not in default run (see note above)
│   │   ├── ecg_experiment.py            # unused, kept for reference (leaky split, see note above)
│   │   ├── ecg_patient_experiment.py    # active ECG experiment (patient-independent split)
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
        └── models/       (per experiment run — default: ecg/, hapt/; har/ if run with --exp har)
            ├── ecg/    (dendritic_uncompressed.pt, dendritic_snowflake.pt, mlp.pt)
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
| UCI HAR *(opt-in — not in default run, see Results)* | [UCI ML Repository](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones) | Manual — place in `data/har/` |
| ECG Heartbeat | Kaggle `shayanfazeli/heartbeat` | Via Kaggle CLI on first load |
| UCI HAPT | [UCI ML Repository](https://archive.ics.uci.edu/dataset/341/smartphone+based+recognition+of+human+activities+and+postural+transitions) | Manual — place in `data/hapt/` |
| St Petersburg INCART ECG *(patient-split, 4-class AAMI)* | [PhysioNet](https://physionet.org/content/incartdb/1.0.0/) | Yes — downloaded and cached on first load |
| EEG Brainwave *(unused, kept for reference)* | Kaggle `birdy654/eeg-brainwave-dataset-feeling-emotions` | Not wired into the experiment CLI — see leakage note above |

For Kaggle datasets, set up `~/.kaggle/kaggle.json` with your credentials. `.npy` cache files are auto-generated on first load alongside the raw data.

---

## Usage

```bash
# Run the default experiments (ecg + hapt)
python main.py

# Run specific experiments — e.g. include HAR (opt-in, not in default run)
python main.py --exp har ecg hapt

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
| `--exp` | `ecg hapt` | Experiments to run |
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
- `figures/` — per-dataset plots (confusion matrix, ROC/PR, training curves, per-class F1, compression delta, weight distribution, branch diversity, val accuracy)
- `figures/combined/` — cross-dataset plots, one chart per metric across all 3 datasets (accuracy, F1, model size, architecture/component/regularization ablation, Pareto frontier, inference time, edge profile)
- `models/{dataset}/` — best model weights per dataset:
  - `dendritic_uncompressed.pt` — float32 state dict (best seed)
  - `dendritic_snowflake.pt` — compressed quantized dict (best seed)
  - `mlp.pt` — float32 MLP state dict (best seed)
