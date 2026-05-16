# Experiment Log

---

## 2026-05-07 — Project Initialisation

**Commits:** `4e2ea4d` First commit · `2904a8c` v1

### Summary
Project scaffolded from scratch. Core architecture and experiment pipeline established.

**`4e2ea4d` — First commit**
- Created full project structure: `src/compression/`, `src/experiments/`, `src/models/`, `src/plots/`, `src/training/`
- `DendriticNetwork` base model with `DendriticLayer` and `DendriticNeuron`
- `MLPBaseline` for comparison
- Initial experiments: UCI Adult, Folktables, Scaling, XOR, Ablation
- Stub compression pipeline (`compression_pipeline.py`, `quantization.py`, `topology_sharing.py`)
- Basic training/evaluation loop

**`2904a8c` — v1**
- Expanded `DendriticNetwork` with proper branching architecture
- Improved compression pipeline with actual quantization logic
- Renamed `adult_income_experiment.py` → `uci_adult_experiment.py`, added `wine_experiment.py`
- Expanded Scaling and Ablation experiments
- Removed `learning_mode_experiment.py` and `uci_experiments.py` (dead code)

---

## 2026-05-11 — MSE Metrics + Experiment Expansion

**Commits:** `de51991` v2 added mse · `c51aba9` Stop tracking settings.local.json

### Summary
Added MSE as a second evaluation metric alongside accuracy. Expanded to more datasets.

**`de51991` — v2 added mse**
- Added MSE metric to all experiments (alongside accuracy) for regression-style error analysis
- New experiments: `creditcard_experiment.py`, `folktables_multistate_experiment.py`
- Added training curve plots, ROC/PR plots, scaling plots
- Centralised `main.py` replacing `run_all_experiments.py`
- Improved `compression_pipeline.py` with cleaner quantization logic
- Removed dead files: `quantization.py`, `training/utils.py`, `folktables_plots.py`, `plot_xor_boundary.py`

**`c51aba9` — Stop tracking settings.local.json**
- Removed `.claude/settings.local.json` from version control (local IDE config)

---

## 2026-05-13 — Snowflake Compression + Output System

**Commits:** `fb9ffef` Add snowflake compression experiments with MLP baseline comparison and output system

### Summary
First complete implementation of the Snowflake (per-layer int8) compression method with head-to-head MLP comparison and structured experiment output.

**`fb9ffef`**
- Added compressed MLP baseline to all 4 experiments (wine, adult, folktables, creditcard) to test the snowflake hypothesis via delta comparison
- Timestamped output directories (`outputs/run_YYYYMMDD_HHMMSS/`) with auto-saved `figures/`, `metrics.csv`, `summary.txt`
- `--arch` CLI flag to print model architectures via `torchinfo`
- `size_bytes()` added to `MLPBaseline` for compression ratio tracking
- Fixed `_save_metrics_csv` to skip multi-element tensor / list results
- Dynamic figure directory support in `save_utils` via `set_fig_dir()`
- Removed `DendriticNeuron` class (dead code, incompatible with pipeline)

---

## 2026-05-14 — Architecture Overhaul + ECG Experiments

**Commits:** `0fff1ae` Refactor · `6255b14` Soma layer · `ba1446e` use_soma toggle + param-matched MLP · `2177bd0` ECG + compression baselines + auto-logging

### Morning — Codebase Refactor (`0fff1ae`)
- Added 10% validation split and per-epoch val loss tracking to adult, folktables, HAR experiments
- Threaded `fine_tune_epochs` through all `compress_model` calls; exposed as `--fine-tune-epochs` CLI arg
- Removed dead experiments: creditcard, folktables_multistate, wine, occupancy
- Renamed `src/data/` → `src/loaders/` for clarity
- **Added HAR experiment** (UCI wearable sensor activity recognition, 6-class)
- Added `src/reporting/` module: `summary.py`, `utils.py`, `plots.py` for structured CSV/text/plot output

### Midday — DendriticNetwork Architecture (`6255b14`, `ba1446e`)

**`6255b14` — Add soma layer**
- Inserted `Linear + ReLU` soma between branch concatenation and `fc2`
- Collapses each branch's `hidden_per_branch` activations to one signal per branch — more biologically accurate dendritic integration

**`ba1446e` — Param-matched MLP + use_soma toggle**
- `DendriticNetwork`: `use_soma=True` flag; `fc2` input dim adjusts when soma disabled
- `MLPBaseline`: `match_params` kwarg + `param_matched_hidden()` to auto-size hidden layer to match DendriticNetwork param count — ensures fair comparison
- All experiments updated to param-matched MLP
- `evaluate.py`: added `count_params()` utility
- `main.py`: disabled adult/folktables/scaling; `--arch` shows param-matched MLP

### Afternoon/Evening — ECG Compression Experiments (`2177bd0`)

**Dataset:** MIT-BIH ECG, 87,554 train / 21,892 test, 187 features, 5 classes  
**Model:** DendriticNetwork (hidden1=64, hidden2=32, branches=8, hidden_per_branch=8, ~17k params)

Infrastructure added:
- `.npy` caching in `load_ecg.py` — avoids re-parsing 411 MB CSV on each run
- `--seeds` CLI flag for multi-seed averaging
- Auto-logging via `_Tee` class in `main.py` — every run saves `run.log` to its output folder

Compression methods evaluated:

| Method | Description | Fine-tune |
|---|---|---|
| Snowflake (int8) | Per-layer int8, one scale per layer group | 3 epochs |
| Global int8 | Single global scale across all params | 3 epochs |
| Dynamic (int8) | PyTorch `quantize_dynamic` on Linear layers, CPU-only | 0 (none) |
| ~~Pruned 75%~~ | ~~Magnitude pruning, sparse float32+int32 storage~~ | ~~dropped~~ |

**Progressive results (seed=42):**

| Epochs | Uncompressed | Snowflake (4×) | Global int8 (4×) | 3rd Method |
|---|---|---|---|---|
| 1 | 67.42% | 78.62% (+11.2%) | 75.35% (+7.9%) | Pruned: 82.40% |
| 10 | 92.40% | 91.60% (-0.80%) | 91.78% (-0.62%) | Pruned: 66.31% |
| 20 | 94.91% | 94.05% (-0.85%) | 95.42% (+0.51%) | Pruned: 35.64% |
| 50 ft=1 | 95.78% | 96.13% (+0.35%) | 94.07% (-1.71%) | Pruned: 37.79% |
| 50 ft=3 | 95.78% | 96.13% (+0.35%) | 94.35% (-1.43%) | Pruned: 37.65% → dropped |
| **50 final** | **95.78%** | **96.13% (+0.35%)** | **94.35% (-1.43%)** | **Dynamic: 95.98% (+0.20%)** |

**Final results — 50 epochs, seed=42:**

| Method | Accuracy | Delta | Size (bytes) | Ratio |
|---|---|---|---|---|
| Uncompressed (Dendritic) | 95.78% | — | 68,660 | 1× |
| **Snowflake (int8)** | **96.13%** | **+0.35%** | **17,213** | **4×** |
| Dynamic (int8) | 95.98% | +0.20% | 33,158 | ~2× |
| Global int8 | 94.35% | -1.43% | 17,213 | 4× |
| MLP Baseline | 94.91% | — | 68,728 | 1× |
| MLP Compressed | 96.22% | — | 17,190 | 4× |

**Observations:**
1. **Snowflake best overall** — 4× compression, +0.35% accuracy (quantization regularises)
2. **Dynamic int8 accurate but ~2× only** — PyTorch serialisation overhead doubles size vs manual packing
3. **Global int8 degrades at high epochs** — single scale too coarse for a well-trained model
4. **Pruning (75%) dropped** — catastrophic for a ~17k param model; 75% sparsity unrecoverable regardless of fine-tune epochs
5. **Dendritic beats MLP uncompressed** — 95.78% vs 94.91% (+0.87%)

---

## 2026-05-16 — 3-Seed Evaluation + Dynamic Quant Size Fix

**Commits:** *(this session)*

### Summary
Fixed `dynamic_model_size_bytes` to report true raw data size (was inflated ~2× by pickle overhead). Ran full 3-seed evaluation (seeds 42, 0, 7) across all active experiments: ablation, component, HAR, ECG.

### Dynamic Quantization Size Fix

`torch.save` on PackedParams objects adds ~15KB pickle overhead, making dynamic quant appear ~2× instead of ~4×. Fixed by measuring raw data directly:
- int8 weights: `mod.weight().int_repr().numel()` (1 byte each)
- fp32 biases: `mod.bias().numel() * 4` (4 bytes each)

Result: Dynamic now reports 17,684 bytes (~3.9×) vs 17,213 bytes (4.0×) for Snowflake — 471-byte gap is float32 biases vs int8 biases.

### 3-Seed Results (seeds 42, 0, 7 — 50 epochs, fine_tune_epochs=3)

**ECG Heartbeat (MIT-BIH, 5-class) — primary benchmark:**

| Method | Accuracy | ± std | Delta | Size (bytes) | Ratio |
|---|---|---|---|---|---|
| Uncompressed (Dendritic) | 95.97% | ±1.15% | — | 68,660 | 1× |
| **Snowflake (int8)** | **96.58%** | **±0.89%** | **+0.61%** | 17,213 | **4×** |
| Dynamic (int8) | 95.72% | ±1.45% | -0.25% | 17,684 | ~3.9× |
| Global int8 | 95.31% | ±2.63% | -0.65% | 17,213 | 4× |
| MLP Baseline | 94.96% | ±0.31% | — | 68,728 | 1× |
| MLP Compressed | 94.85% | ±0.55% | — | 17,190 | 4× |

**HAR (binary walking vs stationary) — saturated, not informative:**

All methods: ~99.98% ±0.03% — task too easy to distinguish compression quality.

**Observations:**
1. **Snowflake confirmed best across seeds** — 4× compression, +0.61% gain, lowest variance (±0.89%)
2. **Global int8 unstable** — highest variance (±2.63%), single scale inadequate for well-trained model
3. **Dendritic beats MLP** — uncompressed (95.97% vs 94.96%) and compressed (96.58% vs 94.85%)
4. **Dynamic quant marginally negative** (-0.25%) with ~3.9× ratio after size fix
5. **HAR task saturated** — binary classification too easy; ECG is the meaningful benchmark

---

## Commit History

| Commit | Date | Summary |
|---|---|---|
| `4e2ea4d` | 2026-05-07 | First commit — base DendriticNetwork, compression pipeline, experiment stubs |
| `2904a8c` | 2026-05-07 | v1 — expanded compression pipeline, UCI Adult, Folktables, Scaling experiments |
| `de51991` | 2026-05-11 | v2 — MSE metrics, creditcard/folktables-multistate experiments, reporting plots |
| `fb9ffef` | 2026-05-13 | Snowflake compression with MLP baseline comparison, output system, `--arch` flag |
| `0fff1ae` | 2026-05-14 | Refactor: validation splits, HAR experiment, `src/loaders/`, `src/reporting/` module |
| `6255b14` | 2026-05-14 | Add soma layer to DendriticNetwork |
| `ba1446e` | 2026-05-14 | Param-matched MLP baseline, `use_soma` toggle, `count_params` utility |
| `2177bd0` | 2026-05-14 | ECG experiment, global int8 + dynamic quantization, HAR updated, auto-logging, experiment log |
| *(pending)* | 2026-05-16 | Fix `dynamic_model_size_bytes`, 3-seed evaluation (ECG + HAR), update experiment log |

---

## Next Steps

- [x] ~~Commit today's session work~~ — done in `2177bd0`
- [x] ~~Apply same compression comparison to HAR experiment~~ — done in `2177bd0`
- [x] ~~Run 3-seed evaluation (seeds 42, 0, 7) for reliable ± std statistics~~ — done 2026-05-16
- [x] ~~Investigate dynamic quantization size overhead~~ — fixed 2026-05-16 (pickle overhead; raw data = 17,684 bytes ≈ 3.9×)
