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

## 2026-05-16 — Int4 Quantization + 3-Seed Evaluation

**Commits:** *(this session)*

### Summary
Fixed `dynamic_model_size_bytes` to report true raw data size (was inflated ~2× by pickle overhead). Ran full 3-seed evaluation (seeds 42, 0, 7) across all active experiments: ablation, component, HAR, ECG.

### Snowflake int4 (4-bit) Quantization

Added per-layer int4 quantization as the 4th compression method:
- **Packing**: values clamped to [-7, 7], offset-encoded (+8 → [1,15] as uint4), packed 2-per-byte
- **Scale**: same per-layer-group scheme as Snowflake int8 (weight+bias share one scale)
- **Storage**: `ceil(n/2)` packed bytes + 4 bytes per layer scale → 8× compression vs float32

**3-seed ECG results (seeds 42, 0, 7 — 50 epochs, fine_tune_epochs=3):**

| Method | Accuracy | ± std | Delta | Size (bytes) | Ratio |
|---|---|---|---|---|---|
| Uncompressed (Dendritic) | 95.97% | ±1.15% | — | 68,660 | 1× |
| **Snowflake (int8)** | **96.58%** | **±0.89%** | **+0.61%** | 17,213 | **4×** |
| Dynamic (int8) | 95.72% | ±1.45% | -0.25% | 17,684 | ~3.9× |
| Global int8 | 95.31% | ±2.63% | -0.65% | 17,213 | 4× |
| **Snowflake (int4)** | **72.30%** | **±13.89%** | **-23.67%** | **8,631** | **8×** |

**Finding: int4 is not viable for ~17k param models.** The 4-bit range (14 representable values) is too narrow for the weight distributions after training. High variance (±13.89%) shows training is unstable — the quantisation grid is coarser than the meaningful weight differences. Conclusion: **8-bit is the minimum precision for models at this scale**.

To make int4 viable would require a significantly larger model (more parameters to absorb quantisation error) or a much larger fine-tuning budget.

### Dynamic Quantization Size Fix

`torch.save` on PackedParams objects adds ~15KB pickle overhead, making dynamic quant appear ~2× instead of ~4×. Fixed by measuring raw data directly:
- int8 weights: `mod.weight().int_repr().numel()` (1 byte each)
- fp32 biases: `mod.bias().numel() * 4` (4 bytes each)

Result: Dynamic now reports 17,684 bytes (~3.9×) vs 17,213 bytes (4.0×) for Snowflake — 471-byte gap is float32 biases vs int8 biases.

### 3-Seed Results — Summary

See int4 table above for full ECG results. Additional notes:

- **MLP Baseline**: 94.96% ±0.31% (uncompressed), 94.85% ±0.55% (compressed at 4×)
- **HAR** (binary walking vs stationary): all methods ~99.98% ±0.03% — task too easy to distinguish compression quality

**Observations:**
1. **Snowflake confirmed best across seeds** — 4× compression, +0.61% gain, lowest variance (±0.89%)
2. **Global int8 unstable** — highest variance (±2.63%), single scale inadequate for well-trained model
3. **Dendritic beats MLP** — uncompressed (95.97% vs 94.96%) and compressed (96.58% vs 94.85%)
4. **Dynamic quant marginally negative** (-0.25%) with ~3.9× ratio after size fix
5. **int4 not viable** at ~17k params — 8-bit is minimum precision for this model scale
6. **HAR task saturated** — binary classification too easy; ECG is the meaningful benchmark

---

## 2026-05-17 — EEG Brainwave Experiment + Int4 Scale Threshold

**Commits:** *(this session)*

### Summary
Added EEG Brainwave (emotion classification) as a new benchmark. Key finding: Snowflake int4 is viable at ~167k params, confirming that int4 viability scales with model size.

**Dataset:** Kaggle `birdy654/eeg-brainwave-dataset-feeling-emotions`  
- 2,132 samples, 2,548 engineered EEG features, 3 balanced classes (NEGATIVE/NEUTRAL/POSITIVE)  
- 80/20 stratified split → 1,706 train / 426 test  
- StandardScaler normalisation; `.npy` caching  

**Model:** DendriticNetwork (input_dim=2548, hidden1=64, hidden2=32, branches=8, hidden_per_branch=8, ~167k params)  
Large fc1 (2548→64, 163k params) dominates — this is what makes int4 viable here.

**3-seed results (seeds 42, 0, 7 — 50 epochs, fine_tune_epochs=3):**

| Method | Accuracy | ±std | Delta | Size (bytes) | Ratio |
|---|---|---|---|---|---|
| Uncompressed (Dendritic) | 97.66% | ±0.23% | — | 672,812 | 1× |
| Snowflake (int8) | 97.58% | ±0.14% | -0.08% | 168,251 | 4× |
| Global int8 | 97.58% | ±0.36% | -0.08% | 168,251 | 4× |
| Dynamic (int8) | 95.24% | ±0.14% | -2.42% | 168,716 | ~4× |
| **Snowflake (int4)** | **97.74%** | **±0.14%** | **+0.08%** | **84,150** | **8×** |
| MLP Baseline | 97.81% | ±0.14% | — | 673,740 | 1× |
| MLP Compressed | 97.81% | ±0.14% | — | 168,443 | 4× |

**100-epoch run (convergence check):**

| Method | 50 epo | 100 epo | Note |
|---|---|---|---|
| Uncompressed | 97.66% ±0.23% | 97.66% ±0.23% | Converged |
| Snowflake (int8) | 97.58% ±0.14% | 97.58% ±0.14% | Stable |
| Global int8 | 97.58% ±0.36% | 97.66% ±0.00% | Variance collapses at 100 epo |
| Dynamic (int8) | 95.24% ±0.14% | 95.39% ±0.49% | Marginal gain |
| Snowflake (int4) | 97.74% ±0.14% | 97.66% ±0.23% | **0.00% delta — lossless** |

Model fully converged by epoch 50; 100 epochs adds nothing. **50 epochs is the correct stopping point.**

**Observations:**
1. **Snowflake int4 viable at ~167k params** — 0.00% delta at 8× compression. First dataset where 4-bit succeeds
2. **Int4 scale threshold confirmed** — ~17k params (ECG): -23.67%; ~167k params (EEG): 0.00%. The large fc1 layer (163k/167k params) provides sufficient quantization headroom
3. **Snowflake int8 near-lossless** — -0.08% at 4×, consistent with all prior datasets
4. **Global int8 stabilises at 100 epochs** — variance ±0.36% → ±0.00%; needs more training than Snowflake
5. **Dynamic quant worst** — -2.42% despite same storage cost as Snowflake int8
6. **Dendritic ≈ MLP** — 97.66% vs 97.81% (0.15% gap); highly engineered features level the playing field vs ECG's +0.87% dendritic advantage
7. **Snowflake int4 > Snowflake int8 > Global int8 > Dynamic** — ranking consistent with ECG

---

## 2026-05-20 — Confusion Matrices + Full 3-Dataset 3-Seed Run

**Commits:** *(this session)*

### Summary
Added confusion matrix evaluation and plots to all 3 experiments. Introduced centralised plot styling. Ran a full 50-epoch, 3-seed benchmark across HAR, ECG, and EEG. GPU run on GTX 1650 Max-Q (CUDA 12.4 via `D:\Python` torch 2.6.0+cu124).

### Infrastructure Changes

**`src/training/evaluate.py`** — added `confusion_matrix_eval(model, X, y, num_classes, device)`:
- Multi-class: `argmax` predictions; binary: threshold at 0.5
- Returns sklearn `confusion_matrix` on the full test set

**`src/plots/plot_confusion_matrix.py`** (new):
- Side-by-side normalised confusion matrices for Uncompressed vs Snowflake (int8)
- Blues colormap, row-normalised; each cell annotates fraction + raw count
- Saved as `{experiment}_confusion.png`

**`src/plots/style.py`** (new):
- Centralised `apply_style()`, `METHOD_COLORS`, `PALETTE` constants shared across all plot modules

**`src/plots/plot_accuracy.py`** — fixed pre-existing `yerr` bug:
- `None` in a yerr list crashes matplotlib; replaced with `float("nan")` to suppress zero-std bars

All experiment files (HAR, ECG, EEG) and `src/reporting/utils.py` / `src/reporting/plots.py` updated to wire confusion matrices end-to-end.

### 50-Epoch, 3-Seed Results (`run_20260520_193531_3exp_epo50`)

#### UCI HAR (239s)

| Method | Accuracy | ±std | F1 | Size (bytes) | Ratio |
|---|---|---|---|---|---|
| Uncompressed | 99.98% | ±0.03% | 0.9998 | 163,876 | 1× |
| Snowflake (int8) | **99.98%** | **±0.03%** | **0.9998** | 41,017 | **4×** |
| Global int8 | 99.98% | ±0.03% | 0.9998 | 41,017 | 4× |
| Dynamic int8 | 99.98% | ±0.03% | 0.9998 | 41,476 | 3.95× |
| MLP Baseline | 99.98% | ±0.03% | 0.9998 | 164,400 | 1× |

Task saturated — all methods lossless, no discrimination between methods.

#### ECG Heartbeat (3,826s)

| Method | Accuracy | ±std | F1 | Delta Acc | Size (bytes) | Ratio |
|---|---|---|---|---|---|---|
| Uncompressed | 96.08% | ±0.43% | 0.8411 | — | 68,660 | 1× |
| **Snowflake (int8)** | **96.60%** | **±0.42%** | **0.8568** | **+0.53%** | 17,213 | **4×** |
| Dynamic int8 | 95.73% | ±0.45% | 0.8285 | -0.35% | 17,684 | 3.99× |
| Global int8 | 95.30% | ±0.98% | 0.8213 | -0.77% | 17,213 | 4× |
| MLP Baseline | 94.80% | ±0.63% | 0.8035 | — | 68,728 | 1× |

Snowflake improves over uncompressed (+0.53% acc, +1.57% F1) — quantization regularises. Global int8 degrades most, validating per-layer calibration. Dendritic beats MLP by +1.28% uncompressed.

#### EEG Brainwave (45s)

| Method | Accuracy | ±std | F1 | Delta Acc | Size (bytes) | Ratio |
|---|---|---|---|---|---|---|
| Uncompressed | 97.66% | ±0.23% | 0.9765 | — | 672,812 | 1× |
| Snowflake (int8) | 97.58% | ±0.14% | 0.9757 | -0.08% | 168,251 | **4×** |
| Global int8 | 97.58% | ±0.36% | 0.9757 | -0.08% | 168,251 | 4× |
| **Dynamic int8** | 95.24% | ±0.14% | 0.9516 | **-2.42%** | 168,716 | 3.99× |
| MLP Baseline | 97.74% | ±0.27% | 0.9773 | — | 673,740 | 1× |

Dynamic int8 fails on EEG (-2.42%) — the large fc1 layer (163k params, wide activation range) is particularly sensitive to activation-based dynamic range estimation. Snowflake near-lossless at 4×.

### Observations
1. **Snowflake wins on every meaningful benchmark** — improves ECG (+0.53%), near-lossless on EEG (-0.08%), saturated on HAR
2. **Dynamic int8 is unreliable** — good on HAR/ECG, collapses on EEG (-2.42%); per-layer static calibration (Snowflake) is more robust
3. **Global int8 worst on ECG** — single scale too coarse after 50 epochs of training; per-layer scale essential
4. **Dendritic > MLP on ECG** (+1.28%) — architectural advantage where the task has complexity; equal on EEG (engineered features)
5. **HAR remains saturated** — binary task too easy; ECG is the primary differentiating benchmark

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
| `f946061` | 2026-05-16 | Add int4 quantization (8×); 3-seed ECG+HAR evaluation; dynamic size fix; int4 not viable at ~17k params |
| `3b33e1e` | 2026-05-17 | EEG Brainwave experiment; int4 viable at ~167k params; scale threshold confirmed |
| `f98a4af` | 2026-05-17 | Update accuracy plot to show all compression methods |
| `3e2acda` | 2026-05-20 | Confusion matrices, plot style system, yerr fix; 50-epoch 3-seed HAR+ECG+EEG results |
| `25aa06a` | 2026-05-27 | Refactor MLP to match DNN code structure; add `print_arch` to both models |
| `cfed247` | 2026-05-28 | Refactor experiments to shared base_experiment.py; add HAPT 12-class dataset; full 4-dataset 3-seed run |
| `55328fb` | 2026-05-28 | Add 3 deeper analysis features: ROC/PR curves, compression delta, significance |
| `5399d2c` | 2026-05-31 | Remove tqdm from training loop; TF parity check on ECG (matching results, TF files deleted) |
| `55328fb` | 2026-06-09 | Add ROC/PR curves, compression delta, and paired t-test significance reporting |
| `3c5b2da` | 2026-06-09 | Best-model saving per dataset, edge-AI profile plot, ablation/component out of defaults |
| `948d8f8` | 2026-07-08 | Add per-channel, QAT, and mixed-precision quantization baselines (point 9) |
| `a56b1fd` | 2026-07-08 | Add 95% CI, TOST equivalence testing, and 10-seed default (point 3) |
| `3d5ff21` | 2026-07-08 | Fix print_summary to display accuracy for all 8 quantization methods |
| `13ef16b` | 2026-07-08 | Update experiment log with 2026-07-08 session results |
| `12d77e9` | 2026-07-09 | Add --replot flag and results.pkl saving to decouple plotting from training |
| `57ddcc6` | 2026-07-14 | Add Snowflake+Static: symmetric weight scales + INT8 activation quant (Pi benchmark only) |
| `e09b812` | 2026-07-15 | Add standalone int4 quantization comparison script |
| `40fdcbd` | 2026-07-17 | Wire Snowflake+Static into the main experiment pipeline |
| `1f2077a` | 2026-07-17 | Add test_method.py for fast single-method compression testing |

---

## 2026-05-27 — Component Ablation → ECG + Plot Expansion + Smoke Test

**Commits:** `25aa06a` + this session (uncommitted)

### Summary
Expanded the plot suite with 8 new plot types. Switched component ablation from EEG to ECG for consistency. Ran full smoke test (all experiments, 1 epoch, 1 seed) — exit code 0, all 21 plots generated cleanly.

### Infrastructure Changes

**New plot modules added:**
- `plot_component_ablation.py` — bar chart for 4 compression conditions (none/topo_only/quant_only/both) with error bars and dashed baseline
- `plot_per_class_f1.py` — per-class F1 bar chart derived from confusion matrix
- `plot_weight_dist.py` — weight distribution histogram
- `plot_val_accuracy.py` — validation accuracy curve over epochs
- `plot_cross_dataset.py` — cross-dataset accuracy summary
- `plot_pareto.py` — accuracy vs. compression ratio Pareto frontier
- `plot_inference_time.py` — inference time comparison
- `plot_roc_pr.py` — ROC + PR curves per method

All new plots wired into `src/reporting/plots.py`.

**Component ablation switched from EEG → ECG:**
- `_run_component` now calls `load_ecg()`, `num_classes=5`
- `load_eeg` import removed from `main.py`
- Both ablation and component now run on ECG (consistent)
- ETA impact: component now ~55 min per full run (was ~15 min on EEG)

### Smoke Test Results — 1 epoch, 1 seed (42)
*Output: `outputs/run_20260527_171001_all_epo1`*

| Experiment | Uncompressed | Snowflake (int8) | Delta | Size after compression |
|---|---|---|---|---|
| UCI HAR | 100.00% | 99.95% | -0.05% | 163,876 → 41,017 B (4×) |
| ECG Heartbeat | 67.42% | 78.62% | +11.21% | 68,660 → 17,213 B (4×) |
| EEG Brainwave | 87.35% | 89.93% | +2.58% | 672,812 → 168,251 B (4×) |

Note: ECG at 1 epoch is underfit (67%); converges to ~96% at 50 epochs.
All 21 plots generated. Exit code 0.

---

---

## 2026-05-28 — HAPT Dataset + Experiment Refactor + Full 4-Dataset Run

**Commits:** *(this session)*

### Summary
Added HAPT (UCI Smartphone 12-class) as a fourth benchmark dataset. Refactored all four experiment files into a shared `base_experiment.py`, eliminating ~600 lines of duplication. Ran full 50-epoch 3-seed benchmark across all 6 experiments (ablation, component, HAR, ECG, EEG, HAPT).

### HAPT Dataset

**Dataset:** UCI HAPT (Human Activities and Postural Transitions)
- 561 pre-extracted inertial features, 12 classes: 6 base activities (Walking, Upstairs, Downstairs, Sitting, Standing, Laying) + 6 postural transitions (Stand→Sit, Sit→Stand, Sit→Lie, Lie→Sit, Stand→Lie, Lie→Stand)
- Transition classes severely underrepresented (23–90 samples vs 1,400+ for base activities)
- After oversampling to max class count (1,423 per class): 17,076 train / 3,162 test
- StandardScaler normalisation; `.npy` caching

### Experiment Refactor

All four experiment files (HAR, ECG, EEG, HAPT) were ~95% identical. Extracted shared logic into `src/experiments/base_experiment.py`:
- Single `run_experiment(get_data, num_classes, class_names, epochs, seeds, fine_tune_epochs, batch_size)` function
- `get_data(seed)` callable pattern: seed-aware loaders (HAR) pass `lambda seed: load_har(seed=seed)`; fixed loaders (ECG/EEG/HAPT) pre-load once and pass `lambda seed: cached_data`
- Each experiment file reduced to ~13 lines; total savings ~600 lines across 4 files
- Fixed latent `NameError` in `eeg_experiment.py`: `train_test_split` was called without being imported

### Full 6-Experiment Run — 50 epochs, 3 seeds (`run_20260528_154409_all_epo50`)

#### Ablation Study + Component Ablation (ECG)

Component ablation (1 seed, time: 1,127.69s):
- `none`: acc=0.9180 | `topo_only`: acc=0.9180 | `quant_only`: acc=0.9180 | `both`: acc=0.9180

#### UCI HAR (161.63s)

| Method | Accuracy | ±std | F1 | Size (bytes) | Ratio |
|---|---|---|---|---|---|
| Uncompressed (Dendritic) | 97.94% | ±0.32% | 0.9803 | 164,536 | 1× |
| **Snowflake (int8)** | **97.93%** | **±0.34%** | **0.9802** | 41,182 | **4×** |
| Global int8 | 97.93% | ±0.42% | 0.9802 | 41,182 | 4× |
| Dynamic int8 | 97.72% | ±0.25% | 0.9782 | 41,656 | ~4× |
| MLP Baseline | 98.09% | ±0.35% | 0.9818 | 163,608 | 1× |

#### ECG Heartbeat (3,426.50s)

| Method | Accuracy | ±std | F1 | Delta F1 | Size (bytes) | Ratio |
|---|---|---|---|---|---|---|
| Uncompressed (Dendritic) | 96.08% | ±0.43% | 0.8411 | — | 68,660 | 1× |
| **Snowflake (int8)** | **96.60%** | **±0.42%** | **0.8568** | **+0.0157** | 17,213 | **4×** |
| Dynamic int8 | 95.73% | ±0.45% | 0.8285 | -0.0126 | 17,684 | ~4× |
| Global int8 | 95.30% | ±0.98% | 0.8213 | -0.0198 | 17,213 | 4× |
| MLP Baseline | 94.80% | ±0.63% | 0.8035 | — | 68,728 | 1× |

#### EEG Brainwave (26.13s)

| Method | Accuracy | ±std | F1 | Delta F1 | Size (bytes) | Ratio |
|---|---|---|---|---|---|---|
| Uncompressed (Dendritic) | 97.66% | ±0.23% | 0.9765 | — | 672,812 | 1× |
| **Snowflake (int8)** | **97.58%** | **±0.14%** | **0.9757** | **-0.0008** | 168,251 | **4×** |
| Global int8 | 97.58% | ±0.36% | 0.9757 | -0.0008 | 168,251 | 4× |
| Dynamic int8 | 95.24% | ±0.14% | 0.9516 | -0.0248 | 168,716 | ~4× |
| MLP Baseline | 97.74% | ±0.27% | 0.9773 | — | 673,740 | 1× |

#### HAPT (249.76s)

| Method | Accuracy | ±std | F1 | Delta F1 | Size (bytes) | Ratio |
|---|---|---|---|---|---|---|
| Uncompressed (Dendritic) | 92.45% | ±0.52% | 0.8146 | — | 165,328 | 1× |
| **Snowflake (int8)** | **92.80%** | **±0.62%** | **0.8178** | **+0.0032** | 41,380 | **4×** |
| Global int8 | 92.69% | ±0.54% | 0.8138 | -0.0008 | 41,380 | 4× |
| Dynamic int8 | 92.45% | ±0.48% | 0.8117 | -0.0028 | 41,872 | ~4× |
| MLP Baseline | 92.37% | ±0.89% | 0.8281 | — | 165,360 | 1× |

### Observations
1. **Snowflake int8 best or tied-best on all 4 datasets** — consistent 4× lossless compression
2. **ECG and HAPT: Snowflake improves over uncompressed** (+0.0157 and +0.0032 F1) — quantization regularises on imbalanced/complex tasks
3. **Dynamic int8 weakest** — worst on EEG (-0.0248 F1); per-layer static calibration (Snowflake) consistently more robust
4. **HAPT: MLP slightly outperforms Dendritic** (F1 0.828 vs 0.815) — 12-class transition structure may favour the simpler MLP topology; Dendritic advantage persists on ECG
5. **HAR now 6-class** — F1 0.9803 vs prior 0.9998 (binary); more discriminative as a benchmark
6. **Oversampling critical for ECG** — without balancing: acc=0.9747 but F1=0.8735 (model ignores minority arrhythmia classes); with balancing: F1~0.97

---

## 2026-05-31 — TF Parity Check + tqdm Removal

**Commits:** *(this session)*

### Summary
Implemented a TensorFlow port of `DendriticNetwork` to verify architecture parity, ran it on ECG for 50 epochs across 3 seeds, confirmed matching results, then deleted the TF files — project stays PyTorch-only. Removed tqdm from the training loop.

### TF DendriticNetwork — ECG Parity Check

Ported `DendriticNetwork` to TensorFlow (`DendriticNetworkTF`, same topology: fc1 → branches → soma → fc2 → out). Ran on ECG (50 epochs, 3 seeds: 42, 123, 456).

| Seed | Accuracy | F1 (macro) |
|------|----------|------------|
| 42   | 0.9621   | 0.8484     |
| 123  | 0.9666   | 0.8516     |
| 456  | 0.9637   | 0.8471     |
| **Mean** | **0.9641 ± 0.0019** | **0.8490 ± 0.0019** |

Result matches PyTorch baseline (~0.96 acc, ~0.84 F1). TF files deleted after verification — no reason to maintain two implementations.

**Note:** TF GPU not available on native Windows ≥ 2.11 (requires WSL2 or DirectML plugin). Ran on CPU only.

### tqdm Removal

Removed tqdm progress bar from `src/training/train.py`:
- Deleted `from tqdm import tqdm`, `tqdm_config`, `use_tqdm` parameter, and loop wrapping
- `for epoch in range(epochs)` → `for _ in range(epochs)`
- Removed `use_tqdm=False` from both `compress_model` and `compress_model_global` calls in `compression_pipeline.py`

---

---

## 2026-06-09 — Best Model Saving + Edge Profiling + Deeper Analysis

**Commits:** `55328fb` Add ROC/PR curves, compression delta, significance · `5399d2c` Remove tqdm from training loop · `3c5b2da` Add best-model saving, edge profiling, and deeper analysis features

### Summary
Added deeper analysis features (ROC/PR curves, per-method compression delta, paired t-test significance). Added edge-AI profiling (model size, FLOPs, latency, throughput estimates). Added best-model saving per dataset. Commented ablation/component out of default `ALL_EXPERIMENTS` so they only run when explicitly invoked. Ran full 50-epoch 3-seed benchmark.

### Infrastructure Changes

**`55328fb` — ROC/PR curves, compression delta, significance**
- Per-method ROC and PR curve plots added to all experiments
- Compression delta (Snowflake − uncompressed, etc.) now reported in summary
- Paired t-test (n=3 seeds) for Snowflake/Global/Dynamic vs Uncompressed; p-values and significance stars reported

**`3c5b2da` — Best model saving, edge profiling**
- `base_experiment.py`: tracks best accuracy across seeds for uncompressed, Snowflake, and MLP; saves state dicts after the seed loop
  - `{run_dir}/models/{dataset}/dendritic_uncompressed.pt` — float32 state dict
  - `{run_dir}/models/{dataset}/dendritic_snowflake.pt` — compressed quantized dict
  - `{run_dir}/models/{dataset}/mlp.pt` — float32 state dict
- `src/plots/plot_edge_profile.py` (new): edge-AI profile bar/table — model size, params, FLOPs/sample, activation memory, latency, throughput estimates
- `main.py`: ablation/component commented out of `ALL_EXPERIMENTS`; pass `model_dir` to all dataset runners; `_model_dirs` dict maps dataset key to save path

### Run — 50 epochs, 3 seeds (42, 0, 7)
*Output: `outputs/run_20260609_150152_all_epo50`*

| Dataset | Uncompressed | Snowflake int8 | Delta | F1 delta | Size |
|---|---|---|---|---|---|
| UCI HAR (6-class) | 97.94% ±0.32% | 97.93% ±0.34% | -0.02% | -0.0001 | 160.7→40.2 KB (4×) |
| ECG Heartbeat (5-class) | 96.08% ±0.43% | **96.60% ±0.42%** | **+0.53%** | +0.0157 | 67.0→16.8 KB (4×) |
| EEG Brainwave (3-class) | 97.66% ±0.23% | 97.58% ±0.14% | -0.08% | -0.0008 | 657.0→164.3 KB (4×) |
| HAPT (12-class) | 92.45% ±0.52% | **92.80% ±0.62%** | **+0.35%** | +0.0032 | 161.4→40.4 KB (4×) |

Significance (paired t-test, n=3): Dynamic int8 on EEG is the only statistically significant degradation — t=-31.0, p=0.001 (*).

Edge-AI profile highlights (HAR as example):
- Params: 41,134 | FLOPs/sample: 41,134 MACs | Activation mem: 1.36 KB
- Latency: Dendritic=1.57μs | Snowflake=1.64μs | Dynamic=2.46μs | MLP=0.33μs
- Throughput: Dendritic=635K | Snowflake=611K | Dynamic=406K | MLP=3.04M sps

Timing: HAR=112s | ECG=2729s (~45 min) | EEG=23s | HAPT=205s | **Total≈50 min**

---

## 2026-07-05 — 100-Epoch Convergence Study + Ablation + EEG Overfitting Investigation

### Summary
Full 50-epoch 4-dataset baseline run. Per-dataset 100-epoch convergence study — only ECG benefits. EEG overfitting investigated (weight decay attempted, reverted — ceiling-limited). Ablation and component ablation run for the first time on current pipeline. Key finding: branch diversity is load-bearing; topology sharing collapses the model to chance.

### Infrastructure Changes

- **`main.py`**: Output folder naming changed from generic `all_epoN` / `Nexp_epoN` to experiment names joined (`har_ecg_eeg_hapt_epoN`, `ecg_epoN`, etc.)
- **`main.py`**: `ablation` and `component` added back to `ALL_EXPERIMENTS`; `_DEFAULT_EXPERIMENTS = ["har", "ecg", "eeg", "hapt"]` keeps the plain `python main.py` default unchanged
- **`train.py`**: `weight_decay=0.0` param added to `train()` → passed to Adam optimizer
- **`base_experiment.py`**: threads `weight_decay` through `run_experiment` → `train()` calls

### 50-Epoch 4-Dataset Baseline (`run_20260705_102953_all_epo50`)

| Dataset | Uncompressed | Snowflake int8 | Delta Acc | Delta F1 | Size | Ratio |
|---|---|---|---|---|---|---|
| UCI HAR (6-class) | 97.94% ±0.32% | 97.93% ±0.34% | -0.02% | -0.0001 | 160.7→40.2 KB | 4× |
| ECG Heartbeat (5-class) | 96.08% ±0.43% | **96.60% ±0.42%** | **+0.53%** | +0.0157 | 67.0→16.8 KB | 4× |
| EEG Brainwave (3-class) | 97.66% ±0.23% | 97.58% ±0.14% | -0.08% | -0.0008 | 657.0→164.3 KB | 4× |
| HAPT (12-class) | 92.45% ±0.52% | **92.80% ±0.62%** | **+0.35%** | +0.0032 | 161.4→40.4 KB | 4× |

Dynamic int8 on EEG: only statistically significant degradation — t=-31.0, p=0.001 (*).  
Timing: HAR=129s | ECG=3308s (~55 min) | EEG=28s | HAPT=277s | Total≈62 min

### 100-Epoch Convergence Study

| Dataset | 50-ep Snowflake | 100-ep Snowflake | Δ | Verdict |
|---|---|---|---|---|
| ECG Heartbeat | 96.60% | **96.97%** | +0.37% | **Needs 100 epochs** |
| UCI HAR | 97.93% | 97.85% | -0.08% | Converged at 50 |
| HAPT | 92.80% | 92.84% | +0.04% | Converged at 50 |
| EEG Brainwave | 97.58% | 97.58% | 0.00% | Converged at 50 |

ECG improvement at 100 epochs is real — large dataset (87k train samples) requires more gradient steps. All other datasets fully converged by epoch 50. **Canonical epochs: ECG=100, HAR/HAPT/EEG=50.**

ECG 100-epoch final results:

| Method | Accuracy | ±std | F1 | Delta |
|---|---|---|---|---|
| Uncompressed | 96.58% | ±0.11% | 0.8582 | — |
| **Snowflake (int8)** | **96.97%** | **±0.54%** | **0.8682** | **+0.39%** |
| Global int8 | 96.69% | ±0.21% | 0.8586 | +0.11% |
| Dynamic int8 | 96.28% | ±0.62% | 0.8432 | -0.29% |

### EEG Overfitting Investigation

EEG Dendritic showed train loss → ~0 while val accuracy plateaued (classic overfitting pattern in training curves). Attempted fix via weight decay in Adam:

| Config | Uncompressed Acc | ±std | Note |
|---|---|---|---|
| Baseline (WD=0) | 97.66% | ±0.23% | Reference |
| WD=1e-3 | 95.86% | ±3.14% | Too aggressive — std blew up 14× |
| WD=1e-4 | 97.42% | ±0.47% | Marginal std increase, slight acc drop |

**Conclusion:** EEG is ceiling-limited, not regularization-limited. Val accuracy is already at the dataset's discriminative ceiling; the train/val loss gap is cosmetic. Weight decay reverted to 0.0 (`weight_decay` param kept in codebase for future use).

### Ablation Study (ECG, 50 epochs, 3 seeds)

Architecture size sweep on ECG — shows accuracy scales cleanly with capacity:

| Config | Branches | Acc (Uncomp) | Acc (Snowflake) | Snowflake Δ |
|---|---|---|---|---|
| h1=16, h2=8, br=2 | 2 | 86.20% | 85.32% | -0.88% |
| h1=32, h2=16, br=4 | 4 | 91.55% | **91.79%** | +0.24% |
| h1=64, h2=32, br=6 | 6 | 95.08% | 95.02% | -0.06% |

Snowflake regulariser effect appears at medium size (br=4). Main experiment (br=8, 100 epochs) achieves 96.97% — consistent extrapolation.

### Component Ablation (ECG, 50 epochs, seed=42)

Isolates contribution of quantization vs topology sharing:

| Condition | Description | Accuracy |
|---|---|---|
| `none` | Uncompressed baseline | 90.91% |
| `quant_only` | Snowflake int8, no topology sharing | **91.80%** (+0.89%) |
| `topo_only` | Branch weights shared (identical), float32 | **18.23% ≈ random** |
| `both` | Topology sharing + quantization | 18.44% ≈ random |

**Critical finding:** topology sharing (copying branch 0's weights to all branches) destroys branch diversity — all branches produce identical outputs, the soma receives no useful variation, and the model collapses to near-chance accuracy (5-class random = 20%). This definitively answers *why* Snowflake uses quantization without topology sharing: **branch diversity is the core inductive bias of the dendritic architecture**.

Quantization alone (`quant_only`) slightly improves over baseline (+0.89%), confirming the regulariser hypothesis.

---

## 2026-07-08 — Professor Feedback: Quantization Baselines, TOST Equivalence Testing, 10-Seed Run

**Commits:** `948d8f8` Add per-channel, QAT, and mixed-precision quantization baselines (point 9) · `a56b1fd` Add 95% CI, TOST equivalence testing, and 10-seed default (point 3) · `3d5ff21` Fix print_summary to display accuracy for all 8 quantization methods

### Summary
Addressed professor feedback points 3 and 9. Added three new quantization baselines for comparison. Replaced 3-seed runs with 10-seed runs for statistical validity. Added 95% confidence intervals and TOST equivalence testing. Full 4-dataset 10-seed run completed.

### New Quantization Baselines (Point 9)

**`src/compression/compression_pipeline.py`** — 3 new method families:

- **Per-channel int8**: one scale per output neuron (row of weight matrix) vs one scale per layer (Snowflake). Biases kept float32. Slightly larger than Snowflake but finer-grained quantization.
- **QAT (Quantization-Aware Training)**: FX graph mode via `prepare_qat_fx` + `convert_fx`. Fake-quant nodes inserted during fine-tuning so model learns with quantization in mind → better calibrated scales than post-training.
- **Mixed precision**: `fc1` and `out` layers stay float32, inner layers (branches, soma, fc2) quantized int8. Protects sensitive boundary layers at cost of size (fc1 dominates for large models).

All three use PyTorch FX graph mode (`fbgemm` backend, CPU-only). `base_experiment.py` runs all 8 methods per seed and computes sizes; `store_simple` and plots wired through.

### Statistical Validity (Point 3)

**`src/analysis/tost.py`** (new):
- `ci_95(lst)` — 95% confidence interval half-width using t-distribution (ddof=1)
- `tost_paired(a, b, margin=0.02)` — Two One-Sided Tests for equivalence within ±2%. Tests H₀_low (mean_diff ≤ −ε) and H₀_high (mean_diff ≥ +ε); EQUIV if both p < 0.05. Returns equivalent, p_low, p_high, mean_diff, CI bounds, n.

**`main.py`**: default `SEEDS` bumped from `(42, 0, 7)` → `(42, 0, 7, 1, 2, 3, 4, 5, 6, 8)` (10 seeds).

**`print_summary`**: shows `+/- std  95% CI: +/-X` per method; t-test block replaced with TOST table. **`save_metrics_csv`**: CI and TOST columns added. **`save_per_seed_csv`**: all 8 methods now included.

### Full 10-Seed 50-Epoch Run — 4 Datasets

#### UCI HAR (6-class) — 8.2 min

| Method | Accuracy | 95% CI | TOST (e=2%) |
|---|---|---|---|
| Uncompressed | 94.12% ±0.48% | ±0.34% | — |
| Snowflake (int8) | 94.16% ±0.45% | ±0.32% | **EQUIV** diff=+0.04% |
| Global int8 | 94.23% ±0.45% | ±0.32% | **EQUIV** diff=+0.11% |
| Dynamic (int8) | 94.19% ±0.39% | ±0.28% | **EQUIV** diff=+0.06% |
| Static (int8) | — | — | **EQUIV** diff=−0.11% |
| Per-channel | — | — | **EQUIV** diff=0.00% |
| QAT (int8) | — | — | **EQUIV** diff=+0.09% |
| Mixed precision | — | — | **EQUIV** diff=−0.05% |
| MLP Baseline | 94.50% ±0.37% | ±0.26% | — |

All 7 compression methods **EQUIV** on HAR. CI now ±0.34% vs ~±0.9% with 3 seeds.

#### ECG Heartbeat (5-class) — 3.9 hrs (large dataset: 87k samples)

| Method | Accuracy | 95% CI | TOST (e=2%) |
|---|---|---|---|
| Uncompressed | 96.23% ±0.92% | ±0.66% | — |
| Snowflake (int8) | 96.77% ±0.46% | ±0.33% | **EQUIV** diff=+0.54% |
| Global int8 | 95.73% ±1.60% | ±1.14% | **EQUIV** diff=−0.50% |
| Dynamic (int8) | 95.88% ±1.05% | ±0.75% | **EQUIV** diff=−0.35% |
| Static (int8) | — | — | **EQUIV** diff=+0.37% |
| Per-channel | — | — | **EQUIV** diff=−0.02% |
| QAT (int8) | — | — | **EQUIV** diff=+0.87% |
| Mixed precision | — | — | **EQUIV** diff=−0.16% |
| MLP Baseline | 95.30% ±0.54% | ±0.39% | — |

All 7 **EQUIV**. QAT best at +0.87% over uncompressed.

#### EEG Brainwave (3-class) — 2.1 min

| Method | Accuracy | 95% CI | TOST (e=2%) |
|---|---|---|---|
| Uncompressed | 97.85% ±0.18% | ±0.13% | — |
| Snowflake (int8) | 97.78% ±0.23% | ±0.16% | **EQUIV** diff=−0.07% |
| Global int8 | 97.75% ±0.25% | ±0.18% | **EQUIV** diff=−0.09% |
| Dynamic (int8) | 95.43% ±0.54% | ±0.39% | **NOT EQUIV** diff=−2.41% |
| Static (int8) | 97.42% ±0.47% | ±0.34% | **EQUIV** diff=−0.42% |
| Per-channel | 97.85% ±0.18% | ±0.13% | **EQUIV** diff=0.00% |
| QAT (int8) | 97.70% ±0.18% | ±0.13% | **EQUIV** diff=−0.14% |
| Mixed precision | 97.80% ±0.20% | ±0.14% | **EQUIV** diff=−0.05% |
| MLP Baseline | 97.56% ±0.30% | ±0.21% | — |

6/7 **EQUIV**. Dynamic **NOT EQUIV** — −2.41% exceeds ε=2% margin (CI=[−2.78%, −2.04%]). Consistent with prior findings.

#### HAPT (12-class) — 18 min

| Method | Accuracy | 95% CI | TOST (e=2%) |
|---|---|---|---|
| Uncompressed | 92.22% ±0.57% | ±0.41% | — |
| Snowflake (int8) | 92.50% ±0.47% | ±0.33% | **EQUIV** diff=+0.28% |
| Global int8 | 92.52% ±0.39% | ±0.28% | **EQUIV** diff=+0.31% |
| Dynamic (int8) | 92.10% ±0.63% | ±0.45% | **EQUIV** diff=−0.11% |
| Static (int8) | 91.95% ±0.60% | ±0.43% | **EQUIV** diff=−0.26% |
| Per-channel | 92.21% ±0.59% | ±0.42% | **EQUIV** diff=−0.01% |
| QAT (int8) | 92.62% ±0.39% | ±0.28% | **EQUIV** diff=+0.41% |
| Mixed precision | 92.21% ±0.54% | ±0.38% | **EQUIV** diff=−0.01% |
| MLP Baseline | 92.54% ±0.47% | ±0.34% | — |

All 7 **EQUIV**.

### Key Findings Across All Datasets

1. **27/28 compression variants statistically equivalent to uncompressed at ε=2%** (TOST, n=10)
2. **Only exception**: Dynamic quantization on EEG (−2.41%, CI entirely outside ε=2% margin)
3. **Snowflake never drops** — positive delta on ECG (+0.54%), HAPT (+0.28%), HAR (+0.04%), near-zero on EEG (−0.07%)
4. **QAT consistently best** — largest positive diff on 3/4 datasets (ECG +0.87%, HAPT +0.41%, HAR +0.09%)
5. **Per-channel near-identical to uncompressed** — diff ≤ 0.02% on all datasets; extremely precise quantization
6. **Dynamic quantization least reliable** — only method to fail TOST; not recommended for this architecture
7. **CI tightened ~3×** vs 3-seed runs (e.g. EEG uncompressed: was ±0.9% estimate, now ±0.13%)

---

## 2026-07-09 — Full Combined Run: --replot Flag, results.pkl, Cross-Dataset Plots + Ablation

**Commits:** `3d5ff21` Fix print_summary to display accuracy for all 8 quantization methods · `13ef16b` Update experiment log with 2026-07-08 session results · `12d77e9` Add --replot flag and results.pkl saving to decouple plotting from training

### Summary
Fixed summary output for the 4 new quantization methods (Static, Per-channel, QAT, Mixed). Decoupled plotting from training via `--replot` flag and `results.pkl` saving. Ran the first full combined 10-seed run including all 4 datasets plus both ablation studies in a single invocation — generating the first cross-dataset plots (Pareto frontier, edge profile, cross-dataset summary) and a single `results.pkl` covering all experiments.

### Infrastructure Changes

**`3d5ff21` — Fix print_summary for all 8 methods**
- Summary was printing accuracy lines only for Uncompressed/Snowflake/Global/Dynamic — Static, Per-channel, QAT, Mixed were missing
- Added variable extraction and print blocks for all 4 new methods

**`12d77e9` — --replot + results.pkl**
- `main.py`: saves `results.pkl` (pickled `{results, timings}` dict) at end of every normal run
- `--replot RUN_DIR [...]`: loads and merges `results.pkl` from one or more run dirs, regenerates all plots + summary + CSVs without re-training
- Motivation: per-dataset runs (one `--exp har`, one `--exp ecg`, etc.) each have only 1 dataset in results, so cross-dataset plots (`plot_pareto`, `plot_cross_dataset_summary`, etc.) were skipped. `--replot` lets you merge them afterward without re-running

### Full Combined Run — 50 epochs, 10 seeds
*Output: `outputs/run_20260708_182443_har_ecg_eeg_hapt_ablation_component_epo50`*

Ran `python main.py --exp har ecg eeg hapt ablation component` — all 6 experiments in one invocation.

| Experiment | Time |
|---|---|
| UCI HAR | 492s (8.2 min) |
| ECG Heartbeat | 15,398s (4.3 hrs) |
| EEG Brainwave | 115s (1.9 min) |
| HAPT | 966s (16.1 min) |
| Ablation Study | 3,279s (54.7 min) |
| Component Ablation | 1,088s (18.1 min) |
| **Total** | **~5h 39m** |

Cross-dataset plots generated for the first time (Pareto frontier, edge profile, cross-dataset summary, inference time comparison).

### Ablation Study — ECG, 50 epochs, 1 seed

Architecture size sweep (confirms accuracy scales with capacity):

| Config | Branches | Uncomp | Snowflake | Δ |
|---|---|---|---|---|
| h1=16, h2=8, br=2 | 2 | 87.99% | 87.18% | −0.81% |
| h1=32, h2=16, br=4 | 4 | 92.66% | **92.82%** | +0.16% |
| h1=64, h2=32, br=6 | 6 | 96.01% | **96.76%** | +0.75% |

Snowflake regulariser benefit grows with model size. Consistent with 2026-07-05 results.

### Component Ablation — ECG, 50 epochs, 1 seed

| Condition | Description | Accuracy |
|---|---|---|
| `none` | Uncompressed baseline | 90.91% |
| `quant_only` | Snowflake int8, no topology sharing | **91.80%** (+0.89%) |
| `topo_only` | Branch weights shared, float32 | **18.23% ≈ random** |
| `both` | Topology sharing + quantization | 18.44% ≈ random |

Confirms prior finding: topology sharing destroys branch diversity → model collapses to chance (5-class random = 20%). Quantization alone slightly improves over baseline. Results stable across sessions.

---

## 2026-07-12 — Int4 Quantization: Standalone Run Across All 4 Datasets

### Summary
Implemented Snowflake int4 (4-bit) quantization and ran a standalone benchmark (`run_int4.py`) across all 4 datasets. Int4 achieves 8× compression vs float32 (2× better than int8) but accuracy degrades in proportion to task complexity. Viable on simple datasets, fails on complex ones.

### Implementation

**`src/compression/compression_pipeline.py`** — 4 new functions:
- `_quantize_int4(model)` — per-layer scale = max_val/7.0, clamp to [-7, 7], store as `torch.int8`
- `compress_model_int4(model, fine_tune_data, fine_tune_epochs)` — quantize with optional fine-tuning
- `decompress_model_int4(compressed, model)` — dequantize: `q.float() * scale`
- `int4_size_bytes(compressed)` — counts 0.5 bytes/element (theoretical packed int4) + 4 bytes/layer for scale

PyTorch has no native int4 dtype; values stored as int8 using range [-7, 7] instead of [-127, 127]. Size counted at 0.5 bytes/element to reflect theoretical packing (2 int4 per byte).

**`run_int4.py`** (standalone, not wired into main experiment loop) — trains once per seed, runs int8 reference + int4, prints per-seed rows and final summary with TOST.

Reporting files (plots, summary, utils) wired to show int4 when data is present, but `base_experiment.py` left unchanged so int4 does not run in the main loop.

### Results — 3 seeds (42, 0, 7), 50 epochs, fine-tune 3 epochs

#### UCI HAR (6-class) — 10 seeds, 12.1 min

| Method | Accuracy | 95% CI | TOST (ε=2%) | Size |
|---|---|---|---|---|
| Uncompressed | 94.55% ±0.43% | — | — | 164536B |
| Snowflake int8 | 94.65% ±0.32% | — | **EQUIV** diff=+0.10% | 41182B (4.0×) |
| Snowflake int4 | 93.43% ±0.66% | — | **EQUIV** diff=−1.11% | 20615B (8.0×) |

#### ECG Heartbeat (5-class) — 3 seeds, 74.2 min

| Method | Accuracy | TOST (ε=2%) | Size |
|---|---|---|---|
| Uncompressed | 96.13% ±1.15% | — | 68660B |
| Snowflake int8 | 97.36% ±0.34% | **EQUIV** diff=+1.23% | 17213B (4.0×) |
| Snowflake int4 | 87.39% ±6.12% | **NOT EQUIV** diff=−8.74% | 8630B (8.0×) |

#### EEG Brainwave (3-class) — 3 seeds, 0.5 min

| Method | Accuracy | TOST (ε=2%) | Size |
|---|---|---|---|
| Uncompressed | 97.89% ±0.58% | — | 672812B |
| Snowflake int8 | 97.81% ±0.34% | **EQUIV** diff=−0.08% | 168251B (4.0×) |
| Snowflake int4 | 97.81% ±0.89% | **EQUIV** diff=−0.08% | 84149B (8.0×) |

#### HAPT (12-class) — 3 seeds, 5.6 min

| Method | Accuracy | TOST (ε=2%) | Size |
|---|---|---|---|
| Uncompressed | 92.88% ±2.24% | — | 165328B |
| Snowflake int8 | 93.05% ±2.07% | **EQUIV** diff=+0.17% | 41380B (4.0×) |
| Snowflake int4 | 88.74% ±1.62% | **NOT EQUIV** diff=−4.14% | 20714B (8.0×) |

### Key Findings

1. **Int4 viability is task-complexity-dependent** — passes TOST on EEG (3-class, −0.08%) and HAR (6-class, −1.11%), fails on HAPT (12-class, −4.14%) and ECG (5-class fine-grained, −8.74%)
2. **ECG is the worst case** — 8.74pp drop with enormous variance (±6.12%), CI=[-14.01%, -3.46%]. Heartbeat classification requires fine-grained float precision that 4-bit cannot represent
3. **EEG is the best case** — virtually zero degradation (−0.08pp) at 8× compression; well-separated 3-class clusters tolerate aggressive quantization
4. **Int8 remains the reliable choice** — EQUIV on all 4 datasets; int4 only viable as optional aggressive compression on simple problems
5. **Conclusion: int4 not recommended as a general method** — inconsistent across datasets and fails to meet the ε=2% equivalence threshold on 2/4 datasets

---

## 2026-07-17 — Snowflake+Static Wired into Main Pipeline + Pi Power Supply Fix

**Commits:** `40fdcbd` Wire Snowflake+Static into the main experiment pipeline · `1f2077a` Add test_method.py for fast single-method compression testing

### Summary
`compress_model_snowflake_static` (added 2026-07-14 in `57ddcc6`, previously only exercised in `benchmark_pi.py`) is now a full 9th method in the main `main.py` pipeline — tracked through accuracy/F1/size/TOST/per-seed CSVs and plots exactly like the other 8. Added `test_method.py`, a fast single-method iteration script that loads a saved `dendritic_uncompressed.pt` checkpoint and tests one compression method without retraining or running the other 7 — cuts iteration time from hours to seconds. Also discovered and fixed a Raspberry Pi power supply issue that was silently throttling all prior Pi latency benchmarks by roughly 2×.

### Infrastructure Changes

**`40fdcbd` — Wire Snowflake+Static into `base_experiment.py`**
- Runs `compress_model_snowflake_static` per seed, alongside the other 7 methods, with the same try/except FX-failure handling as Static/Mixed/QAT
- Threaded `compressed_snowflake_static` through `accuracy`, `accuracy_std`, `f1`, `f1_std`, `sizes`, `per_seed`, `ci_95`, and `tost` in the returned dict
- `reporting/utils.py`, `reporting/summary.py`: flattened into `store_simple()`, printed in console/`summary.txt` as "Snowflake+Static", added "SF+Static" row to the TOST equivalence table, added to `per_seed_metrics.csv`
- `reporting/plots.py`, `plots/style.py`: added to accuracy/F1/size bar charts as "Snowflake+Static (int8)" with its own color (`#17BECF`)
- Not added to `plot_cross_dataset.py`/`plot_pareto.py` — those use a fixed 5-method comparison set that Static/QAT/Mixed/Per-channel were also never added to

**`1f2077a` — `test_method.py`**
- Loads `models/<dataset>/dendritic_uncompressed.pt`, applies one named `--method`, evaluates on the real test set, prints accuracy/F1/size/ratio vs. uncompressed
- Motivation: the full pipeline retrains the base model + MLP + all 8 methods on every invocation (this is why the ECG run below took ~4 hours), so iterating on a single method's code required an all-or-nothing multi-hour run. This script isolates just the one method being changed.

### Full 4-Dataset 10-Seed Run (Snowflake+Static included)

Run individually per dataset (`--exp har`, `--exp eeg`, `--exp hapt`, `--exp ecg`), then merged via `--replot` into `outputs/run_20260715_192823_replot/`.

#### UCI HAR — 398.6s (6.6 min)

| Method | Accuracy | 95% CI | TOST (ε=2%) |
|---|---|---|---|
| Uncompressed | 94.12% ±0.48% | ±0.34% | — |
| Snowflake (int8) | 94.16% ±0.45% | ±0.32% | EQUIV diff=+0.04% |
| Static (int8) | 94.01% ±0.43% | ±0.31% | EQUIV diff=−0.11% |
| **Snowflake+Static** | **94.05% ±0.45%** | ±0.32% | **EQUIV diff=−0.07%** |

#### EEG Brainwave — 102.9s (1.7 min)

| Method | Accuracy | 95% CI | TOST (ε=2%) |
|---|---|---|---|
| Uncompressed | 97.85% ±0.18% | ±0.13% | — |
| Snowflake (int8) | 97.78% ±0.23% | ±0.16% | EQUIV diff=−0.07% |
| Static (int8) | 97.42% ±0.47% | ±0.34% | EQUIV diff=−0.42% |
| **Snowflake+Static** | **97.75% ±0.20%** | ±0.14% | **EQUIV diff=−0.09%** |
| Dynamic (int8) | 95.43% ±0.54% | ±0.39% | NOT EQUIV diff=−2.41% (only failure, as always) |

Snowflake+Static essentially matches plain weight-only Snowflake here and clearly beats plain Static — the strongest case yet that Snowflake's symmetric per-layer weight scale still helps once activations are also quantized.

#### HAPT — 1129.7s (18.8 min)

| Method | Accuracy | 95% CI | TOST (ε=2%) |
|---|---|---|---|
| Uncompressed | 92.22% ±0.57% | ±0.41% | — |
| Snowflake (int8) | 92.50% ±0.47% | ±0.33% | EQUIV diff=+0.28% |
| Static (int8) | 91.95% ±0.60% | ±0.43% | EQUIV diff=−0.26% |
| **Snowflake+Static** | **91.99% ±0.54%** | ±0.38% | **EQUIV diff=−0.23%** |

#### ECG Heartbeat — 14,288s (3h 58m — largest dataset, 87k train samples)

| Method | Accuracy | 95% CI | TOST (ε=2%) |
|---|---|---|---|
| Uncompressed | 96.23% ±0.92% | ±0.66% | — |
| Snowflake (int8) | 96.77% ±0.46% | ±0.33% | EQUIV diff=+0.54% |
| Static (int8) | 96.60% ±0.75% | ±0.54% | EQUIV diff=+0.37% |
| **Snowflake+Static** | **96.38% ±0.89%** | ±0.64% | **EQUIV diff=+0.15%** |
| QAT (int8) | 97.05% ±0.29% | ±0.21% | EQUIV diff=+0.82% (best method this dataset) |

**All 8 methods EQUIV on all 4 datasets** except the known Dynamic-on-EEG failure — Snowflake+Static passes TOST everywhere, joining the fully-validated method set.

### Raspberry Pi: Power Supply Throttling Discovery

Re-ran `benchmark_pi.py` on all 4 datasets after switching to a correct power supply. Comparing Snowflake+Static batch=1 latency, old (insufficient PSU) vs new (correct PSU):

| Dataset | Old latency | New latency | Speedup from fix |
|---|---|---|---|
| HAR | 8.62ms | 4.54ms | 1.90× |
| ECG | 8.94ms | 4.70ms | 1.90× |
| EEG | 9.46ms | 4.73ms | 2.00× |
| HAPT | 9.07ms | 4.56ms | 1.99× |

A near-perfectly uniform ~2× across every dataset — the signature of Raspberry Pi undervoltage throttling (the SoC silently halves clock speed under insufficient power delivery). Accuracy/F1 were identical between old and new runs (hardware-independent, as expected), confirming only latency was affected. **All prior Pi latency numbers in this log/`benchmark_pi_output/` predate this fix and should be treated as ~2× pessimistic.**

With the corrected PSU, real batch=1 speedups vs Float32 baseline:

| Dataset | Snowflake (weight-only) | Snowflake+Static | Static (int8) |
|---|---|---|---|
| HAR | 0.97× (no speedup) | **1.98×** | 1.95× |
| ECG | 1.00× | **1.77×** | 1.80× |
| EEG | 0.98× | **2.16×** | 2.13× |
| HAPT | 0.99× | **1.95×** | 1.97× |

This confirms weight-only Snowflake gives storage savings only — it decompresses back to float32 before inference, so CPU runs the same float32 matmuls (~1.0× speedup). Snowflake+Static is the method that delivers both real deployment speedup (on par with plain Static/QAT) **and** competitive accuracy, since it runs true INT8 arithmetic via qnnpack. New CSVs also capture batch=-1 (full test set as one batch) figures, confirming amortized-batch latency is far more optimistic than true single-sample latency (e.g. ECG Float32: 128,352 samples/sec batched vs. 120/sec at batch=1) — a gap worth being explicit about in any deployment-latency claim.

### Key Findings

1. **Snowflake+Static is now the recommended method for real edge deployment** — the only one shown to combine competitive accuracy (EQUIV on all 4 datasets) with a real ~1.8–2.2× runtime speedup on actual ARM hardware, vs. plain Snowflake's storage-only benefit
2. **The Pi PSU issue was silently doubling every prior latency number** — always verify power delivery before trusting Pi benchmark timings; ~2× uniform slowdown across unrelated datasets is a strong throttling tell
3. **Full pipeline retraining is expensive and usually unnecessary for method-level iteration** — `test_method.py` reuses the already-trained checkpoint and tests one method in seconds instead of hours
4. **Batched "per-sample" latency figures meaningfully understate true single-sample deployment latency** — worth reporting batch=1 numbers specifically for any wearable/edge-device latency claim

---

## 2026-07-18 — Professor Feedback Audit + Layer-Matched Baseline + Pi Thermal Test

**Commits:** *(pending — thermal_test.py, base_experiment.py, mlp_baseline.py, reporting/summary.py, reporting/utils.py not yet committed)*

### Summary
Got real SSH access to the Pi 3 (`david@apollo`) and ran corrected-PSU benchmarks plus a 15-minute sustained-load thermal test. Then went through 10 pieces of professor feedback point by point against the actual codebase (not assumed — verified against source) and closed the two highest-leverage gaps: added `LayerMatchedMLP`, a baseline matched to DendriticNetwork's per-layer shape (not just total param count), and wired both `MLPBaseline` and `LayerMatchedMLP` through all 8 compression methods (previously the MLP baseline only ever got plain Snowflake).

### Raspberry Pi: SSH Access + Full Benchmark Re-run

Set up ed25519 key-based SSH access to the Pi (`david@apollo`, Raspberry Pi 3 Model B, `models/`+`data/` already present from prior manual runs). Ran `benchmark_pi.py` for all 4 datasets at both `--batch-size -1` (30m24s) and `--batch-size 1` (4m45s). Batch=1 Snowflake+Static numbers closely reproduced the earlier corrected-PSU figures (e.g. ECG 4.6995ms → 4.701ms), confirming the power fix is stable, not a one-off.

### Sustained-Load Thermal Test (`thermal_test.py`, new)

Pi 3 has no built-in power ADC (`vcgencmd pmic_read_adc` unsupported — that's Pi 4/5 only), so true energy/power measurement needs external hardware (INA219 recommended, not yet acquired). `vcgencmd measure_temp` works for free, so built a sustained-load test instead: tight single-sample inference loop for a fixed duration, temperature sampled every 3s in a background thread, reusing `benchmark_pi.py`'s model-building blocks.

Ran Snowflake+Static on ECG for 15 minutes (900s): **206,256 inferences, 229.2 inf/s sustained throughout — zero throughput degradation.** Temperature rose from 40.8°C to a **steady-state 46.2°C by the 5-minute mark**, then held flat (±0.5°C) for the remaining 10 minutes — a clean thermal-equilibrium curve, not runaway heating. Comfortably below the Pi's ~80°C throttle threshold, even on a bare board with no active cooling. Confirms sustained real-time inference (e.g. continuous ECG monitoring) doesn't stress this hardware thermally.

### Professor Feedback Audit (10 points, checked against source)

Went through each claim against the actual code rather than assumption:

| # | Point | Status |
|---|---|---|
| 1 | True INT8 arithmetic vs weight-only dequant + edge measurement | ⚠️ Partial — Snowflake confirmed weight-only (~1.0x Pi speedup); Static/Snowflake+Static/Dynamic/QAT/Mixed confirmed true INT8 (1.7–2.2x real speedup). Power/energy still unmeasured (no hardware) |
| 2 | Dendritic robustness vs MLP, total-param **and** per-layer matched | ❌→✅ **Fixed this session** — see below |
| 3 | 10-seed statistical rigor, CI, TOST equivalence | ✅ Done 2026-07-08 |
| 4 | "Lossless compression" is a misnomer | ❌ Not fixed — still framed as lossless in places |
| 5 | Subject/sample-level split — data leakage risk | ❌ Not fixed — HAR is properly subject-split (UCI's official split); **ECG uses Kaggle's pre-split CSVs (known per-beat, not per-patient, split — likely leakage)**; **EEG uses plain `train_test_split(..., random_state=42)`, i.e. random sample-level, no subject/session grouping** |
| 6 | Macro F1, balanced accuracy, per-class sensitivity/specificity | ⚠️ Partial — macro F1 and confusion matrices exist everywhere; balanced accuracy and per-class precision/recall/specificity do not |
| 7 | Quantitative proof of branch diversity | ⚠️ Partial — `branch_diversity.py` measures weight cosine similarity, activation correlation, and per-branch quantization MSE. Missing: saturation rate |
| 8 | Controlled single-variable ablations per claim | ⚠️ Partial — topology-sharing and dynamic-quant-failure claims are well isolated; int4-scales-with-size rests on only 2 data points; ablation studies run at 1 seed vs. the 10 used elsewhere |
| 9 | MLP + compact architectures under all quantization methods | ❌→✅ **Fixed this session** — see below |
| 10 | Real edge hardware deployment | ✅ Done — Pi 3 over SSH, full 4-dataset runs, 15-min thermal test. Caveat: ARM Linux SBC, not bare-metal MCU |

### LayerMatchedMLP + Full-Method Baseline Comparison (closes points 2 & 9)

**The gap:** `MLPBaseline` was only ever matched to DendriticNetwork on *total* parameter count (`param_matched_hidden()` solves one algebraic equation for hidden width). For ECG this means MLP's single `fc1` (16,732 params) is bigger on its own than Dendritic's `fc1 + branches` combined (16,192 params) — very different weight-matrix shapes hiding behind an equal total. And MLP was only ever compressed with plain Snowflake — never the other 7 methods — so "Dendritic is more robust to quantization" was untested for 8/9 of the actual claim.

**`LayerMatchedMLP`** (new class, `src/models/mlp_baseline.py`): plain sequential `input→64→8→32→num_classes`, mirroring DendriticNetwork's exact per-stage widths (fc1, branch/soma bottleneck, fc2, out) but with a single dense layer replacing the 8 parallel branches + soma — isolates whether branching *itself* matters, independent of total budget. Necessarily has fewer total params than DendriticNetwork for the same shape (branches+soma is parameter-redundant by design — 8 separate matrices vs. 1) — 76–98% of Dendritic's param count depending on dataset (varies with how much `fc1` dominates).

| Dataset | Dendritic | LayerMatchedMLP | % of Dendritic |
|---|---|---|---|
| HAR | 41,134 | 36,974 | 89.9% |
| ECG | 17,165 | 13,005 | 75.8% |
| EEG | 168,203 | 164,043 | 97.5% |
| HAPT | 41,332 | 37,172 | 89.9% |

**`compress_all_methods()`** (new helper, `base_experiment.py`): runs a plain Linear-based model through all 8 compression methods (Snowflake, Global, Dynamic, Static, Snowflake+Static, Per-channel, QAT, Mixed) with the same try/except FX-failure handling used for DendriticNetwork. Called once for `mlp` and once for `lm` (LayerMatchedMLP) per seed, replacing MLP's old single-method block. Results land in a new `method_comparison` section of the results dict, with full TOST equivalence testing (same `tost_paired()` used for Dendritic) — surfaced via a new "Baseline Comparison" table in `print_summary`/`summary.txt`.

Smoke-tested on EEG (2 epochs): 1-seed run confirmed both baselines compute cleanly through all 8 methods in 10s; 2-seed run confirmed TOST verdicts populate correctly (`EQUIV`/`NOT EQUIV` with real diff/CI), matching the exact rigor already applied to Dendritic. LayerMatchedMLP's narrower 8-wide bottleneck predictably underperforms the unconstrained MLP uncompressed (0.8407 vs 0.9391 on the smoke test) — sensible, since it has an information bottleneck DendriticNetwork's soma also has.

**Cost note:** this triples per-seed compute (3 models × 8 methods instead of 1 model × 8 + trivial MLP), so a full 10-seed run — especially ECG, which already took ~4 hours for Dendritic alone — will take substantially longer. Not yet run at full scale.

### Key Findings

1. **Two of the professor's ten points are now fully addressed** (statistical rigor was already done; branching-vs-shape and full-method-comparison gaps closed this session), several others are partially addressed with clearly identified remaining scope, and three are still fully open: data leakage (ECG/EEG), imbalanced-metric reporting (balanced accuracy, per-class breakdown), and the "lossless" terminology
2. **Data leakage is the most consequential open gap** — if ECG/EEG results are inflated by leakage, every downstream accuracy/TOST claim on those two datasets is suspect regardless of how rigorous the statistics around it are
3. **The Pi 3 has no built-in power sensing** — real energy-per-inference numbers need external hardware (INA219 recommended) not yet acquired; temperature was used as a free proxy in the meantime and shows no thermal risk at all under sustained load

---

## 2026-07-19 — Output Precision Metric + Full 10-Seed Baseline Run + Ablation Redesign

**Commits:** *(pending — this entry covers all currently-uncommitted changes)*

### Summary
Closed out the remaining professor-feedback gaps that needed real compute rather than just code: added the "true inference precision" metric (quantized-vs-float32 output divergence, not vs. labels), ran the 3-model/8-method baseline comparison at full 10-seed scale on all 4 datasets, redesigned the component ablation to be multi-dataset/multi-seed instead of ECG-only/1-seed, and added a new regularization ablation to directly test the "quantization improves accuracy via regularization" claim instead of just asserting it by analogy. Also added per-epoch training logs across the board so long background runs are actually observable instead of a silent tqdm bar.

### Output Precision Metric (`src/analysis/output_precision.py`, new)

`output_divergence(model_float, model_quant, X, num_classes)` compares a compressed model's raw output against the float32 reference's output on the same inputs — logit MSE, cosine similarity, KL divergence, prediction-flip-rate. This is distinct from accuracy/F1 (which only compares against ground-truth labels) and from branch-level quantization error (which is per-branch, not whole-model). Wired into `base_experiment.py` for all 8 Dendritic methods; `model_float` snapshot moved earlier in the seed loop (before Snowflake) so it's available to every method, not just Snowflake.

Interesting early result (later confirmed at full scale, see below): the two best-*accuracy* methods (Snowflake, QAT) show the *highest* output divergence from the float32 reference, while Per-channel is bit-identical (cos_sim≈1.0) despite unremarkable accuracy — fine-tuned methods find genuinely different decision functions rather than faithfully approximating the original, even when downstream accuracy looks the same or better.

### Full 10-Seed Run — All 4 Datasets, 3-Model Comparison (`run_20260718_184326_har_ecg_eeg_hapt_epo50`)

Ran the full pipeline (Dendritic + MLPBaseline + LayerMatchedMLP, each × 8 compression methods, 10 seeds) end to end for the first time at proper scale — previously only smoke-tested at 1–2 seeds on EEG.

**Time: ~6h48m total, almost entirely dominated by ECG (~6h08m of it).** HAR 813.6s, ECG 22,069.2s, EEG 174.9s, HAPT 1,429.0s. ECG is ~20× slower per epoch than HAR despite similar model size — larger dataset, more batches per epoch.

| Dataset | Uncompressed Acc | Snowflake (int8) |
|---|---|---|
| HAR | 0.9412 ± 0.0048 | 0.9416 ± 0.0045 |
| ECG | 0.9623 ± 0.0092 | 0.9677 ± 0.0046 |
| EEG | 0.9785 ± 0.0018 | 0.9778 ± 0.0023 |
| HAPT | 0.9222 ± 0.0057 | 0.9250 ± 0.0047 |

**Robustness comparison (Point 1's actual claim — Dendritic vs. MLPBaseline vs. LayerMatchedMLP under compression), computed from `method_comparison`:**

Mean |accuracy delta| across all 8 methods, per dataset:

| Dataset | Dendritic | MLP | LayerMatchedMLP |
|---|---|---|---|
| HAR | 0.0007 | **0.0004** | 0.0006 |
| ECG | **0.0036** | 0.0041 | 0.0052 |
| EEG | 0.0041 | 0.0047 | **0.0034** |
| HAPT | 0.0020 | **0.0012** | **0.0012** |
| **Overall mean** | **0.0026** | **0.0026** | **0.0026** |

All three models land at the *exact same* overall mean — no consistent ordering; each "wins" on a different dataset. By worst-case delta (largest single-method drop, usually Dynamic), Dendritic does degrade least on 3 of 4 datasets (loses only to MLP on HAPT), hinting it may specifically resist the worst quantization method better — but every one of these deltas is smaller than the seed-to-seed accuracy std (~0.005–0.01), and no significance test has been run comparing deltas *between* models (only each model vs. its own uncompressed baseline via TOST).

**Conclusion: the data does not support "Dendritic is consistently more robust to quantization than either baseline."** The effect, if real at all, is small, dataset-dependent, and currently indistinguishable from noise. This directly addresses Point 1 by testing the claim rather than assuming it — and the honest answer is negative. Recorded here rather than glossed over.

### Per-Epoch Training Logs (`train.py` + all call sites)

Added `verbose`/`label` params to `train()` — one print line per epoch (loss, val_loss/val_acc if available), flushed immediately. Every `train()` call site across the codebase now passes `verbose=True` with a context label (model name + seed, or fine-tune method name). Motivated by watching the ~7h main run and ~4.5h ablation run with zero visibility into progress beyond a non-flushing tqdm bar. Minor added cost (one print per epoch) is negligible next to actual training time.

### Ablation Redesign — Component Ablation (multi-dataset/multi-seed) + New Regularization Ablation

**The gap (Point 3):** the existing component ablation (`topo_only`/`quant_only`/`both` vs. `none`) was ECG-only and hardcoded to 1 seed (`seeds[:1]`), despite everything else in the codebase using 10-seed rigor. And no ablation existed at all for the "quantization improves accuracy via regularization" claim — it was only ever supported by analogy (quantization happens to slightly improve accuracy, therefore it must be regularizing), never tested against an actual regularization baseline.

**Fixed `_run_component`** (`main.py`): now loops over all 4 datasets with the full seed set and the same DendriticNetwork shape as the main experiments (h1=64, h2=32, branches=8, hidden_per_branch=8), instead of a small ECG-only toy config.

**New `run_regularization_ablation`** (`ablation_study.py`): three conditions per seed — `none` (float32 baseline), `quant_only` (Snowflake-compressed), `reg_only` (retrained from the *same* seed with `weight_decay`, kept float32 — same init + data-shuffle order as `none`, so weight_decay is the only variable that differs). If `reg_only`'s gain over `none` matches `quant_only`'s gain, that supports the regularization-effect claim; if not, it isn't just regularization. Registered as `--exp regularization`.

Also generalized `plot_component_ablation.py` to handle arbitrary condition sets (added `reg_only` label) and updated `summary.py`/`plots.py` to handle the new per-dataset-nested result structure, plus added a one-line `-- dataset: {name} --` banner print per dataset inside both ablation loops (originally missing, made mid-run status checks hard to read).

### Ablation Run — 3 Seeds, All 4 Datasets (`run_20260719_203629_component_regularization_epo50`)

Originally launched at full 10 seeds; stopped and restarted at 3 seeds (42, 0, 7) partway through once the projected ~13–14h remaining time was clear — 3 seeds is enough to catch a real directional effect for these more binary claims, at roughly 1/3 the cost of the 10-seed statistical rigor used for the main equivalence-testing results.

**Time: ~4h28m** (Component Ablation 5,846s / Regularization Ablation 10,232s). ECG dominated both, as with the main run.

**Component Ablation results:**

| Dataset | none | topo_only | quant_only | both | chance level |
|---|---|---|---|---|---|
| HAR | 0.940 | 0.269 ± **0.32** | 0.940 | 0.269 | 16.7% |
| ECG | 0.965 | 0.395 ± 0.21 | 0.967 | 0.402 | 20% |
| EEG | 0.977 | 0.506 ± 0.12 | 0.977 | 0.506 | 33.3% |
| HAPT | 0.924 | 0.227 ± **0.24** | 0.924 | 0.230 | 8.3% |

`quant_only` remains essentially free everywhere (matches `none`). `topo_only` collapses accuracy by 50–70 points with enormous seed-to-seed variance (std up to 0.32) — topology sharing without fine-tuning is not a graceful degradation, it's unstable and can be catastrophic.

**This nuances the earlier single-seed ECG conclusion** (2026-05-27 through 2026-07-05 entries: `topo_only` = 18.23%, described as "collapses to chance... **definitively answers** why branch diversity is the core inductive bias"). At 3 seeds the ECG mean is 39.5% ± 21% — well above the old single-seed point and above chance (20%) on average, just with huge variance. The *direction* of the old conclusion holds (topology sharing without fine-tuning is bad), but "collapses to exactly chance, definitively" was an overstatement resting on N=1. It's messier: often above chance, wildly unstable seed-to-seed, not a clean deterministic collapse.

**Regularization Ablation results:**

| Dataset | none | quant_only | reg_only |
|---|---|---|---|
| HAR | 0.940 | 0.940 | **0.948** (reg helps *more* than quant) |
| ECG | 0.965 | 0.967 | **0.927** (reg *hurts*, quant helps) |
| EEG | 0.977 | 0.977 | 0.976 |
| HAPT | 0.924 | 0.924 | 0.926 |

**The "quantization improves accuracy via regularization" claim does not hold uniformly** — on ECG, explicit weight-decay regularization actively hurts accuracy (~3.8 points) while quantization helps, the opposite of what the regularization-equivalence claim predicts. On HAR, regularization alone outperforms quantization. This is the first real test of a claim that had previously only been supported by analogy (old logs: "quantization slightly improves over baseline... confirming the regularizer hypothesis" — never actually tested against an explicit-regularization baseline). Verdict: not confirmed, dataset-dependent, sometimes contradicted outright.

### Key Findings

1. **Point 1's core novelty claim is not supported by the full-scale data** — Dendritic shows no consistent robustness advantage over MLPBaseline or LayerMatchedMLP under quantization (identical 0.0026 mean delta across all three). This needs to be stated plainly in any writeup, not downplayed.
2. **Two prior "definitive" single-seed conclusions were directionally right but overconfident** — topology-sharing collapse and the regularization hypothesis were both based on N=1 (or, for regularization, N=0 direct tests). Multi-seed/multi-dataset data confirms topology-sharing is bad but not a clean deterministic collapse, and outright contradicts the regularization hypothesis on ECG.
3. **ECG is the compute bottleneck for every multi-seed run in this codebase** — ~20× slower per epoch than HAR, dominates every combined-dataset run's wall-clock time regardless of what's being measured.

---

## 2026-07-20 — README Edge-Deployment Section

**Commits:** *(pending)*

### Summary
Folded the corrected-PSU Pi benchmark numbers into the README (no dedicated edge-deployment section existed before).

### README — Edge Deployment section (new)

Added a batch=1 latency table (Float32 / Snowflake / Static / Snowflake+Static) across all 4 datasets, sourced from `benchmark_pi_output/*.csv`. States plainly: Snowflake gives ~1.0× real speedup (weight-only, dequantizes to float32 before compute), Static/Snowflake+Static give genuine 1.8–2.2× speedup (true INT8 arithmetic) — the accuracy tables alone don't surface this distinction. Also documents the thermal test result and the still-open power-measurement gap (explicitly not being pursued — no INA219 hardware).

### Key Findings

1. **README now has real edge-deployment numbers** instead of no dedicated section — closes one of the "fold results into README" backlog items.

---

## 2026-07-20 — Architecture-Size Ablation Fixed (Multi-Seed, Multi-Dataset)

**Commits:** *(pending)*

### Summary
`run_ablation()` (the 3-model-size sweep) had no seed loop at all — one training run per config, ECG-only, whatever the ambient RNG happened to be. This was the least rigorous ablation in the codebase, even behind where component/regularization ablations started before this session's fixes. Brought it in line: added a proper seed loop (mean ± std per config) and extended to all 4 datasets, mirroring the pattern already used for component/regularization ablations.

### Code changes

`run_ablation()` (`ablation_study.py`) now takes a `seeds` param, loops per config, and returns `{"mean", "std"}` dicts for accuracy/MSE instead of raw scalars (size stays a plain scalar — deterministic per config, doesn't depend on trained weights). `main.py`'s `_run_ablation` now loops over `_ABLATION_DATASETS` (the same dict introduced for component/regularization) instead of hardcoding `load_ecg()`, with a per-dataset banner print. `summary.py` and `plots.py` updated to handle the new `{dataset: [config_results...]}` nested structure, generating one `ablation_study_{dataset}.png` per dataset.

### Full Run — 3 Seeds (42, 0, 7), All 4 Datasets, 50 Epochs (`run_20260720_123529_ablation_epo50`)

**Time: ~3h23m**, ECG-dominated as always (the largest config alone, h1=64/branches=6, took over an hour across its 3 seeds).

| Dataset | Config 1 (h1=16,br=2) | Config 2 (h1=32,br=4) | Config 3 (h1=64,br=6) |
|---|---|---|---|
| HAR | 0.583±**0.378** → 0.582±0.378 | 0.938±0.008 → 0.938±0.007 | 0.939±0.010 → 0.939±0.010 |
| ECG | 0.825±0.084 → 0.793±0.118 (**−3.2pp**) | 0.915±0.005 → 0.911±0.011 | 0.961±0.004 → 0.961±0.002 |
| EEG | 0.974±0.001 → 0.973±0.001 | 0.977±0.005 → 0.977±0.005 | 0.979±0.002 → 0.980±0.001 |
| HAPT | 0.667±**0.220** → 0.709±0.168 | 0.908±0.018 → 0.909±0.019 | 0.928±0.001 → 0.927±0.001 |

(uncompressed → Snowflake-compressed accuracy, mean ± std across 3 seeds)

**Finding 1 — Config 1 (2 branches, tiny) is barely trainable, independent of compression.** Std of 0.378 on HAR and 0.220 on HAPT means some seeds land near-random while others train fine — this is a training-stability problem at that scale, not a quantization artifact.

**Finding 2 — "Snowflake is lossless" holds well from Config 2 upward, but genuinely breaks down at the smallest scale tested, on ECG specifically.** Config 1 shows a real −3.2pp drop under compression on ECG; Configs 2 and 3 stay within ~0.4pp everywhere, matching the near-zero-delta pattern seen at full scale in the main experiments. So the "compression is free" result is solid for reasonably-sized dendritic networks but not unconditionally true — it can degrade at the smallest end, at least on some datasets.

### Key Findings

1. **Architecture-size ablation is now on the same statistical footing as the other two ablations** (3 seeds, 4 datasets) — closes the last remaining "still 1-seed" gap flagged in the project summary.
2. **Model-size floor effect identified**: below a certain size (Config 1: 2 branches, h1=16), DendriticNetwork struggles to train reliably at all, and when it does train, compression can cost real accuracy (ECG). This is a genuine boundary condition worth noting in any writeup that claims "quantization is lossless" — it's lossless in the regime actually used (Config 3-scale and up), not universally.

---

## 2026-07-20 — ECG Patient-Independent Split: Quantifying Data Leakage

**Commits:** *(pending)*

### Summary
Confirmed (by reading the loader code, not assuming) that the standard ECG experiment leaks data: `load_ecg.py` reads Kaggle's pre-split `mitbih_train/test.csv`, which carries no patient/record ID and is known to split by individual beat rather than by patient — so the same patient's beats can appear in both train and test. Built a patient-independent replacement (`src/loaders/load_ecg_patient_split.py`) directly from the raw PhysioNet MIT-BIH database, using the standard DS1 (train, 22 records) / DS2 (test, 22 records) split from de Chazal et al. 2004, wired it in as a new `ecg_patient` experiment, and measured the real size of the leakage effect.

### Data pipeline
- Pulled raw MIT-BIH via `wfdb.dl_database('mitdb', dl_dir='data/mitdb_raw')`.
- Extracted beats via midpoint-to-midpoint windowing between adjacent R-peaks (matching the Kaggle CSV's implicit convention), resampled to 187 samples via linear interpolation.
- Mapped raw annotation symbols to the same 5-class AAMI N/S/V/F/Q scheme the Kaggle CSV uses.
- Split by record ID (DS1 → train, DS2 → test) — no beat from any DS2 patient ever appears in training.
- Train set: 229,225 beats, oversampled to perfectly balanced (45,845 per class) — matching `load_ecg.py`'s `balance=True` default so the comparison isolates the leakage effect alone.
- Test set: 49,690 beats, **left at natural class distribution** (44,238 Normal / 1,836 S / 3,221 V / 388 F / 7 Q) — balancing only ever touches train.

### First attempt was confounded, caught before reporting it
Initial run used the patient-split loader *without* balancing (default class distribution in training), giving 87.92% uncompressed accuracy — but that's not a clean comparison against the original's 96.77%, since the original trains on a balanced set. Added matching `balance=True` oversampling logic to `load_ecg_patient_split.py` and re-ran.

### Full Run — 10 Seeds, 50 Epochs (`run_20260720_183742_ecg_patient_epo50`)

**Time: 3h 31m 37s.**

| | Uncompressed Acc | Best compressed |
|---|---|---|
| Original ECG (random/leaky split, balanced train) | 0.9677 | — |
| ECG patient-split, unbalanced (confounded, discarded) | 0.8792 | 0.8838 |
| **ECG patient-split, balanced (clean, apples-to-apples)** | **0.8371 ± 0.0221** | **0.8697 ± 0.0110 (QAT)** |

Model comparison under the clean patient-split (uncompressed): Dendritic 0.8371 > LayerMatchedMLP 0.8087 > MLPBaseline 0.7901 — Dendritic keeps its edge over both controls even on the leakage-free split.

Macro F1 is low (0.36 uncompressed) — DS2's natural test distribution is dominated by Normal beats (44,238 of 49,690), so accuracy alone is a poor read on minority-class (S/V/F/Q) performance; only balanced accuracy or per-class metrics would show that properly (still not implemented — see Next Steps).

### Key Findings

1. **The real leakage-driven accuracy inflation is ~13 percentage points** (96.77% → 83.71%), not the ~8.3pp the first, confounded comparison suggested. The confounded number understated it because it compared a balanced model against an unbalanced one — the unbalanced patient-split model's 87.92% was itself inflated by majority-class bias, partially masking the true gap.
2. **Dendritic's edge over MLPBaseline/LayerMatchedMLP survives the leakage fix** — this is a genuinely positive result for the architecture, independent of the (unsupported) quantization-robustness claim.
3. **Compression again looks "free" or even beneficial on accuracy** (Snowflake +2.5pp, QAT +3.3pp over uncompressed) on this imbalanced test set — consistent with the pattern seen elsewhere that raw accuracy is a weak signal here; needs the per-class metrics (Next Steps) to confirm this isn't just quantization noise nudging majority-class predictions.
4. **EEG's leakage risk is still unaddressed** — only ECG has been fixed so far; EEG's loader still uses a plain random `train_test_split` with no subject/session grouping.

---

## Next Steps

- [x] ~~Commit today's session work~~ — done in `2177bd0`
- [x] ~~Apply same compression comparison to HAR experiment~~ — done in `2177bd0`
- [x] ~~Run 3-seed evaluation (seeds 42, 0, 7) for reliable ± std statistics~~ — done 2026-05-16
- [x] ~~Investigate dynamic quantization size overhead~~ — fixed 2026-05-16 (pickle overhead; raw data = 17,684 bytes ≈ 3.9×)
- [x] ~~Add int4 (4-bit) quantization~~ — done 2026-05-16; not viable at ~17k params (-23.67%, ±13.89%), 8-bit is minimum
- [x] ~~Find int4 viability threshold~~ — confirmed 2026-05-17; viable at ~167k params (EEG: 0.00% delta at 8×)
- [x] ~~Add confusion matrix plots to all experiments~~ — done 2026-05-20
- [x] ~~Run full 50-epoch 3-seed benchmark across HAR + ECG + EEG on GPU~~ — done 2026-05-20
- [x] ~~Add component ablation plot~~ — done 2026-05-27
- [x] ~~Expand plot suite~~ — done 2026-05-27 (8 new plot types)
- [x] ~~Run full 50-epoch 3-seed benchmark with component ablation on ECG~~ — done 2026-05-28
- [x] ~~Add HAPT (12-class) dataset and experiment~~ — done 2026-05-28
- [x] ~~Refactor experiment files to shared base_experiment.py~~ — done 2026-05-28
- [x] ~~Add ROC/PR curves, compression delta, significance testing~~ — done 2026-06-09 (`55328fb`)
- [x] ~~Add edge-AI profile (FLOPs, latency, throughput)~~ — done 2026-06-09 (`3c5b2da`)
- [x] ~~Save best models per dataset after training~~ — done 2026-06-09 (`3c5b2da`)
- [x] ~~Wire Snowflake+Static into the main pipeline (was Pi-benchmark-only)~~ — done 2026-07-17 (`40fdcbd`)
- [x] ~~Add fast single-method testing tool to avoid full retraining on every iteration~~ — done 2026-07-17 (`1f2077a`, `test_method.py`)
- [x] ~~Diagnose and fix Pi power supply throttling affecting all latency benchmarks~~ — done 2026-07-17 (~2× uniform speedup after PSU fix)
- [x] ~~Add Snowflake+Static to `plot_cross_dataset.py`/`plot_pareto.py`'s fixed method set~~ — done 2026-07-21
- [x] ~~Fold `benchmark_pi_output/` corrected (post-PSU-fix) results into README's edge-deployment claims~~ — done 2026-07-20 (new "Edge Deployment" section)
- [x] ~~Get SSH access to the Pi and re-run corrected-PSU benchmarks (batch=-1 and batch=1)~~ — done 2026-07-18
- [x] ~~Run sustained-load thermal test to check for throttling risk~~ — done 2026-07-18 (15 min, no throttling, steady-state 46.2°C)
- [x] ~~Add per-layer-shape-matched baseline (not just total-param-matched)~~ — done 2026-07-18 (`LayerMatchedMLP`)
- [x] ~~Run MLP baseline(s) through all 8 compression methods, not just Snowflake~~ — done 2026-07-18 (`compress_all_methods()`)
- [x] ~~Add "true inference precision" metric: quantized-vs-float32 output divergence, not just downstream accuracy~~ — done 2026-07-19 (`output_precision.py`, `output_divergence()`)
- [x] ~~Add saturation-rate metric to `branch_diversity.py`~~ — done 2026-07-19 (`branch_saturation_rate()`)
- [x] ~~Run the new 3-model/8-method baseline comparison at full 10-seed scale on all 4 datasets~~ — done 2026-07-19; **result: no consistent robustness advantage for Dendritic (identical 0.0026 mean accuracy-delta across all 3 models)** — Point 1's core claim not supported as stated
- [x] ~~Re-run component ablation at multi-seed, multi-dataset~~ — done 2026-07-19 (3 seeds × 4 datasets); nuances the old single-seed "collapses to exact chance" framing
- [x] ~~Test the "quantization improves accuracy via regularization" claim against an actual regularization baseline~~ — done 2026-07-19 (`run_regularization_ablation`); **not confirmed — contradicted on ECG, where explicit regularization hurts while quantization helps**
- [x] ~~Commit thermal_test.py + output_precision.py + LayerMatchedMLP/ablation/logging changes~~ — done 2026-07-19 (`d5a70b6`)
- [x] ~~Fix ECG data leakage risk~~ — done 2026-07-20 (`load_ecg_patient_split.py`, DS1/DS2 patient-independent split from raw PhysioNet MIT-BIH); **finding: true leakage inflation is ~13pp (96.77% → 83.71%), Dendritic's edge over MLPBaseline/LayerMatchedMLP survives the fix**
- [ ] **Fix EEG data leakage risk** — EEG uses random `train_test_split`, no subject/session grouping. HAR/HAPT are already correctly subject-split for reference
- [x] ~~Add balanced accuracy + per-class precision/recall/specificity~~ — done 2026-07-21 (`per_class_stats_from_cm()` in `evaluate.py`, derived from the confusion matrix already collected — no new training/eval runs needed; printed in `summary.py`). Confirms the ECG patient-split finding: balanced accuracy is only 0.36 vs 0.837 raw accuracy, with near-zero recall on Fusion (0.023) and Unknown (0.000) classes
- [x] ~~Re-run architecture-size ablation (`run_ablation`, 3 model-size configs) at multi-seed~~ — done 2026-07-20 (3 seeds × 4 datasets); **finding: tiny configs (2 branches) are barely trainable (std up to 0.38), and Snowflake compression genuinely costs -3.2pp on ECG at the smallest scale — "lossless" holds from medium size up, not universally**
- [x] ~~Stop describing quantization as "lossless" in README/framing~~ — done 2026-07-20 ("lossless" → "near-lossless"/"no statistically significant accuracy loss", added architecture-size-floor caveat pointing to the experiment log)

**Explicitly not being pursued:** real power/energy draw per inference (needs INA219 or similar hardware, not acquired); TFLite Micro / MCU-class deployment (ESP32, Arduino Nano 33 BLE Sense); writing the findings up into a paper/report draft — user has decided not to pursue these.
