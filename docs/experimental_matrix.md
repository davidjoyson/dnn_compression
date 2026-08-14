# Complete Experimental Matrix

This is the single source of truth for the paper's experimental design. A
check mark means the code path exists; **rerun required** means previously
saved results predate the current multi-seed confusion-matrix aggregation or
factorized ablation implementation. Planned items must not be reported as
completed results.

## Master matrix

| Study block | Dataset / split | Models | Compression / conditions | Training / repetitions | Predictive metrics | Efficiency / diagnostic metrics | Hardware | Implementation status |
|---|---|---|---|---|---|---|---|---|
| Main ECG | MIT-BIH, patient-independent DS1/DS2, 5 AAMI classes; balanced training | Dendritic (`64/32`, 8 branches x width 8); total-parameter-matched MLP; layer-matched MLP; compact ECG 1D-CNN | Float32; Snowflake INT8; global INT8; dynamic INT8; static INT8; Snowflake+Static INT8; per-channel INT8; QAT INT8; mixed precision | 50 epochs; 3 fine-tune epochs; batch 256; 10 seeds: 42, 0, 7, 1, 2, 3, 4, 5, 6, 8 | Accuracy; macro-F1; balanced accuracy; per-class precision, recall, specificity, F1; confusion matrix; ROC/PR; 95% CI and paired TOST (+/-0.02) | Parameters; stored size; compression ratio; output divergence; branch diagnostics; FLOPs, activation memory and latency where available | Desktop CPU; Raspberry Pi 3 for selected methods | Implemented; **rerun required** for aggregate per-class plots |
| Main HAR | HAPT, official subject-independent train/test split, 12 classes; natural training distribution | Dendritic (`64/32`, 8 branches x width 8); total-parameter-matched MLP; layer-matched MLP; compact HAR MLP | Same main method set as ECG | 50 epochs; 3 fine-tune epochs; batch 128; same 10 seeds | Same as main ECG | Same as main ECG | Desktop CPU; Raspberry Pi 3 for selected methods | Implemented; **rerun required** for aggregate per-class plots |
| Independent ECG validation | INCART, patient-independent split, 4 AAMI classes; balanced training | Dendritic (`64/32`, 8 branches x width 8); total-parameter-matched MLP; layer-matched MLP; compact ECG 1D-CNN | Same main method set as ECG | 50 epochs; 3 fine-tune epochs; batch 256; same 10 seeds | Same as main ECG | Same as main ECG | Desktop CPU; Raspberry Pi not yet rerun for current checkpoint | Implemented; opt-in via `--exp incart`; **rerun required** |
| Precision sweep | MIT-BIH and HAPT | Dendritic, with one shared float checkpoint per seed | Snowflake INT8 vs INT6 vs INT4; weight storage only; equal fine-tuning budget | 50 epochs; 3 fine-tune epochs; 3 seeds by default: 42, 0, 7 | Accuracy; macro-F1; 95% CI; paired TOST against float32 | Packed-size estimate and compression ratio | No native low-bit benchmark: values are dequantized for float32 computation | Implemented separately in `run_precision_comparison.py`; results not yet run |
| Architecture ablation: branch count | MIT-BIH and HAPT | Dendritic | Branches = 2, 4, 8, 12; branch width and trunk fixed | User-selected seeds; paper target 10 | Float32 and Snowflake accuracy; compression delta | Parameters and stored size | Desktop CPU | Implemented; results not yet rerun |
| Architecture ablation: branch width | MIT-BIH and HAPT | Dendritic | `hidden_per_branch` = 2, 4, 8, 16; branch count and trunk fixed | User-selected seeds; paper target 10 | Float32 and Snowflake accuracy; compression delta | Parameters and stored size | Desktop CPU | Implemented; results not yet rerun |
| Architecture ablation: trunk size | MIT-BIH and HAPT | Dendritic | Paired (`h1`, `h2`) = (16,8), (32,16), (64,32), (128,64); branch axes fixed | User-selected seeds; paper target 10 | Float32 and Snowflake accuracy; compression delta | Parameters and stored size | Desktop CPU | Implemented; results not yet rerun |
| Compression-component ablation | MIT-BIH and HAPT | Dendritic | Components of the Snowflake procedure enabled/disabled under a fixed architecture | User-selected seeds; paper target 10 | Accuracy and compression delta | Stored size | Desktop CPU | Implemented; results should be rerun with final seeds |
| Regularization control | MIT-BIH and HAPT | Dendritic | Baseline, weight decay only, quantization only, and combined condition | User-selected seeds; paper target 10 | Accuracy and compression delta | Stored size | Desktop CPU | Implemented; results should be rerun with final seeds |
| Snowflake mechanism evidence | MIT-BIH, HAPT and INCART main runs | Dendritic plus structurally matched non-branching control | Float32 vs Snowflake INT8 | First trained seed for branch diagnostics; accuracy uses all seeds | Accuracy after quantization | Per-branch weight range/std; cosine similarity; activation correlation; quantization error; clipping/saturation; output divergence | Desktop CPU | Implemented; verify final plots after rerun |
| Statistical equivalence | Each main dataset and applicable method | Dendritic and baselines | Compressed method paired with its own float32 model by seed | 10 paired seeds | Accuracy, macro-F1 and balanced-accuracy CI/TOST, margin ±2 percentage points | Mean paired difference; 95% CI; both one-sided p-values | Not applicable | Implemented; final results require rerun |
| Resource fairness | Each main dataset | Dendritic; matched MLPs; compact ECG/HAR baseline | Float32 models under one profiling protocol; compressed stored sizes retained per method | 1 warm-up and 30 timed full-test calls | Predictive metrics linked to resource point | Parameters; FLOPs/MACs; stored size; activation memory; latency | Desktop CPU | Implemented for every baseline; final values require rerun |
| Edge latency and memory | Current trained ECG/HAPT checkpoints where available | Dendritic variants | Float32 and storage-only/true-INT8 methods kept distinct | 50 warm-ups; 500 timed calls; separate untimed peak-RSS loop; batch 1 and full-batch in separate runs; configurable threads/affinity/governor | Not an accuracy experiment | Latency mean/std; throughput; RSS before/peak/delta; model size; speedup; CPU metadata; optional before/after temperature | Raspberry Pi 3 Model B, ARM Cortex-A53, QNNPACK | Strengthened protocol implemented; hardware values require rerun |
| Sustained thermal test | ECG representative checkpoint | Dendritic | Snowflake+Static or selected method | 15-minute load; temperature sampled every 2 s | Not applicable | Temperature and sustained throughput | Raspberry Pi 3 Model B | Implemented for representative method, not every method |
| Energy / microcontroller | Not assigned | Not assigned | Not assigned | — | — | Energy per inference | External power meter or MCU-class target | Not performed; explicitly outside current claims |

## Interpretation boundaries

- Snowflake INT8, INT6 and INT4 are weight-storage formats in this code. They
  are dequantized before float32 matrix multiplication and do not establish a
  native low-bit latency improvement.
- Dynamic, static, Snowflake+Static and QAT INT8 can exercise real INT8 compute;
  latency claims must come from the hardware benchmark rather than stored size.
- Raspberry Pi 3 is treated as an edge-gateway SBC, not a microcontroller or
  very-low-power wearable platform.
- Balanced-accuracy CI/TOST and complete float-model baseline resource profiles
  are implemented; final values require the planned multi-seed rerun.
