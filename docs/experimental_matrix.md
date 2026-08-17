# Complete Experimental Matrix

This is the single source of truth for the paper's experimental design and
completed protocol. Results below refer to the final August 2026 runs unless
explicitly marked as outside the current claims.

## Master matrix

| Study block | Dataset / split | Models | Compression / conditions | Training / repetitions | Predictive metrics | Efficiency / diagnostic metrics | Hardware | Implementation status |
|---|---|---|---|---|---|---|---|---|
| Main ECG | MIT-BIH, patient-independent DS1/DS2, 5 AAMI classes; balanced training | Dendritic (`64/32`, 8 branches x width 8); total-parameter-matched MLP; layer-matched MLP; compact ECG 1D-CNN | Float32; Snowflake INT8; global INT8; dynamic INT8; static INT8; Snowflake+Static INT8; per-channel INT8; QAT INT8; mixed precision | 50 epochs; 3 fine-tune epochs; batch 256; 10 seeds for Dendritic/matched MLPs; ECG-CNN limited to seeds 42, 0, 7 | Accuracy; macro-F1; balanced accuracy; per-class precision, recall, specificity, F1; confusion matrix; ROC/PR; 95% CI and paired TOST (+/-0.02) | Parameters; stored size; compression ratio; output divergence; branch diagnostics; FLOPs, activation memory and latency where available | Desktop CPU; Raspberry Pi 3 for selected methods | Implemented; current aggregate results generated |
| Main HAR | HAPT, official subject-independent train/test split, 12 classes; natural training distribution | Dendritic (`64/32`, 8 branches x width 8); total-parameter-matched MLP; layer-matched MLP; compact HAR MLP | Same main method set as ECG | 50 epochs; 3 fine-tune epochs; batch 128; same 10 seeds | Same as main ECG | Same as main ECG | Desktop CPU; Raspberry Pi 3 for selected methods | Implemented; current aggregate results generated |
| Independent ECG validation | INCART, patient-independent split, 4 AAMI classes; balanced training | Dendritic (`64/32`, 8 branches x width 8); total-parameter-matched MLP; layer-matched MLP; compact ECG 1D-CNN | Same main method set as ECG | 50 epochs; 3 fine-tune epochs; batch 256; 10 seeds for Dendritic/matched MLPs; ECG-CNN limited to seeds 42, 0, 7 | Same as main ECG | Same as main ECG | Desktop CPU and Raspberry Pi 3 | Complete; opt-in locally via `--exp incart` |
| Precision sweep | MIT-BIH and HAPT | Dendritic, with one shared float checkpoint per seed | Snowflake INT8 vs INT6 vs INT4; weight storage only; equal fine-tuning budget | 50 epochs; 3 fine-tune epochs; seeds 42, 0, 7 | Accuracy and macro-F1 | Packed-size estimate and compression ratio | No native low-bit benchmark: values are dequantized for float32 computation | Complete in `run_precision_comparison.py` |
| Architecture ablation: branch count | MIT-BIH and HAPT | Dendritic | Branches = 2, 4, 8, 12; branch width and trunk fixed | 20 epochs; seeds 42, 0, 7 | Float32 and Snowflake accuracy; compression delta | Parameters and stored size | Desktop CPU | Complete |
| Architecture ablation: branch width | MIT-BIH and HAPT | Dendritic | `hidden_per_branch` = 2, 4, 8, 16; branch count and trunk fixed | 20 epochs; seeds 42, 0, 7 | Float32 and Snowflake accuracy; compression delta | Parameters and stored size | Desktop CPU | Complete |
| Architecture ablation: trunk size | MIT-BIH and HAPT | Dendritic | Paired (`h1`, `h2`) = (16,8), (32,16), (64,32), (128,64); branch axes fixed | 20 epochs; seeds 42, 0, 7 | Float32 and Snowflake accuracy; compression delta | Parameters and stored size | Desktop CPU | Complete |
| Compression-component ablation | MIT-BIH and HAPT | Dendritic | Baseline; naive post-training branch copying; INT8 weight quantization; both | 50 epochs; 10 seeds | Accuracy and compression delta | Stored size | Desktop CPU | Complete; topology condition must be described as post-training branch replacement, not tied-weight training |
| Regularization control | MIT-BIH and HAPT | Dendritic | Baseline; INT8 weight quantization; weight decay (`1e-3`) | 50 epochs; seeds 42, 0, 7 | Accuracy and compression delta | Stored size | Desktop CPU | Complete; exploratory three-seed control |
| Snowflake mechanism evidence | MIT-BIH, HAPT and INCART main runs | Dendritic plus structurally matched non-branching control | Float32 vs Snowflake INT8 | First trained seed for branch diagnostics; accuracy uses all seeds | Accuracy after quantization | Per-branch weight range/std; cosine similarity; activation correlation; quantization error; clipping/saturation; output divergence | Desktop CPU | Complete |
| Statistical equivalence | Each main dataset and applicable method | Dendritic and baselines | Compressed method paired with its own float32 model by seed | 10 paired seeds | Accuracy, macro-F1 and balanced-accuracy CI/TOST, margin &plusmn;2 percentage points | Mean paired difference; 95% CI; both one-sided p-values | Not applicable | Complete for main runs |
| Resource fairness | Each main dataset | Dendritic; matched MLPs; compact ECG/HAR baseline | Float32 models under one profiling protocol; compressed stored sizes retained per method | 1 warm-up and 30 timed full-test calls | Predictive metrics linked to resource point | Parameters; FLOPs/MACs; stored size; activation memory; latency | Desktop CPU | Complete for main runs |
| Edge latency and memory | MIT-BIH, INCART and HAPT checkpoints | Dendritic variants | Float32 and storage-only/true-INT8 methods kept distinct | 50 warm-ups; 500 timed calls; separate untimed peak-RSS loop; batch 1 and full-batch in separate runs; 4 threads on cores 0-3; performance governor at 1.2 GHz | Not an accuracy experiment | Latency mean/std; throughput; RSS before/peak/delta; model size; speedup; CPU metadata; before/after temperature | Raspberry Pi 3 Model B, ARM Cortex-A53, QNNPACK | Complete for all three datasets |
| Sustained thermal test | MIT-BIH ECG checkpoint | Dendritic | All 10 benchmark methods | 5-minute continuous single-sample load per method; 2-second sampling; 60-second cooldown | Not applicable | Temperature rise, peak temperature and sustained throughput | Raspberry Pi 3 Model B; 4 cores; performance governor | Complete; maximum observed temperature 55.8°C; no throttling flag observed before or after the run |
| Energy / microcontroller | Not assigned | Not assigned | Not assigned | — | — | Energy per inference | External power meter or MCU-class target | Not performed; explicitly outside current claims |

## Main predictive-metric snapshot

| Dataset | Float accuracy (95% CI half-width) | Snowflake accuracy (95% CI half-width) | Float -> Snowflake macro-F1 | Float -> Snowflake balanced accuracy | Float -> Snowflake storage |
|---|---:|---:|---:|---:|---:|
| MIT-BIH ECG | 83.71% (+/-1.58 pp) | 86.21% (+/-0.74 pp) | 0.3595 -> 0.3681 | 0.3850 -> 0.3839 | 68,660 -> 17,213 B |
| HAPT | 92.51% (+/-0.53 pp) | 92.77% (+/-0.40 pp) | 0.8191 -> 0.8267 | 0.8216 -> 0.8278 | 165,328 -> 41,380 B |
| INCART ECG | 78.44% (+/-1.90 pp) | 79.00% (+/-1.57 pp) | 0.4053 -> 0.4061 | 0.5377 -> 0.5399 | 68,528 -> 17,180 B |

## Baseline and resource snapshot

| Dataset | Matched MLP accuracy | Layer-matched MLP accuracy | Compact baseline accuracy | Dendritic MACs/sample | Activation memory | Desktop Dendritic latency |
|---|---:|---:|---:|---:|---:|---:|
| MIT-BIH ECG | 79.06% | 80.69% | ECG 1D-CNN: 67.75% (3 seeds) | 17,165 | 1.35 KB | 0.46 us/sample |
| HAPT | 93.21% | 93.14% | Compact HAPT MLP: 92.80% | 41,332 | 1.41 KB | 1.56 us/sample |
| INCART ECG | 76.53% | 76.62% | ECG 1D-CNN: 54.13% (3 seeds) | 17,132 | 1.34 KB | 0.52 us/sample |

## Raspberry Pi metric snapshot

| Dataset | Float32 latency | Static INT8 latency (speedup) | Snowflake+Static latency (speedup) | Float32 peak RSS | Snowflake+Static peak RSS |
|---|---:|---:|---:|---:|---:|
| MIT-BIH ECG | 7.95 ms | 4.27 ms (1.86x) | 4.29 ms (1.85x) | 709.4 MB | 364.0 MB |
| INCART ECG | 8.54 ms | 4.35 ms (1.96x) | 4.36 ms (1.96x) | 449.2 MB | 538.2 MB |
| HAPT | 8.36 ms | 4.41 ms (1.90x) | 4.34 ms (1.93x) | 384.3 MB | 483.6 MB |

The sustained ECG thermal test recorded 123.2 inferences/s for Float32 and
230.5 inferences/s for QAT INT8. The maximum observed CPU temperature was
55.8 degrees C, and `get_throttled=0x0` was observed before and after the run.

## Completed ablation snapshot

| Finding | MIT-BIH ECG | HAPT |
|---|---:|---:|
| Best branch-count setting | 8 branches: 85.13% | 12 branches: 93.27% |
| Best branch-width setting | width 8: 85.13% | width 2: 92.83% |
| Best trunk setting | 64/32: 85.13% | 128/64: 92.54% |
| Quantization-only component control | 85.31% vs 85.34% float | 92.99% vs 93.01% float |
| Post-training topology replacement | 38.21% | 23.55% |
| Weight-decay control | 78.28% vs 84.87% float | 92.09% vs 92.93% float |

Architecture entries are means over three seeds and 20 epochs. Component
entries use ten seeds and 50 epochs. Regularization entries use three seeds and
50 epochs. The dataset-specific optima should therefore be treated as
exploratory trends rather than a universal architecture ranking.

## Interpretation boundaries

- Snowflake INT8, INT6 and INT4 are weight-storage formats in this code. They
  are dequantized before float32 matrix multiplication and do not establish a
  native low-bit latency improvement.
- Dynamic, static, Snowflake+Static and QAT INT8 can exercise real INT8 compute;
  latency claims must come from the hardware benchmark rather than stored size.
- Raspberry Pi 3 is treated as an edge-gateway SBC, not a microcontroller or
  very-low-power wearable platform.
- The architecture and regularization controls use three exploratory seeds and
  should not be presented with the inferential strength of the ten-seed main
  comparisons.
- The thermal run records temperature and aggregate five-minute throughput. It
  does not continuously log CPU frequency, throttling flags, ambient
  temperature, or windowed latency, so it supports a bounded thermal-risk
  statement rather than a full thermal-stability claim.
