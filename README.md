# DNN Compression with Dendritic Networks

This repository evaluates whether a compact, biologically inspired dendritic
network can retain predictive performance after quantization while reducing
storage and, where the runtime supports it, inference latency.

The final evaluation covers:

- MIT-BIH ECG with a patient-independent DS1/DS2 split;
- INCART ECG with a patient-independent 22/4/6-patient split; and
- HAPT with its official subject-independent train/test split.

The complete paper-ready design is recorded in
[`docs/experimental_matrix.md`](docs/experimental_matrix.md), with a CSV copy
in [`docs/complete_experimental_matrix.csv`](docs/complete_experimental_matrix.csv).

## Main findings

- Snowflake INT8 reduces weight storage by approximately 4x with small or
  positive mean changes in accuracy on all three datasets.
- Storage-only quantization does not imply faster inference. Snowflake INT8,
  INT6, and INT4 are dequantized for float computation in this code.
- Static, Snowflake+Static, and QAT INT8 use real INT8 kernels on Raspberry Pi
  and provide approximately 1.9x batch-1 speedup.
- Dendritic branches have lower weight standard deviation than a structurally
  matched non-branching control and near-zero mean inter-branch cosine
  similarity. The robustness advantage is method-dependent, not universal.
- Rare-class ECG performance remains difficult. Accuracy is therefore always
  accompanied by macro-F1, balanced accuracy, per-class metrics, and confusion
  matrices.
- The Raspberry Pi 3 results characterize an edge-gateway SBC, not a
  microcontroller or very-low-power wearable.

## Architecture

```text
Input
  -> FC1 (shared trunk)
  -> N parallel Linear + ReLU branches
  -> concatenate branch activations
  -> Soma integration layer
  -> FC2
  -> classification output
```

The default DendriticNetwork uses:

- trunk widths `h1=64`, `h2=32`;
- 8 branches;
- 8 hidden units per branch; and
- a dataset-specific output dimension.

Two dense controls are included:

- `MLPBaseline`: matched to the DendriticNetwork's total parameter budget;
- `LayerMatchedMLP`: follows the same stage widths without parallel branches.

Architecture-diverse baselines are also included:

- `ECGCNNBaseline`: a compact 1D-CNN for 187-sample ECG beats;
- `CompactHARMLP`: a compact MLP for HAPT's 561 pre-extracted features.

## Quantization methods

| Method | Storage | Compute path | Pi batch-1 behavior |
|---|---:|---|---:|
| Snowflake INT8 | about 4x smaller | Storage-only; dequantized to float | about 1.0x |
| Global INT8 | about 4x smaller | Storage-only; dequantized to float | about 1.0x |
| Per-channel INT8 | about 4x smaller | Storage-only; dequantized to float | about 1.0x |
| Snowflake INT6 | about 5.3x smaller | Storage-only packed-size estimate | No native speedup claimed |
| Snowflake INT4 | about 8x smaller | Storage-only; dequantized to float | about 1.0x |
| Dynamic INT8 | about 4x smaller | Runtime activation quantization | Batch-dependent |
| Static INT8 | about 4x smaller | True INT8 weights and activations | about 1.9x |
| Snowflake+Static INT8 | about 4x smaller | True INT8 weights and activations | about 1.9x |
| QAT INT8 | about 4x smaller | True INT8 after QAT | about 1.9x |
| Mixed precision | about 1.3x smaller | INT8 inner layers; float endpoints | about 1.2-1.4x |

INT4, INT6, and INT8 storage precision are compared separately with:

```bash
python run_precision_comparison.py
```

INT4 QAT was deliberately not added because it would require a separate
low-bit training and deployment pipeline. INT6 provides the requested
intermediate-precision comparison without expanding the experiment suite.

## Final predictive results

Results below are means over 10 seeds. The `+/-` values are seed-to-seed
standard deviations; confidence intervals and paired TOST outputs are stored
in each run summary.

| Dataset | Float accuracy | Snowflake accuracy | Delta | Float macro-F1 | Snowflake macro-F1 | Float balanced accuracy | Snowflake balanced accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|
| MIT-BIH ECG | 83.71% +/- 2.21% | 86.21% +/- 1.04% | +2.50 pp | 0.3595 | 0.3681 | 0.3850 | 0.3839 |
| HAPT | 92.51% +/- 0.74% | 92.77% +/- 0.56% | +0.26 pp | 0.8191 | 0.8267 | 0.8216 | 0.8278 |
| INCART ECG | 78.44% +/- 2.66% | 79.00% +/- 2.19% | +0.56 pp | 0.4053 | 0.4061 | 0.5377 | 0.5399 |

For accuracy TOST with a fixed +/-2 percentage-point margin, 20 of 24
method-dataset comparisons are equivalent: 5/8 on MIT-BIH, 8/8 on HAPT, and
7/8 on INCART. A non-equivalent result does not automatically mean harm: the
three MIT-BIH failures occur because compression improves accuracy beyond the
upper equivalence bound.

The +/-2-point margin is a conventional practical-indifference threshold, not
a clinically derived minimum important difference. It is fixed across
datasets and is generous relative to HAPT's seed variance but comparatively
tight for MIT-BIH. This limitation is stated explicitly in the report.

## Baseline and resource comparison

| Dataset | Dendritic accuracy | Parameter-matched MLP | Layer-matched MLP | Compact baseline |
|---|---:|---:|---:|---:|
| MIT-BIH ECG | 83.71% | 79.06% | 80.69% | ECG 1D-CNN: 67.75% (3 seeds) |
| HAPT | 92.51% | 93.21% | 93.14% | Compact HAPT MLP: 92.80% |
| INCART ECG | 78.44% | 76.53% | 76.62% | ECG 1D-CNN: 54.13% (3 seeds) |

The INCART MLP result should not be interpreted as uniformly worse. Its raw
accuracy is lower than Dendritic, but its macro-F1 is higher (0.4168 versus
0.4053) and its balanced accuracy is higher (0.5955 versus 0.5377), indicating
a different majority/minority-class trade-off.

| Dataset | Dendritic parameters/MACs | Stored size | Activation memory | Desktop latency |
|---|---:|---:|---:|---:|
| MIT-BIH ECG | 17,165 | 67.05 KB | 1.35 KB | 0.46 us/sample |
| HAPT | 41,332 | 161.45 KB | 1.41 KB | 1.56 us/sample |
| INCART ECG | 17,132 | 66.92 KB | 1.34 KB | 0.52 us/sample |

Every float baseline is profiled under the same protocol for parameters,
MACs, stored size, activation memory, and latency. Parameter matching alone is
not treated as sufficient evidence of deployment fairness.

## ECG rare-class evaluation

The patient-independent ECG tasks are strongly imbalanced. For example, the
INCART test set contains 27,001 Normal beats but only 225 Supraventricular and
84 Fusion beats. The final evaluation reports:

- macro-F1 and balanced accuracy;
- per-class precision, recall, specificity, F1, and support;
- normalized confusion matrices;
- ROC and precision-recall curves; and
- supported-class aggregates for classes with sufficient support.

Balancing is applied to training data only. Validation and test distributions
remain natural. Rare-class performance, especially Fusion and Unknown on
MIT-BIH, remains a documented limitation rather than being hidden by raw
accuracy.

## Factorized architecture ablation

The final architecture study uses 20 epochs and seeds 42, 0, and 7. Each sweep
changes exactly one axis around the default configuration.

| Factor | Values | Best MIT-BIH result | Best HAPT result |
|---|---|---:|---:|
| Branch count | 2, 4, 8, 12 | 8 branches: 85.13% | 12 branches: 93.27% |
| Branch width | 2, 4, 8, 16 | width 8: 85.13% | width 2: 92.83% |
| Trunk size | 16/8, 32/16, 64/32, 128/64 | 64/32: 85.13% | 128/64: 92.54% |

The optima are dataset-specific and exploratory; the three-seed ablation does
not have the inferential strength of the 10-seed main comparison.

## Component and regularization controls

INT8 weight quantization alone is nearly accuracy-neutral:

| Dataset | Float | Quantization only | Post-training topology replacement |
|---|---:|---:|---:|
| MIT-BIH ECG | 85.34% | 85.31% | 38.21% |
| HAPT | 93.01% | 92.99% | 23.55% |

The topology condition copies the first trained branch over every other branch
after training. It is therefore evidence that naive post-training replacement
is destructive, not evidence against a model trained with genuinely tied
weights.

The three-seed regularization control also shows that generic weight decay does
not explain quantization robustness:

| Dataset | Float | Quantization only | Weight decay (`1e-3`) |
|---|---:|---:|---:|
| MIT-BIH ECG | 84.87% | 84.87% | 78.28% |
| HAPT | 92.93% | 92.91% | 92.09% |

## Snowflake mechanism diagnostics

The reporting pipeline includes:

- per-branch weight range and standard deviation;
- inter-branch cosine-similarity matrices;
- activation correlation;
- quantization mean-squared error;
- activation clipping/saturation rate;
- logit MSE, output cosine similarity, KL divergence, and prediction flips;
- accuracy, macro-F1, and balanced-accuracy changes after quantization.

Across the final datasets, branch weight standard deviation is approximately
19-21% lower than the matched non-branching control, and mean off-diagonal
inter-branch cosine similarity is near zero. These are descriptive mechanism
results, not a causal proof.

## Raspberry Pi 3 benchmarking

Reported batch-1 runs use four PyTorch threads with affinity to cores 0-3, the
`performance` governor, and an observed 1.2-1.2 GHz frequency range.

| Dataset | Float32 | Static INT8 | Snowflake+Static INT8 |
|---|---:|---:|---:|
| MIT-BIH ECG | 7.95 ms | 4.27 ms (1.86x) | 4.29 ms (1.85x) |
| INCART ECG | 8.54 ms | 4.35 ms (1.96x) | 4.36 ms (1.96x) |
| HAPT | 8.36 ms | 4.41 ms (1.90x) | 4.34 ms (1.93x) |

Protocol:

- 50 warm-up forward passes per method;
- 500 timed calls per method;
- separate batch-1 and full-test-set processes;
- 20 untimed memory runs with RSS sampled outside the latency loop;
- effective thread count, affinity, governor, and frequency stored per row;
- temperature measured immediately before and after every method.

The sustained ECG thermal test runs every method for five minutes, samples
temperature every two seconds, and uses a 60-second cooldown. Float32 sustained
123.2 inferences/s; QAT INT8 sustained 230.5 inferences/s. The maximum observed
temperature was 55.8 C, with `get_throttled=0x0` before and after the run.

Energy per inference is not measured. Raspberry Pi 3 has no suitable built-in
power sensor, and an external meter was unavailable. Temperature is a thermal
risk indicator, not a substitute for energy measurement.

## Setup

```bash
pip install -r requirements.txt
```

Raw MIT-BIH and INCART records are downloaded from PhysioNet through `wfdb`
when absent. HAPT must be placed under `data/hapt/` in the structure expected
by `src/loaders/load_hapt.py`. Generated datasets, caches, models, outputs, and
`final_output/` are ignored by Git.

## Usage

```bash
# Default: MIT-BIH ECG + HAPT
python main.py

# Include independent INCART validation
python main.py --exp ecg incart hapt

# Select epochs and seeds
python main.py --epochs 50 --seeds 42 0 7

# Factorized architecture, component, and regularization studies
python main.py --exp ablation component regularization --epochs 20 --seeds 42 0 7

# Print model architectures
python main.py --arch

# Regenerate plots from one or more completed runs
python main.py --replot outputs/run_A outputs/run_B

# Separate INT4/INT6/INT8 storage comparison
python run_precision_comparison.py

# Plot existing Pi benchmark CSV files
python plot_pi_benchmark.py

# Plot a completed thermal run
python plot_thermal_results.py --input final_output/pi_benchmark/thermal_ecg_YYYYMMDD
```

CLI experiment choices are `ecg`, `incart`, `hapt`, `ablation`, `component`,
and `regularization`. Architecture ablation always uses seeds 42, 0, and 7;
other experiments use the seeds supplied through `--seeds`.

## Outputs

Each run creates `outputs/run_YYYYMMDD_HHMMSS_<tag>/` containing:

- `run.log`;
- `metrics.csv` and `per_seed_metrics.csv` where applicable;
- `summary.txt`;
- `results.pkl` for `--replot`;
- `figures/` and `figures/combined/`;
- saved model checkpoints.

`final_output/` is a local, ignored collection of paper-ready artifacts. The
canonical tracked methodology files are:

- [`docs/experimental_matrix.md`](docs/experimental_matrix.md);
- [`docs/complete_experimental_matrix.csv`](docs/complete_experimental_matrix.csv);
- [`docs/experiment_log.md`](docs/experiment_log.md).

## Repository layout

```text
main.py                         Main experiment CLI
run_precision_comparison.py     Separate INT4/INT6/INT8 study
benchmark_pi.py                 Raspberry Pi latency/RAM benchmark
thermal_test.py                 Sustained Pi thermal benchmark
plot_pi_benchmark.py            Pi benchmark plots
plot_thermal_results.py         Thermal plots
src/models/                     Dendritic, MLP, CNN, and compact HAPT models
src/compression/                Storage and real-INT8 compression pipelines
src/experiments/                Shared and dataset-specific experiments
src/loaders/                    Patient/subject-independent data loaders
src/training/                   Training, loss, and evaluation functions
src/analysis/                   TOST, branch, and output diagnostics
src/reporting/                  Summaries, CSV export, and plot dispatch
src/plots/                      Individual plotting modules
docs/                           Experimental matrix, results CSV, and run log
```

## Scope boundaries

- Raspberry Pi 3 is an ARM Linux edge-gateway device, not a bare-metal MCU.
- No microcontroller/TFLite Micro deployment is claimed.
- No energy-per-inference result is claimed.
- INT4 QAT is outside the current experiment scope.
- Post-training branch copying is not equivalent to training tied weights.
