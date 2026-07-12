"""
Edge inference benchmark for Raspberry Pi (ARMv8, qnnpack backend).

Measures latency, throughput, and memory for:
  1. Float32 (uncompressed)
  2. Snowflake int8 (dequant before inference — our method)
  3. Dynamic int8 (true INT8 arithmetic via qnnpack NEON)

Usage:
  python benchmark_pi.py --dataset har
  python benchmark_pi.py --dataset ecg --model-dir outputs/run_.../models/ecg
  python benchmark_pi.py --dataset eeg --runs 1000 --batch-size 8
"""
import argparse
import os
import time

import torch
import torch.quantization

# qnnpack = ARM NEON INT8 (Pi); fbgemm = x86 fallback for local testing
_supported = torch.backends.quantized.supported_engines
BACKEND = "qnnpack" if "qnnpack" in _supported else "fbgemm"
torch.backends.quantized.engine = BACKEND

from src.models.dendritic_network import DendriticNetwork
from src.compression.compression_pipeline import (
    compress_model, decompress_model, compressed_size_bytes,
)

DATASETS = {
    "har":  (561,  6),
    "ecg":  (187,  5),
    "eeg":  (2548, 3),
    "hapt": (561,  12),
}


def make_model(input_dim, num_classes):
    return DendriticNetwork(input_dim=input_dim, hidden_neurons1=64,
                            hidden_neurons2=32, branches=8,
                            hidden_per_branch=8, num_classes=num_classes)


def mem_rss_mb():
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024
    except OSError:
        return float("nan")
    return float("nan")


def run_benchmark(model, X, n_warmup=50, n_runs=500):
    model.eval()
    with torch.no_grad():
        for _ in range(n_warmup):
            model(X)
        t_start = time.perf_counter()
        latencies = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            model(X)
            latencies.append(time.perf_counter() - t0)
        total = time.perf_counter() - t_start
    mean_ms = sum(latencies) / len(latencies) * 1000
    std_ms  = (sum((t * 1000 - mean_ms) ** 2 for t in latencies) / len(latencies)) ** 0.5
    throughput = (n_runs * X.shape[0]) / total
    return mean_ms, std_ms, throughput


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=list(DATASETS), required=True)
    parser.add_argument("--model-dir", default=None,
                        help="Path to saved models dir (e.g. outputs/.../models/har). "
                             "If omitted, uses random weights (latency still valid).")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--runs", type=int, default=500)
    parser.add_argument("--warmup", type=int, default=50)
    args = parser.parse_args()

    input_dim, num_classes = DATASETS[args.dataset]
    X = torch.randn(args.batch_size, input_dim)

    rows = []  # (name, lat_ms, std_ms, throughput, size_bytes, mem_delta_mb)

    # ── 1. Float32 ──────────────────────────────────────────────────────────
    model_f32 = make_model(input_dim, num_classes)
    if args.model_dir:
        path = os.path.join(args.model_dir, "dendritic_uncompressed.pt")
        if os.path.exists(path):
            model_f32.load_state_dict(torch.load(path, map_location="cpu"))
            print(f"Loaded {path}")
        else:
            print(f"[warn] {path} not found — using random weights")

    m0 = mem_rss_mb()
    lat, std, tput = run_benchmark(model_f32, X, args.warmup, args.runs)
    mem_delta = mem_rss_mb() - m0
    rows.append(("Float32 (uncompressed)", lat, std, tput, model_f32.size_bytes(), mem_delta))

    # ── 2. Snowflake int8 (dequant) ─────────────────────────────────────────
    model_i8 = make_model(input_dim, num_classes)
    size_i8 = None

    if args.model_dir:
        path = os.path.join(args.model_dir, "dendritic_snowflake.pt")
        if os.path.exists(path):
            compressed = torch.load(path, map_location="cpu")
            decompress_model(compressed, model_i8)
            size_i8 = compressed_size_bytes(compressed)
            print(f"Loaded {path}")
        else:
            print(f"[warn] {path} not found — using random weights")

    if size_i8 is None:
        # ponytail: approximate size without a saved model
        size_i8 = model_f32.size_bytes() // 4

    m0 = mem_rss_mb()
    lat, std, tput = run_benchmark(model_i8, X, args.warmup, args.runs)
    mem_delta = mem_rss_mb() - m0
    rows.append(("Snowflake int8 (dequant)", lat, std, tput, size_i8, mem_delta))

    # ── 3. Dynamic int8 (true qnnpack INT8 arithmetic) ──────────────────────
    model_dyn = make_model(input_dim, num_classes)
    if args.model_dir:
        path = os.path.join(args.model_dir, "dendritic_uncompressed.pt")
        if os.path.exists(path):
            model_dyn.load_state_dict(torch.load(path, map_location="cpu"))

    model_dyn = torch.quantization.quantize_dynamic(
        model_dyn, {torch.nn.Linear}, dtype=torch.qint8
    )
    # int8 weights (1B) + float32 biases (4B) per quantized Linear
    size_dyn = 0
    for mod in model_dyn.modules():
        if hasattr(mod, "_packed_params"):
            try:
                size_dyn += mod.weight().int_repr().nelement()
                if mod.bias() is not None:
                    size_dyn += mod.bias().nelement() * 4
            except Exception:
                pass
    if size_dyn == 0:
        size_dyn = model_f32.size_bytes() // 4  # fallback approximation

    m0 = mem_rss_mb()
    lat, std, tput = run_benchmark(model_dyn, X, args.warmup, args.runs)
    mem_delta = mem_rss_mb() - m0
    rows.append(("Dynamic int8 (qnnpack)", lat, std, tput, size_dyn, mem_delta))

    # ── Print table ──────────────────────────────────────────────────────────
    f32_lat   = rows[0][1]
    f32_size  = rows[0][4]

    print(f"\n{'='*72}")
    print(f"  Dataset: {args.dataset.upper()}  |  batch={args.batch_size}  |  "
          f"n={args.runs}  |  backend={BACKEND}")
    print(f"{'='*72}")
    print(f"{'Method':<28} {'Latency':>9} {'±std':>7} {'Throughput':>12} "
          f"{'Size':>9} {'Speedup':>8}")
    print(f"{'-'*72}")
    for name, lat, std, tput, size, _ in rows:
        speedup = f32_lat / lat
        print(f"{name:<28} {lat:>7.3f}ms {std:>5.3f}ms {tput:>10.0f}/s "
              f"{size:>7}B {speedup:>7.2f}x")
    print(f"{'='*72}")
    print(f"  Size ratios vs float32:  "
          + "  |  ".join(f"{r[0].split()[0]}: {f32_size/r[4]:.1f}x" if r[4] else f"{r[0].split()[0]}: n/a" for r in rows))
    print(f"{'='*72}\n")


if __name__ == "__main__":
    main()
