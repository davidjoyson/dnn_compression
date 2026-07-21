"""
Edge inference benchmark for Raspberry Pi (ARMv8, qnnpack backend).

Measures latency, throughput, and size for all compression methods:
  1. Float32          — uncompressed baseline
  2. Snowflake int8   — per-layer, our main method (4×)
  3. Global int8      — single scale across all layers (4×)
  4. Per-channel int8 — one scale per output neuron (4×)
  5. Snowflake int4   — per-layer, 8× compression
  6. Dynamic int8     — true INT8 matmul via qnnpack (no calibration)
  7. Static W+A       — FX graph, pre-calibrated activations
  8. Mixed precision  — FX graph, fc1+out stay float32
  9. QAT              — Quantization-Aware Training (FX graph)

Usage:
  python benchmark_pi.py --dataset har
  python benchmark_pi.py --dataset ecg --model-dir outputs/.../models/ecg
  python benchmark_pi.py --dataset har --skip-qat
"""
import argparse
import os
import time

import torch

# qnnpack = ARM NEON INT8 (Pi); fbgemm = x86 fallback for local testing
_supported = torch.backends.quantized.supported_engines
BACKEND = "qnnpack" if "qnnpack" in _supported else "fbgemm"
torch.backends.quantized.engine = BACKEND

from src.models.dendritic_network import DendriticNetwork
from src.compression.compression_pipeline import (
    compress_model, decompress_model, compressed_size_bytes,
    compress_model_global,
    compress_model_per_channel, decompress_model_per_channel, per_channel_size_bytes,
    compress_model_int4, decompress_model_int4, int4_size_bytes,
    compress_model_dynamic, dynamic_model_size_bytes,
    compress_model_static, static_model_size_bytes,
    compress_model_mixed, mixed_model_size_bytes,
    compress_model_qat,
    compress_model_snowflake_static,
)
from src.loaders.load_har  import load_har
from src.loaders.load_ecg_patient_split import load_ecg_patient_split
from src.loaders.load_eeg  import load_eeg
from src.loaders.load_hapt import load_hapt

DATASETS = {
    "har":  (561,  6),
    "ecg":  (187,  5),
    "eeg":  (2548, 3),
    "hapt": (561,  12),
}

LOADERS = {
    "har":  load_har,
    "ecg":  load_ecg_patient_split,
    "eeg":  load_eeg,
    "hapt": load_hapt,
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


def global_size_bytes(compressed):
    """int8 weights (1B each) + one global scale (4B)."""
    return sum(e["q"].nelement() for e in compressed.values()) + 4


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=list(DATASETS), default=None,
                        help="Dataset to benchmark. Omit to run all 4.")
    parser.add_argument("--model-dir", default=None,
                        help="Path to saved models dir. Defaults to models/<dataset>.")
    parser.add_argument("--batch-size", type=int, default=-1,
                        help="Samples per forward pass. -1 = all test samples (default).")
    parser.add_argument("--runs",       type=int, default=500)
    parser.add_argument("--warmup",     type=int, default=50)
    parser.add_argument("--skip-qat",   action="store_true",
                        help="Skip QAT (trains for --qat-epochs, slow on Pi).")
    parser.add_argument("--qat-only",   action="store_true",
                        help="Run QAT method only, skip all others.")
    parser.add_argument("--qat-epochs", type=int, default=2)
    parser.add_argument("--output", default=None,
                        help="CSV file to append results to (default: results_<dataset>.csv).")
    args = parser.parse_args()

    datasets = [args.dataset] if args.dataset else list(DATASETS)
    for dataset in datasets:
        args.dataset = dataset
        _run(args)


def _run(args):
    input_dim, num_classes = DATASETS[args.dataset]

    model_dir = args.model_dir or os.path.join("models", args.dataset)

    # Load train + test data (train needed for QAT/calibration)
    X_tr_np, y_tr_np, X_te_np, y_te_np = LOADERS[args.dataset]()
    X_tr = torch.tensor(X_tr_np, dtype=torch.float32)
    y_tr = torch.tensor(y_tr_np, dtype=torch.long)
    X_all = torch.tensor(X_te_np, dtype=torch.float32)
    y_all = torch.tensor(y_te_np, dtype=torch.long)
    X = X_all if args.batch_size == -1 else X_all[:args.batch_size]
    y = y_all if args.batch_size == -1 else y_all[:args.batch_size]
    print(f"Loaded {len(X_te_np)} test samples — using {len(X)}\n")

    rows = []  # (name, lat_ms, std_ms, throughput, size_bytes, acc, f1)

    def fresh():
        """Return a model with trained weights if model-dir given, else random."""
        m = make_model(input_dim, num_classes)
        path = os.path.join(model_dir, "dendritic_uncompressed.pt")
        if os.path.exists(path):
            m.load_state_dict(torch.load(path, map_location="cpu"))
        return m

    def evaluate(model_infer):
        # always evaluate on full test set, regardless of --batch-size
        model_infer.eval()
        with torch.no_grad():
            logits = model_infer(X_all)
        preds = logits.argmax(dim=1)
        acc = (preds == y_all).float().mean().item()
        # macro F1
        classes = y_all.unique()
        f1s = []
        for c in classes:
            tp = ((preds == c) & (y_all == c)).sum().item()
            fp = ((preds == c) & (y_all != c)).sum().item()
            fn = ((preds != c) & (y_all == c)).sum().item()
            p = tp / (tp + fp) if (tp + fp) else 0.0
            r = tp / (tp + fn) if (tp + fn) else 0.0
            f1s.append(2 * p * r / (p + r) if (p + r) else 0.0)
        return acc, sum(f1s) / len(f1s)

    def bench(name, model_infer, size_bytes):
        lat, std, tput = run_benchmark(model_infer, X, args.warmup, args.runs)
        acc, f1 = evaluate(model_infer)
        mem = mem_rss_mb()
        rows.append((name, lat, std, tput, size_bytes, acc, f1, mem))
        print(f"  ok  {name}")

    def skip(name, reason):
        nan = float("nan")
        rows.append((name, nan, nan, nan, 0, nan, nan, nan))
        print(f"  --  {name}  [{reason}]")

    # ── 1. Float32 ─────────────────────────────────────────────────────────
    m_f32 = fresh()
    bench("Float32 (baseline)", m_f32, m_f32.size_bytes())
    f32_size = m_f32.size_bytes()

    if not args.qat_only:
        # ── 2. Snowflake int8 (per-layer) ──────────────────────────────────
        path = os.path.join(model_dir, "dendritic_snowflake.pt")
        c8 = torch.load(path, map_location="cpu") if os.path.exists(path) \
             else compress_model(fresh())
        m_i8 = fresh()
        decompress_model(c8, m_i8)
        bench("Snowflake int8 (per-layer)", m_i8, compressed_size_bytes(c8))

        # ── 3. Global int8 ─────────────────────────────────────────────────
        c_g = compress_model_global(fresh())
        m_g = fresh()
        decompress_model(c_g, m_g)
        bench("Global int8", m_g, global_size_bytes(c_g))

        # ── 4. Per-channel int8 ────────────────────────────────────────────
        c_pc = compress_model_per_channel(fresh())
        m_pc = fresh()
        decompress_model_per_channel(c_pc, m_pc)
        bench("Per-channel int8", m_pc, per_channel_size_bytes(c_pc))

        # ── 5. Snowflake int4 (per-layer) ──────────────────────────────────
        c_i4 = compress_model_int4(fresh())
        m_i4 = fresh()
        decompress_model_int4(c_i4, m_i4)
        bench("Snowflake int4 (per-layer)", m_i4, int4_size_bytes(c_i4))

        # ── 6. Dynamic int8 (qnnpack) ──────────────────────────────────────
        m_dyn = compress_model_dynamic(fresh())
        bench("Dynamic int8 (qnnpack)", m_dyn, dynamic_model_size_bytes(m_dyn))

        # ── 7. Static W+A (FX graph) ───────────────────────────────────────
        try:
            m_st = compress_model_static(fresh(), (X_tr, y_tr), backend=BACKEND)
            bench("Static W+A int8 (FX)", m_st, static_model_size_bytes(m_st))
        except Exception as e:
            skip("Static W+A int8 (FX)", str(e)[:72])

        # ── 8. Snowflake+Static (Snowflake weight scales + INT8 activations) ─
        try:
            m_sws = compress_model_snowflake_static(fresh(), (X_tr, y_tr), backend=BACKEND)
            bench("Snowflake+Static int8", m_sws, static_model_size_bytes(m_sws))
        except Exception as e:
            skip("Snowflake+Static int8", str(e)[:72])

        # ── 9. Mixed precision (FX graph) ──────────────────────────────────
        try:
            m_mx = compress_model_mixed(fresh(), (X_tr, y_tr), backend=BACKEND)
            bench("Mixed precision (FX)", m_mx, mixed_model_size_bytes(m_mx))
        except Exception as e:
            skip("Mixed precision (FX)", str(e)[:72])

    # ── 10. QAT (FX graph) ─────────────────────────────────────────────────
    if args.skip_qat:
        skip("QAT int8 (FX)", "--skip-qat")
    else:
        qat_path = os.path.join(model_dir, "dendritic_qat.pt")
        try:
            if os.path.exists(qat_path):
                m_qat = torch.load(qat_path, map_location="cpu", weights_only=False)
            else:
                m_qat = compress_model_qat(fresh(), (X_tr, y_tr),
                                            epochs=args.qat_epochs,
                                            num_classes=num_classes, backend=BACKEND)
            bench("QAT int8 (FX)", m_qat, static_model_size_bytes(m_qat))
        except Exception as e:
            skip("QAT int8 (FX)", str(e)[:72])

    # ── Print table ─────────────────────────────────────────────────────────
    f32_lat = rows[0][1]

    W = 107
    print(f"\n{'='*W}")
    print(f"  Dataset: {args.dataset.upper()}  |  batch={args.batch_size}  |  "
          f"n={args.runs}  |  backend={BACKEND}")
    print(f"{'='*W}")
    print(f"{'Method':<30} {'Latency':>9} {'+-std':>7} {'Throughput':>12} "
          f"{'Size':>9} {'Speedup':>8} {'Compress':>8} {'Acc':>6} {'F1':>6} {'RSS':>7}")
    print(f"{'-'*W}")
    for name, lat, std, tput, size, acc, f1, mem in rows:
        if lat != lat:  # nan = failed/skipped
            print(f"{name:<30} {'N/A':>9}")
            continue
        speedup  = f32_lat / lat
        compress = f"{f32_size / size:.1f}x" if size else "n/a"
        acc_s = f"{acc*100:.1f}%" if acc == acc else "n/a"
        f1_s  = f"{f1*100:.1f}%" if f1 == f1 else "n/a"
        mem_s = f"{mem:.0f}MB" if mem == mem else "n/a"
        print(f"{name:<30} {lat:>7.3f}ms {std:>5.3f}ms {tput:>10.0f}/s "
              f"{size:>7}B {speedup:>7.2f}x {compress:>8} {acc_s:>6} {f1_s:>6} {mem_s:>7}")
    print(f"{'='*W}\n")

    # ── Save CSV ─────────────────────────────────────────────────────────────
    import csv
    csv_path = args.output or os.path.join("outputs", f"results_{args.dataset}.csv")
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(["dataset", "backend", "batch", "method", "latency_ms", "std_ms",
                        "throughput", "size_bytes", "speedup", "compression", "acc", "f1", "rss_mb"])
        for name, lat, std, tput, size, acc, f1, mem in rows:
            speedup  = f32_lat / lat if lat == lat else ""
            compress = round(f32_size / size, 2) if size else ""
            w.writerow([args.dataset, BACKEND, args.batch_size, name,
                        round(lat, 4) if lat == lat else "",
                        round(std, 4) if std == std else "",
                        round(tput, 1) if tput == tput else "",
                        size,
                        round(speedup, 4) if speedup != "" else "",
                        compress,
                        round(acc, 4) if acc == acc else "",
                        round(f1, 4) if f1 == f1 else "",
                        round(mem, 1) if mem == mem else ""])
    print(f"Results saved to {csv_path}")


if __name__ == "__main__":
    main()
