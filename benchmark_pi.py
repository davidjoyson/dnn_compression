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
import ctypes
import gc
import os
import platform
import re
import subprocess
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
from src.loaders.load_ecg_patient_split import load_ecg_patient_split
from src.loaders.load_hapt import load_hapt
from src.loaders.load_incart import load_incart

DATASETS = {
    "ecg":  (187,  5),
    "incart": (187, 4),
    "hapt": (561,  12),
}

LOADERS = {
    "ecg":  load_ecg_patient_split,
    # The shared benchmark expects train/test arrays only; INCART's loader
    # returns a separate validation partition by default.
    "incart": lambda balance: load_incart(
        balance=balance, return_validation=False
    ),
    "hapt": load_hapt,
}

# Must match each dataset's canonical training config (see docs/experiment_log.md,
# 2026-07-27 entries) -- ECG is balance=True, HAPT is balance=False.
BALANCE = {
    "ecg":  True,
    "incart": True,
    "hapt": False,
}


def make_model(input_dim, num_classes):
    return DendriticNetwork(input_dim=input_dim, hidden_neurons1=64,
                            hidden_neurons2=32, branches=8,
                            hidden_per_branch=8, num_classes=num_classes)


def mem_rss_mb(trim=True):
    # Force glibc to release freed arena pages back to the OS first, so this
    # reads live usage instead of a high-water mark left over from an earlier
    # method's transient allocations (e.g. FX calibration forward passes).
    if trim:
        gc.collect()
        try:
            ctypes.CDLL(None).malloc_trim(0)
        except (OSError, AttributeError, TypeError):
            pass
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024
    except OSError:
        return float("nan")
    return float("nan")


def measure_peak_rss(model, X, n_runs=20):
    """Observed per-method RSS peak in a separate, untimed inference loop."""
    before = mem_rss_mb(trim=True)
    peak = before
    model.eval()
    with torch.no_grad():
        for _ in range(n_runs):
            model(X)
            peak = max(peak, mem_rss_mb(trim=False))
    return before, peak, peak - before


_TEMP_RE = re.compile(r"temp=([\d.]+)")


def read_temperature_c():
    try:
        result = subprocess.run(
            ["vcgencmd", "measure_temp"], capture_output=True,
            text=True, timeout=3, check=False,
        )
        match = _TEMP_RE.search(result.stdout)
        return float(match.group(1)) if match else float("nan")
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return float("nan")


def read_cpu_metadata():
    governors, frequencies = set(), []
    cpu_root = "/sys/devices/system/cpu"
    try:
        cpu_names = sorted(name for name in os.listdir(cpu_root)
                           if re.fullmatch(r"cpu\d+", name))
    except OSError:
        cpu_names = []
    for cpu_name in cpu_names:
        cpufreq = os.path.join(cpu_root, cpu_name, "cpufreq")
        try:
            with open(os.path.join(cpufreq, "scaling_governor")) as f:
                governors.add(f.read().strip())
        except OSError:
            pass
        try:
            with open(os.path.join(cpufreq, "scaling_cur_freq")) as f:
                frequencies.append(int(f.read().strip()) / 1000)
        except (OSError, ValueError):
            pass
    return {
        "governor": ",".join(sorted(governors)) or "unknown",
        "frequency_mhz_min": min(frequencies) if frequencies else None,
        "frequency_mhz_max": max(frequencies) if frequencies else None,
    }


def configure_cpu(threads, affinity, set_governor):
    torch.set_num_threads(threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    if affinity:
        cpus = {int(cpu.strip()) for cpu in affinity.split(",") if cpu.strip()}
        try:
            os.sched_setaffinity(0, cpus)
        except (AttributeError, OSError) as exc:
            print(f"[warn] Could not set CPU affinity to {sorted(cpus)}: {exc}")
    if set_governor:
        changed = 0
        for path in (f"/sys/devices/system/cpu/cpu{i}/cpufreq/scaling_governor"
                     for i in range(os.cpu_count() or 1)):
            try:
                with open(path, "w") as f:
                    f.write(set_governor)
                changed += 1
            except OSError:
                pass
        if not changed:
            print(f"[warn] Could not set CPU governor to {set_governor!r}; "
                  "run with suitable OS permissions or record the existing governor.")


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
                        help="Dataset to benchmark. Omit to run all configured datasets.")
    parser.add_argument("--model-dir", default=None,
                        help="Path to saved models dir. Defaults to models/<dataset>.")
    parser.add_argument("--batch-size", type=int, default=-1,
                        help="Samples per forward pass. -1 = all test samples (default).")
    parser.add_argument("--runs",       type=int, default=500)
    parser.add_argument("--warmup",     type=int, default=50)
    parser.add_argument("--threads", type=int, default=1,
                        help="PyTorch intra-op threads (default: 1 for reproducibility).")
    parser.add_argument("--cpu-affinity", default=None,
                        help="Comma-separated logical CPUs, e.g. 0 or 0,1. Linux only.")
    parser.add_argument("--set-governor", choices=["performance", "powersave"], default=None,
                        help="Attempt to set the CPU governor; usually requires root.")
    parser.add_argument("--memory-runs", type=int, default=20,
                        help="Separate forwards used to observe peak RSS (default: 20).")
    parser.add_argument("--temperature", action="store_true",
                        help="Record vcgencmd temperature immediately before/after each method.")
    parser.add_argument("--skip-qat",   action="store_true",
                        help="Skip QAT (trains for --qat-epochs, slow on Pi).")
    parser.add_argument("--qat-only",   action="store_true",
                        help="Run QAT method only, skip all others.")
    parser.add_argument("--qat-epochs", type=int, default=2)
    parser.add_argument("--output", default=None,
                        help="CSV file to append results to (default: results_<dataset>.csv).")
    args = parser.parse_args()

    if args.threads < 1 or args.runs < 1 or args.warmup < 0 or args.memory_runs < 1:
        parser.error("threads/runs/memory-runs must be positive and warmup non-negative")
    configure_cpu(args.threads, args.cpu_affinity, args.set_governor)

    datasets = [args.dataset] if args.dataset else list(DATASETS)
    for dataset in datasets:
        args.dataset = dataset
        _run(args)


def _run(args):
    input_dim, num_classes = DATASETS[args.dataset]
    cpu_meta = read_cpu_metadata()
    affinity = (sorted(os.sched_getaffinity(0))
                if hasattr(os, "sched_getaffinity") else [])
    print("Benchmark environment:")
    print(f"  platform={platform.platform()}  torch={torch.__version__}  backend={BACKEND}")
    print(f"  threads={torch.get_num_threads()}  affinity={affinity or 'unavailable'}")
    print(f"  governor={cpu_meta['governor']}  "
          f"frequency={cpu_meta['frequency_mhz_min'] or '?'}-"
          f"{cpu_meta['frequency_mhz_max'] or '?'} MHz\n")

    model_dir = args.model_dir or os.path.join("models", args.dataset)

    # Load train + test data (train needed for QAT/calibration)
    X_tr_np, y_tr_np, X_te_np, y_te_np = LOADERS[args.dataset](balance=BALANCE[args.dataset])
    X_tr = torch.tensor(X_tr_np, dtype=torch.float32)
    y_tr = torch.tensor(y_tr_np, dtype=torch.long)
    X_all = torch.tensor(X_te_np, dtype=torch.float32)
    y_all = torch.tensor(y_te_np, dtype=torch.long)
    X = X_all if args.batch_size == -1 else X_all[:args.batch_size]
    y = y_all if args.batch_size == -1 else y_all[:args.batch_size]
    print(f"Loaded {len(X_te_np)} test samples — using {len(X)}\n")

    rows = []

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
        temp_before = read_temperature_c() if args.temperature else float("nan")
        lat, std, tput = run_benchmark(model_infer, X, args.warmup, args.runs)
        acc, f1 = evaluate(model_infer)
        rss_before, rss_peak, rss_delta = measure_peak_rss(
            model_infer, X, args.memory_runs
        )
        temp_after = read_temperature_c() if args.temperature else float("nan")
        rows.append({
            "method": name, "latency_ms": lat, "std_ms": std,
            "throughput": tput, "size_bytes": size_bytes,
            "acc": acc, "f1": f1, "rss_mb": rss_peak,
            "rss_before_mb": rss_before, "rss_peak_mb": rss_peak,
            "rss_delta_mb": rss_delta, "temp_before_c": temp_before,
            "temp_after_c": temp_after,
        })
        print(f"  ok  {name}")

    def skip(name, reason):
        nan = float("nan")
        rows.append({
            "method": name, "latency_ms": nan, "std_ms": nan,
            "throughput": nan, "size_bytes": 0, "acc": nan, "f1": nan,
            "rss_mb": nan, "rss_before_mb": nan, "rss_peak_mb": nan,
            "rss_delta_mb": nan, "temp_before_c": nan, "temp_after_c": nan,
        })
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
    f32_lat = rows[0]["latency_ms"]

    W = 107
    print(f"\n{'='*W}")
    print(f"  Dataset: {args.dataset.upper()}  |  batch={args.batch_size}  |  "
          f"n={args.runs}  |  backend={BACKEND}")
    print(f"{'='*W}")
    print(f"{'Method':<30} {'Latency':>9} {'+-std':>7} {'Throughput':>12} "
          f"{'Size':>9} {'Speedup':>8} {'Compress':>8} {'Acc':>6} {'F1':>6} {'PeakRSS':>8}")
    print(f"{'-'*W}")
    for row in rows:
        name, lat, std = row["method"], row["latency_ms"], row["std_ms"]
        tput, size = row["throughput"], row["size_bytes"]
        acc, f1, mem = row["acc"], row["f1"], row["rss_peak_mb"]
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
    output_dir = os.path.dirname(csv_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    fieldnames = [
        "schema_version", "dataset", "backend", "batch", "method",
        "latency_ms", "std_ms", "throughput", "size_bytes", "speedup",
        "compression", "acc", "f1", "rss_mb", "rss_before_mb",
        "rss_peak_mb", "rss_delta_mb", "temp_before_c", "temp_after_c",
        "warmup_runs", "timed_runs", "memory_runs", "torch_threads",
        "cpu_affinity", "cpu_governor", "cpu_freq_min_mhz",
        "cpu_freq_max_mhz", "torch_version", "platform",
    ]
    if os.path.exists(csv_path):
        with open(csv_path, newline="") as existing:
            existing_header = next(csv.reader(existing), [])
        if existing_header != fieldnames:
            stem, ext = os.path.splitext(csv_path)
            csv_path = f"{stem}_v2{ext or '.csv'}"
            print(f"[warn] Existing CSV uses the old schema; writing {csv_path} instead.")
    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        for row in rows:
            lat, size = row["latency_ms"], row["size_bytes"]
            record = {
                "schema_version": 2, "dataset": args.dataset, "backend": BACKEND,
                "batch": args.batch_size, "method": row["method"],
                "latency_ms": round(lat, 4) if lat == lat else "",
                "std_ms": round(row["std_ms"], 4) if row["std_ms"] == row["std_ms"] else "",
                "throughput": round(row["throughput"], 1) if row["throughput"] == row["throughput"] else "",
                "size_bytes": size,
                "speedup": round(f32_lat / lat, 4) if lat == lat else "",
                "compression": round(f32_size / size, 2) if size else "",
                "acc": round(row["acc"], 4) if row["acc"] == row["acc"] else "",
                "f1": round(row["f1"], 4) if row["f1"] == row["f1"] else "",
                "rss_mb": round(row["rss_peak_mb"], 1) if row["rss_peak_mb"] == row["rss_peak_mb"] else "",
                "rss_before_mb": round(row["rss_before_mb"], 1) if row["rss_before_mb"] == row["rss_before_mb"] else "",
                "rss_peak_mb": round(row["rss_peak_mb"], 1) if row["rss_peak_mb"] == row["rss_peak_mb"] else "",
                "rss_delta_mb": round(row["rss_delta_mb"], 1) if row["rss_delta_mb"] == row["rss_delta_mb"] else "",
                "temp_before_c": row["temp_before_c"] if row["temp_before_c"] == row["temp_before_c"] else "",
                "temp_after_c": row["temp_after_c"] if row["temp_after_c"] == row["temp_after_c"] else "",
                "warmup_runs": args.warmup, "timed_runs": args.runs,
                "memory_runs": args.memory_runs, "torch_threads": torch.get_num_threads(),
                "cpu_affinity": ",".join(map(str, affinity)),
                "cpu_governor": cpu_meta["governor"],
                "cpu_freq_min_mhz": cpu_meta["frequency_mhz_min"] or "",
                "cpu_freq_max_mhz": cpu_meta["frequency_mhz_max"] or "",
                "torch_version": torch.__version__, "platform": platform.platform(),
            }
            writer.writerow(record)
    print(f"Results saved to {csv_path}")


if __name__ == "__main__":
    main()
