"""Sustained-load thermal test for the Pi.

Runs one compression method (or all of them back to back) in a tight,
single-sample inference loop for a fixed duration, sampling CPU temperature
(`vcgencmd measure_temp`) in the background. benchmark_pi.py's 500-run burst
is too short to reveal thermal throttling; this simulates a continuous-
monitoring wearable workload instead.

Usage:
  python thermal_test.py --dataset ecg --method snowflake_static --duration 300
  python thermal_test.py --dataset ecg --method all --duration 300 --cooldown 60
"""
import argparse
import csv
import os
import re
import subprocess
import threading
import time

import torch

from benchmark_pi import DATASETS, LOADERS, make_model, BACKEND
from src.compression.compression_pipeline import (
    compress_model, decompress_model,
    compress_model_global,
    compress_model_dynamic,
    compress_model_static,
    compress_model_snowflake_static,
    compress_model_per_channel, decompress_model_per_channel,
    compress_model_qat,
    compress_model_mixed,
    compress_model_int4, decompress_model_int4,
)

METHODS = ["float32", "snowflake", "global", "dynamic", "static",
           "snowflake_static", "perchan", "qat", "mixed", "int4"]

_TEMP_RE = re.compile(r"temp=([\d.]+)")


def read_temp():
    try:
        out = subprocess.run(["vcgencmd", "measure_temp"], capture_output=True,
                             text=True, timeout=5).stdout
        m = _TEMP_RE.search(out)
        return float(m.group(1)) if m else float("nan")
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return float("nan")


def build_model(method, model_dir, input_dim, num_classes, X_cal, y_cal, ft_epochs):
    def fresh():
        m = make_model(input_dim, num_classes)
        path = os.path.join(model_dir, "dendritic_uncompressed.pt")
        if os.path.exists(path):
            m.load_state_dict(torch.load(path, map_location="cpu"))
        return m

    if method == "float32":
        return fresh().eval()
    if method == "snowflake":
        m = fresh()
        c = compress_model(m, fine_tune_data=(X_cal, y_cal), fine_tune_epochs=ft_epochs)
        decompress_model(c, m)
        return m.eval()
    if method == "global":
        m = fresh()
        c = compress_model_global(m, fine_tune_data=(X_cal, y_cal), fine_tune_epochs=ft_epochs)
        decompress_model(c, m)
        return m.eval()
    if method == "perchan":
        m = fresh()
        c = compress_model_per_channel(m)
        decompress_model_per_channel(c, m)
        return m.eval()
    if method == "int4":
        m = fresh()
        c = compress_model_int4(m, fine_tune_data=(X_cal, y_cal), fine_tune_epochs=ft_epochs)
        decompress_model_int4(c, m)
        return m.eval()
    if method == "dynamic":
        return compress_model_dynamic(fresh())
    if method == "static":
        return compress_model_static(fresh(), (X_cal, y_cal), backend=BACKEND)
    if method == "snowflake_static":
        return compress_model_snowflake_static(fresh(), (X_cal, y_cal), backend=BACKEND)
    if method == "mixed":
        return compress_model_mixed(fresh(), (X_cal, y_cal), backend=BACKEND)
    if method == "qat":
        return compress_model_qat(fresh(), (X_cal, y_cal), epochs=ft_epochs,
                                  num_classes=num_classes, backend=BACKEND)
    raise ValueError(method)


def run_sustained(model, X_sample, duration, interval, log_path):
    """Tight single-sample inference loop for `duration` seconds, sampling temp
    every `interval` seconds in a background thread. Returns a summary dict."""
    readings = []
    stop_flag = threading.Event()

    def sampler():
        t0 = time.time()
        while not stop_flag.is_set():
            readings.append((round(time.time() - t0, 2), read_temp()))
            time.sleep(interval)

    start_temp = read_temp()
    sampler_thread = threading.Thread(target=sampler, daemon=True)
    sampler_thread.start()

    n_infer = 0
    t_start = time.time()
    with torch.no_grad():
        while time.time() - t_start < duration:
            model(X_sample)
            n_infer += 1
    elapsed = time.time() - t_start

    stop_flag.set()
    sampler_thread.join(timeout=interval + 1)
    end_temp = read_temp()
    peak_temp = max((t for _, t in readings), default=end_temp)

    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["elapsed_s", "temp_c"])
        w.writerows(readings)

    return {
        "n_infer": n_infer, "elapsed_s": round(elapsed, 1),
        "inf_per_s": round(n_infer / elapsed, 1) if elapsed else 0.0,
        "start_temp": start_temp, "end_temp": end_temp, "peak_temp": peak_temp,
        "rise_c": round(end_temp - start_temp, 1),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=list(DATASETS))
    parser.add_argument("--method", required=True, choices=METHODS + ["all"])
    parser.add_argument("--model-dir", default=None)
    parser.add_argument("--duration", type=int, default=300,
                        help="Sustained-load duration per method, in seconds (default: 300)")
    parser.add_argument("--interval", type=float, default=2.0,
                        help="Temperature sampling interval, in seconds (default: 2.0)")
    parser.add_argument("--cooldown", type=int, default=60,
                        help="Pause between methods when --method all, in seconds (default: 60)")
    parser.add_argument("--fine-tune-epochs", type=int, default=3)
    args = parser.parse_args()

    input_dim, num_classes = DATASETS[args.dataset]
    model_dir = args.model_dir or os.path.join("models", args.dataset)

    X_tr_np, y_tr_np, X_te_np, y_te_np = LOADERS[args.dataset]()
    X_tr = torch.tensor(X_tr_np, dtype=torch.float32)
    y_tr = torch.tensor(y_tr_np, dtype=torch.long)
    X_sample = torch.tensor(X_te_np[:1], dtype=torch.float32)

    methods = METHODS if args.method == "all" else [args.method]
    summary_rows = []

    for i, method in enumerate(methods):
        print(f"\n[{i+1}/{len(methods)}] Building {method} model for {args.dataset}...")
        model = build_model(method, model_dir, input_dim, num_classes, X_tr, y_tr,
                            args.fine_tune_epochs)

        log_path = os.path.join("thermal_logs", f"{args.dataset}_{method}.csv")
        print(f"  Running sustained load for {args.duration}s (logging every {args.interval}s)...")
        result = run_sustained(model, X_sample, args.duration, args.interval, log_path)
        result["method"] = method
        summary_rows.append(result)

        print(f"  {result['n_infer']} inferences in {result['elapsed_s']}s "
              f"({result['inf_per_s']} inf/s sustained)")
        print(f"  Temp: start={result['start_temp']:.1f}C  end={result['end_temp']:.1f}C  "
              f"peak={result['peak_temp']:.1f}C  rise={result['rise_c']:+.1f}C")
        print(f"  Log -> {log_path}")

        if args.method == "all" and i < len(methods) - 1:
            print(f"  Cooling down {args.cooldown}s before next method...")
            time.sleep(args.cooldown)

    if args.method == "all":
        summary_path = os.path.join("thermal_logs", f"{args.dataset}_all_summary.csv")
        with open(summary_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["method", "n_infer", "elapsed_s", "inf_per_s",
                                              "start_temp", "end_temp", "peak_temp", "rise_c"])
            w.writeheader()
            w.writerows(summary_rows)
        print(f"\nSummary saved -> {summary_path}")


if __name__ == "__main__":
    main()
