import csv
import io
import math
import os
from datetime import datetime

from .utils import to_float


def make_run_dir(label=None):
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = f"_{label}" if label else ""
    run_dir = os.path.join("outputs", f"run_{stamp}{suffix}")
    os.makedirs(os.path.join(run_dir, "figures"), exist_ok=True)
    return run_dir


def save_metrics_csv(results, run_dir):
    def _delta(a, b):
        return round(a - b, 6) if not (math.isnan(a) or math.isnan(b)) else ""

    def _ratio(u, c):
        return round(u / c, 3) if c and not math.isnan(u) else ""

    rows = []
    for name, r in results.items():
        if not isinstance(r, dict) or "accuracy_uncompressed" not in r:
            continue
        chk = r["accuracy_uncompressed"]
        if isinstance(chk, list) or (hasattr(chk, "numel") and chk.numel() > 1):
            continue

        acc_u     = to_float(r.get("accuracy_uncompressed",   float("nan")))
        acc_c     = to_float(r.get("accuracy_compressed",     float("nan")))
        acc_mlp   = to_float(r.get("accuracy_mlp_baseline",   float("nan")))
        acc_mlp_c = to_float(r.get("accuracy_mlp_compressed", float("nan")))

        rows.append({
            "experiment":                  name,
            "acc_uncompressed":            round(acc_u, 6),
            "acc_compressed":              round(acc_c, 6),
            "acc_mlp_baseline":            round(acc_mlp, 6)   if not math.isnan(acc_mlp)   else "",
            "acc_mlp_compressed":          round(acc_mlp_c, 6) if not math.isnan(acc_mlp_c) else "",
            "acc_delta_dendritic":         _delta(acc_c,     acc_u),
            "acc_delta_mlp":               _delta(acc_mlp_c, acc_mlp),
            "std_uncompressed":            r.get("std_uncompressed",   0.0),
            "std_compressed":              r.get("std_compressed",     0.0),
            "std_mlp_baseline":            r.get("std_mlp_baseline",   0.0),
            "std_mlp_compressed":          r.get("std_mlp_compressed", 0.0),
            "mse_uncompressed":            r.get("mse_uncompressed",   ""),
            "mse_compressed":              r.get("mse_compressed",     ""),
            "mse_mlp_baseline":            r.get("mse_mlp_baseline",   ""),
            "mse_mlp_compressed":          r.get("mse_mlp_compressed", ""),
            "size_dendritic_u":            r.get("size_uncompressed",     ""),
            "size_dendritic_c":            r.get("size_compressed",       ""),
            "compression_ratio_dendritic": _ratio(r.get("size_uncompressed", float("nan")),
                                                   r.get("size_compressed",   float("nan"))),
            "size_mlp_u":                  r.get("size_mlp_uncompressed", ""),
            "size_mlp_c":                  r.get("size_mlp_compressed",   ""),
            "compression_ratio_mlp":       _ratio(r.get("size_mlp_uncompressed", float("nan")),
                                                   r.get("size_mlp_compressed",   float("nan"))),
            "time_seconds":                round(r.get("time_seconds", 0.0), 2),
            "num_seeds":                   r.get("num_seeds", 1),
        })

    if not rows:
        return
    csv_path = os.path.join(run_dir, "metrics.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Metrics saved -> {csv_path}")


def print_summary(results, timings):
    print("\n=== Experiment Summary ===\n")

    for name, r in results.items():
        _t = (r.get("time_seconds") if isinstance(r, dict) else None) or timings.get(name)
        time_str = f"  Time             : {_t:.2f} sec\n" if _t is not None else "\n"

        if isinstance(r, list):
            print(f"{name} ({len(r)} configurations):")
            for i, cfg_r in enumerate(r):
                cfg = cfg_r.get("config", {})
                au  = cfg_r.get("accuracy_uncompressed", float("nan"))
                ac  = cfg_r.get("accuracy_compressed",   float("nan"))
                mu  = cfg_r.get("mse_uncompressed",      float("nan"))
                mc  = cfg_r.get("mse_compressed",        float("nan"))
                tag = f"h1={cfg.get('h1','?')} h2={cfg.get('h2','?')} br={cfg.get('branches','?')}"
                print(f"  Config {i+1} {tag}: acc_u={au:.4f} acc_c={ac:.4f}  mse_u={mu:.4f} mse_c={mc:.4f}")
            print(time_str, end="")
            continue

        if name == "Component Ablation" and isinstance(r, dict) and "none" in r:
            print(f"{name}:")
            for condition, stats in r.items():
                mse_m = stats.get("mse_mean", float("nan"))
                mse_s = stats.get("mse_std",  0.0)
                print(f"  {condition:<12}: acc={stats['mean']:.4f} +/- {stats['std']:.4f}  mse={mse_m:.4f} +/- {mse_s:.4f}")
            print(time_str, end="")
            continue

        if name == "Folktables Multi-State" and isinstance(r, dict) and "test_states" in r:
            print(f"{name} (trained on {r['train_state']}):")
            mse_u_list = r.get("mse_uncompressed", [float("nan")] * len(r["test_states"]))
            mse_c_list = r.get("mse_compressed",   [float("nan")] * len(r["test_states"]))
            for state, au, ac, mu, mc in zip(
                r["test_states"], r["accuracy_uncompressed"], r["accuracy_compressed"],
                mse_u_list, mse_c_list,
            ):
                print(f"  {state}: u_acc={au:.4f} u_mse={mu:.4f}  c_acc={ac:.4f} c_mse={mc:.4f}")
            print(f"  Size (U -> C)    : {r['size_uncompressed']} -> {r['size_compressed']} bytes")
            print(time_str, end="")
            continue

        if not isinstance(r, dict) or "accuracy_uncompressed" not in r:
            print(f"{name}: Completed (complex output)")
            print(time_str, end="")
            continue

        acc_u_val = r["accuracy_uncompressed"]
        if hasattr(acc_u_val, "numel") and acc_u_val.numel() != 1:
            print(f"{name}: Completed (multi-dimensional results)")
            print(time_str, end="")
            continue

        acc_u   = to_float(acc_u_val)
        acc_c   = to_float(r["accuracy_compressed"])
        acc_mlp = to_float(r.get("accuracy_mlp_baseline",   float("nan")))
        std_u   = to_float(r.get("std_uncompressed",  0.0))
        std_c   = to_float(r.get("std_compressed",    0.0))
        std_mlp = to_float(r.get("std_mlp_baseline",  0.0))
        acc_mlp_c     = to_float(r.get("accuracy_mlp_compressed", float("nan")))
        std_mlp_c     = to_float(r.get("std_mlp_compressed",      0.0))
        acc_global   = to_float(r.get("accuracy_compressed_global",  float("nan")))
        std_global   = to_float(r.get("std_compressed_global",       0.0))
        acc_dynamic  = to_float(r.get("accuracy_compressed_dynamic", float("nan")))
        std_dynamic  = to_float(r.get("std_compressed_dynamic",      0.0))
        acc_int4     = to_float(r.get("accuracy_compressed_int4",    float("nan")))
        std_int4     = to_float(r.get("std_compressed_int4",         0.0))
        mse_u         = to_float(r.get("mse_uncompressed",     float("nan")))
        mse_c         = to_float(r.get("mse_compressed",       float("nan")))
        mse_mlp       = to_float(r.get("mse_mlp_baseline",     float("nan")))
        mse_mlp_c     = to_float(r.get("mse_mlp_compressed",   float("nan")))
        std_mse_u     = to_float(r.get("mse_std_uncompressed",   0.0))
        std_mse_c     = to_float(r.get("mse_std_compressed",     0.0))
        std_mse_mlp   = to_float(r.get("mse_std_mlp_baseline",   0.0))
        std_mse_mlp_c = to_float(r.get("mse_std_mlp_compressed", 0.0))

        n_seeds   = r.get("num_seeds", 1)
        seed_note = f" (mean over {n_seeds} seeds)" if n_seeds > 1 else ""

        print(f"{name}{seed_note}:")
        print(f"  Uncompressed Acc : {acc_u:.4f} +/- {std_u:.4f}")
        print(f"  Snowflake (int8) : {acc_c:.4f} +/- {std_c:.4f}  [{r.get('size_uncompressed','?')} -> {r.get('size_compressed','?')} bytes]")
        if acc_global == acc_global:
            print(f"  Global int8      : {acc_global:.4f} +/- {std_global:.4f}  [{r.get('size_uncompressed','?')} -> {r.get('size_compressed_global','?')} bytes]")
        if acc_dynamic == acc_dynamic:
            print(f"  Dynamic (int8)   : {acc_dynamic:.4f} +/- {std_dynamic:.4f}  [{r.get('size_uncompressed','?')} -> {r.get('size_compressed_dynamic','?')} bytes]")
        if acc_int4 == acc_int4:
            print(f"  Snowflake (int4) : {acc_int4:.4f} +/- {std_int4:.4f}  [{r.get('size_uncompressed','?')} -> {r.get('size_compressed_int4','?')} bytes]")
        if acc_global == acc_global and acc_dynamic == acc_dynamic:
            delta_line = f"  -- Compression delta: Snowflake(8b)={acc_c - acc_u:+.4f} | Global(8b)={acc_global - acc_u:+.4f} | Dynamic(8b)={acc_dynamic - acc_u:+.4f}"
            if acc_int4 == acc_int4:
                delta_line += f" | Snowflake(4b)={acc_int4 - acc_u:+.4f}"
            print(delta_line)
        if acc_mlp == acc_mlp:
            print(f"  MLP Baseline Acc : {acc_mlp:.4f} +/- {std_mlp:.4f}")
        if acc_mlp_c == acc_mlp_c:
            print(f"  MLP Compressed   : {acc_mlp_c:.4f} +/- {std_mlp_c:.4f}")
        if mse_u == mse_u:
            print(f"  Uncompressed MSE : {mse_u:.4f} +/- {std_mse_u:.4f}")
            print(f"  Compressed MSE   : {mse_c:.4f} +/- {std_mse_c:.4f}")
        if mse_mlp == mse_mlp:
            print(f"  MLP Baseline MSE : {mse_mlp:.4f} +/- {std_mse_mlp:.4f}")
        if mse_mlp_c == mse_mlp_c:
            print(f"  MLP Compressed MSE:{mse_mlp_c:.4f} +/- {std_mse_mlp_c:.4f}")
        if r.get("size_mlp_uncompressed") is not None:
            print(f"  MLP Size         : {r['size_mlp_uncompressed']} -> {r['size_mlp_compressed']} bytes")
        print(time_str, end="")


def save_summary_txt(results, timings, run_dir):
    buf = io.StringIO()
    import sys
    old_stdout, sys.stdout = sys.stdout, buf
    print_summary(results, timings)
    sys.stdout = old_stdout
    txt_path = os.path.join(run_dir, "summary.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(buf.getvalue())
    print(f"  Summary saved  -> {txt_path}")
