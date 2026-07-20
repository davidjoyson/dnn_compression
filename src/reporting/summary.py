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
            "ci_95_uncompressed":          round(r.get("ci_95", {}).get("uncompressed", 0.0), 6),
            "ci_95_compressed":            round(r.get("ci_95", {}).get("compressed",   0.0), 6),
            "tost_snowflake_equiv":        r.get("tost", {}).get("compressed",        {}).get("equivalent"),
            "tost_snowflake_mean_diff":    r.get("tost", {}).get("compressed",        {}).get("mean_diff"),
            "tost_snowflake_ci_low":       r.get("tost", {}).get("compressed",        {}).get("ci_low"),
            "tost_snowflake_ci_high":      r.get("tost", {}).get("compressed",        {}).get("ci_high"),
            "tost_snowflake_p_low":        r.get("tost", {}).get("compressed",        {}).get("p_low"),
            "tost_snowflake_p_high":       r.get("tost", {}).get("compressed",        {}).get("p_high"),
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

        if name == "Ablation Study" and isinstance(r, dict):
            print(f"{name}:")
            for dataset, configs in r.items():
                print(f"  [{dataset}]")
                for i, cfg_r in enumerate(configs):
                    cfg = cfg_r.get("config", {})
                    au = cfg_r.get("accuracy_uncompressed", {})
                    ac = cfg_r.get("accuracy_compressed", {})
                    tag = f"h1={cfg.get('h1','?')} h2={cfg.get('h2','?')} br={cfg.get('branches','?')}"
                    print(f"    Config {i+1} {tag}: "
                          f"acc_u={au.get('mean', float('nan')):.4f} +/- {au.get('std', 0.0):.4f}  "
                          f"acc_c={ac.get('mean', float('nan')):.4f} +/- {ac.get('std', 0.0):.4f}")
            print(time_str, end="")
            continue

        if name in ("Component Ablation", "Regularization Ablation") and isinstance(r, dict):
            print(f"{name}:")
            for dataset, conditions in r.items():
                print(f"  [{dataset}]")
                for condition, stats in conditions.items():
                    mse_m = stats.get("mse_mean", float("nan"))
                    mse_s = stats.get("mse_std",  0.0)
                    mse_str = f"  mse={mse_m:.4f} +/- {mse_s:.4f}" if "mse_mean" in stats else ""
                    print(f"    {condition:<12}: acc={stats['mean']:.4f} +/- {stats['std']:.4f}{mse_str}")
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
        _ci = r.get("ci_95", {})
        ci_u   = to_float(_ci.get("uncompressed",       0.0))
        ci_c   = to_float(_ci.get("compressed",         0.0))
        ci_mlp = to_float(_ci.get("mlp_baseline",       0.0))
        acc_mlp_c     = to_float(r.get("accuracy_mlp_compressed", float("nan")))
        std_mlp_c     = to_float(r.get("std_mlp_compressed",      0.0))
        acc_global   = to_float(r.get("accuracy_compressed_global",  float("nan")))
        std_global   = to_float(r.get("std_compressed_global",       0.0))
        acc_dynamic  = to_float(r.get("accuracy_compressed_dynamic", float("nan")))
        std_dynamic  = to_float(r.get("std_compressed_dynamic",      0.0))
        acc_static   = to_float(r.get("accuracy_compressed_static",  float("nan")))
        std_static   = to_float(r.get("std_compressed_static",       0.0))
        acc_sfstatic = to_float(r.get("accuracy_compressed_snowflake_static", float("nan")))
        std_sfstatic = to_float(r.get("std_compressed_snowflake_static",      0.0))
        acc_perchan  = to_float(r.get("accuracy_compressed_perchan", float("nan")))
        std_perchan  = to_float(r.get("std_compressed_perchan",      0.0))
        acc_qat      = to_float(r.get("accuracy_compressed_qat",     float("nan")))
        std_qat      = to_float(r.get("std_compressed_qat",          0.0))
        acc_mixed    = to_float(r.get("accuracy_compressed_mixed",   float("nan")))
        std_mixed    = to_float(r.get("std_compressed_mixed",        0.0))
        acc_int4     = to_float(r.get("accuracy_compressed_int4",   float("nan")))
        std_int4     = to_float(r.get("std_compressed_int4",        0.0))
        f1_u      = to_float(r.get("f1_uncompressed",       float("nan")))
        f1_c      = to_float(r.get("f1_compressed",         float("nan")))
        f1_global = to_float(r.get("f1_compressed_global",  float("nan")))
        f1_dyn    = to_float(r.get("f1_compressed_dynamic", float("nan")))
        f1_mlp    = to_float(r.get("f1_mlp_baseline",       float("nan")))
        f1_mlp_c  = to_float(r.get("f1_mlp_compressed",     float("nan")))
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

        def _ci_str(ci_val):
            return f"  95% CI: +/-{ci_val:.4f}" if ci_val > 0 else ""

        print(f"{name}{seed_note}:")
        print(f"  Uncompressed Acc : {acc_u:.4f} +/- {std_u:.4f}{_ci_str(ci_u)}")
        print(f"  Snowflake (int8) : {acc_c:.4f} +/- {std_c:.4f}{_ci_str(ci_c)}  [{r.get('size_uncompressed','?')} -> {r.get('size_compressed','?')} bytes]")
        if not math.isnan(acc_global):
            ci_gl = to_float(_ci.get("compressed_global", 0.0))
            print(f"  Global int8      : {acc_global:.4f} +/- {std_global:.4f}{_ci_str(ci_gl)}  [{r.get('size_uncompressed','?')} -> {r.get('size_compressed_global','?')} bytes]")
        if not math.isnan(acc_dynamic):
            ci_dy = to_float(_ci.get("compressed_dynamic", 0.0))
            print(f"  Dynamic (int8)   : {acc_dynamic:.4f} +/- {std_dynamic:.4f}{_ci_str(ci_dy)}  [{r.get('size_uncompressed','?')} -> {r.get('size_compressed_dynamic','?')} bytes]")
        if not math.isnan(acc_static):
            ci_st = to_float(_ci.get("compressed_static", 0.0))
            print(f"  Static (int8)    : {acc_static:.4f} +/- {std_static:.4f}{_ci_str(ci_st)}  [{r.get('size_uncompressed','?')} -> {r.get('size_compressed_static','?')} bytes]")
        if not math.isnan(acc_sfstatic):
            ci_sfst = to_float(_ci.get("compressed_snowflake_static", 0.0))
            print(f"  Snowflake+Static : {acc_sfstatic:.4f} +/- {std_sfstatic:.4f}{_ci_str(ci_sfst)}  [{r.get('size_uncompressed','?')} -> {r.get('size_compressed_snowflake_static','?')} bytes]")
        if not math.isnan(acc_perchan):
            ci_pc = to_float(_ci.get("compressed_perchan", 0.0))
            print(f"  Per-channel      : {acc_perchan:.4f} +/- {std_perchan:.4f}{_ci_str(ci_pc)}  [{r.get('size_uncompressed','?')} -> {r.get('size_compressed_perchan','?')} bytes]")
        if not math.isnan(acc_qat):
            ci_qt = to_float(_ci.get("compressed_qat", 0.0))
            print(f"  QAT (int8)       : {acc_qat:.4f} +/- {std_qat:.4f}{_ci_str(ci_qt)}  [{r.get('size_uncompressed','?')} -> {r.get('size_compressed_qat','?')} bytes]")
        if not math.isnan(acc_mixed):
            ci_mx = to_float(_ci.get("compressed_mixed", 0.0))
            print(f"  Mixed precision  : {acc_mixed:.4f} +/- {std_mixed:.4f}{_ci_str(ci_mx)}  [{r.get('size_uncompressed','?')} -> {r.get('size_compressed_mixed','?')} bytes]")
        if not math.isnan(acc_int4):
            ci_i4 = to_float(_ci.get("compressed_int4", 0.0))
            print(f"  Snowflake (int4) : {acc_int4:.4f} +/- {std_int4:.4f}{_ci_str(ci_i4)}  [{r.get('size_uncompressed','?')} -> {r.get('size_compressed_int4','?')} bytes]")
        if not math.isnan(acc_global) and not math.isnan(acc_dynamic):
            print(f"  -- Compression delta: Snowflake(8b)={acc_c - acc_u:+.4f} | Global(8b)={acc_global - acc_u:+.4f} | Dynamic(8b)={acc_dynamic - acc_u:+.4f}")
        if not math.isnan(acc_mlp):
            print(f"  MLP Baseline Acc : {acc_mlp:.4f} +/- {std_mlp:.4f}{_ci_str(ci_mlp)}")
        if not math.isnan(acc_mlp_c):
            ci_mlp_c = to_float(_ci.get("mlp_compressed", 0.0))
            print(f"  MLP Compressed   : {acc_mlp_c:.4f} +/- {std_mlp_c:.4f}{_ci_str(ci_mlp_c)}")
        if not math.isnan(f1_u):
            print(f"  Uncompressed F1  : {f1_u:.4f}")
            print(f"  Snowflake F1     : {f1_c:.4f}  (delta={f1_c - f1_u:+.4f})")
        if not math.isnan(f1_global):
            print(f"  Global int8 F1   : {f1_global:.4f}  (delta={f1_global - f1_u:+.4f})")
        if not math.isnan(f1_dyn):
            print(f"  Dynamic F1       : {f1_dyn:.4f}  (delta={f1_dyn - f1_u:+.4f})")
        if not math.isnan(f1_mlp):
            print(f"  MLP Baseline F1  : {f1_mlp:.4f}")
        if not math.isnan(f1_mlp_c):
            print(f"  MLP Compressed F1: {f1_mlp_c:.4f}")
        if not math.isnan(mse_u):
            print(f"  Uncompressed MSE : {mse_u:.4f} +/- {std_mse_u:.4f}")
            print(f"  Compressed MSE   : {mse_c:.4f} +/- {std_mse_c:.4f}")
        if not math.isnan(mse_mlp):
            print(f"  MLP Baseline MSE : {mse_mlp:.4f} +/- {std_mse_mlp:.4f}")
        if not math.isnan(mse_mlp_c):
            print(f"  MLP Compressed MSE:{mse_mlp_c:.4f} +/- {std_mse_mlp_c:.4f}")
        if r.get("size_mlp_uncompressed") is not None:
            print(f"  MLP Size         : {r['size_mlp_uncompressed']} -> {r['size_mlp_compressed']} bytes")

        tost_r = r.get("tost", {})
        if tost_r:
            _TOST_LABELS = [
                ("compressed",         "Snowflake"),
                ("compressed_global",  "Global   "),
                ("compressed_dynamic", "Dynamic  "),
                ("compressed_static",  "Static   "),
                ("compressed_snowflake_static", "SF+Static"),
                ("compressed_perchan", "Per-chan "),
                ("compressed_qat",     "QAT      "),
                ("compressed_mixed",   "Mixed    "),
                ("compressed_int4",    "Int4     "),
            ]
            rows = [(lbl, tost_r[k]) for k, lbl in _TOST_LABELS
                    if k in tost_r and tost_r[k].get("equivalent") is not None]
            if rows:
                print(f"  Equivalence (TOST, e=2%, n={n_seeds}):")
                for lbl, t in rows:
                    verdict = "EQUIV    " if t["equivalent"] else "NOT EQUIV"
                    print(f"    {lbl}: {verdict}  diff={t['mean_diff']:+.4f}"
                          f"  CI=[{t['ci_low']:+.4f}, {t['ci_high']:+.4f}]"
                          f"  (p_low={t['p_low']:.4f}, p_high={t['p_high']:.4f})")

        mc = r.get("method_comparison")
        if mc:
            _MC_LABELS = [
                ("snowflake",        "Snowflake"),
                ("global",           "Global   "),
                ("dynamic",          "Dynamic  "),
                ("static",           "Static   "),
                ("snowflake_static", "SF+Static"),
                ("perchan",          "Per-chan "),
                ("qat",              "QAT      "),
                ("mixed",            "Mixed    "),
            ]
            print(f"  Baseline Comparison (all methods, n={n_seeds}):")
            for model_key, model_label in [("mlp", "MLP (total-param-matched)"),
                                           ("layer_matched", "LayerMatchedMLP (per-layer-matched)")]:
                mdl = mc.get(model_key)
                if not mdl:
                    continue
                au = to_float(mdl.get("accuracy_uncompressed", float("nan")))
                print(f"    {model_label}: uncompressed={au:.4f}")
                for key, lbl in _MC_LABELS:
                    acc = to_float(mdl.get("accuracy", {}).get(key, float("nan")))
                    if math.isnan(acc):
                        continue
                    t = mdl.get("tost", {}).get(key, {})
                    verdict = "EQUIV    " if t.get("equivalent") else ("NOT EQUIV" if t.get("equivalent") is False else "n/a      ")
                    diff = to_float(t.get("mean_diff", float("nan")))
                    diff_str = f"  diff={diff:+.4f}" if not math.isnan(diff) else ""
                    print(f"      {lbl}: {acc:.4f}  {verdict}{diff_str}")

        op = r.get("output_precision")
        if op:
            print(f"  Output Precision (Dendritic, vs float32 reference, not vs labels):")
            _OP_LABELS = [
                ("snowflake",        "Snowflake"),
                ("global",           "Global   "),
                ("dynamic",          "Dynamic  "),
                ("static",           "Static   "),
                ("snowflake_static", "SF+Static"),
                ("perchan",          "Per-chan "),
                ("qat",              "QAT      "),
                ("mixed",            "Mixed    "),
            ]
            for key, lbl in _OP_LABELS:
                d = op.get(key)
                if not d:
                    continue
                kl = d.get("kl_divergence")
                kl_str = f"  KL={kl:.6f}" if kl is not None else ""
                print(f"    {lbl}: logit_MSE={d['logit_mse']:.6f}  cos_sim={d['cosine_similarity']:.4f}"
                      f"{kl_str}  pred_flip={d['pred_flip_rate']:.4f}")

        ep = r.get("edge_profile")
        if ep:
            su = ep.get("model_size_kb")
            sc = ep.get("compressed_size_kb")
            cr = ep.get("compression_ratio")
            cr_str = f"  ({cr:.1f}x ratio)" if cr else ""
            print(f"  Edge Profile (edge-AI):")
            if su is not None and sc is not None:
                print(f"    Model size     : {su:.1f} KB  ->  {sc:.1f} KB{cr_str}")
            if ep.get("params"):
                print(f"    Params         : {ep['params']:,}")
            if ep.get("flops_per_sample"):
                print(f"    FLOPs/sample   : {ep['flops_per_sample']:,} MACs")
            if ep.get("activation_mem_kb"):
                print(f"    Activation mem : {ep['activation_mem_kb']:.2f} KB  (per-sample overhead)")
            lat = ep.get("latency_us", {})
            if lat.get("uncompressed"):
                def _fmt_us(v): return f"{v:.2f}us" if v else "-"
                lat_parts = [
                    f"Dendritic={_fmt_us(lat.get('uncompressed'))}",
                    f"Snowflake={_fmt_us(lat.get('compressed'))}",
                    f"Dynamic={_fmt_us(lat.get('dynamic'))}",
                    f"MLP={_fmt_us(lat.get('mlp'))}",
                ]
                print(f"    Latency/sample : " + "  |  ".join(lat_parts))
            tp = ep.get("throughput_sps", {})
            if tp.get("uncompressed"):
                def _fmt_tp(v):
                    if not v: return "—"
                    return f"{v/1e6:.2f}M" if v >= 1e6 else f"{v/1e3:.0f}K"
                tp_parts = [
                    f"Dendritic={_fmt_tp(tp.get('uncompressed'))}",
                    f"Snowflake={_fmt_tp(tp.get('compressed'))}",
                    f"Dynamic={_fmt_tp(tp.get('dynamic'))}",
                    f"MLP={_fmt_tp(tp.get('mlp'))}",
                ]
                print(f"    Throughput     : " + "  |  ".join(tp_parts) + " sps")

        print(time_str, end="")


def save_per_seed_csv(results, run_dir):
    rows = []
    for name, r in results.items():
        if not isinstance(r, dict) or r.get("per_seed") is None:
            continue
        ps = r["per_seed"]
        n = len(ps["acc_uncompressed"])
        def _ps(key, i):
            v = ps.get(key, [None] * n)[i]
            return round(v, 6) if v is not None else ""
        for i in range(n):
            rows.append({
                "experiment":              name,
                "seed_index":              i,
                "acc_uncompressed":        _ps("acc_uncompressed", i),
                "acc_compressed":          _ps("acc_compressed", i),
                "acc_compressed_global":   _ps("acc_compressed_global", i),
                "acc_compressed_dynamic":  _ps("acc_compressed_dynamic", i),
                "acc_compressed_static":   _ps("acc_compressed_static", i),
                "acc_compressed_snowflake_static": _ps("acc_compressed_snowflake_static", i),
                "acc_compressed_perchan":  _ps("acc_compressed_perchan", i),
                "acc_compressed_qat":      _ps("acc_compressed_qat", i),
                "acc_compressed_mixed":    _ps("acc_compressed_mixed", i),
                "acc_compressed_int4":     _ps("acc_compressed_int4", i),
                "f1_uncompressed":         _ps("f1_uncompressed", i),
                "f1_compressed":           _ps("f1_compressed", i),
                "f1_compressed_global":    _ps("f1_compressed_global", i),
                "f1_compressed_dynamic":   _ps("f1_compressed_dynamic", i),
                "f1_compressed_static":    _ps("f1_compressed_static", i),
                "f1_compressed_snowflake_static": _ps("f1_compressed_snowflake_static", i),
                "f1_compressed_perchan":   _ps("f1_compressed_perchan", i),
                "f1_compressed_qat":       _ps("f1_compressed_qat", i),
                "f1_compressed_mixed":     _ps("f1_compressed_mixed", i),
                "f1_compressed_int4":      _ps("f1_compressed_int4", i),
            })
    if not rows:
        return
    csv_path = os.path.join(run_dir, "per_seed_metrics.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Per-seed metrics -> {csv_path}")


def save_summary_txt(results, timings, run_dir):
    import sys
    buf = io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = buf
    try:
        print_summary(results, timings)
    finally:
        sys.stdout = old_stdout
    txt_path = os.path.join(run_dir, "summary.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(buf.getvalue())
    print(f"  Summary saved  -> {txt_path}")
