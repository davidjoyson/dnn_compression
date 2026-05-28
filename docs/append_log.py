#!/usr/bin/env python3
"""
Append a structured results entry to docs/experiment_log.md from a run directory.

Usage:
    python docs/append_log.py outputs/run_20260528_154409_all_epo50
    python docs/append_log.py outputs/run_20260528_154409_all_epo50 --log docs/experiment_log.md
"""

import argparse
import csv
import re
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
LOG_PATH = REPO_ROOT / "docs" / "experiment_log.md"

MAIN_EXPS = ["UCI HAR", "ECG Heartbeat", "EEG Brainwave", "HAPT"]


# ------------------------------------------------------------------ #
# Parsers
# ------------------------------------------------------------------ #

def parse_summary(path):
    """
    Returns dict with keys:
      "main"      -> {exp_name: {acc_u, std_u, acc_sf, std_sf, acc_gl, std_gl,
                                  acc_dy, std_dy, acc_mlp, std_mlp,
                                  f1_u, f1_sf, f1_sf_d, f1_gl, f1_gl_d,
                                  f1_dy, f1_dy_d, f1_mlp,
                                  size_u, size_sf, size_gl, size_dy,
                                  size_mlp_u, time, seeds}}
      "ablation"  -> {configs: [...], time}
      "component" -> {none, topo_only, quant_only, both, time}
    """
    text = path.read_text(encoding="utf-8")
    result = {"main": {}, "ablation": None, "component": None}

    # ---- ablation ----
    abl_m = re.search(
        r'Ablation Study \((\d+) configurations\):(.*?)Time\s*:\s*([\d.]+)\s*sec',
        text, re.DOTALL,
    )
    if abl_m:
        configs = []
        for m in re.finditer(
            r'Config \d+ (h1=\d+ h2=\d+ br=\d+): acc_u=([\d.]+) acc_c=([\d.]+)',
            abl_m.group(2),
        ):
            configs.append({"label": m.group(1), "acc_u": float(m.group(2)), "acc_c": float(m.group(3))})
        result["ablation"] = {"configs": configs, "time": float(abl_m.group(3))}

    # ---- component ----
    comp_m = re.search(r'Component Ablation:(.*?)Time\s*:\s*([\d.]+)\s*sec', text, re.DOTALL)
    if comp_m:
        body = comp_m.group(1)
        comp = {"time": float(comp_m.group(2))}
        for cond in ("none", "topo_only", "quant_only", "both"):
            m = re.search(rf'{cond}\s*:\s*acc=([\d.]+)', body)
            if m:
                comp[cond] = float(m.group(1))
        result["component"] = comp

    # ---- main experiments ----
    # Split text into per-experiment blocks
    exp_header = re.compile(
        r'^(UCI HAR|ECG Heartbeat|EEG Brainwave|HAPT)\s+\(mean over (\d+) seeds\):',
        re.MULTILINE,
    )
    positions = [(m.start(), m.group(1), int(m.group(2))) for m in exp_header.finditer(text)]
    for i, (start, name, seeds) in enumerate(positions):
        end = positions[i + 1][0] if i + 1 < len(positions) else len(text)
        body = text[start:end]

        def _f(pattern, default=None):
            m = re.search(pattern, body)
            return float(m.group(1)) if m else default

        def _i(pattern, default=None):
            m = re.search(pattern, body)
            return int(m.group(1)) if m else default

        entry = {
            "seeds": seeds,
            # Accuracies
            "acc_u":   _f(r'Uncompressed Acc\s*:\s*([\d.]+)'),
            "std_u":   _f(r'Uncompressed Acc\s*:.*?\+/-\s*([\d.]+)'),
            "acc_sf":  _f(r'Snowflake \(int8\)\s*:\s*([\d.]+)'),
            "std_sf":  _f(r'Snowflake \(int8\)\s*:.*?\+/-\s*([\d.]+)'),
            "acc_gl":  _f(r'Global int8\s*:\s*([\d.]+)'),
            "std_gl":  _f(r'Global int8\s*:.*?\+/-\s*([\d.]+)'),
            "acc_dy":  _f(r'Dynamic \(int8\)\s*:\s*([\d.]+)'),
            "std_dy":  _f(r'Dynamic \(int8\)\s*:.*?\+/-\s*([\d.]+)'),
            "acc_mlp": _f(r'MLP Baseline Acc\s*:\s*([\d.]+)'),
            "std_mlp": _f(r'MLP Baseline Acc\s*:.*?\+/-\s*([\d.]+)'),
            # F1
            "f1_u":    _f(r'Uncompressed F1\s*:\s*([\d.]+)'),
            "f1_sf":   _f(r'Snowflake F1\s*:\s*([\d.]+)'),
            "f1_sf_d": _f(r'Snowflake F1\s*:.*?\(delta=([+-][\d.]+)\)'),
            "f1_gl":   _f(r'Global int8 F1\s*:\s*([\d.]+)'),
            "f1_gl_d": _f(r'Global int8 F1\s*:.*?\(delta=([+-][\d.]+)\)'),
            "f1_dy":   _f(r'Dynamic F1\s*:\s*([\d.]+)'),
            "f1_dy_d": _f(r'Dynamic F1\s*:.*?\(delta=([+-][\d.]+)\)'),
            "f1_mlp":  _f(r'MLP Baseline F1\s*:\s*([\d.]+)'),
            # Sizes (from Snowflake line, same as Global)
            "size_u":     _i(r'Snowflake \(int8\)\s*:.*?\[(\d+)\s*->'),
            "size_sf":    _i(r'Snowflake \(int8\)\s*:.*?->\s*(\d+)\s*bytes\]'),
            "size_mlp_u": _i(r'MLP Size\s*:\s*(\d+)\s*->'),
            "size_mlp_c": _i(r'MLP Size\s*:.*?->\s*(\d+)\s*bytes'),
            # Time
            "time": _f(r'Time\s*:\s*([\d.]+)\s*sec'),
        }
        result["main"][name] = entry

    return result


# ------------------------------------------------------------------ #
# Markdown builder
# ------------------------------------------------------------------ #

def pct(v):
    return f"{v * 100:.2f}%" if v is not None else "—"

def std(v):
    return f"±{v * 100:.2f}%" if v is not None else ""

def f4(v, sign=False):
    if v is None:
        return "—"
    return f"{v:+.4f}" if sign else f"{v:.4f}"

def ratio(size_u, size_c):
    if size_u and size_c:
        return f"{size_u / size_c:.1f}×"
    return "—"


def build_section(run_dir, summary):
    run_name = run_dir.name
    parts = run_name.split("_")
    date = datetime.strptime(parts[1], "%Y%m%d").strftime("%Y-%m-%d")

    lines = [
        "",
        "---",
        "",
        f"## {date} — Run `{run_name}`",
        "",
        f"*Auto-generated by `docs/append_log.py`. Add narrative observations below.*",
        "",
    ]

    # ---- Summary table (one row per experiment) ----
    main = summary["main"]
    present = [e for e in MAIN_EXPS if e in main]

    if present:
        lines += [
            "### Results Summary",
            "",
            "| Dataset | Seeds | Dendritic F1 | Snowflake F1 | ΔF1 | Size | Ratio | Time |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for exp in present:
            e = main[exp]
            lines.append(
                f"| **{exp}** | {e['seeds']} | {f4(e['f1_u'])} | {f4(e['f1_sf'])} | "
                f"{f4(e['f1_sf_d'], sign=True)} | "
                f"{e['size_u']:,}→{e['size_sf']:,} B | {ratio(e['size_u'], e['size_sf'])} | "
                f"{e['time']:.0f}s |"
            )
        lines.append("")

    # ---- Per-experiment detail ----
    for exp in present:
        e = main[exp]
        lines += [
            f"#### {exp}",
            "",
            "| Method | Accuracy | ±std | F1 | ΔF1 | Size (bytes) | Ratio |",
            "|---|---|---|---|---|---|---|",
            f"| Uncompressed (Dendritic) | {pct(e['acc_u'])} | {std(e['std_u'])} | {f4(e['f1_u'])} | — | {e['size_u']:,} | 1× |",
            f"| **Snowflake (int8)** | **{pct(e['acc_sf'])}** | **{std(e['std_sf'])}** | **{f4(e['f1_sf'])}** | **{f4(e['f1_sf_d'], sign=True)}** | {e['size_sf']:,} | **{ratio(e['size_u'], e['size_sf'])}** |",
            f"| Global int8 | {pct(e['acc_gl'])} | {std(e['std_gl'])} | {f4(e['f1_gl'])} | {f4(e['f1_gl_d'], sign=True)} | {e['size_sf']:,} | {ratio(e['size_u'], e['size_sf'])} |",
            f"| Dynamic int8 | {pct(e['acc_dy'])} | {std(e['std_dy'])} | {f4(e['f1_dy'])} | {f4(e['f1_dy_d'], sign=True)} | — | — |",
            f"| MLP Baseline | {pct(e['acc_mlp'])} | {std(e['std_mlp'])} | {f4(e['f1_mlp'])} | — | {e['size_mlp_u']:,} | 1× |",
            "",
        ]

    # ---- Ablation ----
    if summary["ablation"]:
        abl = summary["ablation"]
        lines += ["#### Ablation Study (ECG)", ""]
        for cfg in abl["configs"]:
            lines.append(f"- `{cfg['label']}`: acc_u={cfg['acc_u']:.4f} → acc_c={cfg['acc_c']:.4f}")
        lines += [f"", f"Time: {abl['time']:.0f}s", ""]

    # ---- Component ablation ----
    if summary["component"]:
        comp = summary["component"]
        lines += ["#### Component Ablation (ECG)", ""]
        for cond in ("none", "topo_only", "quant_only", "both"):
            if cond in comp:
                lines.append(f"- `{cond}`: acc={comp[cond]:.4f}")
        lines += [f"", f"Time: {comp['time']:.0f}s", ""]

    # ---- Observations placeholder ----
    lines += [
        "### Observations",
        "",
        "*(Add observations here)*",
        "",
    ]

    return "\n".join(lines)


# ------------------------------------------------------------------ #
# Entry point
# ------------------------------------------------------------------ #

def main():
    parser = argparse.ArgumentParser(
        description="Append run results to experiment_log.md"
    )
    parser.add_argument("run_dir", help="Path to run output directory")
    parser.add_argument("--log", default=str(LOG_PATH), help="Path to experiment_log.md")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.is_dir():
        print(f"Error: {run_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    summary_path = run_dir / "summary.txt"
    if not summary_path.exists():
        print(f"Error: {summary_path} not found", file=sys.stderr)
        sys.exit(1)

    summary = parse_summary(summary_path)
    section = build_section(run_dir, summary)

    log_path = Path(args.log)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(section)

    print(f"Appended results for '{run_dir.name}' to {log_path}")


if __name__ == "__main__":
    main()
