"""
TOST equivalence testing on macro-F1, alongside the existing accuracy TOST,
plus a forest plot of both.

Reuses per-seed accuracy/macro-F1 already logged in per_seed_metrics.csv by
the main 10-seed run (base_experiment.py) -- no retraining, no new
experiments. Balanced accuracy is NOT included here: it isn't currently
captured per-seed anywhere in the pipeline (only a single best-seed snapshot
in summary.py), so extending TOST to it needs a pipeline change plus a fresh
run to populate.

Usage: python analyze_tost_f1.py [path/to/per_seed_metrics.csv]
"""
import csv
import os
import sys

from src.analysis.tost import tost_paired

RUN = sys.argv[1] if len(sys.argv) > 1 else "outputs/run_20260727_204917_ecg_balanced_hapt_unbalanced/per_seed_metrics.csv"
OUT_DIR = "outputs/tost_forest"

METHODS = [
    ("compressed",              "Snowflake"),
    ("compressed_global",       "Global"),
    ("compressed_dynamic",      "Dynamic"),
    ("compressed_static",       "Static"),
    ("compressed_snowflake_static", "SF+Static"),
    ("compressed_perchan",      "Per-chan"),
    ("compressed_qat",          "QAT"),
    ("compressed_mixed",        "Mixed"),
    ("compressed_int4",         "Int4"),
]
MARGIN = 0.02


def load(path):
    by_exp = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            by_exp.setdefault(row["experiment"], []).append(row)
    return by_exp


def compute_tost(rows, metric):
    """metric: 'acc' or 'f1'. Returns [(label, tost_dict), ...] for methods with data."""
    base = [float(r[f"{metric}_uncompressed"]) if r.get(f"{metric}_uncompressed") else None for r in rows]
    out = []
    for key, label in METHODS:
        m = [float(r[f"{metric}_{key}"]) if r.get(f"{metric}_{key}") else None for r in rows]
        if all(v is None for v in m):
            continue
        t = tost_paired(base, m, margin=MARGIN)
        if t["equivalent"] is None:
            continue
        out.append((label, t))
    return out


def print_report(by_exp):
    print(f"Source: {RUN}\n")
    for exp, rows in by_exp.items():
        for metric, metric_label in [("f1", "macro-F1"), ("acc", "accuracy")]:
            results = compute_tost(rows, metric)
            if not results:
                continue
            print(f"=== {exp} (n={len(rows)} seeds) — TOST on {metric_label}, margin={MARGIN} ===")
            for label, t in results:
                verdict = "EQUIV    " if t["equivalent"] else "NOT EQUIV"
                print(f"  {label:<10}: {verdict}  diff={t['mean_diff']:+.4f}  "
                      f"CI=[{t['ci_low']:+.4f}, {t['ci_high']:+.4f}]  "
                      f"(p_low={t['p_low']:.4f}, p_high={t['p_high']:.4f}, n={t['n']})")
            print()


def plot(by_exp, out_dir):
    import matplotlib.pyplot as plt
    from src.plots.style import apply_style

    apply_style()
    GOOD = "#0ca30c"      # dataviz skill status palette: EQUIV
    CRITICAL = "#d03b3b"  # dataviz skill status palette: NOT EQUIV

    exps = list(by_exp.keys())
    metrics = [("acc", "Accuracy"), ("f1", "Macro-F1")]
    fig, axes = plt.subplots(len(exps), len(metrics), figsize=(11, 3.4 * len(exps)), squeeze=False)

    for row, exp in enumerate(exps):
        for col, (metric, metric_label) in enumerate(metrics):
            ax = axes[row][col]
            results = compute_tost(by_exp[exp], metric)
            labels = [lbl for lbl, _ in results][::-1]
            diffs  = [t["mean_diff"] for _, t in results][::-1]
            los    = [t["mean_diff"] - t["ci_low"] for _, t in results][::-1]
            his    = [t["ci_high"] - t["mean_diff"] for _, t in results][::-1]
            colors = [GOOD if t["equivalent"] else CRITICAL for _, t in results][::-1]
            y = range(len(labels))

            ax.axvspan(-MARGIN, MARGIN, color="#F0F0F0", zorder=1)
            ax.axvline(0, color="#999999", linewidth=1, zorder=2)
            ax.axvline(-MARGIN, color="#999999", linewidth=1, linestyle="--", zorder=2)
            ax.axvline(MARGIN, color="#999999", linewidth=1, linestyle="--", zorder=2,
                       label=f"±{MARGIN:.0%} margin")
            ax.errorbar(diffs, y, xerr=[los, his], fmt="o", markersize=6, capsize=3,
                        color="#333333", ecolor="#333333", zorder=4)
            for yi, (d, c) in enumerate(zip(diffs, colors)):
                ax.scatter([d], [yi], s=55, color=c, zorder=5, edgecolor="white", linewidth=0.8)

            ax.set_yticks(list(y))
            ax.set_yticklabels(labels)
            ax.set_xlabel("Mean diff. vs. uncompressed")
            ax.set_title(f"{exp} — {metric_label}")

    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=GOOD, markersize=8, label="EQUIV"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=CRITICAL, markersize=8, label="NOT EQUIV"),
        Line2D([0], [0], color="#999999", linestyle="--", label=f"±{MARGIN:.0%} margin"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("TOST Equivalence — Mean Difference vs. Uncompressed (95% CI)",
                 fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "tost_forest.png")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {path}")


def main():
    by_exp = load(RUN)
    print_report(by_exp)
    plot(by_exp, OUT_DIR)


if __name__ == "__main__":
    main()
