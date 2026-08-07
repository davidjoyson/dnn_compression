"""
Branch-wise evidence for the Snowflake quantization idea.

For each dendritic branch's weight matrix, reports the range/std that drive
its per-layer int8/int4 scale, the resulting quantization error and clipping
rate, and how similar branches are to each other (cosine similarity) -- the
geometric property the "branch diversity is load-bearing" component-ablation
finding (docs/experiment_log.md, 2026-07-28) rests on but never measured
directly. No training involved: reuses saved checkpoints only.

Usage: python analyze_snowflake_branches.py
"""
import csv
import itertools
import os

import torch

from src.compression.compression_pipeline import compress_model_int4

DATASETS = ["ecg", "hapt"]
N_BRANCHES = 8

# Known post-quantization accuracy (3-seed mean, run_int4.py, 2026-07-30 --
# see docs/experiment_log.md). Uses the multi-seed rerun rather than a single
# Pi-benchmark checkpoint: int4 accuracy is high-variance (see int4 RMSE
# below), so a single checkpoint can get lucky and hide the real spread --
# side-by-side context only, not a statistical claim (n=2 datasets can't
# support a correlation test).
ACC_BASELINE = {"ecg": 0.8450, "hapt": 0.9287}
ACC_INT8     = {"ecg": 0.8707, "hapt": 0.9269}
ACC_INT4     = {"ecg": 0.7348, "hapt": 0.9104}


def branch_stats(w, q, scale, qmax):
    """One branch's weight matrix vs. its quantized/dequantized reconstruction."""
    w_dq = q.float() * scale
    err = w - w_dq
    return {
        "w_min": w.min().item(), "w_max": w.max().item(),
        "w_range": (w.max() - w.min()).item(), "w_std": w.std().item(),
        "scale": float(scale),
        "quant_mae": err.abs().mean().item(),
        "quant_rmse": err.pow(2).mean().sqrt().item(),
        "clip_rate": (q.abs() == qmax).float().mean().item(),
    }


def cosine_matrix(branch_weights):
    """NxN pairwise cosine similarity between branches' flattened weight vectors."""
    vecs = [w.flatten() for w in branch_weights]
    n = len(vecs)
    mat = torch.eye(n)
    for i, j in itertools.combinations(range(n), 2):
        sim = torch.nn.functional.cosine_similarity(vecs[i], vecs[j], dim=0).item()
        mat[i, j] = mat[j, i] = sim
    return mat


def analyze(dataset):
    uncompressed = torch.load(f"models/{dataset}/dendritic_uncompressed.pt", map_location="cpu")
    int8 = torch.load(f"models/{dataset}/dendritic_snowflake.pt", map_location="cpu")

    # int4 has no saved checkpoint -- deterministic, no fine-tune, cheap to recompute.
    import copy
    from src.models.dendritic_network import DendriticNetwork
    input_dim, num_classes = (187, 5) if dataset == "ecg" else (561, 12)
    m = DendriticNetwork(input_dim=input_dim, hidden_neurons1=64, hidden_neurons2=32,
                          branches=N_BRANCHES, hidden_per_branch=8, num_classes=num_classes)
    m.load_state_dict(uncompressed)
    int4 = compress_model_int4(copy.deepcopy(m))

    rows = []
    branch_weights = []
    for b in range(N_BRANCHES):
        key = f"branches.{b}.weight"
        w = uncompressed[key]
        branch_weights.append(w)

        s8 = branch_stats(w, int8[key]["q"], int8[key]["scale"], qmax=127)
        s4 = branch_stats(w, int4[key]["q"], int4[key]["scale"], qmax=7)

        rows.append({"dataset": dataset, "branch": b, "bits": 8, **s8})
        rows.append({"dataset": dataset, "branch": b, "bits": 4, **s4})

    cos_mat = cosine_matrix(branch_weights)
    off_diag = [cos_mat[i, j].item() for i, j in itertools.combinations(range(N_BRANCHES), 2)]
    summary = {
        "dataset": dataset,
        "cosine_sim_mean": sum(off_diag) / len(off_diag),
        "cosine_sim_min": min(off_diag),
        "cosine_sim_max": max(off_diag),
        "quant_rmse_mean_int8": sum(r["quant_rmse"] for r in rows if r["bits"] == 8) / N_BRANCHES,
        "quant_rmse_mean_int4": sum(r["quant_rmse"] for r in rows if r["bits"] == 4) / N_BRANCHES,
        "acc_baseline": ACC_BASELINE[dataset],
        "acc_int8": ACC_INT8[dataset],
        "acc_int4": ACC_INT4[dataset],
        "acc_delta_int8": ACC_INT8[dataset] - ACC_BASELINE[dataset],
        "acc_delta_int4": ACC_INT4[dataset] - ACC_BASELINE[dataset],
    }
    return rows, summary, cos_mat


def plot(all_rows, all_summaries, cos_mats, out_dir):
    import matplotlib.pyplot as plt
    import numpy as np
    from src.plots.style import apply_style

    apply_style()
    COLOR_INT8 = "#4878CF"   # PALETTE[0], matches METHOD_COLORS["Snowflake (int8)"]-family
    COLOR_INT4 = "#637939"   # matches METHOD_COLORS["Snowflake (int4)"]

    # --- Per-branch quantization RMSE, int8 vs int4, one panel per dataset ---
    fig, axes = plt.subplots(1, len(DATASETS), figsize=(11, 4.5), sharey=False)
    for ax, ds in zip(axes, DATASETS):
        ds_rows = [r for r in all_rows if r["dataset"] == ds]
        branches = sorted(set(r["branch"] for r in ds_rows))
        rmse8 = [next(r["quant_rmse"] for r in ds_rows if r["branch"] == b and r["bits"] == 8) for b in branches]
        rmse4 = [next(r["quant_rmse"] for r in ds_rows if r["branch"] == b and r["bits"] == 4) for b in branches]
        x = np.arange(len(branches))
        w = 0.36
        ax.bar(x - w / 2, rmse8, w, label="int8", color=COLOR_INT8, zorder=3,
               edgecolor="white", linewidth=0.6)
        ax.bar(x + w / 2, rmse4, w, label="int4", color=COLOR_INT4, zorder=3,
               edgecolor="white", linewidth=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels([str(b) for b in branches])
        ax.set_xlabel("Branch")
        ax.set_ylabel("Quantization RMSE")
        ax.set_title(ds.upper())
        ax.legend(frameon=False)
    fig.suptitle("Per-Branch Quantization Error — Snowflake int8 vs. int4", fontsize=13, fontweight="bold", y=1.03)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "branch_quant_rmse.png"), bbox_inches="tight")
    plt.close(fig)

    # --- Branch-pairwise cosine similarity heatmap, one panel per dataset ---
    fig, axes = plt.subplots(1, len(DATASETS), figsize=(10.5, 4.6))
    for ax, ds in zip(axes, DATASETS):
        mat = cos_mats[ds].numpy()
        im = ax.imshow(mat, cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_xticks(range(N_BRANCHES))
        ax.set_yticks(range(N_BRANCHES))
        ax.set_xlabel("Branch")
        ax.set_ylabel("Branch")
        ax.set_title(ds.upper())
        ax.grid(False)
        for i in range(N_BRANCHES):
            for j in range(N_BRANCHES):
                v = mat[i, j]
                txt_color = "white" if abs(v) > 0.6 else "#333333"
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6.5, color=txt_color)
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Cosine similarity")
    fig.suptitle("Branch-Pairwise Weight Cosine Similarity (near-0 = geometrically diverse)",
                 fontsize=12.5, fontweight="bold", y=1.03)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "branch_cosine_similarity.png"), bbox_inches="tight")
    plt.close(fig)

    print(f"Saved -> {out_dir}/branch_quant_rmse.png")
    print(f"Saved -> {out_dir}/branch_cosine_similarity.png")


def main():
    out_dir = "outputs/snowflake_branch_evidence"
    os.makedirs(out_dir, exist_ok=True)
    all_rows, all_summaries, cos_mats = [], [], {}
    for ds in DATASETS:
        rows, summary, cos_mat = analyze(ds)
        all_rows.extend(rows)
        all_summaries.append(summary)
        cos_mats[ds] = cos_mat

    with open(os.path.join(out_dir, "branch_stats.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        w.writeheader()
        w.writerows(all_rows)

    with open(os.path.join(out_dir, "summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_summaries[0].keys()))
        w.writeheader()
        w.writerows(all_summaries)

    plot(all_rows, all_summaries, cos_mats, out_dir)

    print(f"{'Dataset':<8}{'Bits':>5}{'Branch':>7}{'Range':>9}{'Std':>8}{'Scale':>9}"
          f"{'MAE':>9}{'RMSE':>9}{'ClipRate':>10}")
    for r in all_rows:
        print(f"{r['dataset']:<8}{r['bits']:>5}{r['branch']:>7}{r['w_range']:>9.3f}"
              f"{r['w_std']:>8.3f}{r['scale']:>9.5f}{r['quant_mae']:>9.5f}"
              f"{r['quant_rmse']:>9.5f}{r['clip_rate']:>10.1%}")

    print(f"\n{'Dataset':<8}{'CosSim(mean/min/max)':>24}{'RMSE int8':>12}{'RMSE int4':>12}"
          f"{'dAcc int8':>12}{'dAcc int4':>12}")
    for s in all_summaries:
        cs = f"{s['cosine_sim_mean']:.3f}/{s['cosine_sim_min']:.3f}/{s['cosine_sim_max']:.3f}"
        print(f"{s['dataset']:<8}{cs:>24}{s['quant_rmse_mean_int8']:>12.5f}"
              f"{s['quant_rmse_mean_int4']:>12.5f}{s['acc_delta_int8']:>+12.4f}"
              f"{s['acc_delta_int4']:>+12.4f}")

    print("\nNote: clip_rate is near-1/n_elements by construction -- each layer's scale is set to "
          "exactly the max |weight|, so that one element always lands at the boundary; it is not "
          "evidence of information loss beyond the intended range, unlike quant_rmse.")
    print(f"Saved -> {out_dir}/{{branch_stats,summary}}.csv")


if __name__ == "__main__":
    main()
