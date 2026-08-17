"""Build the concise figure pack used in the revision-response email."""

from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


ROOT = Path(__file__).resolve().parents[1]
FINAL = ROOT / "final_output"
OUTPUT = FINAL / "email_revision_figures.pdf"

FIGURES = [
    (
        "Architecture ablation — branch count",
        FINAL / "experiment_results/architecture_ablation_20260816/figures/combined/ablation_branch_count.png",
        "Branch count is varied while branch width and trunk dimensions remain fixed.",
    ),
    (
        "Architecture ablation — branch width",
        FINAL / "experiment_results/architecture_ablation_20260816/figures/combined/ablation_branch_width.png",
        "Branch width is varied independently of branch count and trunk size.",
    ),
    (
        "Architecture ablation — trunk hidden size",
        FINAL / "experiment_results/architecture_ablation_20260816/figures/combined/ablation_hidden_size.png",
        "The paired trunk dimensions are varied while both branch axes remain fixed.",
    ),
    (
        "Baseline resource comparison",
        FINAL / "experiment_results/figures/combined/edge_profile.png",
        "The common profile reports parameters, model size, FLOPs/MACs, activation memory and latency.",
    ),
    (
        "MIT-BIH ECG per-class F1",
        FINAL / "experiment_results/figures/ecg_heartbeat_per_class_f1.png",
        "Per-class reporting makes the remaining Fusion and Unknown limitations explicit.",
    ),
    (
        "Snowflake branch diagnostics",
        FINAL / "experiment_results/figures/ecg_heartbeat_branch_diversity.png",
        "Branch ranges, variability and cosine structure provide mechanism-level quantization evidence.",
    ),
    (
        "Four-core Raspberry Pi speedup",
        FINAL / "pi_benchmark/pi_speedup_all_methods.png",
        "MIT-BIH, INCART and HAPT results distinguish storage-only methods from true INT8 execution.",
    ),
    (
        "Sustained Raspberry Pi thermal test",
        FINAL / "pi_benchmark/thermal_ecg_20260816/thermal_summary.png",
        "Five-minute ECG loads across all ten methods peaked at 55.8°C; no throttling flag was observed.",
    ),
]


def cover_page(pdf: PdfPages) -> None:
    fig = plt.figure(figsize=(11.69, 8.27), facecolor="white")
    fig.text(0.07, 0.91, "Experimental Revision — Figure Pack", fontsize=24, weight="bold")
    fig.text(0.07, 0.86, "Selected evidence accompanying the response to reviewer feedback", fontsize=13)

    rows = [
        ("Baselines", "ECG 1D-CNN, compact HAPT model, matched MLP controls"),
        ("Independent data", "INCART patient-independent ECG validation"),
        ("Evaluation", "Macro-F1, balanced accuracy and per-class rare-class metrics"),
        ("Ablation", "Branch count, branch width and trunk size varied independently"),
        ("Snowflake evidence", "Branch statistics, cosine similarity, quantization error and clipping"),
        ("Precision", "Separate INT4 / INT6 / INT8 storage-only comparison"),
        ("Statistics", "95% CIs and paired TOST for accuracy, macro-F1 and balanced accuracy"),
        ("Hardware", "Four-core Pi latency/RAM protocol and sustained thermal evaluation"),
    ]
    y = 0.77
    for heading, detail in rows:
        fig.text(0.08, y, heading, fontsize=12, weight="bold")
        fig.text(0.27, y, detail, fontsize=11)
        y -= 0.075

    fig.text(
        0.07,
        0.09,
        "Scope boundaries: energy per inference and microcontroller deployment were not measured; "
        "the Raspberry Pi is treated as an edge-gateway SBC.",
        fontsize=10,
        style="italic",
        wrap=True,
    )
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def image_page(pdf: PdfPages, title: str, path: Path, caption: str) -> None:
    if not path.exists():
        raise FileNotFoundError(path)
    image = mpimg.imread(path)
    fig = plt.figure(figsize=(11.69, 8.27), facecolor="white")
    ax = fig.add_axes((0.04, 0.12, 0.92, 0.78))
    ax.imshow(image)
    ax.axis("off")
    fig.suptitle(title, fontsize=17, weight="bold", y=0.96)
    fig.text(0.06, 0.055, caption, fontsize=10, wrap=True)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    with PdfPages(OUTPUT) as pdf:
        cover_page(pdf)
        for title, path, caption in FIGURES:
            image_page(pdf, title, path, caption)
    print(f"Saved {len(FIGURES) + 1}-page PDF -> {OUTPUT}")


if __name__ == "__main__":
    main()
