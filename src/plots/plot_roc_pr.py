import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_curve, auc,
    precision_recall_curve, average_precision_score,
)
from .save_utils import fig_path
from .style import apply_style


_COLORS = {"Uncompressed": "#4878CF", "Compressed": "#E87722"}


def plot_roc_pr(curve_data, title="", filename=None):
    apply_style()

    y_true = curve_data["y_true"]
    scores = {
        "Uncompressed": curve_data["y_score_uncompressed"],
        "Compressed":   curve_data["y_score_compressed"],
    }

    fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(12, 4.8))

    for label, score in scores.items():
        color = _COLORS[label]
        fpr, tpr, _ = roc_curve(y_true, score)
        roc_auc = auc(fpr, tpr)
        ax_roc.plot(fpr, tpr, color=color, lw=2, label=f"{label} (AUC={roc_auc:.4f})")
        ax_roc.fill_between(fpr, tpr, alpha=0.08, color=color)

    ax_roc.plot([0, 1], [0, 1], color="#999999", linestyle="--", lw=1, label="Chance")
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.set_title(f"ROC Curve{' — ' + title if title else ''}")
    ax_roc.legend(loc="lower right")

    for label, score in scores.items():
        color = _COLORS[label]
        prec, rec, _ = precision_recall_curve(y_true, score)
        ap = average_precision_score(y_true, score)
        ax_pr.plot(rec, prec, color=color, lw=2, label=f"{label} (AP={ap:.4f})")
        ax_pr.fill_between(rec, prec, alpha=0.08, color=color)

    ax_pr.set_xlabel("Recall")
    ax_pr.set_ylabel("Precision")
    ax_pr.set_title(f"Precision-Recall{' — ' + title if title else ''}")
    ax_pr.legend(loc="upper right")

    plt.tight_layout()
    if filename is None:
        slug = title.lower().replace(" ", "_").replace("/", "_")
        filename = f"{slug}_roc_pr.png"
    plt.savefig(fig_path(filename), dpi=150, bbox_inches="tight")
    plt.close()
