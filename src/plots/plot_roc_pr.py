import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_curve, auc,
    precision_recall_curve, average_precision_score,
)
from .save_utils import fig_path


def plot_roc_pr(curve_data, title="", filename=None):
    y_true = curve_data["y_true"]
    scores = {
        "Uncompressed": curve_data["y_score_uncompressed"],
        "Compressed":   curve_data["y_score_compressed"],
    }
    colors = {"Uncompressed": "#4C72B0", "Compressed": "#DD8452"}

    fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(12, 5))

    for label, score in scores.items():
        fpr, tpr, _ = roc_curve(y_true, score)
        roc_auc = auc(fpr, tpr)
        ax_roc.plot(fpr, tpr, color=colors[label], lw=2,
                    label=f"{label} (AUC = {roc_auc:.4f})")
    ax_roc.plot([0, 1], [0, 1], "k--", lw=1)
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.set_title(f"ROC Curve{' — ' + title if title else ''}")
    ax_roc.legend(loc="lower right")
    ax_roc.grid(linestyle="--", alpha=0.4)

    for label, score in scores.items():
        prec, rec, _ = precision_recall_curve(y_true, score)
        ap = average_precision_score(y_true, score)
        ax_pr.plot(rec, prec, color=colors[label], lw=2,
                   label=f"{label} (AP = {ap:.4f})")
    ax_pr.set_xlabel("Recall")
    ax_pr.set_ylabel("Precision")
    ax_pr.set_title(f"Precision-Recall Curve{' — ' + title if title else ''}")
    ax_pr.legend(loc="upper right")
    ax_pr.grid(linestyle="--", alpha=0.4)

    plt.tight_layout()
    if filename is None:
        slug = title.lower().replace(" ", "_").replace("/", "_")
        filename = f"{slug}_roc_pr.png"
    plt.savefig(fig_path(filename), dpi=150)
    plt.close()
