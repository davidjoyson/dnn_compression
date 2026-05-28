import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.preprocessing import label_binarize
from .save_utils import fig_path
from .style import apply_style


_COLORS = {"Uncompressed": "#4878CF", "Compressed": "#E87722"}


def plot_roc_pr(curve_data, title="", filename=None):
    apply_style()

    y_true = np.asarray(curve_data["y_true"])
    num_classes = curve_data.get("num_classes", 2)
    scores = {
        "Uncompressed": np.asarray(curve_data["y_score_uncompressed"]),
        "Compressed":   np.asarray(curve_data["y_score_compressed"]),
    }

    fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(12, 4.8))

    if num_classes > 2:
        classes = list(range(num_classes))
        y_bin = label_binarize(y_true, classes=classes)  # (N, C)
        mean_fpr = np.linspace(0, 1, 200)

        for label, score_mat in scores.items():
            color = _COLORS[label]

            # Macro-average ROC
            interp_tprs = []
            for c in classes:
                fpr_c, tpr_c, _ = roc_curve(y_bin[:, c], score_mat[:, c])
                interp_tprs.append(np.interp(mean_fpr, fpr_c, tpr_c))
            mean_tpr = np.mean(interp_tprs, axis=0)
            macro_auc = auc(mean_fpr, mean_tpr)
            ax_roc.plot(mean_fpr, mean_tpr, color=color, lw=2,
                        label=f"{label} (macro AUC={macro_auc:.4f})")
            ax_roc.fill_between(mean_fpr, mean_tpr, alpha=0.08, color=color)

            # Micro-average PR (flatten OvR)
            macro_ap = np.mean([
                average_precision_score(y_bin[:, c], score_mat[:, c]) for c in classes
            ])
            prec, rec, _ = precision_recall_curve(y_bin.ravel(), score_mat.ravel())
            ax_pr.plot(rec, prec, color=color, lw=2,
                       label=f"{label} (macro AP={macro_ap:.4f})")
            ax_pr.fill_between(rec, prec, alpha=0.08, color=color)
    else:
        # Binary: use column 1 if 2-D, else use as-is
        for label, score in scores.items():
            if score.ndim == 2:
                score = score[:, 1]
            color = _COLORS[label]
            fpr, tpr, _ = roc_curve(y_true, score)
            ax_roc.plot(fpr, tpr, color=color, lw=2,
                        label=f"{label} (AUC={auc(fpr, tpr):.4f})")
            ax_roc.fill_between(fpr, tpr, alpha=0.08, color=color)
            prec, rec, _ = precision_recall_curve(y_true, score)
            ap = average_precision_score(y_true, score)
            ax_pr.plot(rec, prec, color=color, lw=2,
                       label=f"{label} (AP={ap:.4f})")
            ax_pr.fill_between(rec, prec, alpha=0.08, color=color)

    ax_roc.plot([0, 1], [0, 1], color="#999999", linestyle="--", lw=1, label="Chance")
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.set_title(f"ROC Curve{' — ' + title if title else ''}")
    ax_roc.legend(loc="lower right")

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
