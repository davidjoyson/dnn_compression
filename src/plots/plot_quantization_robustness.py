import numpy as np
import matplotlib.pyplot as plt
from .save_utils import save_fig
from .style import apply_style

# COMPARISON_METHODS internal name -> display label (src/experiments/base_experiment.py)
_METHODS = [
    ("snowflake",        "Snowflake"),
    ("global",           "Global"),
    ("dynamic",          "Dynamic"),
    ("static",           "Static"),
    ("snowflake_static", "SF+Static"),
    ("perchan",          "Per-chan"),
    ("qat",              "QAT"),
    ("mixed",            "Mixed"),
]

# Dendritic's per-method accuracy is flattened onto the top-level result dict
# under these keys (src/reporting/utils.py: store_simple), unlike MLP/
# LayerMatchedMLP which keep the nested {method: value} form under
# "method_comparison" (src/experiments/base_experiment.py).
_DENDRITIC_KEYS = {
    "snowflake":        "accuracy_compressed",
    "global":           "accuracy_compressed_global",
    "dynamic":          "accuracy_compressed_dynamic",
    "static":           "accuracy_compressed_static",
    "snowflake_static": "accuracy_compressed_snowflake_static",
    "perchan":          "accuracy_compressed_perchan",
    "qat":              "accuracy_compressed_qat",
    "mixed":            "accuracy_compressed_mixed",
}

_ARCHS = [
    ("Dendritic",      "#4878CF"),
    ("MLP (param-matched)",   "#D62728"),
    ("LayerMatchedMLP (per-layer)", "#2CA02C"),
]


def _isnan(v):
    return v is None or (isinstance(v, float) and v != v)


def plot_quantization_robustness(r, title="", filename="quantization_robustness.png"):
    """
    Accuracy delta from each architecture's own uncompressed baseline, per
    quantization method. Answers professor point 2/9 directly: does Dendritic
    degrade less under quantization than a parameter- or layer-matched MLP?
    (r["method_comparison"] populated by base_experiment.py's compress_all_methods
    for both MLPBaseline and LayerMatchedMLP.)
    """
    apply_style()
    mc = r.get("method_comparison") or {}
    mlp, lm = mc.get("mlp") or {}, mc.get("layer_matched") or {}

    series = {
        "Dendritic": (r.get("accuracy_uncompressed"),
                      {m: r.get(k) for m, k in _DENDRITIC_KEYS.items()}),
        "MLP (param-matched)": (mlp.get("accuracy_uncompressed"),
                                 mlp.get("accuracy", {})),
        "LayerMatchedMLP (per-layer)": (lm.get("accuracy_uncompressed"),
                                         lm.get("accuracy", {})),
    }

    methods = [(mkey, label) for mkey, label in _METHODS
               if any(not _isnan((series[a][1] or {}).get(mkey)) for a, _ in _ARCHS)]
    if not methods:
        return

    n_m = len(methods)
    width = min(0.25, 0.8 / len(_ARCHS))
    offsets = np.linspace(-(len(_ARCHS) - 1) / 2, (len(_ARCHS) - 1) / 2, len(_ARCHS)) * width
    x = np.arange(n_m)

    fig, ax = plt.subplots(figsize=(max(8, n_m * 1.2), 5))

    for i, (arch_label, color) in enumerate(_ARCHS):
        base, acc = series[arch_label]
        deltas = []
        for mkey, _ in methods:
            v = acc.get(mkey)
            deltas.append((v - base) * 100 if not _isnan(v) and not _isnan(base) else float("nan"))
        ax.bar(x + offsets[i], deltas, width, label=arch_label, color=color,
               zorder=3, edgecolor="white", linewidth=0.6)

    ax.axhline(0, color="black", linewidth=0.8, zorder=2)
    ax.set_ylabel("Accuracy delta vs. own uncompressed (pp)")
    ax.set_title(title, pad=14)
    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label in methods])
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=3, fontsize=8, frameon=False)

    save_fig(filename)
