import numpy as np
import matplotlib.pyplot as plt
from .save_utils import save_fig
from .style import apply_style, METHOD_COLORS


def plot_weight_distribution(weight_dist, title="", filename=None):
    """
    weight_dist: {"before": np.array, "after": np.array}
    Overlaid histograms of weight values before and after int8 quantization.
    """
    apply_style()

    before = weight_dist["before"]
    after  = weight_dist["after"]

    # Clip extreme outliers for readability
    lo = float(np.percentile(before, 0.5))
    hi = float(np.percentile(before, 99.5))
    bins = np.linspace(lo, hi, 80)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(before, bins=bins, alpha=0.65, color=METHOD_COLORS["Uncompressed"],
            label="Float32 (before)", density=True, zorder=3)
    ax.hist(after,  bins=bins, alpha=0.65, color=METHOD_COLORS["Snowflake (int8)"],
            label="int8 (after)",     density=True, zorder=4)

    ax.set_xlabel("Weight Value")
    ax.set_ylabel("Density")
    ax.set_title(f"Weight Distribution{' — ' + title if title else ''}", pad=14)
    ax.legend()

    if filename is None:
        filename = f"{title.lower().replace(' ', '_')}_weight_dist.png"
    save_fig(filename)
