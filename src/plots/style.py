import matplotlib as mpl

PALETTE = [
    "#4878CF",  # steel blue
    "#E87722",  # orange
    "#2CA02C",  # green
    "#9467BD",  # purple
    "#D62728",  # red
    "#8C564B",  # brown
    "#17BECF",  # cyan
]

METHOD_COLORS = {
    "Uncompressed":    "#4878CF",
    "Snowflake (int8)":"#E87722",
    "Global int8":     "#2CA02C",
    "Dynamic (int8)":  "#9467BD",
    "MLP Baseline":    "#D62728",
    "MLP Compressed":  "#8C564B",
    "Snowflake+Static (int8)": "#17BECF",
    "Static (int8)":       "#7F7F7F",
    "Per-channel (int8)":  "#BCBD22",
    "QAT (int8)":           "#E377C2",
    "Mixed precision":      "#393B79",
    "Snowflake (int4)":     "#637939",
}


def apply_style():
    mpl.rcParams.update({
        "figure.facecolor":   "white",
        "axes.facecolor":     "#F7F7F7",
        "axes.spines.top":    False,
        "axes.spines.right":  False,
        "axes.grid":          True,
        "axes.axisbelow":     True,
        "grid.color":         "white",
        "grid.linewidth":     1.1,
        "font.family":        "sans-serif",
        "font.size":          10,
        "axes.titlesize":     12,
        "axes.titleweight":   "bold",
        "axes.labelsize":     10,
        "xtick.labelsize":    9,
        "ytick.labelsize":    9,
        "legend.fontsize":    9,
        "legend.framealpha":  0.85,
        "figure.dpi":         150,
    })
