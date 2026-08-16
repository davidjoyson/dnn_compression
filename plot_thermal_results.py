"""Plot Raspberry Pi sustained-thermal CSV results.

Usage:
    python plot_thermal_results.py \
        --input final_output/pi_benchmark/thermal_ecg_20260816
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.plots.style import PALETTE, apply_style


LABELS = {
    "float32": "Float32",
    "snowflake": "Snowflake INT8",
    "global": "Global INT8",
    "dynamic": "Dynamic INT8",
    "static": "Static W+A INT8",
    "snowflake_static": "Snowflake+Static INT8",
    "perchan": "Per-channel INT8",
    "qat": "QAT INT8",
    "mixed": "Mixed precision",
    "int4": "Snowflake INT4",
}


def plot_results(input_dir: Path, output_dir: Path) -> None:
    apply_style()
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = pd.read_csv(input_dir / "ecg_all_summary.csv")
    colors = {
        method: PALETTE[index % len(PALETTE)]
        for index, method in enumerate(summary["method"])
    }

    fig, ax = plt.subplots(figsize=(10, 5.5))
    for method in summary["method"]:
        series = pd.read_csv(input_dir / f"ecg_{method}.csv")
        ax.plot(
            series["elapsed_s"] / 60.0,
            series["temp_c"],
            linewidth=1.8,
            color=colors[method],
            label=LABELS.get(method, method),
        )
    ax.set_title("Raspberry Pi 3 Temperature During Sustained ECG Inference")
    ax.set_xlabel("Elapsed time within each method (minutes)")
    ax.set_ylabel("CPU temperature (°C)")
    ax.legend(ncol=2, loc="best")
    fig.tight_layout()
    fig.savefig(output_dir / "thermal_temperature_timeseries.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    labels = [LABELS.get(method, method) for method in summary["method"]]
    y = range(len(summary))
    fig, (ax_speed, ax_temp) = plt.subplots(1, 2, figsize=(13, 6), sharey=True)
    ax_speed.barh(y, summary["inf_per_s"], color=[colors[m] for m in summary["method"]])
    ax_speed.set_yticks(list(y), labels)
    ax_speed.invert_yaxis()
    ax_speed.set_xlabel("Sustained throughput (inferences/s)")
    ax_speed.set_title("Throughput")

    ax_temp.barh(y, summary["peak_temp"], color=[colors[m] for m in summary["method"]])
    ax_temp.set_xlabel("Peak CPU temperature (°C)")
    ax_temp.set_title("Peak temperature")
    ax_temp.set_xlim(0, max(60, float(summary["peak_temp"].max()) + 3))

    fig.suptitle("Five-Minute Sustained ECG Thermal Test — Raspberry Pi 3", fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_dir / "thermal_summary.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Folder containing ECG thermal CSVs")
    parser.add_argument("--output", type=Path, default=None, help="Plot folder; defaults to --input")
    args = parser.parse_args()
    plot_results(args.input, args.output or args.input)


if __name__ == "__main__":
    main()
