"""
Generate a plot from the existing Raspberry Pi benchmark CSVs
(benchmark_pi_output/*.csv) -- these had no visualization before, only
the raw CSVs and the hand-built README table.

Usage:
  python plot_pi_benchmark.py
"""
import pandas as pd

import src.plots.save_utils as save_utils
from src.plots.plot_pi_benchmark import plot_pi_latency

DATASETS = ["har", "ecg", "hapt", "eeg"]

if __name__ == "__main__":
    dataframes = {}
    for ds in DATASETS:
        try:
            dataframes[ds] = pd.read_csv(f"benchmark_pi_output/results_{ds}.csv")
        except FileNotFoundError:
            continue

    save_utils.set_fig_dir("benchmark_pi_output")
    plot_pi_latency(dataframes)
    print("Saved -> benchmark_pi_output/pi_latency_batch1.png")
