"""
Generate a plot from the existing Raspberry Pi benchmark CSVs
(benchmark_pi_output/*.csv) -- these had no visualization before, only
the raw CSVs and the hand-built README table.

Usage:
  python plot_pi_benchmark.py
"""
import pandas as pd

import src.plots.save_utils as save_utils
from src.plots.plot_pi_benchmark import (
    plot_pi_latency, plot_pi_memory, plot_pi_speedup_all_methods,
    plot_pi_batch_comparison, plot_pi_pareto,
)

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
    plot_pi_memory(dataframes)
    print("Saved -> benchmark_pi_output/pi_memory_batch1.png")
    plot_pi_speedup_all_methods(dataframes)
    print("Saved -> benchmark_pi_output/pi_speedup_all_methods.png")
    plot_pi_batch_comparison(dataframes)
    print("Saved -> benchmark_pi_output/pi_batch_comparison.png")
    plot_pi_pareto(dataframes)
    print("Saved -> benchmark_pi_output/pi_pareto_real.png")
