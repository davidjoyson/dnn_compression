import torch
from tqdm import tqdm
from src.experiments.xor_experiment import run_xor
from src.experiments.uci_experiments import run_wine
from src.experiments.ablation_study import run_ablation
from src.experiments.scaling_experiment import scaling_experiment

from src.plots.plot_accuracy import plot_accuracy
from src.plots.plot_compression import plot_compression
from src.plots.plot_ablation import plot_ablation
from src.plots.plot_scaling import plot_scaling
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from src.experiments.adult_income_experiment import run_adult_income
from src.experiments.folktables_experiment import run_folktables_income
from src.plots.folktables_plots import *

from src.plots.save_utils import fig_path

experiments = [
    ("Folktables CA 2018", lambda: run_folktables_income("CA", 2018)),
]

# print("\n=== Running XOR Experiment ===")
# acc_u, acc_c = run_xor()
# print("Uncompressed:", acc_u)
# print("Compressed:", acc_c)
# plot_accuracy(acc_u, acc_c, title="XOR Accuracy", filename="xor_accuracy.png")

# # Fake sizes for XOR (your thesis uses 124 → 27 bytes)
# plot_compression(124, 27, filename="xor_compression.png")


# print("\n=== Running Wine Dataset Experiment ===")
# acc_u, acc_c = run_wine()
# print("Uncompressed:", acc_u)
# print("Compressed:", acc_c)
# plot_accuracy(acc_u, acc_c, title="Wine Accuracy", filename="wine_accuracy.png")


# print("\n=== Running Ablation Study ===")
# data = load_wine()
# X = StandardScaler().fit_transform(data.data)
# y = (data.target == 0).astype(float).reshape(-1,1)

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
# X_train = torch.tensor(X_train, dtype=torch.float32)
# y_train = torch.tensor(y_train, dtype=torch.float32)
# X_test = torch.tensor(X_test, dtype=torch.float32)
# y_test = torch.tensor(y_test, dtype=torch.float32)

# ablation_results = run_ablation(X_train, y_train, X_test, y_test)
# print(ablation_results)
# plot_ablation(ablation_results, filename="ablation_study.png")


# print("\n=== Running Scaling Experiment ===")
# scaling_results = scaling_experiment(X_train, y_train, X_test, y_test)
# plot_scaling(scaling_results, filename="scaling_heatmap.png")

# print("\n=== Running Adult Income Experiment ===")
# acc_u, acc_c = run_adult_income()
# plot_accuracy(acc_u, acc_c, title="Adult Income Accuracy", filename="adult_income_accuracy.png")

print("\n=== Running Folktables ACSIncome (CA, 2018) ===")
for name, fn in tqdm(experiments, desc="Running Experiments"):
    print(f"\n=== {name} ===")
    results = fn()

    plot_folktables_accuracy(results["accuracy"], filename="folktables_accuracy.png")
    plot_folktables_size(
        results["sizes"]["uncompressed"],
        results["sizes"]["compressed"],
        filename="folktables_size.png"
    )
    plot_folktables_tradeoff(
        results["accuracy"]["Dendritic (Uncompressed)"],
        results["accuracy"]["Dendritic (Compressed)"],
        results["sizes"]["uncompressed"],
        results["sizes"]["compressed"],
        filename="folktables_tradeoff.png"
    )
print("Saved:", fig_path("folktables_tradeoff.png"))

print("\nAll Folktables plots saved to figures/")
