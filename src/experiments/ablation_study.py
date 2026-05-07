import torch
from src.models.dendritic_network import DendriticNetwork
from src.training.train import train
from src.training.evaluate import evaluate
from src.compression.topology_sharing import apply_topology_sharing
from src.compression.quantization import quantize_model

def run_ablation(X_train, y_train, X_test, y_test):
    results = {}

    # Baseline
    m = DendriticNetwork(X_train.shape[1])
    train(m, X_train, y_train)
    results["baseline"] = evaluate(m, X_test, y_test)

    # Topology only
    m = DendriticNetwork(X_train.shape[1])
    train(m, X_train, y_train)
    apply_topology_sharing(m)
    results["topology_only"] = evaluate(m, X_test, y_test)

    # Quantization only
    m = DendriticNetwork(X_train.shape[1])
    train(m, X_train, y_train)
    quantize_model(m)
    results["quant_only"] = evaluate(m, X_test, y_test)

    # Full compression
    m = DendriticNetwork(X_train.shape[1])
    train(m, X_train, y_train)
    apply_topology_sharing(m)
    quantize_model(m)
    results["full"] = evaluate(m, X_test, y_test)

    return results
