import torch
from src.training.train import train
from src.training.evaluate import evaluate
from src.models.dendritic_network import DendriticNetwork
from src.compression.compression_pipeline import (
    compress_model,
    decompress_model,
    compressed_size_bytes
)


def run_ablation(configs, X_train, y_train, X_test, y_test, epochs=50):
    """
    configs: list of dicts, each containing:
        {
            "h1": int,
            "h2": int,
            "branches": int,
            "hidden_per_branch": int
        }
    """

    results = []

    # Convert data to tensors once
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)
    X_test  = torch.tensor(X_test,  dtype=torch.float32)
    y_test  = torch.tensor(y_test,  dtype=torch.float32).reshape(-1, 1)

    for cfg in configs:

        # -------------------------
        # 1. Build model
        # -------------------------
        model_u = DendriticNetwork(
            input_dim=X_train.shape[1],
            hidden_neurons1=cfg["h1"],
            hidden_neurons2=cfg["h2"],
            branches=cfg["branches"],
            hidden_per_branch=cfg["hidden_per_branch"]
        )

        # -------------------------
        # 2. Train uncompressed
        # -------------------------
        train(model_u, X_train, y_train, epochs=epochs)
        acc_u = evaluate(model_u, X_test, y_test)

        # -------------------------
        # 3. Compress → dict
        # -------------------------
        compressed = compress_model(model_u)
        size_c = compressed_size_bytes(compressed)

        # -------------------------
        # 4. Decompress → model
        # -------------------------
        model_c = decompress_model(compressed, model_u)
        acc_c = evaluate(model_c, X_test, y_test)

        # -------------------------
        # 5. Uncompressed size
        # -------------------------
        size_u = model_u.size_bytes()

        # -------------------------
        # 6. Store results
        # -------------------------
        results.append({
            "config": cfg,
            "accuracy_uncompressed": acc_u,
            "accuracy_compressed": acc_c,
            "size_uncompressed": size_u,
            "size_compressed": size_c
        })

    return results
