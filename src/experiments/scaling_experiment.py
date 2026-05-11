import time
import torch
from tqdm import tqdm

from src.training.train import train
from src.training.evaluate import evaluate, mse_score
from src.models.dendritic_network import DendriticNetwork
from src.compression.compression_pipeline import (
    compress_model,
    decompress_model,
    compressed_size_bytes
)


def run_scaling_experiment(
    X_train,
    y_train,
    X_test,
    y_test,
    neurons1_list,
    neurons2_list,
    branches_list,
    hidden_per_branch=4,
    epochs=50
):
    """
    Runs a 3D scaling experiment over:
        - hidden_neurons1
        - hidden_neurons2
        - branches

    Returns:
        {
            "accuracy_uncompressed": 3D array,
            "accuracy_compressed":   3D array,
            "size_uncompressed":     3D array,
            "size_compressed":       3D array
        }
    """

    # Convert data once
    X_train = X_train.clone().detach().float()
    y_train = y_train.clone().detach().float().reshape(-1, 1)
    X_test  = X_test.clone().detach().float()
    y_test  = y_test.clone().detach().float().reshape(-1, 1)

    # Allocate result tensors
    A = len(neurons1_list)
    B = len(neurons2_list)
    C = len(branches_list)

    acc_u    = torch.zeros((A, B, C))
    acc_c    = torch.zeros((A, B, C))
    mse_u    = torch.zeros((A, B, C))
    mse_c    = torch.zeros((A, B, C))
    size_u   = torch.zeros((A, B, C))
    size_c   = torch.zeros((A, B, C))
    time_sec = torch.zeros((A, B, C))

    total = A * B * C
    pbar = tqdm(total=total, desc="Scaling Experiment", colour="white")

    # 3D grid search
    idx = 0
    for i, h1 in enumerate(neurons1_list):
        for j, h2 in enumerate(neurons2_list):
            for k, br in enumerate(branches_list):
                _t0 = time.time()

                # -------------------------
                # 1. Build model
                # -------------------------
                model_u = DendriticNetwork(
                    input_dim=X_train.shape[1],
                    hidden_neurons1=h1,
                    hidden_neurons2=h2,
                    branches=br,
                    hidden_per_branch=hidden_per_branch
                )

                # -------------------------
                # 2. Train uncompressed
                # -------------------------
                train(model_u, X_train, y_train, epochs=epochs)
                acc_un = evaluate(model_u, X_test, y_test)
                mse_un = mse_score(model_u, X_test, y_test)

                # -------------------------
                # 3. Compress → dict
                # -------------------------
                compressed = compress_model(model_u)
                size_com = compressed_size_bytes(compressed)

                # -------------------------
                # 4. Decompress → model
                # -------------------------
                model_c = decompress_model(compressed, model_u)
                acc_com = evaluate(model_c, X_test, y_test)
                mse_com = mse_score(model_c, X_test, y_test)

                # -------------------------
                # 5. Uncompressed size
                # -------------------------
                size_un = model_u.size_bytes()

                # -------------------------
                # 6. Store results
                # -------------------------
                acc_u[i, j, k]    = acc_un
                acc_c[i, j, k]    = acc_com
                mse_u[i, j, k]    = mse_un
                mse_c[i, j, k]    = mse_com
                size_u[i, j, k]   = size_un
                size_c[i, j, k]   = size_com
                time_sec[i, j, k] = time.time() - _t0

                # -------------------------
                # 7. Progress bar update
                # -------------------------
                pbar.set_postfix({
                    "h1": h1,
                    "h2": h2,
                    "branches": br,
                    "acc": float(acc_un)
                })
                pbar.update(1)
                idx += 1

    pbar.close()

    return {
        "accuracy_uncompressed": acc_u,
        "accuracy_compressed":   acc_c,
        "mse_uncompressed":      mse_u,
        "mse_compressed":        mse_c,
        "size_uncompressed":     size_u,
        "size_compressed":       size_c,
        "time_per_config":       time_sec,
    }
