import copy
import torch
from src.training.train import train
from src.training.evaluate import evaluate, mse_score
from src.models.dendritic_network import DendriticNetwork
from src.data.load_folktables import load_folktables_income
from src.compression.compression_pipeline import (
    compress_model,
    decompress_model,
    compressed_size_bytes,
)


def run_folktables_multistate(
    train_state="CA",
    test_states=("CA", "TX", "NY", "FL", "WA"),
    year=2018,
    epochs=50,
):
    """
    Trains on one state, tests on multiple states (uncompressed and compressed).
    Measures whether quantization hurts cross-state generalisation.

    Returns:
        {
            "train_state": str,
            "test_states": list[str],
            "accuracy_uncompressed": list[float],
            "accuracy_compressed":   list[float],
            "size_uncompressed": int,
            "size_compressed":   int,
        }
    """
    # --- Train data (single state) ---
    X_tr, y_tr, _, _ = load_folktables_income(train_state, year)
    X_tr = torch.tensor(X_tr, dtype=torch.float32)
    y_tr = torch.tensor(y_tr, dtype=torch.float32).reshape(-1, 1)

    # --- Build and train model ---
    model = DendriticNetwork(
        input_dim=X_tr.shape[1],
        hidden_neurons1=32,
        hidden_neurons2=16,
        branches=6,
        hidden_per_branch=4,
    )
    train(model, X_tr, y_tr, epochs=epochs)

    # --- Compress ---
    original_state = copy.deepcopy(model.state_dict())
    compressed = compress_model(model)
    size_c = compressed_size_bytes(compressed)
    size_u = model.size_bytes()

    # --- Evaluate on each test state ---
    acc_u_list, acc_c_list = [], []
    mse_u_list, mse_c_list = [], []

    for state in test_states:
        _, _, X_te, y_te = load_folktables_income(state, year)
        X_te = torch.tensor(X_te, dtype=torch.float32)
        y_te = torch.tensor(y_te, dtype=torch.float32).reshape(-1, 1)

        # Uncompressed: restore original float32 weights
        model.load_state_dict(original_state)
        acc_u_list.append(evaluate(model, X_te, y_te))
        mse_u_list.append(mse_score(model, X_te, y_te))

        # Compressed: dequantised weights
        decompress_model(compressed, model)
        acc_c_list.append(evaluate(model, X_te, y_te))
        mse_c_list.append(mse_score(model, X_te, y_te))

    return {
        "train_state":           train_state,
        "test_states":           list(test_states),
        "accuracy_uncompressed": acc_u_list,
        "accuracy_compressed":   acc_c_list,
        "mse_uncompressed":      mse_u_list,
        "mse_compressed":        mse_c_list,
        "size_uncompressed":     size_u,
        "size_compressed":       size_c,
    }
