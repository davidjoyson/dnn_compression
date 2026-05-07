import torch
from src.training.train import train
from src.training.evaluate import evaluate
from src.models.dendritic_network import DendriticNetwork
from src.data.load_folktables import load_folktables_income
from src.compression.compression_pipeline import (
    compress_model,
    decompress_model,
    compressed_size_bytes
)


def run_folktables_income(state, year, epochs=50):

    # 1. Load data
    X_train, y_train, X_test, y_test = load_folktables_income(state, year)

    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)
    X_test  = torch.tensor(X_test,  dtype=torch.float32)
    y_test  = torch.tensor(y_test,  dtype=torch.float32).reshape(-1, 1)

    # 2. Build model
    model_u = DendriticNetwork(
        input_dim=X_train.shape[1],
        hidden_neurons1=32,
        hidden_neurons2=16,
        branches=6,
        hidden_per_branch=4
    )

    # 3. Train uncompressed model
    train(model_u, X_train, y_train, epochs=epochs)
    acc_u = evaluate(model_u, X_test, y_test)

    # 4. Compress → returns compressed dict
    compressed = compress_model(model_u)
    size_c = compressed_size_bytes(compressed)

    # 5. Decompress → returns PyTorch model with restored weights
    model_c = decompress_model(compressed, model_u)
    acc_c = evaluate(model_c, X_test, y_test)

    # 6. Uncompressed size
    size_u = model_u.size_bytes()

    return {
        "accuracy": {
            "uncompressed": acc_u,
            "compressed": acc_c
        },
        "sizes": {
            "uncompressed": size_u,
            "compressed": size_c
        }
    }
