import torch
from sklearn.model_selection import train_test_split

from src.data.load_adult import load_adult_income
from src.training.train import train
from src.training.evaluate import evaluate
from src.models.dendritic_network import DendriticNetwork
from src.compression.compression_pipeline import (
    compress_model,
    decompress_model,
    compressed_size_bytes
)


def run_uci_adult_income(epochs=50):

    # 1. Load dataset
    X, y = load_adult_income()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Convert to tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)
    X_test  = torch.tensor(X_test,  dtype=torch.float32)
    y_test  = torch.tensor(y_test,  dtype=torch.float32).reshape(-1, 1)

    # 2. Build uncompressed model
    model_u = DendriticNetwork(
        input_dim=X_train.shape[1],
        hidden_neurons1=32,
        hidden_neurons2=16,
        branches=4,
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
