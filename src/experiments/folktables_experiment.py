import copy
import torch
from src.training.train import train
from src.training.evaluate import evaluate, mse_score, predict_proba
from src.models.dendritic_network import DendriticNetwork
from src.models.mlp_baseline import MLPBaseline
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
    hist_u = train(model_u, X_train, y_train, epochs=epochs)
    acc_u = evaluate(model_u, X_test, y_test)
    mse_u = mse_score(model_u, X_test, y_test)
    original_state = copy.deepcopy(model_u.state_dict())

    # 4. Compress → returns compressed dict
    compressed = compress_model(model_u)
    size_c = compressed_size_bytes(compressed)

    # 5. Decompress → returns PyTorch model with restored weights
    model_c = decompress_model(compressed, model_u)
    acc_c = evaluate(model_c, X_test, y_test)
    mse_c = mse_score(model_c, X_test, y_test)

    # 6. Uncompressed size
    size_u = model_u.size_bytes()

    mlp = MLPBaseline(input_dim=X_train.shape[1], hidden=32)
    hist_mlp = train(mlp, X_train, y_train, epochs=epochs)
    acc_mlp = evaluate(mlp, X_test, y_test)
    mse_mlp = mse_score(mlp, X_test, y_test)

    compressed_mlp = compress_model(mlp)
    size_mlp_u = mlp.size_bytes()
    size_mlp_c = compressed_size_bytes(compressed_mlp)
    decompress_model(compressed_mlp, mlp)
    acc_mlp_c = evaluate(mlp, X_test, y_test)
    mse_mlp_c = mse_score(mlp, X_test, y_test)

    score_c = predict_proba(model_c, X_test)
    model_u.load_state_dict(original_state)
    score_u = predict_proba(model_u, X_test)
    curve_data = {
        "y_true":               y_test.cpu().numpy().ravel(),
        "y_score_uncompressed": score_u,
        "y_score_compressed":   score_c,
    }

    return {
        "accuracy": {
            "uncompressed":   acc_u,
            "compressed":     acc_c,
            "mlp_baseline":   acc_mlp,
            "mlp_compressed": acc_mlp_c,
        },
        "mse": {
            "uncompressed":   mse_u,
            "compressed":     mse_c,
            "mlp_baseline":   mse_mlp,
            "mlp_compressed": mse_mlp_c,
        },
        "sizes": {
            "uncompressed":     size_u,
            "compressed":       size_c,
            "mlp_uncompressed": size_mlp_u,
            "mlp_compressed":   size_mlp_c,
        },
        "curve_data": curve_data,
        "loss_history": {
            "Dendritic (Uncompressed)": hist_u,
            "MLP Baseline":             hist_mlp,
        },
    }
