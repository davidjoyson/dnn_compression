import copy
import torch
from sklearn.model_selection import train_test_split

from src.data.load_adult import load_adult_income
from src.training.train import train
from src.training.evaluate import evaluate, mse_score, predict_proba
from src.models.dendritic_network import DendriticNetwork
from src.models.mlp_baseline import MLPBaseline
from src.compression.compression_pipeline import (
    compress_model,
    decompress_model,
    compressed_size_bytes,
)


def run_uci_adult_income(epochs=50, seeds=(42,)):
    X_raw, y_raw = load_adult_income()

    acc_u_list, acc_c_list, acc_mlp_list = [], [], []
    mse_u_list, mse_c_list, mse_mlp_list = [], [], []
    size_u, size_c = None, None
    curve_data = None
    loss_history = None

    for seed in seeds:
        X_train, X_test, y_train, y_test = train_test_split(
            X_raw, y_raw, test_size=0.2, random_state=seed
        )

        X_train = torch.tensor(X_train, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)
        X_test  = torch.tensor(X_test,  dtype=torch.float32)
        y_test  = torch.tensor(y_test,  dtype=torch.float32).reshape(-1, 1)

        torch.manual_seed(seed)

        # Dendritic model — uncompressed
        model_u = DendriticNetwork(
            input_dim=X_train.shape[1],
            hidden_neurons1=32,
            hidden_neurons2=16,
            branches=4,
            hidden_per_branch=4,
        )
        hist_u = train(model_u, X_train, y_train, epochs=epochs)
        acc_u_list.append(evaluate(model_u, X_test, y_test))
        mse_u_list.append(mse_score(model_u, X_test, y_test))

        original_state = copy.deepcopy(model_u.state_dict())

        # Compress with fine-tuning, then evaluate
        compressed = compress_model(model_u, fine_tune_data=(X_train, y_train))
        decompress_model(compressed, model_u)
        acc_c_list.append(evaluate(model_u, X_test, y_test))
        mse_c_list.append(mse_score(model_u, X_test, y_test))

        if size_u is None:
            size_u = model_u.size_bytes()
            size_c = compressed_size_bytes(compressed)

        if curve_data is None:
            model_u.load_state_dict(original_state)
            score_u = predict_proba(model_u, X_test)
            decompress_model(compressed, model_u)
            score_c = predict_proba(model_u, X_test)
            curve_data = {
                "y_true":               y_test.cpu().numpy().ravel(),
                "y_score_uncompressed": score_u,
                "y_score_compressed":   score_c,
            }

        # MLP baseline (comparable hidden size)
        mlp = MLPBaseline(input_dim=X_train.shape[1], hidden=32)
        hist_mlp = train(mlp, X_train, y_train, epochs=epochs)
        acc_mlp_list.append(evaluate(mlp, X_test, y_test))
        mse_mlp_list.append(mse_score(mlp, X_test, y_test))

        if loss_history is None:
            loss_history = {
                "Dendritic (Uncompressed)": hist_u,
                "MLP Baseline":             hist_mlp,
            }

    n = len(seeds)

    def _mean(lst):
        return float(sum(lst) / n)

    def _std(lst):
        return float(torch.tensor(lst).std().item()) if n > 1 else 0.0

    return {
        "accuracy": {
            "uncompressed": _mean(acc_u_list),
            "compressed":   _mean(acc_c_list),
            "mlp_baseline": _mean(acc_mlp_list),
        },
        "accuracy_std": {
            "uncompressed": _std(acc_u_list),
            "compressed":   _std(acc_c_list),
            "mlp_baseline": _std(acc_mlp_list),
        },
        "mse": {
            "uncompressed": _mean(mse_u_list),
            "compressed":   _mean(mse_c_list),
            "mlp_baseline": _mean(mse_mlp_list),
        },
        "mse_std": {
            "uncompressed": _std(mse_u_list),
            "compressed":   _std(mse_c_list),
            "mlp_baseline": _std(mse_mlp_list),
        },
        "sizes": {
            "uncompressed": size_u,
            "compressed":   size_c,
        },
        "num_seeds": n,
        "curve_data": curve_data,
        "loss_history": loss_history,
    }
