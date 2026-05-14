import copy
import torch
from sklearn.model_selection import train_test_split

from src.loaders.load_adult import load_adult_income
from src.training.train import train
from src.training.evaluate import evaluate, mse_score, predict_proba
from src.models.dendritic_network import DendriticNetwork
from src.models.mlp_baseline import MLPBaseline
from src.compression.compression_pipeline import (
    compress_model,
    decompress_model,
    compressed_size_bytes,
)


def run_uci_adult_income(epochs=50, seeds=(42,), fine_tune_epochs=3):
    X_raw, y_raw = load_adult_income()

    acc_u_list, acc_c_list, acc_mlp_list, acc_mlp_c_list = [], [], [], []
    mse_u_list, mse_c_list, mse_mlp_list, mse_mlp_c_list = [], [], [], []
    size_u, size_c = None, None
    size_mlp_u, size_mlp_c = None, None
    curve_data = None
    loss_history = None

    for seed in seeds:
        X_tr, X_test, y_tr, y_test = train_test_split(
            X_raw, y_raw, test_size=0.2, random_state=seed
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_tr, y_tr, test_size=0.1, random_state=seed
        )

        X_train = torch.tensor(X_train, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)
        X_val   = torch.tensor(X_val,   dtype=torch.float32)
        y_val   = torch.tensor(y_val,   dtype=torch.float32).reshape(-1, 1)
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
        hist_u, val_hist_u = train(model_u, X_train, y_train, epochs=epochs,
                                   X_val=X_val, y_val=y_val)
        acc_u_list.append(evaluate(model_u, X_test, y_test))
        mse_u_list.append(mse_score(model_u, X_test, y_test))

        original_state = copy.deepcopy(model_u.state_dict())

        compressed = compress_model(model_u, fine_tune_data=(X_train, y_train),
                                    fine_tune_epochs=fine_tune_epochs)
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

        # MLP baseline — uncompressed then compressed
        mlp = MLPBaseline(input_dim=X_train.shape[1], match_params=sum(p.numel() for p in model_u.parameters()))
        hist_mlp, val_hist_mlp = train(mlp, X_train, y_train, epochs=epochs,
                                       X_val=X_val, y_val=y_val)
        acc_mlp_list.append(evaluate(mlp, X_test, y_test))
        mse_mlp_list.append(mse_score(mlp, X_test, y_test))

        compressed_mlp = compress_model(mlp, fine_tune_data=(X_train, y_train),
                                        fine_tune_epochs=fine_tune_epochs)
        decompress_model(compressed_mlp, mlp)
        acc_mlp_c_list.append(evaluate(mlp, X_test, y_test))
        mse_mlp_c_list.append(mse_score(mlp, X_test, y_test))

        if size_mlp_u is None:
            size_mlp_u = mlp.size_bytes()
            size_mlp_c = compressed_size_bytes(compressed_mlp)

        if loss_history is None:
            loss_history = {
                "Dendritic (Uncompressed)": hist_u,
                "Dendritic (Val)":          val_hist_u,
                "MLP Baseline":             hist_mlp,
                "MLP (Val)":                val_hist_mlp,
            }

    n = len(seeds)

    def _mean(lst):
        return float(sum(lst) / n)

    def _std(lst):
        return float(torch.tensor(lst).std().item()) if n > 1 else 0.0

    return {
        "accuracy": {
            "uncompressed":   _mean(acc_u_list),
            "compressed":     _mean(acc_c_list),
            "mlp_baseline":   _mean(acc_mlp_list),
            "mlp_compressed": _mean(acc_mlp_c_list),
        },
        "accuracy_std": {
            "uncompressed":   _std(acc_u_list),
            "compressed":     _std(acc_c_list),
            "mlp_baseline":   _std(acc_mlp_list),
            "mlp_compressed": _std(acc_mlp_c_list),
        },
        "mse": {
            "uncompressed":   _mean(mse_u_list),
            "compressed":     _mean(mse_c_list),
            "mlp_baseline":   _mean(mse_mlp_list),
            "mlp_compressed": _mean(mse_mlp_c_list),
        },
        "mse_std": {
            "uncompressed":   _std(mse_u_list),
            "compressed":     _std(mse_c_list),
            "mlp_baseline":   _std(mse_mlp_list),
            "mlp_compressed": _std(mse_mlp_c_list),
        },
        "sizes": {
            "uncompressed":     size_u,
            "compressed":       size_c,
            "mlp_uncompressed": size_mlp_u,
            "mlp_compressed":   size_mlp_c,
        },
        "num_seeds": n,
        "curve_data": curve_data,
        "loss_history": loss_history,
    }
