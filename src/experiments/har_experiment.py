import copy
import torch
from sklearn.model_selection import train_test_split

from src.loaders.load_har import load_har
from src.training.train import train
from src.training.evaluate import evaluate, mse_score, predict_proba
from src.models.dendritic_network import DendriticNetwork
from src.models.mlp_baseline import MLPBaseline
from src.compression.compression_pipeline import (
    compress_model,
    decompress_model,
    compressed_size_bytes,
    compress_model_global,
    compress_model_dynamic,
    dynamic_model_size_bytes,
    compress_model_int4,
    decompress_model_int4,
    compressed_size_bytes_int4,
)


def run_har(epochs=50, seeds=(42,), fine_tune_epochs=3):
    acc_u_list, acc_c_list, acc_mlp_list, acc_mlp_c_list = [], [], [], []
    acc_global_list, acc_dynamic_list, acc_int4_list = [], [], []
    mse_u_list, mse_c_list, mse_mlp_list, mse_mlp_c_list = [], [], [], []
    size_u, size_c = None, None
    size_global, size_dynamic, size_int4 = None, None, None
    size_mlp_u, size_mlp_c = None, None
    curve_data = None
    loss_history = None

    for seed in seeds:
        X_raw_tr, y_raw_tr, X_raw_test, y_raw_test = load_har(seed=seed)

        X_train_np, X_val_np, y_train_np, y_val_np = train_test_split(
            X_raw_tr, y_raw_tr, test_size=0.1, random_state=seed
        )

        X_train = torch.tensor(X_train_np, dtype=torch.float32)
        y_train = torch.tensor(y_train_np, dtype=torch.float32)
        X_val   = torch.tensor(X_val_np,   dtype=torch.float32)
        y_val   = torch.tensor(y_val_np,   dtype=torch.float32)
        X_test  = torch.tensor(X_raw_test, dtype=torch.float32)
        y_test  = torch.tensor(y_raw_test, dtype=torch.float32)

        torch.manual_seed(seed)

        model_u = DendriticNetwork(
            input_dim=X_train.shape[1],
            hidden_neurons1=64,
            hidden_neurons2=32,
            branches=8,
            hidden_per_branch=8,
        )
        hist_u, val_hist_u = train(model_u, X_train, y_train, epochs=epochs,
                                   X_val=X_val, y_val=y_val)
        acc_u_list.append(evaluate(model_u, X_test, y_test))
        mse_u_list.append(mse_score(model_u, X_test, y_test))

        original_state = {k: v.cpu().clone() for k, v in model_u.state_dict().items()}

        # Snowflake: per-layer int8
        compressed = compress_model(model_u, fine_tune_data=(X_train, y_train),
                                    fine_tune_epochs=fine_tune_epochs)
        decompress_model(compressed, model_u)
        acc_c_list.append(evaluate(model_u, X_test, y_test))
        mse_c_list.append(mse_score(model_u, X_test, y_test))

        # Global int8 (single scale for all layers)
        model_u.load_state_dict(original_state)
        compressed_global = compress_model_global(model_u, fine_tune_data=(X_train, y_train),
                                                  fine_tune_epochs=fine_tune_epochs)
        decompress_model(compressed_global, model_u)
        acc_global_list.append(evaluate(model_u, X_test, y_test))

        # PyTorch dynamic quantization (CPU-only, no fine-tuning needed)
        model_u.load_state_dict(original_state)
        model_dynamic = compress_model_dynamic(model_u)
        acc_dynamic_list.append(evaluate(model_dynamic, X_test, y_test, device="cpu"))

        # Snowflake int4: per-layer 4-bit quantization (8× compression)
        model_u.load_state_dict(original_state)
        compressed_int4 = compress_model_int4(model_u, fine_tune_data=(X_train, y_train),
                                              fine_tune_epochs=fine_tune_epochs)
        decompress_model_int4(compressed_int4, model_u)
        acc_int4_list.append(evaluate(model_u, X_test, y_test))

        if size_u is None:
            size_u = model_u.size_bytes()
            size_c = compressed_size_bytes(compressed)
            size_global = compressed_size_bytes(compressed_global)
            size_dynamic = dynamic_model_size_bytes(model_dynamic)
            size_int4 = compressed_size_bytes_int4(compressed_int4)

        if curve_data is None:
            model_u.load_state_dict(original_state)
            score_u = predict_proba(model_u, X_test)
            model_u.load_state_dict(original_state)
            decompress_model(compressed, model_u)
            score_c = predict_proba(model_u, X_test)
            curve_data = {
                "y_true":               y_test.cpu().numpy().ravel(),
                "y_score_uncompressed": score_u,
                "y_score_compressed":   score_c,
            }

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
            "uncompressed":       _mean(acc_u_list),
            "compressed":         _mean(acc_c_list),
            "compressed_global":  _mean(acc_global_list),
            "compressed_dynamic": _mean(acc_dynamic_list),
            "compressed_int4":    _mean(acc_int4_list),
            "mlp_baseline":       _mean(acc_mlp_list),
            "mlp_compressed":     _mean(acc_mlp_c_list),
        },
        "accuracy_std": {
            "uncompressed":       _std(acc_u_list),
            "compressed":         _std(acc_c_list),
            "compressed_global":  _std(acc_global_list),
            "compressed_dynamic": _std(acc_dynamic_list),
            "compressed_int4":    _std(acc_int4_list),
            "mlp_baseline":       _std(acc_mlp_list),
            "mlp_compressed":     _std(acc_mlp_c_list),
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
            "uncompressed":       size_u,
            "compressed":         size_c,
            "compressed_global":  size_global,
            "compressed_dynamic": size_dynamic,
            "compressed_int4":    size_int4,
            "mlp_uncompressed":   size_mlp_u,
            "mlp_compressed":     size_mlp_c,
        },
        "num_seeds": n,
        "curve_data": curve_data,
        "loss_history": loss_history,
    }
