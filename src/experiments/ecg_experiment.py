import torch
from sklearn.model_selection import train_test_split

from src.loaders.load_ecg import load_ecg
from src.training.train import train
from src.training.evaluate import evaluate
from src.models.dendritic_network import DendriticNetwork
from src.models.mlp_baseline import MLPBaseline
from src.compression.compression_pipeline import (
    compress_model,
    decompress_model,
    compressed_size_bytes,
    compress_model_global,
    compress_model_dynamic,
    dynamic_model_size_bytes,
)

NUM_CLASSES = 5


def run_ecg(epochs=50, seeds=(42,), fine_tune_epochs=3):
    acc_u_list, acc_c_list, acc_mlp_list, acc_mlp_c_list = [], [], [], []
    acc_global_list, acc_dynamic_list = [], []
    size_u, size_c = None, None
    size_global, size_dynamic = None, None
    size_mlp_u, size_mlp_c = None, None
    loss_history = None

    X_raw_tr, y_raw_tr, X_raw_test, y_raw_test = load_ecg()

    for seed in seeds:
        X_train_np, X_val_np, y_train_np, y_val_np = train_test_split(
            X_raw_tr, y_raw_tr, test_size=0.1, random_state=seed, stratify=y_raw_tr
        )

        X_train = torch.tensor(X_train_np, dtype=torch.float32)
        y_train = torch.tensor(y_train_np, dtype=torch.long)
        X_val   = torch.tensor(X_val_np,   dtype=torch.float32)
        y_val   = torch.tensor(y_val_np,   dtype=torch.long)
        X_test  = torch.tensor(X_raw_test, dtype=torch.float32)
        y_test  = torch.tensor(y_raw_test, dtype=torch.long)

        torch.manual_seed(seed)

        model_u = DendriticNetwork(
            input_dim=X_train.shape[1],
            hidden_neurons1=64,
            hidden_neurons2=32,
            branches=8,
            hidden_per_branch=8,
            num_classes=NUM_CLASSES,
        )
        hist_u, val_hist_u = train(
            model_u, X_train, y_train, epochs=epochs,
            X_val=X_val, y_val=y_val, num_classes=NUM_CLASSES, batch_size=256,
        )
        acc_u_list.append(evaluate(model_u, X_test, y_test, num_classes=NUM_CLASSES))

        # Save trained weights for fair comparison across compression methods
        original_state = {k: v.cpu().clone() for k, v in model_u.state_dict().items()}

        # Snowflake: per-layer int8
        compressed = compress_model(model_u, fine_tune_data=(X_train, y_train),
                                    fine_tune_epochs=fine_tune_epochs)
        decompress_model(compressed, model_u)
        acc_c_list.append(evaluate(model_u, X_test, y_test, num_classes=NUM_CLASSES))

        # Global int8 (single scale for all layers)
        model_u.load_state_dict(original_state)
        compressed_global = compress_model_global(model_u, fine_tune_data=(X_train, y_train),
                                                  fine_tune_epochs=fine_tune_epochs)
        decompress_model(compressed_global, model_u)
        acc_global_list.append(evaluate(model_u, X_test, y_test, num_classes=NUM_CLASSES))

        # PyTorch dynamic quantization (CPU-only, no fine-tuning needed)
        model_u.load_state_dict(original_state)
        model_dynamic = compress_model_dynamic(model_u)
        acc_dynamic_list.append(evaluate(model_dynamic, X_test, y_test, num_classes=NUM_CLASSES, device="cpu"))

        if size_u is None:
            size_u = model_u.size_bytes()
            size_c = compressed_size_bytes(compressed)
            size_global = compressed_size_bytes(compressed_global)
            size_dynamic = dynamic_model_size_bytes(model_dynamic)

        mlp = MLPBaseline(
            input_dim=X_train.shape[1],
            match_params=sum(p.numel() for p in model_u.parameters()),
            num_classes=NUM_CLASSES,
        )
        hist_mlp, val_hist_mlp = train(
            mlp, X_train, y_train, epochs=epochs,
            X_val=X_val, y_val=y_val, num_classes=NUM_CLASSES, batch_size=256,
        )
        acc_mlp_list.append(evaluate(mlp, X_test, y_test, num_classes=NUM_CLASSES))

        compressed_mlp = compress_model(mlp, fine_tune_data=(X_train, y_train),
                                        fine_tune_epochs=fine_tune_epochs)
        decompress_model(compressed_mlp, mlp)
        acc_mlp_c_list.append(evaluate(mlp, X_test, y_test, num_classes=NUM_CLASSES))

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

    n      = len(seeds)
    _mean  = lambda lst: float(sum(lst) / n)
    _std   = lambda lst: float(torch.tensor(lst).std().item()) if n > 1 else 0.0

    return {
        "accuracy": {
            "uncompressed":        _mean(acc_u_list),
            "compressed":          _mean(acc_c_list),
            "compressed_global":   _mean(acc_global_list),
            "compressed_dynamic":  _mean(acc_dynamic_list),
            "mlp_baseline":        _mean(acc_mlp_list),
            "mlp_compressed":      _mean(acc_mlp_c_list),
        },
        "accuracy_std": {
            "uncompressed":        _std(acc_u_list),
            "compressed":          _std(acc_c_list),
            "compressed_global":   _std(acc_global_list),
            "compressed_dynamic":  _std(acc_dynamic_list),
            "mlp_baseline":        _std(acc_mlp_list),
            "mlp_compressed":      _std(acc_mlp_c_list),
        },
        "sizes": {
            "uncompressed":       size_u,
            "compressed":         size_c,
            "compressed_global":  size_global,
            "compressed_dynamic": size_dynamic,
            "mlp_uncompressed":   size_mlp_u,
            "mlp_compressed":     size_mlp_c,
        },
        "num_seeds":    n,
        "loss_history": loss_history,
    }
