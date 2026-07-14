import copy
import os
import time
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from src.training.train import train
from src.training.evaluate import evaluate, f1_eval, confusion_matrix_eval, predict_proba_multiclass
from src.models.dendritic_network import DendriticNetwork
from src.models.mlp_baseline import MLPBaseline
from src.compression.compression_pipeline import (
    compress_model,
    decompress_model,
    compressed_size_bytes,
    compress_model_global,
    compress_model_dynamic,
    dynamic_model_size_bytes,
    compress_model_static,
    static_model_size_bytes,
    compress_model_per_channel,
    decompress_model_per_channel,
    per_channel_size_bytes,
    compress_model_qat,
    compress_model_mixed,
    mixed_model_size_bytes,
)
from src.analysis.branch_diversity import compute_branch_diversity
from src.analysis.tost import ci_95, tost_paired


def run_experiment(get_data, num_classes, class_names, epochs, seeds, fine_tune_epochs, batch_size=128, model_dir=None, weight_decay=0.0):
    """
    Shared experiment loop for all tabular datasets.

    get_data: callable(seed) -> (X_raw_tr, y_raw_tr, X_raw_test, y_raw_test) as NumPy arrays.
      - For fixed loaders (HAR/ECG/EEG/HAPT): pass lambda seed: load_dataset()
    """
    acc_u_list, acc_c_list, acc_mlp_list, acc_mlp_c_list = [], [], [], []
    acc_global_list, acc_dynamic_list, acc_static_list = [], [], []
    acc_perchan_list, acc_qat_list, acc_mixed_list = [], [], []
    best_acc_u, best_state_u = -1, None
    best_acc_c, best_compressed_c = -1, None
    best_acc_mlp, best_state_mlp = -1, None
    best_acc_qat, best_model_qat = -1, None
    f1_u_list, f1_c_list, f1_global_list, f1_dynamic_list, f1_static_list, f1_mlp_list, f1_mlp_c_list = [], [], [], [], [], [], []
    f1_perchan_list, f1_qat_list, f1_mixed_list = [], [], []
    conf_matrix_u, conf_matrix_c = None, None
    size_u, size_c = None, None
    size_global, size_dynamic, size_static = None, None, None
    size_perchan, size_qat, size_mixed = None, None, None
    size_mlp_u, size_mlp_c = None, None
    loss_history = None
    weight_dist = None
    val_acc_history = None
    inference_times = None
    curve_data = None
    n_params = None
    branch_diversity = None

    for seed in seeds:
        X_raw_tr, y_raw_tr, X_raw_test, y_raw_test = get_data(seed)

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
            num_classes=num_classes,
        )
        if n_params is None:
            n_params = sum(p.numel() for p in model_u.parameters())
        hist_u, val_hist_u = train(
            model_u, X_train, y_train, epochs=epochs,
            X_val=X_val, y_val=y_val, num_classes=num_classes, batch_size=batch_size,
            weight_decay=weight_decay,
        )
        acc_u_list.append(evaluate(model_u, X_test, y_test, num_classes=num_classes))
        f1_u_list.append(f1_eval(model_u, X_test, y_test, num_classes=num_classes))
        conf_matrix_u = confusion_matrix_eval(model_u, X_test, y_test, num_classes=num_classes)

        if curve_data is None:
            _y_score_u = predict_proba_multiclass(model_u, X_test)

        original_state = {k: v.cpu().clone() for k, v in model_u.state_dict().items()}
        if model_dir and acc_u_list[-1] > best_acc_u:
            best_acc_u = acc_u_list[-1]
            best_state_u = original_state

        if weight_dist is None:
            weights_before = np.concatenate([p.data.cpu().numpy().ravel() for p in model_u.parameters()])

        # Snowflake: per-layer int8
        compressed = compress_model(model_u, fine_tune_data=(X_train, y_train),
                                    fine_tune_epochs=fine_tune_epochs)
        decompress_model(compressed, model_u)

        if weight_dist is None:
            weights_after = np.concatenate([p.data.cpu().numpy().ravel() for p in model_u.parameters()])
            weight_dist = {"before": weights_before, "after": weights_after}

        if branch_diversity is None:
            model_float = copy.deepcopy(model_u)
            model_float.load_state_dict(original_state)
            branch_diversity = compute_branch_diversity(model_float, model_u, X_train)
        acc_c_list.append(evaluate(model_u, X_test, y_test, num_classes=num_classes))
        if model_dir and acc_c_list[-1] > best_acc_c:
            best_acc_c = acc_c_list[-1]
            best_compressed_c = compressed
        f1_c_list.append(f1_eval(model_u, X_test, y_test, num_classes=num_classes))
        conf_matrix_c = confusion_matrix_eval(model_u, X_test, y_test, num_classes=num_classes)

        if curve_data is None:
            curve_data = {
                "y_true":               y_raw_test,
                "y_score_uncompressed": _y_score_u,
                "y_score_compressed":   predict_proba_multiclass(model_u, X_test),
                "num_classes":          num_classes,
            }

        # Global int8
        model_u.load_state_dict(original_state)
        compressed_global = compress_model_global(model_u, fine_tune_data=(X_train, y_train),
                                                  fine_tune_epochs=fine_tune_epochs)
        decompress_model(compressed_global, model_u)
        acc_global_list.append(evaluate(model_u, X_test, y_test, num_classes=num_classes))
        f1_global_list.append(f1_eval(model_u, X_test, y_test, num_classes=num_classes))

        # PyTorch dynamic quantization
        model_u.load_state_dict(original_state)
        model_dynamic = compress_model_dynamic(model_u)
        acc_dynamic_list.append(evaluate(model_dynamic, X_test, y_test, num_classes=num_classes, device="cpu"))
        f1_dynamic_list.append(f1_eval(model_dynamic, X_test, y_test, num_classes=num_classes, device="cpu"))

        # Static quantization (true INT8: both weights and activations)
        model_u.load_state_dict(original_state)
        try:
            model_static = compress_model_static(model_u, calibration_data=(X_train, y_train))
            acc_static_list.append(evaluate(model_static, X_test, y_test, num_classes=num_classes, device="cpu"))
            f1_static_list.append(f1_eval(model_static, X_test, y_test, num_classes=num_classes, device="cpu"))
        except Exception as e:
            print(f"[warn] Static quantization failed: {e}")
            acc_static_list.append(None)
            f1_static_list.append(None)
            model_static = None

        # Per-channel int8 (one scale per output neuron)
        model_u.load_state_dict(original_state)
        try:
            compressed_perchan = compress_model_per_channel(model_u)
            decompress_model_per_channel(compressed_perchan, model_u)
            acc_perchan_list.append(evaluate(model_u, X_test, y_test, num_classes=num_classes))
            f1_perchan_list.append(f1_eval(model_u, X_test, y_test, num_classes=num_classes))
        except Exception as e:
            print(f"[warn] Per-channel quantization failed: {e}")
            acc_perchan_list.append(None)
            f1_perchan_list.append(None)
            compressed_perchan = None

        # QAT (quantization-aware training, fine-tuned with fake-quant nodes)
        model_u.load_state_dict(original_state)
        try:
            model_qat = compress_model_qat(model_u, train_data=(X_train, y_train),
                                           epochs=fine_tune_epochs, num_classes=num_classes)
            acc_qat = evaluate(model_qat, X_test, y_test, num_classes=num_classes, device="cpu")
            acc_qat_list.append(acc_qat)
            f1_qat_list.append(f1_eval(model_qat, X_test, y_test, num_classes=num_classes, device="cpu"))
            if model_dir and acc_qat > best_acc_qat:
                best_acc_qat = acc_qat
                best_model_qat = model_qat
        except Exception as e:
            print(f"[warn] QAT failed: {e}")
            acc_qat_list.append(None)
            f1_qat_list.append(None)
            model_qat = None

        # Mixed precision (fc1 and out in float32, inner layers int8)
        model_u.load_state_dict(original_state)
        try:
            model_mixed = compress_model_mixed(model_u, calibration_data=(X_train, y_train))
            acc_mixed_list.append(evaluate(model_mixed, X_test, y_test, num_classes=num_classes, device="cpu"))
            f1_mixed_list.append(f1_eval(model_mixed, X_test, y_test, num_classes=num_classes, device="cpu"))
        except Exception as e:
            print(f"[warn] Mixed precision failed: {e}")
            acc_mixed_list.append(None)
            f1_mixed_list.append(None)
            model_mixed = None

        if size_u is None:
            size_u = model_u.size_bytes()
            size_c = compressed_size_bytes(compressed)
            size_global = compressed_size_bytes(compressed_global)
            size_dynamic = dynamic_model_size_bytes(model_dynamic)
            size_static = static_model_size_bytes(model_static) if model_static is not None else None
            size_perchan = per_channel_size_bytes(compressed_perchan) if compressed_perchan is not None else None
            size_qat = static_model_size_bytes(model_qat) if model_qat is not None else None
            size_mixed = mixed_model_size_bytes(model_mixed) if model_mixed is not None else None

        if inference_times is None:
            model_u.load_state_dict(original_state)
            X_cpu = X_test.cpu()
            n_test = len(X_cpu)
            def _time_ms(m, n_runs=30):
                m_cpu = m.cpu().eval()
                with torch.no_grad():
                    _ = m_cpu(X_cpu[:1])
                    t0 = time.perf_counter()
                    for _ in range(n_runs):
                        _ = m_cpu(X_cpu)
                return (time.perf_counter() - t0) / n_runs * 1000
            try:
                import torchinfo as _torchinfo
                _tinfo = _torchinfo.summary(model_u, input_size=(1, X_test.shape[1]), verbose=0)
                flops_per_sample = _tinfo.total_mult_adds
                activation_kb    = _tinfo.total_output_bytes / 1024
            except Exception:
                flops_per_sample = None
                activation_kb    = None
            inference_times = {
                "uncompressed_ms":  _time_ms(model_u),
                "compressed_ms":    _time_ms(copy.deepcopy(model_u)),
                "dynamic_ms":       _time_ms(model_dynamic),
                "static_ms":        _time_ms(model_static) if model_static is not None else None,
                "n_test":           n_test,
                "flops_per_sample": flops_per_sample,
                "activation_kb":    activation_kb,
            }

        mlp = MLPBaseline(
            input_dim=X_train.shape[1],
            match_params=sum(p.numel() for p in model_u.parameters()),
            num_classes=num_classes,
        )
        hist_mlp, val_hist_mlp = train(
            mlp, X_train, y_train, epochs=epochs,
            X_val=X_val, y_val=y_val, num_classes=num_classes, batch_size=batch_size,
        )
        acc_mlp_list.append(evaluate(mlp, X_test, y_test, num_classes=num_classes))
        if model_dir and acc_mlp_list[-1] > best_acc_mlp:
            best_acc_mlp = acc_mlp_list[-1]
            best_state_mlp = {k: v.cpu().clone() for k, v in mlp.state_dict().items()}
        f1_mlp_list.append(f1_eval(mlp, X_test, y_test, num_classes=num_classes))

        compressed_mlp = compress_model(mlp, fine_tune_data=(X_train, y_train),
                                        fine_tune_epochs=fine_tune_epochs)
        decompress_model(compressed_mlp, mlp)
        acc_mlp_c_list.append(evaluate(mlp, X_test, y_test, num_classes=num_classes))
        f1_mlp_c_list.append(f1_eval(mlp, X_test, y_test, num_classes=num_classes))

        if size_mlp_u is None:
            size_mlp_u = mlp.size_bytes()
            size_mlp_c = compressed_size_bytes(compressed_mlp)

        if "mlp_ms" not in inference_times:
            inference_times["mlp_ms"] = _time_ms(mlp)

        if loss_history is None:
            loss_history = {
                "Dendritic (Uncompressed)": hist_u,
                "Dendritic (Val)":          val_hist_u["loss"],
                "MLP Baseline":             hist_mlp,
                "MLP (Val)":                val_hist_mlp["loss"],
            }
            val_acc_history = {
                "Dendritic": val_hist_u["acc"],
                "MLP":       val_hist_mlp["acc"],
            }

    if model_dir:
        os.makedirs(model_dir, exist_ok=True)
        if best_state_u is not None:
            torch.save(best_state_u, os.path.join(model_dir, "dendritic_uncompressed.pt"))
        if best_compressed_c is not None:
            torch.save(best_compressed_c, os.path.join(model_dir, "dendritic_snowflake.pt"))
        if best_state_mlp is not None:
            torch.save(best_state_mlp, os.path.join(model_dir, "mlp.pt"))
        if best_model_qat is not None:
            torch.save(best_model_qat, os.path.join(model_dir, "dendritic_qat.pt"))

    _n_test = inference_times.get("n_test", 1) if inference_times else 1

    def _lat_us(key):
        ms = inference_times.get(key) if inference_times else None
        return round(ms * 1000 / _n_test, 4) if ms and _n_test else None

    def _tput(key):
        ms = inference_times.get(key) if inference_times else None
        return round(_n_test / (ms / 1000), 1) if ms and _n_test else None

    edge_profile = {
        "params":             n_params,
        "flops_per_sample":   inference_times.get("flops_per_sample") if inference_times else None,
        "model_size_kb":      round(size_u / 1024, 2) if size_u else None,
        "compressed_size_kb": round(size_c / 1024, 2) if size_c else None,
        "compression_ratio":  round(size_u / size_c, 3) if size_u and size_c else None,
        "activation_mem_kb":  round((inference_times.get("activation_kb") or 0), 2)
                              if inference_times and inference_times.get("activation_kb") else None,
        "latency_us": {
            "uncompressed": _lat_us("uncompressed_ms"),
            "compressed":   _lat_us("compressed_ms"),
            "dynamic":      _lat_us("dynamic_ms"),
            "static":       _lat_us("static_ms"),
            "mlp":          _lat_us("mlp_ms"),
        },
        "throughput_sps": {
            "uncompressed": _tput("uncompressed_ms"),
            "compressed":   _tput("compressed_ms"),
            "dynamic":      _tput("dynamic_ms"),
            "static":       _tput("static_ms"),
            "mlp":          _tput("mlp_ms"),
        },
    }

    n     = len(seeds)
    _mean = lambda lst: float(sum(lst) / n)
    _std  = lambda lst: float(torch.tensor(lst).std().item()) if n > 1 else 0.0
    # static quant may fail on some seeds; filter None before aggregating
    _mean_safe = lambda lst: float(sum(x for x in lst if x is not None) / max(1, sum(1 for x in lst if x is not None))) if any(x is not None for x in lst) else None
    _std_safe  = lambda lst: float(torch.tensor([x for x in lst if x is not None]).std().item()) if sum(1 for x in lst if x is not None) > 1 else 0.0
    _ci95      = lambda lst: ci_95(lst)
    _ci95_safe = lambda lst: ci_95([x for x in lst if x is not None])

    return {
        "accuracy": {
            "uncompressed":        _mean(acc_u_list),
            "compressed":          _mean(acc_c_list),
            "compressed_global":   _mean(acc_global_list),
            "compressed_dynamic":  _mean(acc_dynamic_list),
            "compressed_static":   _mean_safe(acc_static_list),
            "compressed_perchan":  _mean_safe(acc_perchan_list),
            "compressed_qat":      _mean_safe(acc_qat_list),
            "compressed_mixed":    _mean_safe(acc_mixed_list),
            "mlp_baseline":        _mean(acc_mlp_list),
            "mlp_compressed":      _mean(acc_mlp_c_list),
        },
        "accuracy_std": {
            "uncompressed":        _std(acc_u_list),
            "compressed":          _std(acc_c_list),
            "compressed_global":   _std(acc_global_list),
            "compressed_dynamic":  _std(acc_dynamic_list),
            "compressed_static":   _std_safe(acc_static_list),
            "compressed_perchan":  _std_safe(acc_perchan_list),
            "compressed_qat":      _std_safe(acc_qat_list),
            "compressed_mixed":    _std_safe(acc_mixed_list),
            "mlp_baseline":        _std(acc_mlp_list),
            "mlp_compressed":      _std(acc_mlp_c_list),
        },
        "f1": {
            "uncompressed":        _mean(f1_u_list),
            "compressed":          _mean(f1_c_list),
            "compressed_global":   _mean(f1_global_list),
            "compressed_dynamic":  _mean(f1_dynamic_list),
            "compressed_static":   _mean_safe(f1_static_list),
            "compressed_perchan":  _mean_safe(f1_perchan_list),
            "compressed_qat":      _mean_safe(f1_qat_list),
            "compressed_mixed":    _mean_safe(f1_mixed_list),
            "mlp_baseline":        _mean(f1_mlp_list),
            "mlp_compressed":      _mean(f1_mlp_c_list),
        },
        "f1_std": {
            "uncompressed":        _std(f1_u_list),
            "compressed":          _std(f1_c_list),
            "compressed_global":   _std(f1_global_list),
            "compressed_dynamic":  _std(f1_dynamic_list),
            "compressed_static":   _std_safe(f1_static_list),
            "compressed_perchan":  _std_safe(f1_perchan_list),
            "compressed_qat":      _std_safe(f1_qat_list),
            "compressed_mixed":    _std_safe(f1_mixed_list),
            "mlp_baseline":        _std(f1_mlp_list),
            "mlp_compressed":      _std(f1_mlp_c_list),
        },
        "sizes": {
            "uncompressed":       size_u,
            "compressed":         size_c,
            "compressed_global":  size_global,
            "compressed_dynamic": size_dynamic,
            "compressed_static":  size_static,
            "compressed_perchan": size_perchan,
            "compressed_qat":     size_qat,
            "compressed_mixed":   size_mixed,
            "mlp_uncompressed":   size_mlp_u,
            "mlp_compressed":     size_mlp_c,
        },
        "num_seeds":        n,
        "loss_history":     loss_history,
        "val_acc_history":  val_acc_history,
        "conf_matrix":      {"uncompressed": conf_matrix_u, "compressed": conf_matrix_c},
        "class_names":      class_names,
        "weight_dist":      weight_dist,
        "branch_diversity": branch_diversity,
        "curve_data":       curve_data,
        "per_seed": {
            "acc_uncompressed":       acc_u_list,
            "acc_compressed":         acc_c_list,
            "acc_compressed_global":  acc_global_list,
            "acc_compressed_dynamic": acc_dynamic_list,
            "acc_compressed_static":  acc_static_list,
            "acc_compressed_perchan": acc_perchan_list,
            "acc_compressed_qat":     acc_qat_list,
            "acc_compressed_mixed":   acc_mixed_list,
            "f1_uncompressed":        f1_u_list,
            "f1_compressed":          f1_c_list,
            "f1_compressed_global":   f1_global_list,
            "f1_compressed_dynamic":  f1_dynamic_list,
            "f1_compressed_static":   f1_static_list,
            "f1_compressed_perchan":  f1_perchan_list,
            "f1_compressed_qat":      f1_qat_list,
            "f1_compressed_mixed":    f1_mixed_list,
        },
        "inference_time_uncompressed_ms": inference_times["uncompressed_ms"] if inference_times else None,
        "inference_time_compressed_ms":   inference_times["compressed_ms"]   if inference_times else None,
        "inference_time_dynamic_ms":      inference_times["dynamic_ms"]      if inference_times else None,
        "inference_time_static_ms":       inference_times.get("static_ms")   if inference_times else None,
        "inference_time_mlp_ms":          inference_times["mlp_ms"]          if inference_times else None,
        "edge_profile":                   edge_profile,
        "ci_95": {
            "uncompressed":       _ci95(acc_u_list),
            "compressed":         _ci95(acc_c_list),
            "compressed_global":  _ci95(acc_global_list),
            "compressed_dynamic": _ci95(acc_dynamic_list),
            "compressed_static":  _ci95_safe(acc_static_list),
            "compressed_perchan": _ci95_safe(acc_perchan_list),
            "compressed_qat":     _ci95_safe(acc_qat_list),
            "compressed_mixed":   _ci95_safe(acc_mixed_list),
            "mlp_baseline":       _ci95(acc_mlp_list),
            "mlp_compressed":     _ci95(acc_mlp_c_list),
        },
        "tost": {
            "compressed":         tost_paired(acc_u_list, acc_c_list),
            "compressed_global":  tost_paired(acc_u_list, acc_global_list),
            "compressed_dynamic": tost_paired(acc_u_list, acc_dynamic_list),
            "compressed_static":  tost_paired(acc_u_list, acc_static_list),
            "compressed_perchan": tost_paired(acc_u_list, acc_perchan_list),
            "compressed_qat":     tost_paired(acc_u_list, acc_qat_list),
            "compressed_mixed":   tost_paired(acc_u_list, acc_mixed_list),
        },
    }
