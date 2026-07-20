import copy
import torch
from src.training.train import train
from src.training.evaluate import evaluate, mse_score
from src.models.dendritic_network import DendriticNetwork
from src.compression.compression_pipeline import (
    compress_model,
    decompress_model,
    compressed_size_bytes,
)
from src.compression.topology_sharing import apply_topology_sharing


def run_ablation(configs, X_train, y_train, X_test, y_test, epochs=50, num_classes=1):
    """
    configs: list of dicts with keys h1, h2, branches, hidden_per_branch.
    Trains one model per config and reports accuracy before/after compression.
    """
    results = []

    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long if num_classes > 1 else torch.float32)
    if num_classes == 1:
        y_train = y_train.reshape(-1, 1)
    X_test  = torch.tensor(X_test,  dtype=torch.float32)
    y_test  = torch.tensor(y_test,  dtype=torch.long if num_classes > 1 else torch.float32)
    if num_classes == 1:
        y_test = y_test.reshape(-1, 1)

    for cfg in configs:
        model_u = DendriticNetwork(
            input_dim=X_train.shape[1],
            hidden_neurons1=cfg["h1"],
            hidden_neurons2=cfg["h2"],
            branches=cfg["branches"],
            hidden_per_branch=cfg["hidden_per_branch"],
            num_classes=num_classes,
        )

        train(model_u, X_train, y_train, epochs=epochs, num_classes=num_classes,
              verbose=True, label=f"h1={cfg['h1']} h2={cfg['h2']} br={cfg['branches']}")
        acc_u = evaluate(model_u, X_test, y_test, num_classes=num_classes)
        mse_u = mse_score(model_u, X_test, y_test) if num_classes == 1 else float("nan")

        compressed = compress_model(model_u)
        size_c = compressed_size_bytes(compressed)

        decompress_model(compressed, model_u)
        acc_c = evaluate(model_u, X_test, y_test, num_classes=num_classes)
        mse_c = mse_score(model_u, X_test, y_test) if num_classes == 1 else float("nan")

        size_u = model_u.size_bytes()

        results.append({
            "config": cfg,
            "accuracy_uncompressed": acc_u,
            "accuracy_compressed":   acc_c,
            "mse_uncompressed":      mse_u,
            "mse_compressed":        mse_c,
            "size_uncompressed": size_u,
            "size_compressed": size_c,
        })

    return results


def run_compression_component_ablation(
    X_train, y_train, X_test, y_test, config, epochs=50, seeds=(42,), num_classes=1
):
    """
    Isolates the contribution of each compression component on a single config.

    Conditions tested per seed:
      none       — uncompressed baseline
      topo_only  — topology sharing applied, weights kept as float32
      quant_only — int8 quantization with no topology sharing
      both       — topology sharing + quantization (standard pipeline)

    Returns dict: condition → {"mean": float, "std": float}
    """
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long if num_classes > 1 else torch.float32)
    if num_classes == 1:
        y_train = y_train.reshape(-1, 1)
    X_test  = torch.tensor(X_test,  dtype=torch.float32)
    y_test  = torch.tensor(y_test,  dtype=torch.long if num_classes > 1 else torch.float32)
    if num_classes == 1:
        y_test = y_test.reshape(-1, 1)

    acc_by_condition = {"none": [], "topo_only": [], "quant_only": [], "both": []}
    mse_by_condition = {"none": [], "topo_only": [], "quant_only": [], "both": []}

    for seed in seeds:
        torch.manual_seed(seed)

        base_model = DendriticNetwork(
            input_dim=X_train.shape[1],
            hidden_neurons1=config["h1"],
            hidden_neurons2=config["h2"],
            branches=config["branches"],
            hidden_per_branch=config["hidden_per_branch"],
            num_classes=num_classes,
        )
        train(base_model, X_train, y_train, epochs=epochs, num_classes=num_classes,
              verbose=True, label=f"component seed={seed}")

        # none: evaluate as-is
        _mse = (lambda m: mse_score(m, X_test, y_test)) if num_classes == 1 else (lambda m: float("nan"))

        acc_by_condition["none"].append(evaluate(base_model, X_test, y_test, num_classes=num_classes))
        mse_by_condition["none"].append(_mse(base_model))

        # topo_only: share branch weights, stay in float32
        m_topo = copy.deepcopy(base_model)
        apply_topology_sharing(m_topo)
        acc_by_condition["topo_only"].append(evaluate(m_topo, X_test, y_test, num_classes=num_classes))
        mse_by_condition["topo_only"].append(_mse(m_topo))

        # quant_only: quantize without topology sharing (now the standard pipeline)
        m_quant = copy.deepcopy(base_model)
        compressed_q = compress_model(m_quant)
        decompress_model(compressed_q, m_quant)
        acc_by_condition["quant_only"].append(evaluate(m_quant, X_test, y_test, num_classes=num_classes))
        mse_by_condition["quant_only"].append(_mse(m_quant))

        # both: topology sharing + quantization
        m_both = copy.deepcopy(base_model)
        apply_topology_sharing(m_both)
        compressed_both = compress_model(m_both)
        decompress_model(compressed_both, m_both)
        acc_by_condition["both"].append(evaluate(m_both, X_test, y_test, num_classes=num_classes))
        mse_by_condition["both"].append(_mse(m_both))

    def _stats(lst):
        n = len(lst)
        mean = float(sum(lst) / n)
        std  = float(torch.tensor(lst).std().item()) if n > 1 else 0.0
        return {"mean": mean, "std": std}

    return {
        condition: {**_stats(acc_by_condition[condition]), **{
            "mse_mean": float(sum(mse_by_condition[condition]) / len(mse_by_condition[condition])),
            "mse_std":  float(torch.tensor(mse_by_condition[condition]).std().item())
                        if len(mse_by_condition[condition]) > 1 else 0.0,
        }}
        for condition in acc_by_condition
    }


def run_regularization_ablation(
    X_train, y_train, X_test, y_test, config, epochs=50, seeds=(42,),
    num_classes=1, weight_decay=1e-3,
):
    """
    Isolates whether quantization's accuracy gain is explainable by
    regularization alone, rather than something quantization-specific.

    Conditions tested per seed (single variable changed: none/quant/reg):
      none       — float32 baseline, no weight decay
      quant_only — same trained model, then int8 quantized (Snowflake)
      reg_only   — retrained from the same seed with weight_decay, kept float32

    If reg_only's accuracy gain over none matches quant_only's gain,
    that supports the "quantization improves accuracy via regularization"
    claim; if not, the effect isn't just regularization.
    """
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long if num_classes > 1 else torch.float32)
    if num_classes == 1:
        y_train = y_train.reshape(-1, 1)
    X_test  = torch.tensor(X_test,  dtype=torch.float32)
    y_test  = torch.tensor(y_test,  dtype=torch.long if num_classes > 1 else torch.float32)
    if num_classes == 1:
        y_test = y_test.reshape(-1, 1)

    acc_by_condition = {"none": [], "quant_only": [], "reg_only": []}

    for seed in seeds:
        torch.manual_seed(seed)
        base_model = DendriticNetwork(
            input_dim=X_train.shape[1],
            hidden_neurons1=config["h1"],
            hidden_neurons2=config["h2"],
            branches=config["branches"],
            hidden_per_branch=config["hidden_per_branch"],
            num_classes=num_classes,
        )
        train(base_model, X_train, y_train, epochs=epochs, num_classes=num_classes,
              verbose=True, label=f"reg none seed={seed}")
        acc_by_condition["none"].append(evaluate(base_model, X_test, y_test, num_classes=num_classes))

        m_quant = copy.deepcopy(base_model)
        compressed_q = compress_model(m_quant)
        decompress_model(compressed_q, m_quant)
        acc_by_condition["quant_only"].append(evaluate(m_quant, X_test, y_test, num_classes=num_classes))

        # Same seed -> identical init + data shuffle order as base_model;
        # weight_decay is the only variable that differs.
        torch.manual_seed(seed)
        m_reg = DendriticNetwork(
            input_dim=X_train.shape[1],
            hidden_neurons1=config["h1"],
            hidden_neurons2=config["h2"],
            branches=config["branches"],
            hidden_per_branch=config["hidden_per_branch"],
            num_classes=num_classes,
        )
        train(m_reg, X_train, y_train, epochs=epochs, num_classes=num_classes, weight_decay=weight_decay,
              verbose=True, label=f"reg_only seed={seed}")
        acc_by_condition["reg_only"].append(evaluate(m_reg, X_test, y_test, num_classes=num_classes))

    def _stats(lst):
        n = len(lst)
        mean = float(sum(lst) / n)
        std  = float(torch.tensor(lst).std().item()) if n > 1 else 0.0
        return {"mean": mean, "std": std}

    return {condition: _stats(acc_by_condition[condition]) for condition in acc_by_condition}
