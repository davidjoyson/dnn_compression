import copy
import io

import torch
from collections import defaultdict
from src.training.train import train


def _compute_layer_scales(model):
    """One shared scale per layer group (weight + bias share a single scale)."""
    groups = defaultdict(list)
    for name, p in model.named_parameters():
        layer = name.rsplit(".", 1)[0] if "." in name else name
        groups[layer].append(p.data)

    scales = {}
    for layer, tensors in groups.items():
        max_val = max(t.abs().max().item() for t in tensors)
        scales[layer] = torch.tensor(
            max_val / 127.0 if max_val > 0 else 1.0, dtype=torch.float32
        )
    return scales


def _quantize(model):
    """Quantize all parameters using per-layer shared scales."""
    layer_scales = _compute_layer_scales(model)
    compressed = {}
    with torch.no_grad():
        for name, p in model.named_parameters():
            layer = name.rsplit(".", 1)[0] if "." in name else name
            scale = layer_scales[layer]
            q = torch.round(p.data / scale).clamp(-127, 127).to(torch.int8)
            compressed[name] = {"q": q.cpu(), "scale": scale.cpu()}
    return compressed


def compress_model(model, fine_tune_data=None, fine_tune_epochs=3, fine_tune_lr=1e-4):
    """
    Compress model via per-layer int8 quantization.

    fine_tune_data: optional (X, y) tensors — if provided, decompresses, fine-tunes
                    for fine_tune_epochs, then re-quantizes before returning.
    """
    compressed = _quantize(model)

    if fine_tune_data is not None:
        X, y = fine_tune_data
        num_classes = getattr(model, "num_classes", 1)
        decompress_model(compressed, model)
        train(model, X, y, epochs=fine_tune_epochs, lr=fine_tune_lr, use_tqdm=False,
              num_classes=num_classes)
        compressed = _quantize(model)

    return compressed


def decompress_model(compressed, model):
    """Load compressed weights back into model for evaluation."""
    with torch.no_grad():
        for name, p in model.named_parameters():
            entry = compressed[name]
            p.data = (entry["q"].float() * entry["scale"]).to(dtype=p.dtype, device=p.device)
    return model


def compressed_size_bytes(compressed):
    """
    Compute compressed size in bytes.
    - int8 weights: 1 byte each
    - scale: one float32 (4 bytes) per unique layer group
    """
    total = 0
    seen_layers = set()
    for name, entry in compressed.items():
        total += entry["q"].nelement()
        layer = name.rsplit(".", 1)[0] if "." in name else name
        if layer not in seen_layers:
            total += 4
            seen_layers.add(layer)
    return total


# ------------------------------------------------------------------ #
# Global int8 quantization (single scale for all parameters)         #
# ------------------------------------------------------------------ #

def _quantize_global(model):
    """Quantize all parameters using one global scale."""
    all_params = [p.data for _, p in model.named_parameters()]
    global_max = max(t.abs().max().item() for t in all_params)
    scale = torch.tensor(global_max / 127.0 if global_max > 0 else 1.0, dtype=torch.float32)
    compressed = {}
    with torch.no_grad():
        for name, p in model.named_parameters():
            q = torch.round(p.data / scale).clamp(-127, 127).to(torch.int8)
            compressed[name] = {"q": q.cpu(), "scale": scale.cpu()}
    return compressed


def compress_model_global(model, fine_tune_data=None, fine_tune_epochs=3, fine_tune_lr=1e-4):
    """Compress via global int8 quantization (single scale for all layers)."""
    compressed = _quantize_global(model)
    if fine_tune_data is not None:
        X, y = fine_tune_data
        num_classes = getattr(model, "num_classes", 1)
        decompress_model(compressed, model)
        train(model, X, y, epochs=fine_tune_epochs, lr=fine_tune_lr, use_tqdm=False,
              num_classes=num_classes)
        compressed = _quantize_global(model)
    return compressed


# ------------------------------------------------------------------ #
# PyTorch dynamic quantization (int8 weights, runtime activation quant)
# ------------------------------------------------------------------ #

def compress_model_dynamic(model):
    """Apply PyTorch dynamic quantization to Linear layers (CPU-only).

    Returns a new quantized model; the original is not modified.
    No fine-tuning or calibration data required.
    """
    model_cpu = copy.deepcopy(model).cpu().eval()
    quantized = torch.ao.quantization.quantize_dynamic(
        model_cpu, {torch.nn.Linear}, dtype=torch.qint8
    )
    return quantized


def dynamic_model_size_bytes(model):
    """True compressed size: int8 weight bytes + float32 bias bytes per Linear layer.

    torch.save inflates size ~2x due to pickle overhead on PackedParams objects;
    this measures the raw data actually stored.
    """
    total = 0
    for _, mod in model.named_modules():
        if hasattr(mod, "weight") and hasattr(mod, "bias"):
            try:
                total += mod.weight().int_repr().numel()      # 1 byte per int8 weight
                total += mod.bias().numel() * 4               # 4 bytes per float32 bias
            except Exception:
                pass
    return total
