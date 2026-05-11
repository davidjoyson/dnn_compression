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
        decompress_model(compressed, model)
        train(model, X, y, epochs=fine_tune_epochs, lr=fine_tune_lr, use_tqdm=False)
        compressed = _quantize(model)

    return compressed


def decompress_model(compressed, model):
    """Load compressed weights back into model for evaluation."""
    with torch.no_grad():
        for name, p in model.named_parameters():
            entry = compressed[name]
            p.data = (entry["q"].float() * entry["scale"]).to(p.dtype)
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
