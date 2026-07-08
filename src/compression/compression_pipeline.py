import copy

import torch
from collections import defaultdict
from src.training.train import train


def _layer_name(param_name):
    return param_name.rsplit(".", 1)[0] if "." in param_name else param_name


def _compute_layer_scales(model):
    """One shared scale per layer group (weight + bias share a single scale)."""
    groups = defaultdict(list)
    for name, p in model.named_parameters():
        groups[_layer_name(name)].append(p.data)

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
            scale = layer_scales[_layer_name(name)]
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
        train(model, X, y, epochs=fine_tune_epochs, lr=fine_tune_lr,
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
        layer = _layer_name(name)
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
        train(model, X, y, epochs=fine_tune_epochs, lr=fine_tune_lr,
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


# ------------------------------------------------------------------ #
# PyTorch static quantization (FX graph mode)                        #
# Both weights AND activations quantized — true INT8 arithmetic      #
# ------------------------------------------------------------------ #

def compress_model_static(model, calibration_data, backend="fbgemm"):
    """
    Static INT8 quantization via FX graph mode.
    Calibrates activation ranges from calibration_data[0] (labels unused).
    backend: "fbgemm" (x86) or "qnnpack" (ARM/edge).
    """
    from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx
    import torch.ao.quantization as tq

    model_copy = copy.deepcopy(model).cpu().eval()
    qconfig_mapping = tq.QConfigMapping().set_global(tq.get_default_qconfig(backend))
    X_cal = calibration_data[0].cpu()
    prepared = prepare_fx(model_copy, qconfig_mapping, (X_cal[:1],))
    with torch.no_grad():
        prepared(X_cal)
    return convert_fx(prepared)


def static_model_size_bytes(model):
    """int8 weights (1B each) + float32 biases (4B each) for FX-quantized Linear layers."""
    total = 0
    for _, mod in model.named_modules():
        if hasattr(mod, "weight") and hasattr(mod, "bias"):
            try:
                total += mod.weight().int_repr().nelement()
                if mod.bias() is not None:
                    total += mod.bias().nelement() * 4
            except (AttributeError, RuntimeError):
                pass
    return total


# ------------------------------------------------------------------ #
# Per-channel int8 quantization                                       #
# One scale per output neuron vs one scale per layer (Snowflake)     #
# ------------------------------------------------------------------ #

def _quantize_per_channel(model):
    compressed = {}
    with torch.no_grad():
        for name, p in model.named_parameters():
            if p.dim() >= 2:  # weight: one scale per output channel (row)
                max_per_ch = p.data.abs().amax(dim=tuple(range(1, p.dim())), keepdim=True)
                scales = (max_per_ch / 127.0).clamp(min=1e-8)
                q = torch.round(p.data / scales).clamp(-127, 127).to(torch.int8)
                compressed[name] = {"q": q.cpu(), "scales": scales.cpu()}
            else:  # bias: keep float32
                compressed[name] = {"float": p.data.cpu()}
    return compressed


def compress_model_per_channel(model):
    return _quantize_per_channel(model)


def decompress_model_per_channel(compressed, model):
    with torch.no_grad():
        for name, p in model.named_parameters():
            entry = compressed[name]
            if "float" in entry:
                p.data = entry["float"].to(dtype=p.dtype, device=p.device)
            else:
                p.data = (entry["q"].float() * entry["scales"]).to(dtype=p.dtype, device=p.device)
    return model


def per_channel_size_bytes(compressed):
    """int8 weights + float32 per-channel scales + float32 biases."""
    total = 0
    for entry in compressed.values():
        if "float" in entry:
            total += entry["float"].nelement() * 4
        else:
            total += entry["q"].nelement()
            total += entry["scales"].nelement() * 4
    return total


# ------------------------------------------------------------------ #
# Quantization-Aware Training (QAT, FX graph mode)                   #
# Fake-quant nodes during training → better calibrated scales        #
# ------------------------------------------------------------------ #

def compress_model_qat(model, train_data, epochs=10, lr=1e-4, num_classes=1, backend="fbgemm"):
    from torch.ao.quantization.quantize_fx import prepare_qat_fx, convert_fx
    import torch.ao.quantization as tq

    model_copy = copy.deepcopy(model).cpu()
    qconfig_mapping = tq.QConfigMapping().set_global(tq.get_default_qat_qconfig(backend))
    X_sample = train_data[0][:1].cpu()
    prepared = prepare_qat_fx(model_copy, qconfig_mapping, (X_sample,))

    X, y = train_data[0].cpu(), train_data[1].cpu()
    train(prepared, X, y, epochs=epochs, lr=lr, num_classes=num_classes)

    prepared = prepared.cpu().eval()
    return convert_fx(prepared)


# ------------------------------------------------------------------ #
# Mixed precision (FX graph mode)                                     #
# fc1 and out stay float32; branches/soma/fc2 quantized to int8      #
# ------------------------------------------------------------------ #

def compress_model_mixed(model, calibration_data, backend="fbgemm"):
    from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx
    import torch.ao.quantization as tq

    model_copy = copy.deepcopy(model).cpu().eval()
    qconfig_mapping = (
        tq.QConfigMapping()
        .set_global(tq.get_default_qconfig(backend))
        .set_module_name("fc1", None)
        .set_module_name("out", None)
    )
    X_cal = calibration_data[0].cpu()
    prepared = prepare_fx(model_copy, qconfig_mapping, (X_cal[:1],))
    with torch.no_grad():
        prepared(X_cal)
    return convert_fx(prepared)


def mixed_model_size_bytes(model):
    """Quantized layers: int8; unquantized layers (fc1, out): float32."""
    total = 0
    for _, mod in model.named_modules():
        if hasattr(mod, "weight") and callable(mod.weight):
            try:
                total += mod.weight().int_repr().nelement()
                if mod.bias() is not None:
                    total += mod.bias().nelement() * 4
            except (AttributeError, RuntimeError):
                pass
        elif isinstance(mod, torch.nn.Linear):
            total += mod.weight.nelement() * 4
            if mod.bias is not None:
                total += mod.bias.nelement() * 4
    return total


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
                if mod.bias() is not None:
                    total += mod.bias().numel() * 4           # 4 bytes per float32 bias
            except (AttributeError, RuntimeError):
                pass
    return total
