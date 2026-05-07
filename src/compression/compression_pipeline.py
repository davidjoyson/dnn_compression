import torch
from .topology_sharing import apply_topology_sharing
from .quantization import quantize_tensor_int8


def compress_model(model):
    """
    Returns a TRUE compressed representation of the model.
    Output is a dictionary containing:
        - int8 weights
        - scale factors
    """

    # 1. Apply topology sharing (ties weights)
    apply_topology_sharing(model)

    compressed = {}

    # 2. Quantize each parameter to int8
    with torch.no_grad():
        for name, p in model.named_parameters():

            q, scale = quantize_tensor_int8(p.data)

            compressed[name] = {
                "q": q.cpu(),        # int8 tensor
                "scale": scale.cpu() # float32 scalar
            }

    return compressed


def decompress_model(compressed, model):
    """
    Loads compressed weights back into a PyTorch model.
    Used for evaluation after compression.
    """

    with torch.no_grad():
        for name, p in model.named_parameters():
            entry = compressed[name]
            q = entry["q"].float()
            scale = entry["scale"]
            p.data = (q * scale).to(p.dtype)

    return model


def compressed_size_bytes(compressed):
    """
    Computes the TRUE compressed size in bytes.
    int8 = 1 byte per element
    scale = 4 bytes (float32)
    """

    total = 0
    for entry in compressed.values():
        q = entry["q"]
        total += q.nelement() * 1  # int8
        total += 4                 # scale (float32)
    return total
