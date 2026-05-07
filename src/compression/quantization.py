import torch


def quantize_tensor_int8(tensor: torch.Tensor):
    """
    Symmetric int8 quantization.
    Returns:
        q     -> int8 quantized tensor
        scale -> float32 scale factor
    """

    # Avoid division by zero
    max_val = tensor.abs().max()
    if max_val == 0:
        # All zeros → trivial quantization
        q = torch.zeros_like(tensor, dtype=torch.int8)
        scale = torch.tensor(1.0, dtype=torch.float32)
        return q, scale

    # Compute scale (symmetric)
    scale = max_val / 127.0

    # Quantize
    q = torch.round(tensor / scale).clamp(-127, 127).to(torch.int8)

    return q, scale
