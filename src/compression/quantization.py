import torch

def quantize_model(model):
    """
    Simple 8-bit linear quantization for all parameters.
    """
    for p in model.parameters():
        min_val = p.data.min()
        max_val = p.data.max()
        scale = (max_val - min_val) / 255

        p.data = torch.round((p.data - min_val) / scale) * scale + min_val
