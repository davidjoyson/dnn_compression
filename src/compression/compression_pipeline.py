from .topology_sharing import apply_topology_sharing
from .quantization import quantize_model

def compress_model(model):
    apply_topology_sharing(model)
    quantize_model(model)
    return model
