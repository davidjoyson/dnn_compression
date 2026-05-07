import torch
from src.models.dendritic_network import DendriticNetwork
from src.training.train import train
from src.training.evaluate import evaluate
from src.compression.compression_pipeline import compress_model

def run_xor():
    X = torch.tensor([[0,0],[0,1],[1,0],[1,1]], dtype=torch.float32)
    y = torch.tensor([[0],[1],[1],[0]], dtype=torch.float32)

    model = DendriticNetwork(input_dim=2)
    train(model, X, y)

    acc_uncompressed = evaluate(model, X, y)

    compress_model(model)
    acc_compressed = evaluate(model, X, y)

    return acc_uncompressed, acc_compressed
