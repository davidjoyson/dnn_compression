import torch
from src.models.dendritic_network import DendriticNetwork
from src.training.train import train
from src.training.evaluate import evaluate
from src.compression.compression_pipeline import compress_model

def run_learning_mode(X_train, y_train, X_test, y_test):
    model = DendriticNetwork(input_dim=X_train.shape[1], hidden_neurons=6, branches=3)
    train(model, X_train, y_train)

    acc_uncompressed = evaluate(model, X_test, y_test)

    compress_model(model)
    acc_compressed = evaluate(model, X_test, y_test)

    return acc_uncompressed, acc_compressed
