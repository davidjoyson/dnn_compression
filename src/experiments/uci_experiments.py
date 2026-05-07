from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import torch
from src.models.dendritic_network import DendriticNetwork
from src.training.train import train
from src.training.evaluate import evaluate
from src.compression.compression_pipeline import compress_model

def run_wine():
    data = load_wine()
    X = StandardScaler().fit_transform(data.data)
    y = (data.target == 0).astype(float).reshape(-1,1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DendriticNetwork(input_dim=X_train.shape[1]).to(device)
    train(model, X_train, y_train, epochs=50, lr=1e-3, batch_size=64, device=device)
    acc_uncompressed = evaluate(model, X_test, y_test, device=device)

    compress_model(model)
    acc_compressed = evaluate(model, X_test, y_test)

    return acc_uncompressed, acc_compressed
