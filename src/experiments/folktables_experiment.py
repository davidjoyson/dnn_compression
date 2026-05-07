import torch
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from folktables import ACSDataSource, ACSIncome

from src.models.dendritic_network import DendriticNetwork
from src.models.mlp_baseline import MLPBaseline
from src.training.train import train
from src.training.evaluate import evaluate
from src.compression.compression_pipeline import compress_model


def load_folktables_income(state="CA", year=2018):
    """
    Loads ACSIncome task for a single state.
    Returns X, y as numpy arrays.
    """

    data_source = ACSDataSource(
        survey_year=year,
        horizon="1-Year",
        survey="person"
    )

    acs_data = data_source.get_data(states=[state], download=True)

    features, labels, _ = ACSIncome.df_to_numpy(acs_data)

    # Standardize numeric features
    scaler = StandardScaler()
    X = scaler.fit_transform(features)

    y = labels.astype(float).reshape(-1, 1)

    return X, y


def run_folktables_income(state="CA", year=2018):
    print(f"Loading Folktables ACSIncome for {state}, {year}...")
    X, y = load_folktables_income(state, year)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Convert to tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)

    input_dim = X_train.shape[1]

    # -------------------------
    # Uncompressed dendritic model
    # -------------------------
    model = DendriticNetwork(
        input_dim=input_dim,
        hidden_neurons1=16,
        hidden_neurons2=8,
        branches=4,
        hidden_per_branch=4
    )

    train(model, X_train, y_train, epochs=5, lr=1e-3, batch_size=256)
    acc_uncompressed = evaluate(model, X_test, y_test)

    # -------------------------
    # Compressed dendritic model
    # -------------------------
    compress_model(model)
    acc_compressed = evaluate(model, X_test, y_test)

    # -------------------------
    # MLP baseline
    # -------------------------
    mlp = MLPBaseline(input_dim=input_dim, hidden=32)
    train(mlp, X_train, y_train, epochs=20, lr=1e-3, batch_size=256)
    acc_mlp = evaluate(mlp, X_test, y_test)

    # -------------------------
    # Model sizes
    # -------------------------
    size_uncompressed = sum(p.numel() for p in model.parameters()) * 4  # float32
    size_compressed = sum(p.numel() for p in model.parameters())        # approx 1 byte per param after quantization

    return {
        "accuracy": {
            "Dendritic (Uncompressed)": acc_uncompressed,
            "Dendritic (Compressed)": acc_compressed,
            "MLP Baseline": acc_mlp
        },
        "sizes": {
            "uncompressed": size_uncompressed,
            "compressed": size_compressed
        }
    }
