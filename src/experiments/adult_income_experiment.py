import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

from src.models.dendritic_network import DendriticNetwork
from src.training.train import train
from src.training.evaluate import evaluate
from src.compression.compression_pipeline import compress_model


def load_adult_income():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
    cols = [
        "age","workclass","fnlwgt","education","education-num","marital-status",
        "occupation","relationship","race","sex","capital-gain","capital-loss",
        "hours-per-week","native-country","income"
    ]

    df = pd.read_csv(url, names=cols, na_values=" ?", skipinitialspace=True)
    df = df.dropna()

    X = df.drop("income", axis=1)
    y = (df["income"] == " >50K").astype(float).values.reshape(-1,1)

    numeric = X.select_dtypes(include=["int64","float64"]).columns
    categorical = X.select_dtypes(include=["object"]).columns

    pre = ColumnTransformer([
        ("num", StandardScaler(), numeric),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical)
    ])

    X = pre.fit_transform(X).toarray()

    return X, y


def run_adult_income():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X, y = load_adult_income()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)

    model = DendriticNetwork(
        input_dim=X_train.shape[1],
        hidden_neurons1=16,
        hidden_neurons2=8,
        branches=4,
        hidden_per_branch=4
    ).to(device)

    train(model, X_train, y_train, epochs=20, lr=1e-3, batch_size=256, device=device)
    acc_uncompressed = evaluate(model, X_test, y_test, device=device)

    compress_model(model)
    acc_compressed = evaluate(model, X_test, y_test, device=device)

    return acc_uncompressed, acc_compressed
