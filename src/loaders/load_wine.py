import numpy as np
from sklearn.datasets import load_wine as _load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def load_wine(test_size=0.2, seed=42):
    """
    Loads the Wine dataset, scales features, converts target to binary
    (class 0 vs others), and returns train/test splits as NumPy arrays.

    Returns:
        X_train, y_train, X_test, y_test
    """

    data = _load_wine()
    X = data.data
    y = (data.target == 0).astype(float).reshape(-1, 1)

    # Scale features
    X = StandardScaler().fit_transform(X)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed
    )

    return X_train, y_train, X_test, y_test
