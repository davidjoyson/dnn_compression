import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

_LABEL_MAP = {"NEGATIVE": 0, "NEUTRAL": 1, "POSITIVE": 2}


def load_eeg(data_dir="data/eeg", test_size=0.2, random_state=42):
    cache_files = {
        "X_train": os.path.join(data_dir, "X_train.npy"),
        "y_train": os.path.join(data_dir, "y_train.npy"),
        "X_test":  os.path.join(data_dir, "X_test.npy"),
        "y_test":  os.path.join(data_dir, "y_test.npy"),
    }

    if all(os.path.exists(p) for p in cache_files.values()):
        return (
            np.load(cache_files["X_train"]),
            np.load(cache_files["y_train"]),
            np.load(cache_files["X_test"]),
            np.load(cache_files["y_test"]),
        )

    csv_path = os.path.join(data_dir, "emotions.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"EEG data not found at {csv_path}.\n"
            "Download from: https://www.kaggle.com/datasets/birdy654/eeg-brainwave-dataset-feeling-emotions"
        )

    df = pd.read_csv(csv_path)
    X = df.iloc[:, :-1].values.astype(np.float32)
    y = df["label"].map(_LABEL_MAP).values.astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_test  = scaler.transform(X_test).astype(np.float32)

    np.save(cache_files["X_train"], X_train)
    np.save(cache_files["y_train"], y_train)
    np.save(cache_files["X_test"],  X_test)
    np.save(cache_files["y_test"],  y_test)

    return X_train, y_train, X_test, y_test
