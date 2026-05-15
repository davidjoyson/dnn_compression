import os
import zipfile
import subprocess
import numpy as np
import pandas as pd
from sklearn.utils import resample

_KAGGLE_DATASET = "shayanfazeli/heartbeat"


def download_ecg(data_dir="data/ecg"):
    """Download MIT-BIH CSVs via the Kaggle CLI (requires kaggle + credentials)."""
    os.makedirs(data_dir, exist_ok=True)
    print(f"Downloading ECG dataset from Kaggle ({_KAGGLE_DATASET}) ...")
    try:
        subprocess.run(
            ["kaggle", "datasets", "download", "-d", _KAGGLE_DATASET, "-p", data_dir],
            check=True,
        )
    except FileNotFoundError:
        raise RuntimeError(
            "kaggle CLI not found. Install with:  pip install kaggle\n"
            "Then add credentials: https://www.kaggle.com/docs/api\n"
            "Or manually download from: https://www.kaggle.com/datasets/shayanfazeli/heartbeat"
        )
    zip_path = os.path.join(data_dir, "heartbeat.zip")
    if os.path.exists(zip_path):
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(data_dir)
        os.remove(zip_path)
    print(f"ECG dataset saved to {data_dir}/")


def load_ecg(data_dir="data/ecg", balance=True):
    cache_suffix = "_balanced" if balance else ""
    cache_files = {
        "X_train": os.path.join(data_dir, f"X_train{cache_suffix}.npy"),
        "y_train": os.path.join(data_dir, f"y_train{cache_suffix}.npy"),
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

    train_path = os.path.join(data_dir, "mitbih_train.csv")
    test_path  = os.path.join(data_dir, "mitbih_test.csv")

    if not os.path.exists(train_path):
        download_ecg(data_dir)

    train_df = pd.read_csv(train_path, header=None)
    test_df  = pd.read_csv(test_path,  header=None)

    X_train = train_df.iloc[:, :187].values.astype(np.float32)
    y_train = train_df.iloc[:, 187].values.astype(int)
    X_test  = test_df.iloc[:,  :187].values.astype(np.float32)
    y_test  = test_df.iloc[:,  187].values.astype(int)

    if balance:
        counts    = np.bincount(y_train)
        max_count = counts.max()
        rng       = np.random.RandomState(42)
        parts_X, parts_y = [], []
        for c in range(len(counts)):
            idx = np.where(y_train == c)[0]
            if len(idx) < max_count:
                idx = resample(idx, n_samples=max_count, random_state=rng, replace=True)
            parts_X.append(X_train[idx])
            parts_y.append(np.full(len(idx), c, dtype=int))
        X_train = np.vstack(parts_X)
        y_train = np.concatenate(parts_y)
        perm    = rng.permutation(len(X_train))
        X_train = X_train[perm]
        y_train = y_train[perm]

    np.save(cache_files["X_train"], X_train)
    np.save(cache_files["y_train"], y_train)
    np.save(cache_files["X_test"],  X_test)
    np.save(cache_files["y_test"],  y_test)

    return X_train, y_train, X_test, y_test
