import io
import os
import urllib.request
import zipfile
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample

_UCI_URL = (
    "https://archive.ics.uci.edu/static/public/341/"
    "smartphone+based+recognition+of+human+activities+and+postural+transitions.zip"
)


def download_hapt(data_dir="data/hapt"):
    os.makedirs(data_dir, exist_ok=True)
    print("Downloading HAPT dataset from UCI ML Repository...")
    with urllib.request.urlopen(_UCI_URL, timeout=120) as r:
        data = r.read()
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        for name in ["Train/X_train.txt", "Train/y_train.txt",
                     "Test/X_test.txt",   "Test/y_test.txt"]:
            dest = os.path.join(data_dir, os.path.basename(name))
            with open(dest, "wb") as f:
                f.write(z.read(name))
    print(f"HAPT dataset saved to {data_dir}/")


def load_hapt(data_dir="data/hapt", balance=True):
    """
    UCI HAPT dataset (smartphone accelerometer/gyroscope).
    561 pre-extracted features, 12 classes:
      0-5: WALKING, WALKING_UPSTAIRS, WALKING_DOWNSTAIRS, SITTING, STANDING, LAYING
      6-11: STAND_TO_SIT, SIT_TO_STAND, SIT_TO_LIE, LIE_TO_SIT, STAND_TO_LIE, LIE_TO_STAND
    Pre-split train/test from UCI; transition classes balanced by oversampling.
    Returns X_train, y_train, X_test, y_test as NumPy arrays.
    """
    cache_suffix = "_balanced" if balance else ""
    cache = {
        "X_train": os.path.join(data_dir, f"X_train{cache_suffix}.npy"),
        "y_train": os.path.join(data_dir, f"y_train{cache_suffix}.npy"),
        "X_test":  os.path.join(data_dir, "X_test.npy"),
        "y_test":  os.path.join(data_dir, "y_test.npy"),
    }

    if all(os.path.exists(p) for p in cache.values()):
        return (
            np.load(cache["X_train"]),
            np.load(cache["y_train"]),
            np.load(cache["X_test"]),
            np.load(cache["y_test"]),
        )

    if not os.path.exists(os.path.join(data_dir, "X_train.txt")):
        download_hapt(data_dir)

    X_train = np.loadtxt(os.path.join(data_dir, "X_train.txt"), dtype=np.float32)
    y_train = np.loadtxt(os.path.join(data_dir, "y_train.txt"), dtype=int) - 1  # 1-12 → 0-11
    X_test  = np.loadtxt(os.path.join(data_dir, "X_test.txt"),  dtype=np.float32)
    y_test  = np.loadtxt(os.path.join(data_dir, "y_test.txt"),  dtype=int) - 1

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

    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_test  = scaler.transform(X_test).astype(np.float32)

    np.save(cache["X_train"], X_train)
    np.save(cache["y_train"], y_train)
    np.save(cache["X_test"],  X_test)
    np.save(cache["y_test"],  y_test)

    return X_train, y_train, X_test, y_test
