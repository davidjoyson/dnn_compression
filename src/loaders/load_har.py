import io
import os
import urllib.request
import zipfile
import numpy as np
from sklearn.preprocessing import StandardScaler

_UCI_URL = (
    "https://archive.ics.uci.edu/static/public/240/"
    "human+activity+recognition+using+smartphones.zip"
)
# Inside the zip, files live under this prefix
_ZIP_PREFIX = "UCI HAR Dataset/"


def download_har(data_dir="data/har"):
    os.makedirs(data_dir, exist_ok=True)
    print("Downloading UCI HAR dataset...")
    with urllib.request.urlopen(_UCI_URL, timeout=120) as r:
        data = r.read()
    # Outer zip contains a nested "UCI HAR Dataset.zip"
    with zipfile.ZipFile(io.BytesIO(data)) as outer:
        inner_bytes = outer.read("UCI HAR Dataset.zip")
    with zipfile.ZipFile(io.BytesIO(inner_bytes)) as z:
        for name in [
            "train/X_train.txt", "train/y_train.txt",
            "test/X_test.txt",   "test/y_test.txt",
        ]:
            dest = os.path.join(data_dir, os.path.basename(name))
            with open(dest, "wb") as f:
                f.write(z.read(_ZIP_PREFIX + name))
    print(f"UCI HAR dataset saved to {data_dir}/")


def load_har(data_dir="data/har"):
    """
    UCI HAR dataset — subject-level train/test split from UCI source.
    21 subjects for training, 9 subjects for testing (no subject overlap).
    561 pre-extracted features, 6-class activity recognition.
    """
    cache = {
        "X_train": os.path.join(data_dir, "X_train.npy"),
        "y_train": os.path.join(data_dir, "y_train.npy"),
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
        download_har(data_dir)

    X_train = np.loadtxt(os.path.join(data_dir, "X_train.txt"), dtype=np.float32)
    y_train = np.loadtxt(os.path.join(data_dir, "y_train.txt"), dtype=int) - 1  # 1-6 → 0-5
    X_test  = np.loadtxt(os.path.join(data_dir, "X_test.txt"),  dtype=np.float32)
    y_test  = np.loadtxt(os.path.join(data_dir, "y_test.txt"),  dtype=int) - 1

    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_test  = scaler.transform(X_test).astype(np.float32)

    np.save(cache["X_train"], X_train)
    np.save(cache["y_train"], y_train)
    np.save(cache["X_test"],  X_test)
    np.save(cache["y_test"],  y_test)

    return X_train, y_train, X_test, y_test
