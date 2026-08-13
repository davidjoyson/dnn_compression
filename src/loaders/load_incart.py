"""Patient-independent loader for the PhysioNet INCART arrhythmia database."""

import os
import re

import numpy as np
import wfdb
from sklearn.utils import resample


_AAMI_MAP = {
    "N": 0, "L": 0, "R": 0, "e": 0, "j": 0,
    "A": 1, "a": 1, "J": 1, "S": 1,
    "V": 2, "E": 2,
    "F": 3,
}

N_SAMPLES = 187

# Source-patient IDs from the INCART headers. Keeping every recording from a
# patient in one partition prevents leakage when patients have multiple files.
TRAIN_PATIENTS = tuple(range(1, 23))
VAL_PATIENTS = tuple(range(23, 27))
TEST_PATIENTS = tuple(range(27, 33))


def download_incart(raw_dir="data/incart_raw"):
    os.makedirs(raw_dir, exist_ok=True)
    wfdb.dl_database("incartdb", dl_dir=raw_dir)


def _patient_id(header):
    text = " ".join(header.comments or [])
    patterns = (
        r"patient(?:\s+number|\s+id)?\s*[:=#-]?\s*(\d+)",
        r"patient\s+(\d+)",
    )
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return int(match.group(1))
    raise ValueError(f"Could not determine source patient for record {header.record_name}: {text!r}")


def _record_names(raw_dir):
    records_file = os.path.join(raw_dir, "RECORDS")
    if os.path.exists(records_file):
        with open(records_file, encoding="ascii") as handle:
            return [line.strip() for line in handle if line.strip()]
    return sorted(os.path.splitext(name)[0] for name in os.listdir(raw_dir) if name.endswith(".hea"))


def _extract_beats(record_name, raw_dir):
    path = os.path.join(raw_dir, record_name)
    record = wfdb.rdrecord(path)
    annotation = wfdb.rdann(path, "atr")

    lead_names = [name.upper() for name in record.sig_name]
    lead_index = lead_names.index("II") if "II" in lead_names else 0
    signal = record.p_signal[:, lead_index]

    beats, labels = [], []
    peaks = annotation.sample
    for index in range(1, len(peaks) - 1):
        label = _AAMI_MAP.get(annotation.symbol[index])
        if label is None:
            continue
        lo = (peaks[index - 1] + peaks[index]) // 2
        hi = (peaks[index] + peaks[index + 1]) // 2
        if hi <= lo:
            continue
        beat = signal[lo:hi]
        beat = np.interp(
            np.linspace(0.0, 1.0, N_SAMPLES),
            np.linspace(0.0, 1.0, len(beat)),
            beat,
        ).astype(np.float32)
        beats.append(beat)
        labels.append(label)
    return np.asarray(beats, dtype=np.float32), np.asarray(labels, dtype=np.int64)


def _oversample(X, y):
    """Median-frequency resampling without changing validation/test data.

    Every class is sampled to the median natural class count. This combines
    majority undersampling with minority oversampling and avoids expanding four
    Unknown beats to more than 100,000 examples as the old five-class loader did.
    """
    counts = np.bincount(y, minlength=4)
    if np.any(counts == 0):
        raise ValueError(f"Cannot balance INCART training data with missing classes: {counts}")
    random_state = np.random.RandomState(42)
    target = int(np.median(counts))
    indices = []
    for class_id in range(4):
        class_indices = np.flatnonzero(y == class_id)
        class_indices = resample(
            class_indices,
            n_samples=target,
            replace=len(class_indices) < target,
            random_state=random_state,
        )
        indices.append(class_indices)
    indices = np.concatenate(indices)
    indices = indices[random_state.permutation(len(indices))]
    return X[indices], y[indices]


def load_incart(data_dir="data/incart", raw_dir="data/incart_raw", balance=True,
                download=True, return_validation=True):
    """Load beat windows using a fixed, leakage-free 22/4/6 patient split."""
    # Versioned cache names prevent reuse of the superseded five-class cache.
    suffix = "_aami4_median" if balance else "_aami4_natural"
    names = ("X_train", "y_train", "X_val", "y_val", "X_test", "y_test")
    cache = {
        name: os.path.join(data_dir, f"{name}{suffix}.npy")
        for name in names
    }
    if all(os.path.exists(path) for path in cache.values()):
        arrays = tuple(np.load(cache[name]) for name in names)
        return arrays if return_validation else arrays[:2] + arrays[4:]

    if not os.path.isdir(raw_dir) or not _record_names(raw_dir):
        if not download:
            raise FileNotFoundError(f"INCART data not found at {raw_dir}")
        print("Downloading the PhysioNet INCART database...")
        download_incart(raw_dir)

    grouped = {"train": (set(TRAIN_PATIENTS), [], []),
               "val": (set(VAL_PATIENTS), [], []),
               "test": (set(TEST_PATIENTS), [], [])}
    patient_records = {}
    for record_name in _record_names(raw_dir):
        header = wfdb.rdheader(os.path.join(raw_dir, record_name))
        patient = _patient_id(header)
        patient_records.setdefault(patient, []).append(record_name)
        partition = next((name for name, (ids, _, _) in grouped.items() if patient in ids), None)
        if partition is None:
            raise ValueError(f"Unexpected INCART patient ID {patient} in {record_name}")
        X, y = _extract_beats(record_name, raw_dir)
        grouped[partition][1].append(X)
        grouped[partition][2].append(y)

    expected = set(TRAIN_PATIENTS + VAL_PATIENTS + TEST_PATIENTS)
    if set(patient_records) != expected:
        raise ValueError(f"INCART patient IDs differ from expected: found {sorted(patient_records)}")

    arrays = []
    for partition in ("train", "val", "test"):
        _, X_parts, y_parts = grouped[partition]
        X = np.concatenate(X_parts)
        y = np.concatenate(y_parts)
        counts = np.bincount(y, minlength=4)
        print(f"INCART {partition}: {len(y):,} beats, class counts {counts.tolist()}")
        arrays.extend((X, y))

    X_train, y_train, X_val, y_val, X_test, y_test = arrays
    if balance:
        X_train, y_train = _oversample(X_train, y_train)
        print(f"INCART balanced train: {len(y_train):,} beats, "
              f"class counts {np.bincount(y_train, minlength=4).tolist()}")

    os.makedirs(data_dir, exist_ok=True)
    final_arrays = (X_train, y_train, X_val, y_val, X_test, y_test)
    for name, array in zip(names, final_arrays):
        np.save(cache[name], array)
    return final_arrays if return_validation else final_arrays[:2] + final_arrays[4:]


if __name__ == "__main__":
    load_incart()
