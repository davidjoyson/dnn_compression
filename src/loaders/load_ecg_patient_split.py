"""
Patient-independent ECG loader — an alternative to load_ecg.py that fixes the
data leakage risk in the Kaggle mitbih_train/test.csv split (which has no
patient ID at all and is known to split by beat, not by patient).

Rebuilds the same 5-class (AAMI N/S/V/F/Q) beat-classification task directly
from the raw PhysioNet MIT-BIH Arrhythmia Database (data/mitdb_raw/, pulled
via `wfdb.dl_database('mitdb', ...)`), splitting by RECORD ID using the
standard patient-independent DS1/DS2 grouping from de Chazal et al. 2004
("Automatic Classification of Heartbeats Using ECG Morphology and Heartbeat
Interval Features") — the split convention used throughout the arrhythmia
classification literature specifically to avoid this leakage.

NOTE: DS1/DS2 record lists below are the standard cited grouping; spot-check
against a primary source if exact literature reproducibility matters. The
4 fully-paced records (102, 104, 107, 217) are excluded, matching standard
AAMI-based classification practice.
"""
import os
import numpy as np
import wfdb
from sklearn.utils import resample

# de Chazal et al. 2004 DS1 (train) / DS2 (test) patient-independent split
DS1_TRAIN = [101, 106, 108, 109, 112, 114, 115, 116, 118, 119, 122, 124,
             201, 203, 205, 207, 208, 209, 215, 220, 223, 230]
DS2_TEST  = [100, 103, 105, 111, 113, 117, 121, 123, 200, 202, 210, 212,
             213, 214, 219, 221, 222, 228, 231, 232, 233, 234]

# AAMI EC57 beat-symbol -> 5-class grouping, matching the Kaggle CSV's label scheme
_AAMI_MAP = {
    "N": 0, "L": 0, "R": 0, "e": 0, "j": 0,   # Normal
    "A": 1, "a": 1, "J": 1, "S": 1,           # Supraventricular ectopic
    "V": 2, "E": 2,                           # Ventricular ectopic
    "F": 3,                                   # Fusion
    "/": 4, "f": 4, "Q": 4,                   # Unknown/paced
}

N_SAMPLES = 187  # match the existing Kaggle-derived feature width


def _extract_beats(record_id, raw_dir):
    path = os.path.join(raw_dir, str(record_id))
    rec = wfdb.rdrecord(path)
    ann = wfdb.rdann(path, "atr")
    sig = rec.p_signal[:, 0]  # channel 0 = MLII lead, standard choice

    r_peaks = ann.sample
    symbols = ann.symbol

    X, y = [], []
    for i in range(1, len(r_peaks) - 1):
        sym = symbols[i]
        if sym not in _AAMI_MAP:
            continue
        lo = (r_peaks[i - 1] + r_peaks[i]) // 2
        hi = (r_peaks[i] + r_peaks[i + 1]) // 2
        if hi <= lo:
            continue
        beat = sig[lo:hi]
        # resample variable-length beat window to a fixed N_SAMPLES via linear interp
        x_old = np.linspace(0, 1, len(beat))
        x_new = np.linspace(0, 1, N_SAMPLES)
        beat_rs = np.interp(x_new, x_old, beat).astype(np.float32)
        X.append(beat_rs)
        y.append(_AAMI_MAP[sym])

    return np.array(X, dtype=np.float32), np.array(y, dtype=int)


def load_ecg_patient_split(data_dir="data/ecg", raw_dir="data/mitdb_raw", balance=True):
    """
    balance: oversample minority classes in the TRAIN set only (never touches
    test), matching load_ecg.py's balance=True default -- so the patient-split
    comparison isolates the leakage effect instead of also being confounded by
    a class-balancing difference.
    """
    cache_suffix = "_balanced" if balance else ""
    cache_files = {
        "X_train": os.path.join(data_dir, f"X_train_patient{cache_suffix}.npy"),
        "y_train": os.path.join(data_dir, f"y_train_patient{cache_suffix}.npy"),
        "X_test":  os.path.join(data_dir, "X_test_patient.npy"),
        "y_test":  os.path.join(data_dir, "y_test_patient.npy"),
    }
    if all(os.path.exists(p) for p in cache_files.values()):
        return (
            np.load(cache_files["X_train"]),
            np.load(cache_files["y_train"]),
            np.load(cache_files["X_test"]),
            np.load(cache_files["y_test"]),
        )

    if not os.path.isdir(raw_dir):
        raise FileNotFoundError(
            f"Raw MIT-BIH data not found at {raw_dir}. "
            "Run: python -c \"import wfdb; wfdb.dl_database('mitdb', dl_dir='data/mitdb_raw')\""
        )

    def _build(record_ids):
        Xs, ys = [], []
        for rid in record_ids:
            X, y = _extract_beats(rid, raw_dir)
            Xs.append(X)
            ys.append(y)
            print(f"  record {rid}: {len(y)} beats")
        return np.concatenate(Xs), np.concatenate(ys)

    print(f"Building patient-independent ECG split (DS1 train={len(DS1_TRAIN)} "
          f"records, DS2 test={len(DS2_TEST)} records)...")
    print("Train records:")
    X_train, y_train = _build(DS1_TRAIN)
    print("Test records:")
    X_test, y_test = _build(DS2_TEST)

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

    os.makedirs(data_dir, exist_ok=True)
    np.save(cache_files["X_train"], X_train)
    np.save(cache_files["y_train"], y_train)
    np.save(cache_files["X_test"],  X_test)
    np.save(cache_files["y_test"],  y_test)

    print(f"Train: {X_train.shape[0]} beats, class counts {np.bincount(y_train)}")
    print(f"Test:  {X_test.shape[0]} beats, class counts {np.bincount(y_test)}")

    return X_train, y_train, X_test, y_test


if __name__ == "__main__":
    load_ecg_patient_split()
