import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler

def load_adult_income():
    """
    Loads UCI Adult Income dataset from OpenML.
    Automatically detects the correct target column.
    Returns: X, y as NumPy arrays.
    """

    data = fetch_openml("adult", version=2, as_frame=True)
    df = data.frame

    # Possible target column names
    target_candidates = ["income", "class", "target", "Income"]

    target_col = None
    for col in target_candidates:
        if col in df.columns:
            target_col = col
            break

    if target_col is None:
        raise ValueError(f"Could not find target column. Available columns: {df.columns}")

    # Convert target to binary 0/1
    df[target_col] = (df[target_col].astype(str).str.contains(">50K")).astype(float)

    # One-hot encode categorical columns
    df = pd.get_dummies(df, drop_first=True)

    y = df[target_col].values.astype(np.float32)
    X = df.drop(columns=[target_col]).values.astype(np.float32)

    X = StandardScaler().fit_transform(X)

    return X, y
