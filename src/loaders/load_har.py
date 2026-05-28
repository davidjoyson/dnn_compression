import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def load_har(test_size=0.2, seed=42):
    """
    UCI Human Activity Recognition dataset (smartphone accelerometer/gyroscope).
    561 features, 6-class: WALKING, WALKING_UPSTAIRS, WALKING_DOWNSTAIRS, SITTING, STANDING, LAYING.
    Returns X_train, y_train, X_test, y_test as NumPy arrays.
    """
    data = fetch_openml("har", version=1, as_frame=False, parser="liac-arff")
    X = data.data.astype(np.float32)
    y = data.target.astype(int) - 1  # 1-6 → 0-5

    X = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )
    return X_train, y_train, X_test, y_test
