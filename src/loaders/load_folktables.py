import numpy as np
from folktables import ACSDataSource, ACSIncome

def load_folktables_income(state="CA", year=2018):
    """
    Loads Folktables ACSIncome dataset for a given state + year.
    Returns: X_train, y_train, X_test, y_test (NumPy arrays)
    """

    data_source = ACSDataSource(
        survey_year=year,
        horizon="1-Year",
        survey="person"
    )

    df = data_source.get_data(states=[state], download=True)

    features, labels, _ = ACSIncome.df_to_numpy(df)

    # Convert to NumPy arrays
    X = np.array(features, dtype=np.float32)
    y = np.array(labels, dtype=np.float32).reshape(-1, 1)

    # Manual 80/20 split
    n = len(X)
    split = int(0.8 * n)

    X_train = X[:split]
    y_train = y[:split]
    X_test = X[split:]
    y_test = y[split:]

    return X_train, y_train, X_test, y_test
