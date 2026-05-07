from src.models.dendritic_network import DendriticNetwork
from src.training.train import train
from src.training.evaluate import evaluate
from src.compression.compression_pipeline import compress_model

def scaling_experiment(X_train, y_train, X_test, y_test):
    results = []

    for neurons in [2, 4, 8, 16]:
        for branches in [2, 4, 6]:
            m = DendriticNetwork(X_train.shape[1], neurons, branches)
            train(m, X_train, y_train, epochs=800)

            acc_uncompressed = evaluate(m, X_test, y_test)

            compress_model(m)
            acc_compressed = evaluate(m, X_test, y_test)

            results.append((neurons, branches, acc_uncompressed, acc_compressed))

    return results
