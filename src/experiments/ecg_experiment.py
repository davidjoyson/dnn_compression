from src.loaders.load_ecg import load_ecg
from src.experiments.base_experiment import run_experiment

_CLASS_NAMES = ["Normal", "Supraventricular", "Ventricular", "Fusion", "Unknown"]


def run_ecg(epochs=50, seeds=(42,), fine_tune_epochs=3):
    data = load_ecg()
    return run_experiment(
        get_data=lambda seed: data,
        num_classes=5,
        class_names=_CLASS_NAMES,
        epochs=epochs,
        seeds=seeds,
        fine_tune_epochs=fine_tune_epochs,
        batch_size=256,
    )
