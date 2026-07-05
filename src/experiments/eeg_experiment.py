from src.loaders.load_eeg import load_eeg
from src.experiments.base_experiment import run_experiment

_CLASS_NAMES = ["Negative", "Neutral", "Positive"]


def run_eeg(epochs=50, seeds=(42,), fine_tune_epochs=3, model_dir=None):
    data = load_eeg()
    return run_experiment(
        get_data=lambda seed: data,
        num_classes=3,
        class_names=_CLASS_NAMES,
        epochs=epochs,
        seeds=seeds,
        fine_tune_epochs=fine_tune_epochs,
        batch_size=128,
        model_dir=model_dir,
        weight_decay=0.0,
    )
