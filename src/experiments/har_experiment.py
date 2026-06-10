from src.loaders.load_har import load_har
from src.experiments.base_experiment import run_experiment

_CLASS_NAMES = ["Walking", "Walking Upstairs", "Walking Downstairs", "Sitting", "Standing", "Laying"]


def run_har(epochs=50, seeds=(42,), fine_tune_epochs=3, model_dir=None):
    return run_experiment(
        get_data=lambda seed: load_har(seed=seed),
        num_classes=6,
        class_names=_CLASS_NAMES,
        epochs=epochs,
        seeds=seeds,
        fine_tune_epochs=fine_tune_epochs,
        batch_size=128,
        model_dir=model_dir,
    )
