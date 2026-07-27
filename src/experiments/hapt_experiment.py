from src.loaders.load_hapt import load_hapt
from src.experiments.base_experiment import run_experiment

_CLASS_NAMES = [
    "Walking", "Walking Upstairs", "Walking Downstairs",
    "Sitting", "Standing", "Laying",
    "Stand→Sit", "Sit→Stand", "Sit→Lie", "Lie→Sit", "Stand→Lie", "Lie→Stand",
]


def run_hapt(epochs=50, seeds=(42,), fine_tune_epochs=3, model_dir=None):
    # balance=True (2026-07-27): kept consistent with ECG's default switch,
    # though HAPT's own uncompressed-accuracy ordering doesn't change either
    # way -- see docs/experiment_log.md.
    data = load_hapt(balance=True)
    return run_experiment(
        get_data=lambda seed: data,
        num_classes=12,
        class_names=_CLASS_NAMES,
        epochs=epochs,
        seeds=seeds,
        fine_tune_epochs=fine_tune_epochs,
        batch_size=128,
        model_dir=model_dir,
    )
