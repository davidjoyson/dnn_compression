from src.experiments.base_experiment import run_experiment
from src.loaders.load_incart import load_incart
from src.models.compact_baselines import ECGCNNBaseline


_CLASS_NAMES = ["Normal", "Supraventricular", "Ventricular", "Fusion"]


def run_incart(epochs=50, seeds=(42,), fine_tune_epochs=3, model_dir=None):
    data = load_incart(balance=True, return_validation=True)
    return run_experiment(
        get_data=lambda seed: data,
        num_classes=4,
        class_names=_CLASS_NAMES,
        epochs=epochs,
        seeds=seeds,
        fine_tune_epochs=fine_tune_epochs,
        batch_size=256,
        model_dir=model_dir,
        lit_baseline_fn=lambda: ECGCNNBaseline(num_classes=4),
        lit_baseline_label="INCART-ECG-CNN",
    )
