from src.loaders.load_ecg_patient_split import load_ecg_patient_split
from src.experiments.base_experiment import run_experiment
from src.models.compact_baselines import ECGCNNBaseline

_CLASS_NAMES = ["Normal", "Supraventricular", "Ventricular", "Fusion", "Unknown"]


def run_ecg_patient(epochs=50, seeds=(42,), fine_tune_epochs=3, model_dir=None):
    """Same task/classes as run_ecg, but split by patient (DS1/DS2) instead of
    by beat -- checks how much the standard ECG split's accuracy is inflated
    by patient-level data leakage.

    balance=True (2026-07-27): oversampling restored as the default after
    finding it's specifically what gives Dendritic a real, reproducible
    quantization-robustness edge over MLPBaseline/LayerMatchedMLP on
    Snowflake/Global/QAT -- see docs/experiment_log.md. Costs TOST
    cleanliness (21/24 vs 24/24) and ~7x training time vs balance=False."""
    data = load_ecg_patient_split(balance=True)
    return run_experiment(
        get_data=lambda seed: data,
        num_classes=5,
        class_names=_CLASS_NAMES,
        epochs=epochs,
        seeds=seeds,
        fine_tune_epochs=fine_tune_epochs,
        batch_size=256,
        model_dir=model_dir,
        lit_baseline_fn=lambda: ECGCNNBaseline(num_classes=5),
        lit_baseline_label="ECG-CNN",
    )
