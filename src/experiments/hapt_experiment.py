from src.loaders.load_hapt import load_hapt
from src.experiments.base_experiment import run_experiment
from src.models.compact_baselines import CompactHARMLP

_CLASS_NAMES = [
    "Walking", "Walking Upstairs", "Walking Downstairs",
    "Sitting", "Standing", "Laying",
    "Stand→Sit", "Sit→Stand", "Sit→Lie", "Lie→Sit", "Stand→Lie", "Lie→Stand",
]


def run_hapt(epochs=50, seeds=(42,), fine_tune_epochs=3, model_dir=None):
    # balance=False (2026-07-27, corrected): balance=True was tried to match
    # ECG's switch, but HAPT's transition classes have only 23-90 train
    # examples -- oversampling them craters accuracy ~20pp (Dendritic
    # 92.5%->72.5%) and blows up seed variance 8x, while balance=False keeps
    # the Snowflake/Global/QAT Dendritic-vs-baseline robustness edge intact
    # (in fact cleaner: baselines go slightly negative, Dendritic stays
    # positive) with normal accuracy/variance and unchanged 8/8 TOST.
    # See docs/experiment_log.md.
    data = load_hapt(balance=False)
    return run_experiment(
        get_data=lambda seed: data,
        num_classes=12,
        class_names=_CLASS_NAMES,
        epochs=epochs,
        seeds=seeds,
        fine_tune_epochs=fine_tune_epochs,
        batch_size=128,
        model_dir=model_dir,
        lit_baseline_fn=lambda: CompactHARMLP(input_dim=data[0].shape[1], num_classes=12),
        lit_baseline_label="HAPT-MLP",
    )
