from pathlib import Path
from experiment_runner import ExperimentConfig, run_many_experiments, get_project_root

if __name__ == "__main__":
    project_root = get_project_root()  # <project_root>/
    data_dir = project_root / "data"
    results_root = project_root / "results" / "step3_augmentation"

    configs = [
        ExperimentConfig(
            data_dir=str(data_dir),
            model="simple_bn",
            optimizer="adam",
            lr=1e-3,
            epochs=20,
            batch_size=16,
            weight_decay=1e-4,
            dropout_p=0.2,
            augment="none"
        ),
        ExperimentConfig(
            data_dir=str(data_dir),
            model="simple_bn",
            optimizer="adam",
            lr=1e-3,
            epochs=20,
            batch_size=16,
            weight_decay=1e-4,
            dropout_p=0.2,
            augment="strong"
        ),
    ]

    summary = run_many_experiments(configs, results_root=str(results_root))
    print(summary)
