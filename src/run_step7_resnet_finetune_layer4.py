from pathlib import Path
from experiment_runner import ExperimentConfig, run_single_experiment, get_project_root

if __name__ == "__main__":
    project_root = get_project_root()
    data_dir = project_root / "data"
    results_root = project_root / "results" / "step7_resnet50_layer4_single"

    cfg = ExperimentConfig(
        data_dir=str(data_dir),
        model="resnet50",
        optimizer="adam",
        lr=1e-4,
        epochs=10,
        batch_size=16,
        weight_decay=1e-4,
        augment="none",
        normalize="imagenet",
        seed=42,
    )

    print(run_single_experiment(cfg, results_root=str(results_root)))
