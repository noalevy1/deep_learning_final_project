from pathlib import Path
from experiment_runner import ExperimentConfig, run_single_experiment, get_project_root

if __name__ == "__main__":
    project_root = get_project_root()
    data_dir = project_root / "data"
    results_root = project_root / "results" / "step6_resnet50_frozen"

    cfg = ExperimentConfig(
        data_dir=str(data_dir),
        model="resnet50",
        optimizer="adam",
        lr=1e-3,
        epochs=8,
        batch_size=16,
        weight_decay=1e-4,
        dropout_p=0.0,
        augment="none",
        normalize="imagenet",
        seed=42,
        pretrained=True,
        freeze_mode="backbone",
    )

    print(run_single_experiment(cfg, results_root=str(results_root)))
