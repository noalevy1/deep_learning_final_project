from pathlib import Path
from experiment_runner import ExperimentConfig, run_many_experiments, get_project_root

if __name__ == "__main__":
    project_root = get_project_root()
    data_dir = project_root / "data"
    results_root = project_root / "results" / "step7_resnet50_finetune"

    base = dict(
        data_dir=str(data_dir),
        model="resnet50",
        optimizer="adam",
        lr=1e-4,
        epochs=10,
        batch_size=16,
        weight_decay=1e-4,
        normalize="imagenet",
        seed=42,
        pretrained=True,
        freeze_mode="layer4",
    )

    configs = [
        ExperimentConfig(**base, augment="none"),
        ExperimentConfig(**base, augment="strong"),
    ]

    print(run_many_experiments(configs, results_root=str(results_root)))
