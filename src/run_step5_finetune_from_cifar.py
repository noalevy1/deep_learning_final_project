from __future__ import annotations
from pathlib import Path
import torch

from experiment_runner import ExperimentConfig, run_single_experiment, get_project_root


def main():
    print("Step 5: Fine-tuning from CIFAR-10 checkpoint", flush=True)

    project_root = get_project_root()
    data_dir = project_root / "data"
    pretrain_path = project_root / "results" / "step4_pretrain_cifar" / "pretrained_cifar10.pt"
    results_root = project_root / "results" / "step5_finetune_cifar"

    cfg = ExperimentConfig(
        data_dir=str(data_dir),
        model="simple_bn",
        optimizer="adam",
        lr=1e-4,
        epochs=10,
        batch_size=16,
        weight_decay=1e-4,
        dropout_p=0.2,
        augment="strong",
        normalize="none",
        seed=42,
        pretrained_path=str(pretrain_path),
    )

    summary = run_single_experiment(cfg, results_root=str(results_root))
    print(summary)


if __name__ == "__main__":
    main()
