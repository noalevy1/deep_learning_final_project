from __future__ import annotations
from pathlib import Path
import torch
import torch.nn as nn

from experiment_runner import ExperimentConfig, run_single_experiment


def main():
    print("Step 5: Fine-tuning from CIFAR-10 checkpoint", flush=True)

    DATA_DIR = r"/Users/noalevy/Desktop/Desktop - Noa’s MacBook Pro/School/MTA/third year/first semester/deep learning/data"
    PRETRAIN_PATH = r"pretrained_cifar10.pt"

    cfg = ExperimentConfig(
        data_dir=DATA_DIR,
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
    )

    ckpt = torch.load(PRETRAIN_PATH, map_location="cpu")
    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt

    cfg.pretrained_path = PRETRAIN_PATH  # type: ignore

    summary = run_single_experiment(cfg, results_root="results")
    print(summary)


if __name__ == "__main__":
    main()
