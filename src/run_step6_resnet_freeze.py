from experiment_runner import ExperimentConfig, run_single_experiment

if __name__ == "__main__":

    cfg = ExperimentConfig(
        model="resnet50",
        optimizer="adam",
        lr=1e-3,
        epochs=8,
        batch_size=16,
        weight_decay=1e-4,
        dropout_p=0.0,          # not used by resnet
        augment="none",
        normalize="imagenet",
        seed=42,
        pretrained=True,
        freeze_mode="backbone", # train only fc
    )

    print(run_single_experiment(cfg, results_root="results"))
