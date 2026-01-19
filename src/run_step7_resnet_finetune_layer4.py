from experiment_runner import ExperimentConfig, run_single_experiment

if __name__ == "__main__":

    cfg = ExperimentConfig(
        model="resnet50",
        optimizer="adam",
        lr=1e-4,                 # base LR; backbone uses 0.1*lr in code
        epochs=10,
        batch_size=16,
        weight_decay=1e-4,
        augment="none",          # start clean
        normalize="imagenet",
        seed=42,
        pretrained=True,
        freeze_mode="layer4",    # open layer4 + fc
    )

    print(run_single_experiment(cfg, results_root="results"))
