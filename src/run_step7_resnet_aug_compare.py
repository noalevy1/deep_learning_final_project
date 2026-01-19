from experiment_runner import ExperimentConfig, run_many_experiments

if __name__ == "__main__":

    base = dict(
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

    print(run_many_experiments(configs, results_root="results"))
