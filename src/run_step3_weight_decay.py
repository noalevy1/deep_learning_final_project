from experiment_runner import ExperimentConfig, run_many_experiments, get_project_root

if __name__ == "__main__":
    project_root = get_project_root()
    data_dir = project_root / "data"
    results_root = project_root / "results" / "step3_weight_decay"

    print("main3 starts", flush=True)

    base = dict(
        data_dir=str(data_dir),
        model="simple_bn",
        optimizer="adam",
        lr=1e-3,
        epochs=20,
        batch_size=16,
        seed=42,
        dropout_p=0.2,
        augment="none",
    )

    configs = [
        ExperimentConfig(**base, weight_decay=0.0),
        ExperimentConfig(**base, weight_decay=1e-4),
        ExperimentConfig(**base, weight_decay=1e-3),
    ]

    print(run_many_experiments(configs, results_root=str(results_root)), flush=True)
