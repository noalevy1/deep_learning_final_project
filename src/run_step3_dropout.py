from experiment_runner import ExperimentConfig, run_many_experiments, get_project_root

if __name__ == "__main__":
    project_root = get_project_root()  # <project_root>/
    data_dir = project_root / "data"
    results_root = project_root / "results" / "step3_dropout"

    print("step 3 (dropout) starts", flush=True)

    base = dict(
        data_dir=str(data_dir),
        model="simple_bn",
        optimizer="adam",
        lr=1e-3,
        epochs=20,
        batch_size=16,
        seed=42,
        weight_decay=1e-4,
        augment="none",
    )

    configs = [
        ExperimentConfig(**base, dropout_p=0.0),
        ExperimentConfig(**base, dropout_p=0.2),
        ExperimentConfig(**base, dropout_p=0.5),
    ]

    print(run_many_experiments(configs, results_root=str(results_root)), flush=True)
