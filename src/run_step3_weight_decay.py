if __name__ == "__main__":
    from experiment_runner import ExperimentConfig, run_many_experiments

    DATA_DIR = r"/Users/noalevy/Desktop/Desktop - Noa’s MacBook Pro/School/MTA/third year/first semester/deep learning/data"

    print("main3 starts")

    base = dict(
        data_dir=DATA_DIR,
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

    print(run_many_experiments(configs, results_root="results"), flush=True)
