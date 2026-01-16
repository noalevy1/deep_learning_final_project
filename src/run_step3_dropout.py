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
        weight_decay=1e-4,
        augment="none",
    )

    configs = [
        ExperimentConfig(data_dir=DATA_DIR, model="simple_bn", optimizer="adam", lr=1e-3, weight_decay=1e-4,
                         dropout_p=0.0, augment="none"),
        ExperimentConfig(data_dir=DATA_DIR, model="simple_bn", optimizer="adam", lr=1e-3, weight_decay=1e-4,
                         dropout_p=0.2, augment="none"),
        ExperimentConfig(data_dir=DATA_DIR, model="simple_bn", optimizer="adam", lr=1e-3, weight_decay=1e-4,
                         dropout_p=0.5, augment="none"),
    ]

    print(run_many_experiments(configs, results_root="results"), flush=True)
