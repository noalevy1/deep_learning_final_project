if __name__ == "__main__":
    from experiment_runner import ExperimentConfig, run_many_experiments


    configs = [
        ExperimentConfig(
            model="simple_bn",
            optimizer="adam",
            lr=1e-3,
            epochs=20,
            batch_size=16,
            weight_decay=1e-4,
            dropout_p=0.2,  # keep constant
            augment="none"  # baseline
        ),
        ExperimentConfig(
            data_dir=DATA_DIR,
            model="simple_bn",
            optimizer="adam",
            lr=1e-3,
            epochs=20,
            batch_size=16,
            weight_decay=1e-4,
            dropout_p=0.2,  # keep constant
            augment="strong"  # augmentation experiment
        ),
    ]

    summary = run_many_experiments(configs, results_root="results")
    print(summary)