from experiment_runner import ExperimentConfig, run_many_experiments

if __name__ == "__main__":
    DATA_DIR = r"/Users/noalevy/Desktop/Desktop - Noa’s MacBook Pro/School/MTA/third year/first semester/deep learning/data"

    print("starting main", flush=True)
    configs = [
        ExperimentConfig(data_dir=DATA_DIR, model="simple",    optimizer="adam", lr=1e-3, epochs=20, batch_size=16, seed=42),
        ExperimentConfig(data_dir=DATA_DIR, model="simple_bn", optimizer="adam", lr=1e-3, epochs=20, batch_size=16, seed=42),
    ]

    summary = run_many_experiments(configs, results_root="results")
    print(summary)
