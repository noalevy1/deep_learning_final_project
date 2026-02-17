from pathlib import Path
from experiment_runner import ExperimentConfig, run_many_experiments, get_project_root

if __name__ == "__main__":
    project_root = get_project_root()
    data_dir = project_root / "data"
    results_root = project_root / "results" / "step2_batchnorm"

    print("starting step 2 (batchnorm)", flush=True)

    configs = [
        ExperimentConfig(data_dir=str(data_dir), model="simple",    optimizer="adam", lr=1e-3, epochs=20, batch_size=16, seed=42),
        ExperimentConfig(data_dir=str(data_dir), model="simple_bn", optimizer="adam", lr=1e-3, epochs=20, batch_size=16, seed=42),
    ]

    summary = run_many_experiments(configs, results_root=str(results_root))
    print(summary)
