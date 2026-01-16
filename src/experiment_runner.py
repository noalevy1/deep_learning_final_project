
from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
from datetime import datetime
import json
import csv

import torch
import torch.nn as nn
#import matplotlib.pyplot as plt

from data import get_dataloaders
from models import SimpleCNN, SimpleCNN_BN, DeeperCNN


@dataclass
class ExperimentConfig:
    data_dir: str
    model: str                 # "simple" | "deeper"
    optimizer: str             # "adam" | "sgd"
    img_size: int = 224
    batch_size: int = 16
    epochs: int = 20
    lr: float = 1e-3
    weight_decay: float = 0.0
    momentum: float = 0.9
    seed: int = 42
    normalize: str = "none"    # "none" | "imagenet"
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    dropout_p: float = 0.2
    augment: str = "none"   # "none" | "strong"
    pretrained_path: str | None = None


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model(name: str, num_classes: int, img_size: int, dropout_p: float):
    name = name.lower()
    if name == "simple":
        return SimpleCNN(num_classes=num_classes, img_size=img_size, dropout_p=dropout_p)
    if name == "simple_bn":
        return SimpleCNN_BN(num_classes=num_classes, img_size=img_size, dropout_p=dropout_p)
    if name == "deeper":
        return DeeperCNN(num_classes=num_classes, img_size=img_size, dropout_p=dropout_p)
    raise ValueError("model must be: simple | deeper")


def build_optimizer(name: str, params, lr: float, weight_decay: float, momentum: float):
    name = name.lower()
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
    raise ValueError("optimizer must be: adam | sgd")


def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train() if train else model.eval()

    total_loss = 0.0
    total_correct = 0
    total_n = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        if train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(train):
            logits = model(x)
            loss = criterion(logits, y)
            if train:
                loss.backward()
                optimizer.step()

        bs = y.size(0)
        total_loss += loss.item() * bs
        total_correct += (logits.argmax(dim=1) == y).sum().item()
        total_n += bs

    return total_loss / total_n, total_correct / total_n


def save_history_csv(history, out_path: Path):
    with out_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc"])
        for i in range(len(history["train_loss"])):
            w.writerow([
                i + 1,
                history["train_loss"][i],
                history["train_acc"][i],
                history["val_loss"][i],
                history["val_acc"][i],
            ])


def plot_history(history, out_dir: Path, title: str):
    import matplotlib.pyplot as plt
    epochs = list(range(1, len(history["train_loss"]) + 1))

    plt.figure()
    plt.plot(epochs, history["train_loss"])
    plt.plot(epochs, history["val_loss"])
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(["train", "val"])
    plt.title(f"Loss — {title}")
    plt.savefig(out_dir / "loss.png", dpi=150, bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.plot(epochs, history["train_acc"])
    plt.plot(epochs, history["val_acc"])
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend(["train", "val"])
    plt.title(f"Accuracy — {title}")
    plt.savefig(out_dir / "accuracy.png", dpi=150, bbox_inches="tight")
    plt.close()


def run_single_experiment(cfg: ExperimentConfig, results_root: str = "results"):
    set_seed(cfg.seed)
    device = get_device()

    normalize = None if cfg.normalize == "none" else cfg.normalize

    train_loader, val_loader, test_loader, class_names = get_dataloaders(
        data_dir=cfg.data_dir,
        batch_size=cfg.batch_size,
        img_size=cfg.img_size,
        train_ratio=cfg.train_ratio,
        val_ratio=cfg.val_ratio,
        test_ratio=cfg.test_ratio,
        seed=cfg.seed,
        normalize=normalize,
        augment=cfg.augment
    )

    num_classes = len(class_names)
    model = build_model(cfg.model, num_classes=num_classes, img_size=cfg.img_size, dropout_p=cfg.dropout_p).to(device)

    # ---- load pretrained weights (Step 5) ----
    if cfg.pretrained_path:
        ckpt = torch.load(cfg.pretrained_path, map_location="cpu")
        state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt

        model_state = model.state_dict()
        filtered = {k: v for k, v in state.items() if k in model_state and v.shape == model_state[k].shape}
        missing, unexpected = model.load_state_dict(filtered, strict=False)

        print(f"Loaded pretrained weights from {cfg.pretrained_path}")
        print(f" - loaded: {len(filtered)} tensors")
        print(f" - missing keys: {len(missing)}")
        print(f" - unexpected keys: {len(unexpected)}")

    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(cfg.optimizer, model.parameters(), cfg.lr, cfg.weight_decay, cfg.momentum)

    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = (
        f"{ts}_step3_model-{cfg.model}_opt-{cfg.optimizer}_lr-{cfg.lr}"
        f"_bs-{cfg.batch_size}_wd-{cfg.weight_decay}"
        f"_drop-{cfg.dropout_p}_aug-{cfg.augment}_norm-{cfg.normalize}"
        f"_pre-{Path(cfg.pretrained_path).stem}" if cfg.pretrained_path else ""
    )
    run_dir = Path(results_root) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    # save config
    config_dict = asdict(cfg) | {"class_names": class_names, "device": str(device), "run_name": run_name}
    (run_dir / "config.json").write_text(json.dumps(config_dict, indent=2))

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = -1.0
    best_path = run_dir / "best.pt"

    for epoch in range(1, cfg.epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, optimizer=None, device=device, train=False)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(
            f"[{run_name}] epoch {epoch:02d}/{cfg.epochs:02d} | "
            f"train loss {train_loss:.4f} acc {train_acc:.4f} | "
            f"val loss {val_loss:.4f} acc {val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "best_val_acc": best_val_acc,
                    "class_names": class_names,
                    "config": config_dict,
                },
                best_path,
            )

    # evaluate best on test
    ckpt = torch.load(best_path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)

    test_loss, test_acc = run_epoch(model, test_loader, criterion, optimizer=None, device=device, train=False)
    (run_dir / "test_metrics.json").write_text(json.dumps({"test_loss": test_loss, "test_acc": test_acc}, indent=2))

    save_history_csv(history, run_dir / "history.csv")
    title = f"{cfg.model} | {cfg.optimizer} | lr={cfg.lr} | bs={cfg.batch_size} | wd={cfg.weight_decay}"
    plot_history(history, run_dir, title=title)

    print(f"saved run to: {run_dir}")
    return {"run_dir": str(run_dir), "best_val_acc": best_val_acc, "test_acc": test_acc}


def run_many_experiments(configs: list[ExperimentConfig], results_root: str = "results"):
    summary = []
    for cfg in configs:
        res = run_single_experiment(cfg, results_root=results_root)
        summary.append({
            "run_dir": res["run_dir"],
            "model": cfg.model,
            "optimizer": cfg.optimizer,
            "lr": cfg.lr,
            "batch_size": cfg.batch_size,
            "weight_decay": cfg.weight_decay,
            "best_val_acc": res["best_val_acc"],
            "test_acc": res["test_acc"],
            "dropout_p": cfg.dropout_p,
            "augment": cfg.augment,
        })

    # save a global summary csv
    out = Path(results_root) / "summary.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)

    return summary

if __name__ == "__main__":
    DATA_DIR = r"/Users/noalevy/Desktop/Desktop - Noa’s MacBook Pro/School/MTA/third year/first semester/deep learning/data"

    print("start running experiments")

    configs = [
        ExperimentConfig(data_dir=DATA_DIR, model="simple", optimizer="adam", lr=1e-3, epochs=20, batch_size=16),
        ExperimentConfig(data_dir=DATA_DIR, model="simple", optimizer="adam", lr=3e-4, epochs=20, batch_size=16),

        ExperimentConfig(data_dir=DATA_DIR, model="simple", optimizer="sgd", lr=1e-2, epochs=20, batch_size=16,
                         momentum=0.9),
        ExperimentConfig(data_dir=DATA_DIR, model="simple", optimizer="sgd", lr=3e-3, epochs=20, batch_size=16,
                         momentum=0.9),

        ExperimentConfig(data_dir=DATA_DIR, model="deeper", optimizer="adam", lr=1e-3, epochs=20, batch_size=16),
        ExperimentConfig(data_dir=DATA_DIR, model="deeper", optimizer="adam", lr=3e-4, epochs=20, batch_size=16),

        ExperimentConfig(data_dir=DATA_DIR, model="deeper", optimizer="sgd", lr=1e-2, epochs=20, batch_size=16,
                         momentum=0.9),
        ExperimentConfig(data_dir=DATA_DIR, model="deeper", optimizer="sgd", lr=3e-3, epochs=20, batch_size=16,
                         momentum=0.9),
    ]

    summary = run_many_experiments(configs, results_root="results")
    summary
