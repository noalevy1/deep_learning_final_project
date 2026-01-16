from __future__ import annotations
from pathlib import Path
import json
import csv
import argparse
from datetime import datetime

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from data import get_dataloaders
from models import SimpleCNN, DeeperCNN


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


def plot_history(history, out_dir: Path, title: str):
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


def build_model(name: str, num_classes: int, img_size: int):
    name = name.lower()
    if name == "simple":
        return SimpleCNN(num_classes=num_classes, img_size=img_size)
    if name == "deeper":
        return DeeperCNN(num_classes=num_classes, img_size=img_size)
    raise ValueError("model must be: simple | deeper")


def build_optimizer(name: str, params, lr: float, weight_decay: float, momentum: float):
    name = name.lower()
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
    raise ValueError("optimizer must be: adam | sgd")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument("--model", type=str, choices=["simple", "deeper"], default="simple")
    p.add_argument("--optimizer", type=str, choices=["adam", "sgd"], default="adam")
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--normalize", type=str, choices=["none", "imagenet"], default="none")
    args = p.parse_args()

    set_seed(args.seed)
    device = get_device()

    normalize = None if args.normalize == "none" else args.normalize

    train_loader, val_loader, test_loader, class_names = get_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        img_size=args.img_size,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        seed=args.seed,
        normalize=normalize,   # <- דורש את שינוי data.py
    )

    num_classes = len(class_names)
    model = build_model(args.model, num_classes=num_classes, img_size=args.img_size).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(
        args.optimizer,
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        momentum=args.momentum,
    )

    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = (
        f"{ts}_model-{args.model}_opt-{args.optimizer}_lr-{args.lr}"
        f"_bs-{args.batch_size}_wd-{args.weight_decay}_norm-{args.normalize}"
    )
    run_dir = Path("results") / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    config = vars(args) | {"class_names": class_names, "device": str(device), "run_name": run_name}
    (run_dir / "config.json").write_text(json.dumps(config, indent=2))

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = -1.0
    best_path = run_dir / "best.pt"

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, optimizer=None, device=device, train=False)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(
            f"epoch {epoch:02d}/{args.epochs:02d} | "
            f"train loss {train_loss:.4f} acc {train_acc:.4f} | "
            f"val loss {val_loss:.4f} acc {val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "img_size": args.img_size,
                    "class_names": class_names,
                    "best_val_acc": best_val_acc,
                    "config": config,
                },
                best_path,
            )
            print(f"saved {best_path} (best_val_acc={best_val_acc:.4f})")

    # test עם המודל הכי טוב
    ckpt = torch.load(best_path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)

    test_loss, test_acc = run_epoch(model, test_loader, criterion, optimizer=None, device=device, train=False)
    print(f"test loss {test_loss:.4f} acc {test_acc:.4f}")
    (run_dir / "test_metrics.json").write_text(json.dumps({"test_loss": test_loss, "test_acc": test_acc}, indent=2))

    save_history_csv(history, run_dir / "history.csv")
    title = f"{args.model} | {args.optimizer} | lr={args.lr} | bs={args.batch_size} | wd={args.weight_decay}"
    plot_history(history, run_dir, title=title)

    print(f"saved run to: {run_dir}")


if __name__ == "__main__":
    main()
