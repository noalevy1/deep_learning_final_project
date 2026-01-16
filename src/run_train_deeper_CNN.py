from pathlib import Path
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from data import get_dataloaders
from models import DeeperCNN


DATA_DIR = Path("/Users/noalevy/Desktop/Desktop - Noa’s MacBook Pro/School/MTA/third year/first semester/deep learning/data")
IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 20
LR = 1e-3
WEIGHT_DECAY = 0.0
SEED = 42

SAVE_DIR = Path("results")
SAVE_DIR.mkdir(parents=True, exist_ok=True)
BEST_PATH = SAVE_DIR / "deepercnn_best.pt"


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
    if train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_correct = 0
    total_n = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        if train and optimizer is not None:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(train):
            logits = model(x)
            loss = criterion(logits, y)
            if train and optimizer is not None:
                loss.backward()
                optimizer.step()

        bs = y.size(0)
        total_loss += loss.item() * bs
        total_correct += (logits.argmax(dim=1) == y).sum().item()
        total_n += bs

    return total_loss / total_n, total_correct / total_n


def plot_history(history, out_path: Path, prefix: str):
    epochs = list(range(1, len(history["train_loss"]) + 1))

    plt.figure()
    plt.plot(epochs, history["train_loss"])
    plt.plot(epochs, history["val_loss"])
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(["train", "val"])
    plt.title("Loss")
    plt.savefig(out_path / f"{prefix}_loss.png", dpi=150, bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.plot(epochs, history["train_acc"])
    plt.plot(epochs, history["val_acc"])
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend(["train", "val"])
    plt.title("Accuracy")
    plt.savefig(out_path / f"{prefix}_accuracy.png", dpi=150, bbox_inches="tight")
    plt.close()


def main():
    set_seed(SEED)
    device = get_device()
    print(device)

    train_loader, val_loader, test_loader, class_names = get_dataloaders(
        data_dir=DATA_DIR,
        batch_size=BATCH_SIZE,
        img_size=IMG_SIZE,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        seed=SEED,
    )

    num_classes = len(class_names)
    model = DeeperCNN(num_classes=num_classes, img_size=IMG_SIZE).to(device)
    print(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = -1.0

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, optimizer=None, device=device, train=False)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(
            f"epoch {epoch:02d}/{EPOCHS:02d} | "
            f"train loss {train_loss:.4f} acc {train_acc:.4f} | "
            f"val loss {val_loss:.4f} acc {val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "img_size": IMG_SIZE,
                    "class_names": class_names,
                    "best_val_acc": best_val_acc,
                },
                BEST_PATH,
            )
            print(f"saved {BEST_PATH} (best_val_acc={best_val_acc:.4f})")

    ckpt = torch.load(BEST_PATH, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)

    test_loss, test_acc = run_epoch(model, test_loader, criterion, optimizer=None, device=device, train=False)
    print(f"test loss {test_loss:.4f} acc {test_acc:.4f}")

    plot_history(history, SAVE_DIR, prefix="deepercnn")
    print(str(SAVE_DIR / "deepercnn_loss.png"))
    print(str(SAVE_DIR / "deepercnn_accuracy.png"))


if __name__ == "__main__":
    main()
