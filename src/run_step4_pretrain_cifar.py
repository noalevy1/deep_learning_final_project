import ssl
import certifi
ssl._create_default_https_context = lambda: ssl.create_default_context(cafile=certifi.where())

from pathlib import Path
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from models import SimpleCNN_BN
from experiment_runner import get_device, set_seed, run_epoch, get_project_root


def main():
    print("Step 4: Pretraining on CIFAR-10")

    project_root = get_project_root()
    results_root = project_root / "results" / "step4_pretrain_cifar"
    results_root.mkdir(parents=True, exist_ok=True)

    set_seed(42)
    device = get_device()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    cifar_root = project_root / "data" / "cifar10"

    train_ds = datasets.CIFAR10(root=str(cifar_root), train=True, download=True, transform=transform)
    test_ds  = datasets.CIFAR10(root=str(cifar_root), train=False, download=True, transform=transform)

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    test_loader  = DataLoader(test_ds, batch_size=64, shuffle=False)

    model = SimpleCNN_BN(
        num_classes=10,
        img_size=224,
        dropout_p=0.2
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    epochs = 10
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = run_epoch(
            model, train_loader, criterion, optimizer, device, train=True
        )
        val_loss, val_acc = run_epoch(
            model, test_loader, criterion, optimizer=None, device=device, train=False
        )

        print(
            f"epoch {epoch:02d}/{epochs} | "
            f"train acc {train_acc:.4f} | val acc {val_acc:.4f}"
        )

    ckpt_path = results_root / "pretrained_cifar10.pt"
    torch.save(model.state_dict(), ckpt_path)
    print(f"Saved pretrained weights to {ckpt_path}")


if __name__ == "__main__":
    main()
