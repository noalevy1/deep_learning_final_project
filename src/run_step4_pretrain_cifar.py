import ssl
import certifi
ssl._create_default_https_context = lambda: ssl.create_default_context(cafile=certifi.where())

import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from models import SimpleCNN_BN
from experiment_runner import get_device, set_seed, run_epoch

def main():
    print("Step 4: Pretraining on CIFAR-10")

    set_seed(42)
    device = get_device()

    # CIFAR-10 transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    train_ds = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
    test_ds  = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)

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

    # save pretrained weights
    torch.save(model.state_dict(), "pretrained_cifar10.pt")
    print("Saved pretrained weights to pretrained_cifar10.pt")

if __name__ == "__main__":
    main()
