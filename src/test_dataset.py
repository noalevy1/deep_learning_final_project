from pathlib import Path
from data import get_dataloaders, split_summary

DATA_DIR = Path("/Users/noalevy/Desktop/Desktop - Noa’s MacBook Pro/School/MTA/third year/first semester/deep learning/data")

if __name__ == "__main__":
    train_loader, val_loader, test_loader, class_names = get_dataloaders(
        data_dir=DATA_DIR,
        batch_size=16,
        img_size=224,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        seed=42,
    )

    print(class_names)
    split_summary(train_loader, val_loader, test_loader, class_names)