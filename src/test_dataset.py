from pathlib import Path
from data import get_dataloaders, split_summary


def get_project_root() -> Path:
    return Path(__file__).resolve().parent.parent


project_root = get_project_root()
DATA_DIR = project_root / "data"

if __name__ == "__main__":
    train_loader, val_loader, test_loader, class_names = get_dataloaders(
        data_dir=str(DATA_DIR),
        batch_size=16,
        img_size=224,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        seed=42,
    )

    print(class_names)
    split_summary(train_loader, val_loader, test_loader, class_names)
