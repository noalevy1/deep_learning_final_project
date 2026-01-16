from pathlib import Path
from collections import Counter
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

def _stratified_split_indices(labels, train_ratio, val_ratio, test_ratio, seed):
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
    g = torch.Generator().manual_seed(seed)
    labels_t = torch.tensor(labels, dtype=torch.long)
    indices_by_class = {}
    for idx, y in enumerate(labels):
        indices_by_class.setdefault(int(y), []).append(idx)

    train_idx, val_idx, test_idx = [], [], []
    for c, idxs in indices_by_class.items():
        idxs_t = torch.tensor(idxs, dtype=torch.long)
        perm = idxs_t[torch.randperm(len(idxs_t), generator=g)].tolist()
        n = len(perm)
        n_train = int(round(n * train_ratio))
        n_val = int(round(n * val_ratio))
        n_train = min(n_train, n)
        n_val = min(n_val, n - n_train)
        n_test = n - n_train - n_val
        train_idx.extend(perm[:n_train])
        val_idx.extend(perm[n_train:n_train + n_val])
        test_idx.extend(perm[n_train + n_val:n_train + n_val + n_test])

    def _shuffle(lst):
        t = torch.tensor(lst, dtype=torch.long)
        return t[torch.randperm(len(t), generator=g)].tolist()

    return _shuffle(train_idx), _shuffle(val_idx), _shuffle(test_idx)

def get_transforms(img_size, normalize=None, augment: str = "none"):
    imagenet_mean = (0.485, 0.456, 0.406)
    imagenet_std  = (0.229, 0.224, 0.225)

    norm_layer = []
    if normalize == "imagenet":
        norm_layer = [transforms.Normalize(imagenet_mean, imagenet_std)]

    # ---- TRAIN transforms ----
    train_ops = [transforms.Resize((img_size, img_size))]

    if augment == "none":
        pass
    elif augment == "strong":
        train_ops += [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02),
        ]
    else:
        raise ValueError("augment must be 'none' or 'strong'")

    train_ops += [
        transforms.ToTensor(),
        *norm_layer,
    ]
    train_tfms = transforms.Compose(train_ops)

    # ---- EVAL transforms (val/test) ----
    eval_tfms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        *norm_layer,
    ])

    return train_tfms, eval_tfms


def get_dataloaders(
    data_dir,
    batch_size=16,
    img_size=224,
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15,
    seed=42,
    num_workers=0,
    pin_memory=False,
    normalize=None,
    augment: str = "none"
):
    data_dir = Path(data_dir)
    train_tfms, eval_tfms = get_transforms(img_size, normalize=normalize, augment=augment)

    base_ds = datasets.ImageFolder(data_dir, transform=eval_tfms)
    labels = [y for _, y in base_ds.samples]
    train_idx, val_idx, test_idx = _stratified_split_indices(labels, train_ratio, val_ratio, test_ratio, seed)

    train_ds = Subset(datasets.ImageFolder(data_dir, transform=train_tfms), train_idx)
    val_ds = Subset(datasets.ImageFolder(data_dir, transform=eval_tfms), val_idx)
    test_ds = Subset(datasets.ImageFolder(data_dir, transform=eval_tfms), test_idx)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

    class_names = base_ds.classes
    return train_loader, val_loader, test_loader, class_names

def split_summary(train_loader, val_loader, test_loader, class_names):
    def _count(loader):
        c = Counter()
        for _, y in loader:
            for t in y.tolist():
                c[int(t)] += 1
        return {class_names[k]: v for k, v in sorted(c.items())}

    train_counts = _count(train_loader)
    val_counts = _count(val_loader)
    test_counts = _count(test_loader)

    print(train_counts)
    print(val_counts)
    print(test_counts)
    print(sum(train_counts.values()), sum(val_counts.values()), sum(test_counts.values()))
