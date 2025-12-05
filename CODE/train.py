import os
import argparse
from pathlib import Path
import numpy as np  # for mixup
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm
import timm

# DEFAULTS (EfficientNet-B1 + size 240)
DEFAULT_MODEL_NAME = "efficientnet_b1"
DEFAULT_IMG_SIZE = 240

# Mixup and TTA config
MIXUP_ALPHA = 0.2       # 0.0 disables mixup
USE_TTA_TEST = True     # Test-time augmentation: avg(original, flipped)

def parse_args():
    parser = argparse.ArgumentParser(description="Train HAR model (EfficientNet-B1)")

    parser.add_argument(
        "--data_dir",
        type=str,
        default="/home/ubuntu/DLGP/Data",
        help="Base data directory that contains train/ and Training_set.csv",
    )

    parser.add_argument(
        "--model_name",
        type=str,
        default=DEFAULT_MODEL_NAME,
        help="Model architecture name (default: efficientnet_b1).",
    )

    parser.add_argument(
        "--img_size",
        type=int,
        default=DEFAULT_IMG_SIZE,
        help="Input image size (default: 240 for EfficientNet-B1).",
    )

    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--lr", type=float, default=1e-4)

    # Gradual unfreezing config
    parser.add_argument(
        "--freeze_backbone_epochs",
        type=int,
        default=0,
        help="If > 0, freeze backbone for this many epochs, then unfreeze and fine-tune.",
    )
    parser.add_argument(
        "--finetune_lr",
        type=float,
        default=3e-5,
        help="Learning rate to use after unfreezing backbone.",
    )

    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--test_ratio", type=float, default=0.15)
    parser.add_argument("--num_workers", type=int, default=4)

    parser.add_argument("--freeze_backbone", action="store_true")
    parser.add_argument("--no_pretrained", action="store_true")
    parser.add_argument("--cpu", action="store_true")

    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs",
        help="Directory to save models, logs, etc.",
    )

    return parser.parse_args()

class HARImageDataset(Dataset):
    """
    Custom dataset for Human Action Recognition.
    Expects a DataFrame with 'filename' and 'label' columns,
    and a root directory containing all images.
    """

    def __init__(self, df, img_root, label_to_idx, transform=None):
        self.df = df.reset_index(drop=True)
        self.img_root = Path(img_root)
        self.label_to_idx = label_to_idx
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = self.img_root / row["filename"]
        image = Image.open(img_path).convert("RGB")
        label = self.label_to_idx[row["label"]]

        if self.transform:
            image = self.transform(image)
        return image, label

def get_transforms(img_size: int):
    """
    Stronger augmentations for train, ImageNet-style preprocessing for val/test.
    """
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(
            img_size,
            scale=(0.6, 1.0),
            ratio=(3/4, 4/3),
        ),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.RandomAffine(
            degrees=0,
            translate=(0.1, 0.1),
        ),
        transforms.RandomPerspective(
            distortion_scale=0.1,
            p=0.3,
        ),
        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.02,
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    eval_transform = transforms.Compose([
        transforms.Resize(int(img_size * 1.15)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    return train_transform, eval_transform

def create_dataloaders(
    data_dir,
    img_size,
    batch_size,
    val_ratio,
    test_ratio,
    num_workers,
):
    """
    Reads Training_set.csv, shuffles, splits into train/val/test, and creates DataLoaders.
    """
    data_dir = Path(data_dir)
    df = pd.read_csv(data_dir / "Training_set.csv")
    img_root = data_dir / "train"

    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)

    unique_labels = sorted(df["label"].unique())
    label_to_idx = {lbl: i for i, lbl in enumerate(unique_labels)}
    idx_to_label = {i: lbl for lbl, i in label_to_idx.items()}

    total = len(df)
    n_test = int(test_ratio * total)
    n_val = int(val_ratio * total)
    n_train = total - n_val - n_test

    df_train = df.iloc[:n_train]
    df_val = df.iloc[n_train:n_train + n_val]
    df_test = df.iloc[n_train + n_val:]

    print(f"Total samples: {total}")
    print(f"Train: {len(df_train)}, Val: {len(df_val)}, Test: {len(df_test)}")

    train_t, eval_t = get_transforms(img_size)

    train_ds = HARImageDataset(df_train, img_root, label_to_idx, train_t)
    val_ds = HARImageDataset(df_val, img_root, label_to_idx, eval_t)
    test_ds = HARImageDataset(df_test, img_root, label_to_idx, eval_t)

    def make_loader(ds, shuffle):
        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
        )

    loaders = {
        "train": make_loader(train_ds, True),
        "val": make_loader(val_ds, False),
        "test": make_loader(test_ds, False),
    }

    return loaders, unique_labels, idx_to_label

def get_model(model_name, num_classes, pretrained=True):
    """
    Create model from timm and replace final layer with Dropout + Linear.
    """
    model = timm.create_model(model_name, pretrained=pretrained)

    if hasattr(model, "classifier") and isinstance(model.classifier, nn.Module):
        in_f = model.classifier.in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_f, num_classes),
        )
    elif hasattr(model, "head") and isinstance(model.head, nn.Module):
        in_f = model.head.in_features
        model.head = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_f, num_classes),
        )
    else:
        raise ValueError("Classifier/head not found in model.")

    return model

def freeze_backbone(model):
    """
    Freeze all layers except classifier/head for feature-extractor training.
    """
    for p in model.parameters():
        p.requires_grad = False

    if hasattr(model, "classifier") and isinstance(model.classifier, nn.Module):
        for p in model.classifier.parameters():
            p.requires_grad = True
    elif hasattr(model, "head") and isinstance(model.head, nn.Module):
        for p in model.head.parameters():
            p.requires_grad = True

def unfreeze_backbone(model):
    """
    Unfreeze all layers (used after initial frozen phase in gradual unfreezing).
    """
    for p in model.parameters():
        p.requires_grad = True

def mixup_data(x, y, alpha=0.2):
    """
    Applies mixup to a batch.
    Returns mixed inputs, paired targets, and lambda.
    """
    if alpha <= 0.0:
        return x, y, y, 1.0

    lam = np.random.beta(alpha, alpha)
    batch_size = x.size(0)
    index = torch.randperm(batch_size, device=x.device)

    mixed_x = lam * x + (1.0 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, preds, y_a, y_b, lam):
    """
    Mixup loss: lam * CE(preds, y_a) + (1-lam) * CE(preds, y_b)
    """
    return lam * criterion(preds, y_a) + (1.0 - lam) * criterion(preds, y_b)

def train_one_epoch(model, loader, criterion, optimizer, device, mixup_alpha=0.0):
    model.train()
    total, correct, loss_sum = 0, 0.0, 0.0

    for X, y in tqdm(loader, desc="Train", leave=False):
        X, y = X.to(device), y.to(device)

        optimizer.zero_grad()

        if mixup_alpha > 0.0:
            X_mixed, y_a, y_b, lam = mixup_data(X, y, mixup_alpha)
            out = model(X_mixed)
            loss = mixup_criterion(criterion, out, y_a, y_b, lam)

            preds = out.argmax(1)
            correct_batch = (
                lam * (preds == y_a).sum().item()
                + (1.0 - lam) * (preds == y_b).sum().item()
            )
            correct += correct_batch
            batch_size = y.size(0)
            total += batch_size
            loss_sum += loss.item() * batch_size
        else:
            out = model(X)
            loss = criterion(out, y)
            preds = out.argmax(1)
            correct += (preds == y).sum().item()
            batch_size = y.size(0)
            total += batch_size
            loss_sum += loss.item() * batch_size

        loss.backward()
        optimizer.step()

    return loss_sum / total, correct / total

def eval_one_epoch(model, loader, criterion, device, desc="Val", tta=False):
    """
    Evaluation loop. If tta=True, we average predictions from original and
    horizontally flipped images (simple TTA).
    """
    model.eval()
    total, correct, loss_sum = 0, 0.0, 0.0
    all_y, all_pred = [], []

    with torch.no_grad():
        for X, y in tqdm(loader, desc=desc, leave=False):
            X, y = X.to(device), y.to(device)

            out = model(X)

            if tta:
                X_flipped = torch.flip(X, dims=[3])
                out_flipped = model(X_flipped)
                out = (out + out_flipped) / 2.0

            loss = criterion(out, y)
            preds = out.argmax(1)

            batch_size = y.size(0)
            loss_sum += loss.item() * batch_size
            correct += (preds == y).sum().item()
            total += batch_size

            all_y.extend(y.cpu().tolist())
            all_pred.extend(preds.cpu().tolist())

    return loss_sum / total, correct / total, all_y, all_pred

def main(args):
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    loaders, class_names, _ = create_dataloaders(
        args.data_dir,
        args.img_size,
        args.batch_size,
        args.val_ratio,
        args.test_ratio,
        args.num_workers,
    )

    print("\nClasses:", class_names)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print("Using:", device)

    model = get_model(
        args.model_name,
        num_classes=len(class_names),
        pretrained=not args.no_pretrained,
    )

    use_gradual_unfreeze = False
    unfrozen = False

    if args.freeze_backbone:
        print("Freezing backbone for all epochs (no unfreezing).")
        freeze_backbone(model)
    elif args.freeze_backbone_epochs > 0:
        print(
            f"Using gradual unfreezing: freezing backbone for "
            f"first {args.freeze_backbone_epochs} epoch(s)."
        )
        freeze_backbone(model)
        use_gradual_unfreeze = True

    model = model.to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=1e-4,
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.1, patience=3
    )

    best_val = 0.0
    ckpt = out_dir / f"{args.model_name}_best.pth"
    logs = []

    for epoch in range(args.epochs):
        if use_gradual_unfreeze and (epoch == args.freeze_backbone_epochs) and not unfrozen:
            print(
                f"\nUnfreezing backbone at epoch {epoch + 1} and "
                f"switching to fine-tuning lr={args.finetune_lr:.1e}"
            )
            unfreeze_backbone(model)
            optimizer = optim.Adam(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=args.finetune_lr,
                weight_decay=1e-4,
            )
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=0.1, patience=3
            )
            unfrozen = True

        print(f"\nEpoch {epoch + 1}/{args.epochs}")

        train_loss, train_acc = train_one_epoch(
            model,
            loaders["train"],
            criterion,
            optimizer,
            device,
            mixup_alpha=MIXUP_ALPHA,
        )
        val_loss, val_acc, _, _ = eval_one_epoch(
            model, loaders["val"], criterion, device, desc="Val", tta=False
        )

        scheduler.step(val_loss)

        print(f"Train Loss {train_loss:.4f} | Acc {train_acc:.4f}")
        print(f"Val   Loss {val_loss:.4f} | Acc {val_acc:.4f}")

        logs.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
        })

        if val_acc > best_val:
            best_val = val_acc
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch + 1,
                    "val_acc": best_val,
                    "class_names": class_names,
                },
                ckpt,
            )
            print(f"Saved best model: val_acc={best_val:.4f}")

    pd.DataFrame(logs).to_csv(
        out_dir / f"training_log_{args.model_name}.csv", index=False
    )

    print("\nLoading best model for final test...")
    checkpoint = torch.load(ckpt, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_loss, test_acc, all_y, all_pred = eval_one_epoch(
        model,
        loaders["test"],
        criterion,
        device,
        desc="Test",
        tta=USE_TTA_TEST,
    )

    print(f"\nTest Loss {test_loss:.4f} | Test Acc {test_acc:.4f}")

    rep = classification_report(
        all_y,
        all_pred,
        target_names=class_names,
        output_dict=True,
    )
    pd.DataFrame(rep).transpose().to_csv(
        out_dir / f"results_{args.model_name}.csv"
    )

    cm = confusion_matrix(all_y, all_pred)
    pd.DataFrame(cm, index=class_names, columns=class_names).to_csv(
        out_dir / f"confusion_matrix_{args.model_name}.csv"
    )

    print("Done.")

if __name__ == "__main__":
    args = parse_args()
    main(args)


