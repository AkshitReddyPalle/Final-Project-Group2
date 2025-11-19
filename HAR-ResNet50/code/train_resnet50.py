
---

### `HAR-ResNet50/code/train_resnet50.py`

```python
import os
import argparse
import random
import numpy as np

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

from sklearn.metrics import classification_report, confusion_matrix


# ---------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------
def set_seed(seed: int = 42) -> None:
    """
    Set random seeds for Python, NumPy and PyTorch to improve reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # For full determinism (slower but reproducible)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------
def build_model(num_classes: int, fine_tune_last_block: bool = True) -> nn.Module:
    """
    Load a pretrained ResNet-50 and adapt the final layer for our number of classes.

    If fine_tune_last_block is True, only the last block + FC are trainable;
    earlier layers are frozen.
    """
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

    # Freeze all parameters by default
    for param in model.parameters():
        param.requires_grad = False

    if fine_tune_last_block:
        # Unfreeze the last layer4 block and the fully-connected layer
        for param in model.layer4.parameters():
            param.requires_grad = True
    else:
        # Unfreeze everything if we truly want full fine-tuning
        for param in model.parameters():
            param.requires_grad = True

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    return model


# ---------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------
def create_dataloaders(
    data_dir: str,
    batch_size: int = 32,
    num_workers: int = 4,
) -> tuple[dict, list[str]]:
    """
    Create train and validation dataloaders from an ImageFolder structure.

    Expected directory structure:
        data_dir/
            train/
                class_1/
                class_2/
                ...
            val/
                class_1/
                class_2/
                ...
    """
    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")

    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    val_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(val_dir, transform=val_transform)

    class_names = train_dataset.classes

    dataloaders = {
        "train": DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        ),
        "val": DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        ),
    }

    return dataloaders, class_names


# ---------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------
def train_model(
    model: nn.Module,
    dataloaders: dict,
    device: torch.device,
    num_epochs: int,
    lr: float,
    step_size: int,
    gamma: float,
    class_names: list[str],
    save_path: str,
) -> None:
    """
    Train the model, track best validation accuracy, and save the best weights.
    Also prints a classification report + confusion matrix for the validation set.
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)

    model.to(device)

    best_acc = 0.0
    best_state_dict = None

    for epoch in range(1, num_epochs + 1):
        print(f"\nEpoch {epoch}/{num_epochs}")
        print("-" * 30)

        for phase in ["train", "val"]:
            if phase == "train":
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0
            running_total = 0

            all_labels = []
            all_preds = []

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels).item()
                running_total += labels.size(0)

                if phase == "val":
                    all_labels.extend(labels.cpu().numpy().tolist())
                    all_preds.extend(preds.cpu().numpy().tolist())

            epoch_loss = running_loss / running_total
            epoch_acc = running_corrects / running_total

            if phase == "train":
                print(f"train Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.4f}")
                scheduler.step()
            else:
                print(f"val   Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.4f}")

                # Track best model
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_state_dict = model.state_dict()
                    torch.save(best_state_dict, save_path)

        # End epoch

    print(f"\nBest Validation Accuracy: {best_acc}")

    # Load best weights for evaluation
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    # Final evaluation on validation set
    model.eval()
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for inputs, labels in dataloaders["val"]:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            all_labels.extend(labels.cpu().numpy().tolist())
            all_preds.extend(preds.cpu().numpy().tolist())

    print("\nClassification Report:")
    print(
        classification_report(
            all_labels,
            all_preds,
            target_names=class_names,
            digits=2,
        )
    )

    print("\nConfusion Matrix:")
    print(confusion_matrix(all_labels, all_preds))


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune ResNet-50 for Human Activity Recognition."
    )

    parser.add_argument(
        "--data_dir",
        type=str,
        default="../data",
        help="Root directory containing 'train' and 'val' subfolders.",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="../model",
        help="Directory where the best model checkpoint will be saved.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for training and validation.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=15,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Initial learning rate.",
    )
    parser.add_argument(
        "--step_size",
        type=int,
        default=5,
        help="Step size for StepLR scheduler.",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.1,
        help="Gamma (decay factor) for StepLR scheduler.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of subprocesses for data loading.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Computation device. 'auto' = use CUDA if available.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--fine_tune_last_block",
        action="store_true",
        help="If set, only the last ResNet block + FC will be trainable.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Decide device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(f"Using device: {device}")

    # Set random seeds
    set_seed(args.seed)

    # Create dataloaders
    dataloaders, class_names = create_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    print("Classes:", class_names)
    print("Number of classes:", len(class_names))

    # Build model
    model = build_model(
        num_classes=len(class_names),
        fine_tune_last_block=args.fine_tune_last_block or True,
    )

    # Prepare save path
    os.makedirs(args.save_dir, exist_ok=True)
    save_path = os.path.join(args.save_dir, "resnet50_har_best_finetuned.pth")

    # Train
    train_model(
        model=model,
        dataloaders=dataloaders,
        device=device,
        num_epochs=args.epochs,
        lr=args.lr,
        step_size=args.step_size,
        gamma=args.gamma,
        class_names=class_names,
        save_path=save_path,
    )


if __name__ == "__main__":
    main()
