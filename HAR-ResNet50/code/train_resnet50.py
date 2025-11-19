import os
import time
import copy
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torchvision import datasets, transforms, models
from sklearn.metrics import classification_report, confusion_matrix


def build_dataloaders(data_dir: str,
                      batch_size: int = 32,
                      num_workers: int = 2,
                      image_size: int = 224):
    """
    Build train / val dataloaders from an ImageFolder-style directory.

    Expected structure:

        data/
          train/
            class1/ img1.jpg ...
          val/
            class1/ ...

    """
    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")

    if not os.path.isdir(train_dir) or not os.path.isdir(val_dir):
        raise FileNotFoundError(
            f"Could not find train/val folders under {data_dir}. "
            "Expected data/train and data/val."
        )

    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(val_dir, transform=val_transform)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    class_names = train_dataset.classes
    return {"train": train_loader, "val": val_loader}, class_names


def build_model(num_classes: int, fine_tune_last_block: bool = True) -> nn.Module:
    """
    Create a ResNet-50 model pre-trained on ImageNet and adapt the final layer
    for our number of classes.

    If fine_tune_last_block=True, only layer4 + fc are unfrozen; everything
    else stays frozen.
    """
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

    # Freeze all parameters first
    for param in model.parameters():
        param.requires_grad = False

    if fine_tune_last_block:
        for name, param in model.named_parameters():
            if "layer4" in name or "fc" in name:
                param.requires_grad = True
    else:
        # Unfreeze everything for full fine-tuning
        for param in model.parameters():
            param.requires_grad = True

    # Replace final classification head
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def train_model(model: nn.Module,
                dataloaders,
                device,
                num_epochs: int = 15,
                lr: float = 1e-3,
                step_size: int = 5,
                gamma: float = 0.1,
                class_names=None,
                save_path: str = "resnet50_har_best_finetuned.pth"):

    criterion = nn.CrossEntropyLoss()

    # Only optimize the parameters that require gradients
    params_to_update = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(params_to_update, lr=lr)

    scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)

    model = model.to(device)

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    since = time.time()

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

            all_labels = []
            all_preds = []

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels).item()

                all_labels.extend(labels.detach().cpu().numpy().tolist())
                all_preds.extend(preds.detach().cpu().numpy().tolist())

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects / len(dataloaders[phase].dataset)

            print(f"{phase} Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.4f}")

            if phase == "val":
                scheduler.step()

                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    # Ensure directory exists
                    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
                    torch.save(best_model_wts, save_path)
                    print(f"  🔥 New best model saved to {save_path}")

    time_elapsed = time.time() - since
    print(f"\nTraining complete in {time_elapsed/60:.1f} min")
    print(f"Best Validation Accuracy: {best_acc:.4f}")

    # Load best weights
    model.load_state_dict(best_model_wts)

    # Final evaluation with classification report + confusion matrix on val set
    if "val" in dataloaders and class_names is not None:
        model.eval()
        all_labels = []
        all_preds = []
        with torch.no_grad():
            for inputs, labels in dataloaders["val"]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                all_labels.extend(labels.detach().cpu().numpy().tolist())
                all_preds.extend(preds.detach().cpu().numpy().tolist())

        print("\nClassification Report:")
        print(classification_report(all_labels, all_preds, target_names=class_names))

        cm = confusion_matrix(all_labels, all_preds)
        print("\nConfusion Matrix:")
        print(cm)

    return model


def main():
    parser = argparse.ArgumentParser(
        description="Train ResNet-50 on Human Action Recognition dataset."
    )
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Path to data folder containing train/ and val/ subfolders.")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size.")
    parser.add_argument("--num_workers", type=int, default=2, help="DataLoader workers.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--save_dir", type=str, default="model",
                        help="Directory to save best model .pth file.")
    parser.add_argument("--fine_tune_last_block", action="store_true",
                        help="Only fine-tune ResNet layer4 + fc instead of full network.")

    args = parser.parse_args()

    # Resolve data_dir
    if args.data_dir is None:
        # Assume script lives in code/ and data/ is sibling of that folder
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        data_dir = os.path.join(project_root, "data")
    else:
        data_dir = args.data_dir

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataloaders, class_names = build_dataloaders(
        data_dir=data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    print("Classes:", class_names)
    print("Number of classes:", len(class_names))

    model = build_model(num_classes=len(class_names),
                        fine_tune_last_block=args.fine_tune_last_block)

    os.makedirs(args.save_dir, exist_ok=True)
    save_path = os.path.join(args.save_dir, "resnet50_har_best_finetuned.pth")

    train_model(
        model=model,
        dataloaders=dataloaders,
        device=device,
        num_epochs=args.epochs,
        lr=args.lr,
        step_size=5,
        gamma=0.1,
        class_names=class_names,
        save_path=save_path,
    )


if __name__ == "__main__":
    main()
