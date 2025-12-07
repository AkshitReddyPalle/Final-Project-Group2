import os
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from torchvision.transforms import RandAugment, RandomErasing
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import f1_score
import torch.optim as optim
import numpy as np
from torch.optim.lr_scheduler import _LRScheduler
import matplotlib.pyplot as plt

# Custom Dataset
class HARCsvDataset(Dataset):
    def __init__(self, csv_path, img_dir, transform=None):
        self.data = pd.read_csv(csv_path)
        self.img_dir = img_dir
        self.transform = transform

        self.classes = sorted(self.data["label"].unique())
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        self.data["label_idx"] = self.data["label"].map(self.class_to_idx)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = os.path.join(self.img_dir, row["filename"])

        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)

        return img, row["label_idx"]


# CNN Model
class BaselineCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        def block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, 3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2)
            )

        self.features = nn.Sequential(
            block(3, 32),
            block(32, 64),
            block(64, 128),
            block(128, 256),
            block(256, 512)
        )

        self.avg = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avg(x)
        return self.classifier(x)

class WarmupCosineLR(_LRScheduler):
    def __init__(self, optimizer, warmup_epochs, max_epochs, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            # Linear warmup
            return [
                base_lr * (self.last_epoch + 1) / self.warmup_epochs
                for base_lr in self.base_lrs
            ]
        # Cosine decay
        progress = (self.last_epoch - self.warmup_epochs) / max(
            1, self.max_epochs - self.warmup_epochs
        )
        return [
            base_lr * 0.5 * (1 + np.cos(np.pi * progress))
            for base_lr in self.base_lrs
        ]


# Train Function

def train_model():
    base_dir = "/home/ubuntu/HAR/Human Action Recognition"
    csv_path = os.path.join(base_dir, "Training_set.csv")
    img_dir = os.path.join(base_dir, "train")

    transform = transforms.Compose([
        transforms.RandomResizedCrop(256, scale=(0.75, 1.0)),
        RandAugment(num_ops=2, magnitude=9),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
        RandomErasing(p=0.25, scale=(0.02, 0.2), ratio=(0.3, 3.3))
    ])

    full_dataset = HARCsvDataset(csv_path, img_dir, transform)

    num_classes = len(full_dataset.classes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Training on:", device)


    # Split dataset

    val_size = int(0.2 * len(full_dataset))
    train_size = len(full_dataset) - val_size

    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])


    # Weighted sampler for train set ONLY

    train_labels = [full_dataset.data.iloc[i]["label_idx"] for i in train_dataset.indices]

    class_counts = np.bincount(train_labels, minlength=num_classes)
    weights = 1.0 / torch.tensor(class_counts, dtype=torch.float32)
    sample_weights = np.array([weights[label] for label in train_labels])

    sampler = torch.utils.data.WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

    train_loader = DataLoader(train_dataset, batch_size=32, sampler=sampler)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # Model
    model = BaselineCNN(num_classes).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)

    warmup_epochs = 3
    total_epochs = 40

    scheduler = WarmupCosineLR(
        optimizer,
        warmup_epochs=warmup_epochs,
        max_epochs=total_epochs
    )

    best_val_f1 = 0

    # Training Loop
    history = {"train_f1": [], "val_f1": []}

    for epoch in range(total_epochs):
        model.train()
        preds, labels = [], []
        train_correct = 0
        total_train = 0

        for imgs, lbls in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{total_epochs}"):
            imgs, lbls = imgs.to(device), lbls.to(device)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, lbls)
            loss.backward()
            optimizer.step()

            pred = outputs.argmax(1)
            preds.extend(pred.cpu().numpy())
            labels.extend(lbls.cpu().numpy())

            train_correct += (pred == lbls).sum().item()
            total_train += lbls.size(0)

        train_f1 = f1_score(labels, preds, average="macro")
        train_acc = train_correct / total_train

        # Validation
        model.eval()
        val_preds, val_labels = [], []
        val_correct = 0
        total_val = 0

        with torch.no_grad():
            for imgs, lbls in val_loader:
                imgs, lbls = imgs.to(device), lbls.to(device)
                outputs = model(imgs)

                pred = outputs.argmax(1)
                val_preds.extend(pred.cpu().numpy())
                val_labels.extend(lbls.cpu().numpy())

                val_correct += (pred == lbls).sum().item()
                total_val += lbls.size(0)

        val_f1 = f1_score(val_labels, val_preds, average="macro")
        val_acc = val_correct / total_val

        #  Log history properly
        history["train_f1"].append(train_f1)
        history["val_f1"].append(val_f1)

        print(f"Epoch {epoch + 1:02d} | "
              f"Train F1: {train_f1:.4f} | Train Acc: {train_acc:.4f} | "
              f"Val F1: {val_f1:.4f} | Val Acc: {val_acc:.4f}")

        scheduler.step()

        # Save best model
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), "best_cnn.pth")
            print(" New best model saved!")

    print("\n Training Completed — Best Val F1:", best_val_f1)

    # Display F1 curve live instead of saving
    plt.figure(figsize=(8, 5))
    plt.plot(history["train_f1"], label="Train F1", linewidth=2)
    plt.plot(history["val_f1"], label="Validation F1", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("F1 Score")
    plt.title("Training vs Validation F1-Score")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
if __name__ == "__main__":
    train_model()
