import os
import time
import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as T
from torchvision.models import inception_v3, Inception_V3_Weights
from torchvision.models.inception import InceptionOutputs

from sklearn.metrics import (
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
)

from Dataset import HARDataset, split_csv  # change to `from dataset import ...` if your file is lowercase

# ---------------- Paths -----------------
BASE_DIR = "/home/ubuntu/Final-Project-Group2/Data/Human Action Recognition"
IMG_DIR = os.path.join(BASE_DIR, "train")
CSV_PATH = os.path.join(BASE_DIR, "Training_set.csv")

# ---------------- Hyperparameters ----------------
BATCH_SIZE = 32
EPOCHS = 15
LR = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------- CSV Splits ----------------
TRAIN_CSV, VAL_CSV, TEST_CSV = split_csv(CSV_PATH)

# ---------------- Transforms ----------------
# Inception v3 expects 299x299 + ImageNet normalization
train_transforms = T.Compose([
    T.Resize((320, 320)),
    T.RandomResizedCrop(299, scale=(0.8, 1.0)),
    T.RandomHorizontalFlip(),
    T.ColorJitter(brightness=0.2, contrast=0.2,
                  saturation=0.2, hue=0.05),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
])

val_transforms = T.Compose([
    T.Resize((299, 299)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
])

# ---------------- Datasets and Loaders ----------------
train_dataset = HARDataset(
    TRAIN_CSV,
    IMG_DIR,
    transform=train_transforms,
    use_mixup=True,
    use_mosaic=True,
)
val_dataset = HARDataset(
    VAL_CSV,
    IMG_DIR,
    transform=val_transforms,
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=4)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

num_classes = len(train_dataset.labels)

# ---------------- Model (Inception v3) ----------------
# IMPORTANT: don't pass aux_logits=False when using weights.
weights = Inception_V3_Weights.IMAGENET1K_V1
model = inception_v3(weights=weights)  # aux_logits=True by default with weights
in_features = model.fc.in_features
model.fc = nn.Linear(in_features, num_classes)
model = model.to(DEVICE)

# ---------------- Loss and Optimizer ----------------
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = optim.Adam(model.parameters(), lr=LR)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="max", factor=0.5, patience=2
)

# ---------------- Training Loop ----------------
best_val_f1 = 0.0
start_time = time.time()

for epoch in range(1, EPOCHS + 1):
    model.train()
    all_preds, all_labels = [], []

    for imgs, labels in train_loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

        # Apply MixUp
        if train_dataset.use_mixup and random.random() < 0.5:
            imgs, labels_onehot = train_dataset.mixup(imgs, labels)
            outputs = model(imgs)
            if isinstance(outputs, InceptionOutputs):
                outputs = outputs.logits
            log_probs = torch.log_softmax(outputs, dim=1)
            loss = -(labels_onehot.to(DEVICE) * log_probs).sum(dim=1).mean()

        # Apply Mosaic
        elif train_dataset.use_mosaic and random.random() < 0.3:
            imgs, labels_onehot = train_dataset.mosaic(imgs, labels)
            outputs = model(imgs)
            if isinstance(outputs, InceptionOutputs):
                outputs = outputs.logits
            log_probs = torch.log_softmax(outputs, dim=1)
            loss = -(labels_onehot.to(DEVICE) * log_probs).sum(dim=1).mean()

        else:
            outputs = model(imgs)
            if isinstance(outputs, InceptionOutputs):
                outputs = outputs.logits
            loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        preds = outputs.argmax(1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    train_acc  = accuracy_score(all_labels, all_preds)
    train_prec = precision_score(all_labels, all_preds, average="macro", zero_division=0)
    train_rec  = recall_score(all_labels, all_preds, average="macro", zero_division=0)
    train_f1   = f1_score(all_labels, all_preds, average="macro", zero_division=0)

    # ---------------- Validation ----------------
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            outputs = model(imgs)
            if isinstance(outputs, InceptionOutputs):
                outputs = outputs.logits
            preds = outputs.argmax(1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    val_acc  = accuracy_score(all_labels, all_preds)
    val_prec = precision_score(all_labels, all_preds, average="macro", zero_division=0)
    val_rec  = recall_score(all_labels, all_preds, average="macro", zero_division=0)
    val_f1   = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    val_cm   = confusion_matrix(all_labels, all_preds)

    print(
        f"Epoch {epoch} - loss:{loss.item():.4f} "
        f"Train -> acc:{train_acc:.4f} prec:{train_prec:.4f} rec:{train_rec:.4f} f1:{train_f1:.4f} | "
        f"Val -> acc:{val_acc:.4f} prec:{val_prec:.4f} rec:{val_rec:.4f} f1:{val_f1:.4f}"
    )

    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        torch.save(model.state_dict(), "best_inception_v3.pth")
        print(f"Saved best model (val_f1={best_val_f1:.4f})")

    scheduler.step(val_f1)
    print(f"Current LR: {scheduler.optimizer.param_groups[0]['lr']}")

# ---------------- Training Time and Model Size ----------------
end_time = time.time()
print(f"\nTraining completed in {(end_time - start_time) / 60:.2f} minutes")
model_size_mb = sum(p.numel() for p in model.parameters()) * 4 / (1024 ** 2)
print(f"Model size: {model_size_mb:.2f} MB")
