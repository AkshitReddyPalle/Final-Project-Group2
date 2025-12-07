import os
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as T
from torchvision.models import vgg16, VGG16_Weights
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from dataset import HARDataset, split_csv

# ---------------- Paths -----------------
BASE_DIR = '/home/ubuntu/.cache/kagglehub/datasets/meetnagadia/human-action-recognition-har-dataset/versions/1/Human Action Recognition'
IMG_DIR = os.path.join(BASE_DIR, 'train')
CSV_PATH = os.path.join(BASE_DIR, 'Training_set.csv')

# ---------------- Hyperparameters ----------------
BATCH_SIZE = 32
EPOCHS = 5
LR = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------- CSV Splits ----------------
TRAIN_CSV, VAL_CSV, TEST_CSV = split_csv(CSV_PATH)

# ---------------- Transforms ----------------
train_transforms = T.Compose([
    T.Resize((224, 224)),
    T.RandomHorizontalFlip(),
    T.ToTensor(),
])

val_transforms = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
])

# ---------------- Datasets ----------------
train_dataset = HARDataset(TRAIN_CSV, IMG_DIR, transform=train_transforms, use_mixup=True, use_mosaic=True)
val_dataset = HARDataset(VAL_CSV, IMG_DIR, transform=val_transforms)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

# ---------------- Model ----------------
model = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
model.classifier[6] = nn.Linear(model.classifier[6].in_features, len(train_dataset.labels))
model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)

# ---------------- Tracking arrays ----------------
train_f1_list = []
val_f1_list = []

best_val_f1 = 0.0
start_time = time.time()

# ---------------- Training Loop ----------------
for epoch in range(1, EPOCHS + 1):
    model.train()
    all_preds, all_labels = [], []

    for imgs, labels in train_loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

        if train_dataset.use_mixup and random.random() < 0.5:
            imgs, labels_onehot = train_dataset.mixup(imgs, labels)
            outputs = model(imgs)
            loss = -(labels_onehot * torch.log_softmax(outputs, dim=1)).sum(dim=1).mean()

        elif train_dataset.use_mosaic and random.random() < 0.3:
            imgs, labels_onehot = train_dataset.mosaic(imgs, labels)
            outputs = model(imgs)
            loss = -(labels_onehot * torch.log_softmax(outputs, dim=1)).sum(dim=1).mean()

        else:
            outputs = model(imgs)
            loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        preds = outputs.argmax(1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    # Train metrics
    train_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    train_f1_list.append(train_f1)

    # ---------------- Validation ----------------
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            outputs = model(imgs)
            preds = outputs.argmax(1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    val_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    val_f1_list.append(val_f1)

    val_cm = confusion_matrix(all_labels, all_preds)

    print(f"Epoch {epoch}/{EPOCHS} - Train F1: {train_f1:.4f} | Val F1: {val_f1:.4f}")

    # Save best model
    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        torch.save(model.state_dict(), "best_vgg16.pth")
        print(f"Saved best model (Val F1 = {best_val_f1:.4f})")

    # LR scheduler
    scheduler.step(val_f1)

# ---------------- F1 Curve ----------------
plt.figure(figsize=(8, 6))
plt.plot(train_f1_list, label="Train Macro-F1")
plt.plot(val_f1_list, label="Validation Macro-F1")
plt.xlabel("Epoch")
plt.ylabel("Macro F1 Score")
plt.title("Training vs Validation Macro-F1 Curve")
plt.legend()
plt.savefig("f1_curve.png")
plt.close()

# ---------------- Validation Confusion Matrix ----------------
plt.figure(figsize=(10, 8))
sns.heatmap(val_cm, annot=False, cmap="viridis")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Validation Confusion Matrix")
plt.savefig("val_confusion_matrix.png")
plt.close()

# ---------------- Training Summary ----------------
end_time = time.time()
print(f"Training completed in {(end_time - start_time)/60:.2f} minutes")
