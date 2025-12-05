import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as T
from torchvision.models import vgg16, VGG16_Weights
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix
import numpy as np
from dataset import HARDataset, split_csv
import random

# ---------------- Paths -----------------
BASE_DIR = '/home/ubuntu/.cache/kagglehub/datasets/meetnagadia/human-action-recognition-har-dataset/versions/1/Human Action Recognition'
IMG_DIR = os.path.join(BASE_DIR, 'train')
CSV_PATH = os.path.join(BASE_DIR, 'Training_set.csv')

# ---------------- Hyperparameters ----------------
BATCH_SIZE = 32
EPOCHS = 5
LR = 1e-4
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

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

# ---------------- Datasets and Loaders ----------------
train_dataset = HARDataset(TRAIN_CSV, IMG_DIR, transform=train_transforms, use_mixup=True, use_mosaic=True)
val_dataset = HARDataset(VAL_CSV, IMG_DIR, transform=val_transforms)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

# ---------------- Model ----------------
model = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
num_features = model.classifier[6].in_features
model.classifier[6] = nn.Linear(num_features, len(train_dataset.labels))
model = model.to(DEVICE)

# ---------------- Loss and Optimizer ----------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)

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
            loss = -(labels_onehot * torch.log_softmax(outputs, dim=1)).sum(dim=1).mean()
        # Apply Mosaic (optional, for batch augmentation)
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

    train_acc = accuracy_score(all_labels, all_preds)
    train_prec = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    train_rec = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    train_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

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

    val_acc = accuracy_score(all_labels, all_preds)
    val_prec = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    val_rec = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    val_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    val_cm = confusion_matrix(all_labels, all_preds)

    print(f"Epoch {epoch} - loss:{loss.item():.4f} "
          f"Train -> acc:{train_acc:.4f} prec:{train_prec:.4f} rec:{train_rec:.4f} f1:{train_f1:.4f} | "
          f"Val -> acc:{val_acc:.4f} prec:{val_prec:.4f} rec:{val_rec:.4f} f1:{val_f1:.4f}")

    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        torch.save(model.state_dict(), 'best_vgg16.pth')
        print(f"Saved best model (val_f1={best_val_f1:.4f})")

    scheduler.step(val_f1)
    print(f"Current LR: {scheduler.optimizer.param_groups[0]['lr']}")

# ---------------- Training Time and Model Size ----------------
end_time = time.time()
print(f"\nTraining completed in {(end_time - start_time) / 60:.2f} minutes")
model_size_mb = sum(p.numel() for p in model.parameters()) * 4 / (1024 ** 2)
print(f"Model size: {model_size_mb:.2f} MB")
