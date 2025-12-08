import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as T
from torchvision.models import vgg16, VGG16_Weights
from sklearn.metrics import f1_score, accuracy_score
import csv
from dataset import HARDataset, split_csv

# ---------------- Paths -----------------
BASE_DIR = '/home/ubuntu/.cache/kagglehub/datasets/meetnagadia/human-action-recognition-har-dataset/versions/1/Human Action Recognition'
IMG_DIR = os.path.join(BASE_DIR, 'train')
CSV_PATH = os.path.join(BASE_DIR, 'Training_set.csv')

# ---------------- Hyperparameters -----------------
BATCH_SIZE = 32
EPOCHS = 10
LR = 1e-4
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ---------------- CSV Splits -----------------
TRAIN_CSV, VAL_CSV, TEST_CSV = split_csv(CSV_PATH)

# ---------------- Transforms -----------------
train_transforms = T.Compose([T.Resize((224,224)), T.RandomHorizontalFlip(), T.ToTensor()])
val_transforms = T.Compose([T.Resize((224,224)), T.ToTensor()])

# ---------------- Dataset and Loader -----------------
train_dataset = HARDataset(TRAIN_CSV, IMG_DIR, transform=train_transforms)
val_dataset = HARDataset(VAL_CSV, IMG_DIR, transform=val_transforms)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

# ---------------- Model -----------------
model = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
num_features = model.classifier[6].in_features
model.classifier[6] = nn.Linear(num_features, len(train_dataset.labels))
model = model.to(DEVICE)

# ---------------- Loss and Optimizer -----------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)

# ---------------- Logging -----------------
epoch_log_file = 'epoch_metrics.csv'
with open(epoch_log_file, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['epoch','train_acc','train_f1','val_acc','val_f1','lr','best_model'])

# ---------------- Training Loop -----------------
best_val_f1 = 0.0
start_time = time.time()

for epoch in range(1, EPOCHS+1):
    model.train()
    all_preds, all_labels = [], []
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        outputs = model(imgs)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        preds = outputs.argmax(1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    train_acc = accuracy_score(all_labels, all_preds)
    train_f1 = f1_score(all_labels, all_preds, average='macro')

    # ---------------- Validation -----------------
    model.eval()
    val_preds, val_labels = [], []
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            outputs = model(imgs)
            preds = outputs.argmax(1)
            val_preds.extend(preds.cpu().numpy())
            val_labels.extend(labels.cpu().numpy())

    val_acc = accuracy_score(val_labels, val_preds)
    val_f1 = f1_score(val_labels, val_preds, average='macro')

    # ---------------- Save Best Model -----------------
    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        torch.save(model.state_dict(), 'best_vgg16.pth')
        best_model_flag = 'Yes'
    else:
        best_model_flag = 'No'

    # ---------------- Scheduler Step -----------------
    scheduler.step(val_f1)

    # ---------------- Print & Log -----------------
    print(f"Epoch {epoch}/{EPOCHS} - Train F1: {train_f1:.4f} | Val F1: {val_f1:.4f} | LR: {scheduler.optimizer.param_groups[0]['lr']:.6f} | Best Model: {best_model_flag}")
    with open(epoch_log_file, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([epoch, train_acc, train_f1, val_acc, val_f1, scheduler.optimizer.param_groups[0]['lr'], best_model_flag])

end_time = time.time()
print(f"\nTraining completed in {(end_time-start_time)/60:.2f} minutes")
