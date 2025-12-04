import os
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models, transforms
from dataset import HARCsvDataset
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Paths
BASE_PATH = "/home/ubuntu/.cache/kagglehub/datasets/meetnagadia/human-action-recognition-har-dataset/versions/1/Human Action Recognition"
TRAIN_CSV = os.path.join(BASE_PATH, "Training_set.csv")
TRAIN_IMG_DIR = os.path.join(BASE_PATH, "train")
TEST_CSV = os.path.join(BASE_PATH, "Testing_set.csv")
TEST_IMG_DIR = os.path.join(BASE_PATH, "test")

# Hyperparameters
batch_size = 32
lr = 1e-4
epochs = 10
patience = 3  # for early stopping
num_classes = 15

# Transforms
transform_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

transform_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Train/Validation split
full_train_df = pd.read_csv(TRAIN_CSV)
train_df, val_df = train_test_split(
    full_train_df,
    test_size=0.2,
    stratify=full_train_df['label'],
    random_state=42
)

# Save temporary CSVs
train_tmp_csv = "train_tmp.csv"
val_tmp_csv = "val_tmp.csv"
train_df.to_csv(train_tmp_csv, index=False)
val_df.to_csv(val_tmp_csv, index=False)

# Datasets and Loaders
train_dataset = HARCsvDataset(train_tmp_csv, TRAIN_IMG_DIR, transform=transform_train, has_labels=True)
val_dataset = HARCsvDataset(val_tmp_csv, TRAIN_IMG_DIR, transform=transform_test, has_labels=True)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

# Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
model.classifier[6] = nn.Linear(4096, num_classes)
model = model.to(device)

# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=2, factor=0.5)

# Early stopping
best_val_f1 = 0.0
counter = 0

# Training loop
for epoch in range(epochs):
    model.train()
    all_labels, all_preds = [], []

    for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} - Training"):
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        preds = outputs.argmax(dim=1)
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())

    train_f1 = f1_score(all_labels, all_preds, average='macro')

    # Validation
    model.eval()
    val_labels, val_preds = [], []
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            preds = outputs.argmax(dim=1)
            val_labels.extend(labels.cpu().numpy())
            val_preds.extend(preds.cpu().numpy())

    val_f1 = f1_score(val_labels, val_preds, average='macro')
    print(f"Epoch {epoch+1}: Train F1: {train_f1:.4f}, Val F1: {val_f1:.4f}")

    # Scheduler step
    scheduler.step(val_f1)

    # Early stopping
    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        counter = 0
        torch.save(model.state_dict(), "best_vgg16.pth")
        print("Saved new best model!")
    else:
        counter += 1
        if counter >= patience:
            print(f"No improvement in {patience} epochs. Early stopping.")
            break

print(f"Training Finished. Best Val F1: {best_val_f1:.4f}")
