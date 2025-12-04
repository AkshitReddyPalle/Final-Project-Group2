import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, models
from dataset import HARCsvDataset
from tqdm import tqdm
from sklearn.metrics import f1_score
import numpy as np

# Paths
base_dir = "/home/ubuntu/.cache/kagglehub/datasets/meetnagadia/human-action-recognition-har-dataset/versions/1/Human Action Recognition"
train_csv = os.path.join(base_dir, "Training_set.csv")
train_img_dir = os.path.join(base_dir, "train")

# Data Transforms
transform_train = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.8,1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(30),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.RandomAffine(degrees=15, translate=(0.1,0.1), scale=(0.9,1.1), shear=10),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

transform_val = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

def train_vgg16(batch_size=32, epochs=10, patience=5, freeze_epochs=3):
    # Dataset
    full_dataset = HARCsvDataset(train_csv, train_img_dir, transform_train)
    num_classes = len(full_dataset.classes)
    print(f"Found {len(full_dataset)} images, {num_classes} classes: {full_dataset.classes}")

    # Split
    val_size = int(0.2 * len(full_dataset))
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    val_dataset.dataset.transform = transform_val

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Device & Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
    model.classifier[6] = nn.Linear(4096, num_classes)
    model = model.to(device)

    # Freeze initial layers
    for param in model.features[:10].parameters():
        param.requires_grad = False

    # Class weights
    train_labels = [full_dataset.data["label_idx"].iloc[i] for i in train_dataset.indices]
    class_counts = np.bincount(train_labels)
    class_weights = torch.tensor(1.0 / class_counts, dtype=torch.float).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)

    best_val_f1 = 0

    for epoch in range(epochs):
        model.train()
        train_preds, train_labels_list = [], []

        for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} - Training"):
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            preds = outputs.argmax(1)
            train_preds.extend(preds.cpu().numpy())
            train_labels_list.extend(labels.cpu().numpy())

        train_f1 = f1_score(train_labels_list, train_preds, average="macro")

        # Validation
        model.eval()
        val_preds, val_labels_list = [], []

        with torch.no_grad():
            for imgs, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} - Validation"):
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                preds = outputs.argmax(1)
                val_preds.extend(preds.cpu().numpy())
                val_labels_list.extend(labels.cpu().numpy())

        val_f1 = f1_score(val_labels_list, val_preds, average="macro")
        print(f"Epoch {epoch+1}: Train F1: {train_f1:.4f}, Val F1: {val_f1:.4f}")

        scheduler.step(val_f1)

        # Save best model
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), "best_vgg16.pth")
            print("Saved new best model!")

        # Unfreeze after initial epochs
        if epoch+1 == freeze_epochs:
            for param in model.features[:10].parameters():
                param.requires_grad = True
            print("Unfroze first 10 conv layers for fine-tuning")

    print("Training Finished. Best Val F1:", best_val_f1)

if __name__ == "__main__":
    train_vgg16(batch_size=32, epochs=10)
