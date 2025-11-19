import os
import copy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ------------------------------------
# 1. Device selection (GPU if available)
# ------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ------------------------------------
# 2. Paths & Hyperparameters
# ------------------------------------
base_dir = os.path.dirname(os.path.abspath(__file__))  # .../code
data_dir = os.path.join(base_dir, "..", "data")        # .../data

batch_size = 32
num_epochs = 10
lr = 1e-3

# ------------------------------------
# 3. Data Transforms (ImageNet standard)
# ------------------------------------
imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std  = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
])

# ------------------------------------
# 4. Datasets & Dataloaders
# ------------------------------------
train_dir = os.path.join(data_dir, "train")
val_dir   = os.path.join(data_dir, "val")

train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
val_dataset   = datasets.ImageFolder(val_dir,   transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, num_workers=0)

class_names = train_dataset.classes
num_classes = len(class_names)

print("Classes:", class_names)
print("Number of classes:", num_classes)

# ------------------------------------
# 5. Load Pretrained ResNet50
# ------------------------------------
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

# Freeze all layers
for param in model.parameters():
    param.requires_grad = False

# Replace classification head
in_features = model.fc.in_features
model.fc = nn.Linear(in_features, num_classes)

model = model.to(device)

# ------------------------------------
# 6. Loss, Optimizer, Scheduler
# ------------------------------------
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.fc.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

# ------------------------------------
# 7. Training Function
# ------------------------------------
def train_model(model, criterion, optimizer, scheduler, num_epochs):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_val_acc = 0.0

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print("-" * 30)
        
        for phase in ["train", "val"]:
            model.train() if phase == "train" else model.eval()
            dataloader = train_loader if phase == "train" else val_loader

            running_loss = 0.0
            preds_all, labels_all = [], []

            for inputs, labels in dataloader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                preds_all.extend(preds.cpu().numpy())
                labels_all.extend(labels.cpu().numpy())

            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_acc = accuracy_score(labels_all, preds_all)

            print(f"{phase} Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.4f}")

            if phase == "val" and epoch_acc > best_val_acc:
                best_val_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        scheduler.step()

    print("\nBest Validation Accuracy:", best_val_acc)
    model.load_state_dict(best_model_wts)
    return model

# ------------------------------------
# 8. Train the model
# ------------------------------------
model = train_model(model, criterion, optimizer, scheduler, num_epochs)

# Save best model
models_dir = os.path.join(base_dir, "models")
os.makedirs(models_dir, exist_ok=True)
save_path = os.path.join(models_dir, "resnet50_har_best.pth")
torch.save(model.state_dict(), save_path)
print("Model saved to:", save_path)

# ------------------------------------
# 9. Final Evaluation
# ------------------------------------
model.eval()
preds_all = []
labels_all = []

with torch.no_grad():
    for inputs, labels in val_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)

        preds_all.extend(preds.cpu().numpy())
        labels_all.extend(labels.cpu().numpy())

val_acc = accuracy_score(labels_all, preds_all)
print("\nFinal Validation Accuracy:", val_acc)

print("\nClassification Report:")
print(classification_report(labels_all, preds_all, target_names=class_names))

print("\nConfusion Matrix:")
print(confusion_matrix(labels_all, preds_all))
