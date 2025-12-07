import os
import torch
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix, classification_report
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image

from train import BaselineCNN  # Import trained model


base_dir = "/home/ubuntu/HAR"
dataset_dir = os.path.join(base_dir, "Human Action Recognition")

train_csv = os.path.join(dataset_dir, "Training_set.csv")
test_csv = os.path.join(dataset_dir, "Testing_set.csv")

train_img_dir = os.path.join(dataset_dir, "train")
test_img_dir = os.path.join(dataset_dir, "test")


# === LOAD CLASSES ===
df_train = pd.read_csv(train_csv)
classes = sorted(df_train["label"].unique())
class_to_idx = {c: i for i, c in enumerate(classes)}


# === VALIDATION DATASET ===
class HARValDataset(Dataset):
    def __init__(self, csv_path, img_dir, class_to_idx, transform=None):
        self.data = pd.read_csv(csv_path)
        val_size = int(0.2 * len(self.data))
        self.data = self.data.tail(val_size)

        self.img_dir = img_dir
        self.transform = transform
        self.class_to_idx = class_to_idx

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img = Image.open(os.path.join(train_img_dir, row["filename"])).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, self.class_to_idx[row["label"]]


# === TRANSFORMS ===
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])


# === LOAD MODEL ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BaselineCNN(num_classes=len(classes)).to(device)
model.load_state_dict(torch.load("best_cnn.pth", map_location=device))
model.eval()


# === EVALUATION ON VALIDATION SPLIT ===
val_dataset = HARValDataset(train_csv, train_img_dir, class_to_idx, transform)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

all_preds, all_labels = [], []
with torch.no_grad():
    for imgs, labels in val_loader:
        imgs = imgs.to(device)
        out = model(imgs)
        preds = out.argmax(1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())

print("\n Classification Report:")
print(classification_report(all_labels, all_preds, target_names=classes))



# CONFUSION MATRIX

cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=False, cmap="Blues",
            xticklabels=classes, yticklabels=classes)
plt.title("Confusion Matrix - Baseline CNN")
plt.tight_layout()
plt.show()



#  PER-CLASS ACCURACY

per_class_acc = cm.diagonal() / cm.sum(axis=1)
plt.figure(figsize=(12, 5))
plt.bar(classes, per_class_acc)
plt.xticks(rotation=50, ha="right")
plt.ylabel("Accuracy")
plt.title("Per-Class Accuracy")
plt.tight_layout()
plt.show()


# SAMPLE TEST PREDICTIONS

df_test = pd.read_csv(test_csv)
sample_files = df_test["filename"].sample(9, random_state=42).tolist()

plt.figure(figsize=(10, 10))
for i, filename in enumerate(sample_files):
    img_path = os.path.join(test_img_dir, filename)
    img = Image.open(img_path).convert("RGB")
    img_in = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        pred = model(img_in).argmax(1).item()

    plt.subplot(3, 3, i+1)
    plt.imshow(img.resize((256, 256)))
    plt.title(f"P:{classes[pred]}", color="blue")
    plt.axis("off")

plt.tight_layout()
plt.show()
print("Test Prediction Visuals Displayed")
