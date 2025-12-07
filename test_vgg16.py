import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
import torchvision.transforms as T
from torchvision.models import vgg16, VGG16_Weights
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

from dataset import HARDataset

# ---------------- Paths ----------------
BASE_DIR = '/home/ubuntu/.cache/kagglehub/datasets/meetnagadia/human-action-recognition-har-dataset/versions/1/Human Action Recognition'
IMG_DIR = os.path.join(BASE_DIR, 'train')
TEST_CSV = os.path.join(BASE_DIR, 'test_split.csv')

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------- Transforms ----------------
test_transforms = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
])

# ---------------- Dataset ----------------
test_dataset = HARDataset(TEST_CSV, IMG_DIR, transform=test_transforms)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

# ---------------- Model ----------------
model = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
model.classifier[6] = torch.nn.Linear(model.classifier[6].in_features, len(test_dataset.labels))

model.load_state_dict(torch.load("best_vgg16.pth", map_location=DEVICE))
model = model.to(DEVICE)
model.eval()

# ---------------- Inference ----------------
all_preds = []
with torch.no_grad():
    for imgs, _ in test_loader:
        imgs = imgs.to(DEVICE)
        outputs = model(imgs)
        preds = outputs.argmax(1)
        all_preds.extend(preds.cpu().numpy())

# True labels
all_true = test_dataset.df["label"].map(test_dataset.label2idx).tolist()

# ---------------- Test Metrics ----------------
test_acc = accuracy_score(all_true, all_preds)
test_prec = precision_score(all_true, all_preds, average="macro", zero_division=0)
test_rec = recall_score(all_true, all_preds, average="macro", zero_division=0)
test_f1 = f1_score(all_true, all_preds, average="macro", zero_division=0)

print("Test Accuracy:", test_acc)
print("Test Precision:", test_prec)
print("Test Recall:", test_rec)
print("Test F1:", test_f1)

# ---------------- Test Confusion Matrix ----------------
test_cm = confusion_matrix(all_true, all_preds)

plt.figure(figsize=(10, 8))
sns.heatmap(test_cm, annot=False, cmap="viridis")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Test Confusion Matrix")
plt.savefig("test_confusion_matrix.png")
plt.close()

# ---------------- Per-Class Metrics ----------------
report = classification_report(
    all_true,
    all_preds,
    target_names=test_dataset.labels,
    output_dict=True
)

pd.DataFrame(report).to_csv("test_class_report.csv")
print("Per-class metrics saved to test_class_report.csv")

# ---------------- Save Predictions ----------------
submission = pd.DataFrame({
    "filename": test_dataset.df["filename"],
    "pred_label": [test_dataset.labels[i] for i in all_preds]
})
submission.to_csv("submission.csv", index=False)
print("Predictions saved to submission.csv")
