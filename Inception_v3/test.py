import os
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as T
import pandas as pd
import numpy as np

from Dataset import HARDataset  
from torchvision.models import inception_v3, Inception_V3_Weights
from torchvision.models.inception import InceptionOutputs

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

# ---------------- Paths ----------------
BASE_DIR = "/home/ubuntu/Final-Project-Group2/Data/Human Action Recognition"
IMG_DIR = os.path.join(BASE_DIR, "train")
TEST_CSV = os.path.join(BASE_DIR, "test_split.csv")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------- Transforms ----------------
test_transforms = T.Compose([
    T.Resize((299, 299)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
])

# ---------------- Dataset and Loader ----------------
test_dataset = HARDataset(TEST_CSV, IMG_DIR, transform=test_transforms)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

num_classes = len(test_dataset.labels)
print("Number of classes:", num_classes)
print("Labels:", test_dataset.labels)

# ---------------- Model ----------------
weights = Inception_V3_Weights.IMAGENET1K_V1
model = inception_v3(weights=weights)  # aux_logits=True
in_features = model.fc.in_features
model.fc = torch.nn.Linear(in_features, num_classes)

model.load_state_dict(torch.load("best_inception_v3.pth", map_location=DEVICE))
model = model.to(DEVICE)
model.eval()

# ---------------- Evaluation ----------------
all_preds = []
all_labels = []

with torch.no_grad():
    for imgs, labels in test_loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        outputs = model(imgs)
        if isinstance(outputs, InceptionOutputs):
            outputs = outputs.logits
        preds = outputs.argmax(1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

all_preds = np.array(all_preds)
all_labels = np.array(all_labels)

# ---------------- Metrics ----------------
acc = accuracy_score(all_labels, all_preds)
prec = precision_score(all_labels, all_preds, average="macro", zero_division=0)
rec = recall_score(all_labels, all_preds, average="macro", zero_division=0)
f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
cm = confusion_matrix(all_labels, all_preds)

print("\n=== Test Split Metrics (Inception v3) ===")
print(f"Accuracy      : {acc:.4f}")
print(f"Precision (macro): {prec:.4f}")
print(f"Recall (macro)   : {rec:.4f}")
print(f"F1-score (macro) : {f1:.4f}")
print("\nConfusion Matrix (rows=true, cols=pred):")
print(cm)

# Optional: detailed per-class report
print("\nClassification report (per class):")
print(classification_report(all_labels, all_preds, target_names=[str(l) for l in test_dataset.labels], zero_division=0))

# ---------------- Save predictions (optional) ----------------
submission = pd.DataFrame({
    "filename": test_dataset.df["filename"],
    "true_label": [test_dataset.labels[i] for i in all_labels],
    "pred_label": [test_dataset.labels[i] for i in all_preds],
})

# Save inside Inception_v3 folder
out_dir = os.path.dirname(__file__)  # this is /home/ubuntu/Final-Project-Group2/Inception_v3
os.makedirs(out_dir, exist_ok=True)  # already exists, but safe

out_path = os.path.join(out_dir, "test_split_predictions_inception_v3.csv")
submission.to_csv(out_path, index=False)
print(f"\nPredictions saved to {out_path}")

