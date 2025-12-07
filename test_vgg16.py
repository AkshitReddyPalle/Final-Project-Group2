import os
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as T
from dataset import HARDataset
from torchvision.models import vgg16, VGG16_Weights
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd

# ---------------- Paths -----------------
BASE_DIR = '/home/ubuntu/.cache/kagglehub/datasets/meetnagadia/human-action-recognition-har-dataset/versions/1/Human Action Recognition'
IMG_DIR = os.path.join(BASE_DIR, 'train')
TEST_CSV = os.path.join(BASE_DIR, 'test_split.csv')
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ---------------- Transforms -----------------
test_transforms = T.Compose([T.Resize((224,224)), T.ToTensor()])

# ---------------- Dataset and Loader -----------------
test_dataset = HARDataset(TEST_CSV, IMG_DIR, transform=test_transforms)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

# ---------------- Model -----------------
model = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
num_features = model.classifier[6].in_features
model.classifier[6] = torch.nn.Linear(num_features, len(test_dataset.labels))
model.load_state_dict(torch.load('best_vgg16.pth', map_location=DEVICE))
model = model.to(DEVICE)
model.eval()

# ---------------- Inference -----------------
all_preds, all_labels = [], []
with torch.no_grad():
    for imgs, labels in test_loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        outputs = model(imgs)
        preds = outputs.argmax(1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# ---------------- Classification Report -----------------
report = classification_report(all_labels, all_preds, target_names=test_dataset.labels, output_dict=True)
report_df = pd.DataFrame(report).transpose()
report_df.to_csv('test_class_report.csv', index=True)
print("Classification report saved: test_class_report.csv")

# ---------------- Confusion Matrix -----------------
cm = confusion_matrix(all_labels, all_preds)
cm_df = pd.DataFrame(cm, index=test_dataset.labels, columns=test_dataset.labels)
cm_df.to_csv('test_confusion_matrix.csv')
print("Confusion matrix saved: test_confusion_matrix.csv")
