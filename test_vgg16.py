import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms, models
from dataset import HARCsvDataset
from sklearn.metrics import f1_score
import pandas as pd

# Paths
base_dir = "/home/ubuntu/.cache/kagglehub/datasets/meetnagadia/human-action-recognition-har-dataset/versions/1/Human Action Recognition"
test_csv = os.path.join(base_dir, "Test_set.csv")
test_img_dir = os.path.join(base_dir, "test")

# Auto-detect CSV
if not os.path.exists(test_csv):
    possible_test_csv = ["Test_set.csv", "testing_set.csv", "test.csv"]
    found = False
    for f in possible_test_csv:
        path = os.path.join(base_dir, f)
        if os.path.exists(path):
            test_csv = path
            found = True
            break
    if not found:
        raise FileNotFoundError(f"No test CSV found in {base_dir}. Checked: {possible_test_csv}")

# Check if labels exist
df_test = pd.read_csv(test_csv)
has_labels = 'label' in df_test.columns

# Transform
transform_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Dataset and loader
test_dataset = HARCsvDataset(test_csv, test_img_dir, transform=transform_test, has_labels=has_labels)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Device & model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
num_classes = 15
model.classifier[6] = torch.nn.Linear(4096, num_classes)
model.load_state_dict(torch.load("best_vgg16.pth", map_location=device))
model = model.to(device)
model.eval()

all_preds, all_labels, all_filenames = [], [], []

with torch.no_grad():
    for imgs, labels_or_names in test_loader:
        imgs = imgs.to(device)
        outputs = model(imgs)
        preds = outputs.argmax(1)
        all_preds.extend(preds.cpu().numpy())

        if has_labels:
            all_labels.extend(labels_or_names.numpy())
        else:
            all_filenames.extend(labels_or_names)

if has_labels:
    f1 = f1_score(all_labels, all_preds, average='macro')
    print(f"Test F1 Score: {f1:.4f}")
else:
    print("Test dataset has no labels. Predictions:")
    for fname, pred in zip(all_filenames, all_preds):
        print(f"{fname} -> {pred}")
