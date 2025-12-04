import os
import torch
import numpy as np
from torchvision import transforms, models, datasets
from sklearn.metrics import f1_score
from dataset import HARCsvDataset

# Paths
base_dir = "/home/ubuntu/.cache/kagglehub/datasets/meetnagadia/human-action-recognition-har-dataset/versions/1/Human Action Recognition"
test_csv = os.path.join(base_dir, "Testing_set.csv")
test_img_dir = os.path.join(base_dir, "test")

# Transforms
transform_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Dataset
test_dataset = HARCsvDataset(test_csv, test_img_dir, transform_test)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

# Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = len(test_dataset.classes)
model = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
model.classifier[6] = torch.nn.Linear(4096, num_classes)
model.load_state_dict(torch.load("best_vgg16.pth"))
model = model.to(device)
model.eval()

# Evaluation
all_preds, all_labels = [], []

with torch.no_grad():
    for imgs, labels in test_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = model(imgs)
        preds = outputs.argmax(1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

f1 = f1_score(all_labels, all_preds, average="macro")
accuracy = (np.array(all_preds) == np.array(all_labels)).mean()
print(f"Test F1 Score: {f1:.4f}, Accuracy: {accuracy:.4f}")
