import os
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision import models, transforms
from dataset import HARCsvDataset
from tqdm import tqdm

# Paths
BASE_PATH = "/home/ubuntu/.cache/kagglehub/datasets/meetnagadia/human-action-recognition-har-dataset/versions/1/Human Action Recognition"
TEST_CSV = os.path.join(BASE_PATH, "Testing_set.csv")
TEST_IMG_DIR = os.path.join(BASE_PATH, "test")

# Transform
transform_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Dataset and Loader
test_dataset = HARCsvDataset(TEST_CSV, TEST_IMG_DIR, transform=transform_test, has_labels=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

# Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = 15
model = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
model.classifier[6] = torch.nn.Linear(4096, num_classes)
model.load_state_dict(torch.load("best_vgg16.pth", map_location=device))
model = model.to(device)
model.eval()

# Predict
all_preds = []
with torch.no_grad():
    for imgs, img_names in tqdm(test_loader, desc="Testing"):
        imgs = imgs.to(device)
        outputs = model(imgs)
        preds = outputs.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())

# Save submission
submission = pd.DataFrame({
    "filename": [name for _, name in test_dataset],
    "label": [HARCsvDataset.index_to_label(idx) for idx in all_preds]
})
submission.to_csv("submission.csv", index=False)
print("Test predictions saved to submission.csv")
