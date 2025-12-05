import os
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as T
from dataset import HARDataset
from torchvision.models import vgg16, VGG16_Weights
import pandas as pd

# ---------------- Paths ----------------
BASE_DIR = '/home/ubuntu/.cache/kagglehub/datasets/meetnagadia/human-action-recognition-har-dataset/versions/1/Human Action Recognition'
IMG_DIR = os.path.join(BASE_DIR, 'train')  # change to 'test' if test images are separate
TEST_CSV = os.path.join(BASE_DIR, 'test_split.csv')

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ---------------- Transforms ----------------
test_transforms = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
])

# ---------------- Dataset and Loader ----------------
test_dataset = HARDataset(TEST_CSV, IMG_DIR, transform=test_transforms)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

# ---------------- Model ----------------
model = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
num_features = model.classifier[6].in_features
model.classifier[6] = torch.nn.Linear(num_features, len(test_dataset.labels))
model.load_state_dict(torch.load('best_vgg16.pth', map_location=DEVICE))
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

# ---------------- Save Predictions ----------------
submission = pd.DataFrame({
    'filename': test_dataset.df['filename'],
    'pred_label': [test_dataset.labels[i] for i in all_preds]
})
submission.to_csv('submission.csv', index=False)
print("Predictions saved to submission.csv")
