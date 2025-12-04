import os
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image

class HARCsvDataset(Dataset):
    def __init__(self, csv_path, img_dir, transform=None):
        self.data = pd.read_csv(csv_path)
        self.img_dir = img_dir
        self.transform = transform

        # Map class names to indices
        self.classes = sorted(self.data["label"].unique())
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.data["label_idx"] = self.data["label"].map(self.class_to_idx)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = os.path.join(self.img_dir, row["filename"])
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, row["label_idx"]
