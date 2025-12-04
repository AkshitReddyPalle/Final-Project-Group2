import os
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image

class HARCsvDataset(Dataset):
    """
    Custom dataset for HAR (Human Action Recognition) CSV files.
    Handles both labeled (train/val) and unlabeled (test) datasets.
    """
    def __init__(self, csv_path, img_dir, transform=None, has_labels=True):
        self.df = pd.read_csv(csv_path)
        self.img_dir = img_dir
        self.transform = transform
        self.has_labels = has_labels

        # Normalize column names
        self.df.columns = [c.lower() for c in self.df.columns]

        if "filename" not in self.df.columns:
            raise ValueError(f"'filename' column not found in CSV: {csv_path}")

        if self.has_labels and "label" not in self.df.columns:
            raise ValueError(f"'label' column missing in labeled CSV: {csv_path}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_name = row["filename"]
        img_path = os.path.join(self.img_dir, img_name)

        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)

        if self.has_labels:
            # Convert label string to integer index
            label = row["label"]
            if isinstance(label, str):
                label = HARCsvDataset.label_to_index(label)
            return img, label
        else:
            return img, img_name

    # Map class names to indices
    classes = ['calling', 'clapping', 'cycling', 'dancing', 'drinking', 'eating',
               'fighting', 'hugging', 'laughing', 'listening_to_music', 'running',
               'sitting', 'sleeping', 'texting', 'using_laptop']
    class_to_idx = {cls_name: idx for idx, cls_name in enumerate(classes)}

    @staticmethod
    def label_to_index(label):
        return HARCsvDataset.class_to_idx[label]

    @staticmethod
    def index_to_label(idx):
        return HARCsvDataset.classes[idx]
