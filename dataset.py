import os
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image

class HARCsvDataset(Dataset):
    def __init__(self, csv_path, img_dir, transform=None, has_labels=True):
        self.csv_path = csv_path
        self.img_dir = img_dir
        self.transform = transform
        self.has_labels = has_labels

        self.data = pd.read_csv(csv_path)
        if self.has_labels:
            if 'label' in self.data.columns:
                self.classes = sorted(self.data['label'].unique())
                self.data['label_idx'] = self.data['label'].apply(lambda x: self.classes.index(x))
            else:
                raise KeyError("CSV does not contain 'label' column")
        else:
            self.classes = []

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = self.data.iloc[idx]['filename']
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        if self.has_labels:
            label = self.data.iloc[idx]['label_idx']
            return image, label
        else:
            return image, img_name
