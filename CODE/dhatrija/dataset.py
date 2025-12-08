import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T
import torch
import numpy as np

class HARDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.df = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        self.labels = sorted(self.df['label'].unique())
        self.label2idx = {label: idx for idx, label in enumerate(self.labels)}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, row['filename'])
        image = Image.open(img_path).convert('RGB')
        label = self.label2idx[row['label']]
        if self.transform:
            image = self.transform(image)
        return image, label

def split_csv(csv_file, train_ratio=0.8, val_ratio=0.1):
    df = pd.read_csv(csv_file)
    from sklearn.model_selection import train_test_split
    train_df, temp_df = train_test_split(df, test_size=(1-train_ratio), stratify=df['label'], random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df['label'], random_state=42)

    base_dir = os.path.dirname(csv_file)
    train_csv = os.path.join(base_dir, 'train_split.csv')
    val_csv = os.path.join(base_dir, 'val_split.csv')
    test_csv = os.path.join(base_dir, 'test_split.csv')

    train_df.to_csv(train_csv, index=False)
    val_df.to_csv(val_csv, index=False)
    test_df.to_csv(test_csv, index=False)
    print(f"Splits saved. Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    return train_csv, val_csv, test_csv
