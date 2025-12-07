import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T
import numpy as np
import torch
import random

class HARDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None, use_mixup=False, use_mosaic=False, alpha=0.4):
        self.df = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        self.use_mixup = use_mixup
        self.use_mosaic = use_mosaic
        self.alpha = alpha
        self.labels = sorted(self.df['label'].unique())
        self.label2idx = {label: idx for idx, label in enumerate(self.labels)}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, self.df.iloc[idx]['filename'])
        image = Image.open(img_name).convert("RGB")
        label = self.label2idx[self.df.iloc[idx]['label']]

        if self.transform:
            image = self.transform(image)

        return image, label

    # ---------------------- MixUp ----------------------
    def mixup(self, imgs, labels):
        lam = np.random.beta(self.alpha, self.alpha)
        idx2 = torch.randperm(imgs.size(0))

        mixed_imgs = lam * imgs + (1 - lam) * imgs[idx2]
        labels_1h = torch.nn.functional.one_hot(labels, len(self.labels)).float()
        labels_2h = torch.nn.functional.one_hot(labels[idx2], len(self.labels)).float()

        mixed_labels = lam * labels_1h + (1 - lam) * labels_2h
        return mixed_imgs, mixed_labels

    # ---------------------- Mosaic ----------------------
    def mosaic(self, imgs, labels):
        b, c, h, w = imgs.size()
        idx = torch.randperm(b)

        new_imgs = torch.zeros_like(imgs)
        new_labels = torch.nn.functional.one_hot(labels, len(self.labels)).float()

        for i in range(b):
            img1 = imgs[i]
            img2 = imgs[idx[i]]
            img3 = imgs[idx[(i+1) % b]]
            img4 = imgs[idx[(i+2) % b]]

            top = torch.cat([img1[:, :h//2, :w//2], img2[:, :h//2, w//2:]], dim=2)
            bottom = torch.cat([img3[:, h//2:, :w//2], img4[:, h//2:, w//2:]], dim=2)
            new_imgs[i] = torch.cat([top, bottom], dim=1)

        return new_imgs, new_labels

def split_csv(csv_file, train_ratio=0.8, val_ratio=0.1):
    df = pd.read_csv(csv_file)
    from sklearn.model_selection import train_test_split

    train_df, temp_df = train_test_split(df, test_size=1 - train_ratio, random_state=42, stratify=df["label"])
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, stratify=temp_df["label"])

    base_dir = os.path.dirname(csv_file)
    train_csv = os.path.join(base_dir, "train_split.csv")
    val_csv = os.path.join(base_dir, "val_split.csv")
    test_csv = os.path.join(base_dir, "test_split.csv")

    train_df.to_csv(train_csv, index=False)
    val_df.to_csv(val_csv, index=False)
    test_df.to_csv(test_csv, index=False)

    print(f"Splits saved. Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")

    return train_csv, val_csv, test_csv
