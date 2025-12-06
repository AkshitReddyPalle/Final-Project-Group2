import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T
import random
import torch
import numpy as np


BASE_DIR = "/home/ubuntu/Final-Project-Group2/Data/Human Action Recognition"

TRAINING_CSV = os.path.join(BASE_DIR, "Training_set.csv")
TRAIN_IMG_DIR = os.path.join(BASE_DIR, "train")


TESTING_CSV = os.path.join(BASE_DIR, "Testing_set.csv")     # no labels
TEST_IMG_DIR = os.path.join(BASE_DIR, "test")               # test images


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
        image = Image.open(img_name).convert('RGB')
        label = self.label2idx[self.df.iloc[idx]['label']]

        if self.transform:
            image = self.transform(image)

        return image, label

    # MixUp helper
    def mixup(self, imgs, labels):
        lam = np.random.beta(self.alpha, self.alpha)
        idx2 = torch.randperm(imgs.size(0))
        mixed_imgs = lam * imgs + (1 - lam) * imgs[idx2]

        labels_onehot = torch.nn.functional.one_hot(labels, num_classes=len(self.labels)).float()
        labels2_onehot = torch.nn.functional.one_hot(labels[idx2], num_classes=len(self.labels)).float()
        mixed_labels = lam * labels_onehot + (1 - lam) * labels2_onehot
        return mixed_imgs, mixed_labels

    # Mosaic helper
    def mosaic(self, imgs, labels):
        # Combine 4 random images into one (2x2 grid)
        batch_size, c, h, w = imgs.size()
        indices = torch.randperm(batch_size)
        new_imgs = torch.zeros_like(imgs)
        new_labels = torch.zeros_like(
            torch.nn.functional.one_hot(labels, num_classes=len(self.labels)).float()
        )

        for i in range(batch_size):
            img1 = imgs[i]
            img2 = imgs[indices[i % batch_size]]
            img3 = imgs[indices[(i + 1) % batch_size]]
            img4 = imgs[indices[(i + 2) % batch_size]]

            # create 2x2 mosaic
            top = torch.cat([img1[:, :h // 2, :w // 2],
                             img2[:, :h // 2, w // 2:]], dim=2)
            bottom = torch.cat([img3[:, h // 2:, :w // 2],
                                img4[:, h // 2:, w // 2:]], dim=2)
            new_imgs[i] = torch.cat([top, bottom], dim=1)

            # simple label (not mixing labels here)
            new_labels[i] = torch.nn.functional.one_hot(
                labels[i], num_classes=len(self.labels)
            ).float()

        return new_imgs, new_labels


def split_csv(csv_file, train_ratio=0.8, val_ratio=0.1):
    df = pd.read_csv(csv_file)
    from sklearn.model_selection import train_test_split

    # First split: train vs (val+test)
    train_df, temp_df = train_test_split(
        df,
        test_size=(1 - train_ratio),
        random_state=42,
        stratify=df['label']
    )

    # Second split: val vs test (50/50 of temp)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,
        random_state=42,
        stratify=temp_df['label']
    )

    base_dir = os.path.dirname(csv_file)
    train_csv = os.path.join(base_dir, 'train_split.csv')
    val_csv = os.path.join(base_dir, 'val_split.csv')
    test_csv = os.path.join(base_dir, 'test_split.csv')

    train_df.to_csv(train_csv, index=False)
    val_df.to_csv(val_csv, index=False)
    test_df.to_csv(test_csv, index=False)

    print(f"Splits saved. Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    print(f"  Train CSV: {train_csv}")
    print(f"  Val CSV:   {val_csv}")
    print(f"  Test CSV:  {test_csv}")
    return train_csv, val_csv, test_csv


# Optional: small example to verify everything runs
if __name__ == "__main__":
    # 1) Make train/val/test splits from Training_set.csv
    train_csv, val_csv, test_csv = split_csv(TRAINING_CSV)

    # 2) Simple transforms (resize + normalize for e.g. Inception v3)
    transform = T.Compose([
        T.Resize((299, 299)),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5],
                    std=[0.5, 0.5, 0.5]),
    ])

    # 3) Create one dataset just to test
    train_dataset = HARDataset(train_csv, TRAIN_IMG_DIR, transform=transform)
    print("Train dataset length:", len(train_dataset))

    img, label = train_dataset[0]
    print("Single sample image shape:", img.shape)
    print("Single sample label index:", label)
