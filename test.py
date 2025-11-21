import os
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm


# Same CNN model as train.py

class BaselineCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        def block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, 3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2)
            )

        self.features = nn.Sequential(
            block(3, 32),
            block(32, 64),
            block(64, 128),
            block(128, 256),
            block(256, 512)
        )

        self.avg = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avg(x)
        return self.classifier(x)



# Dataset for Test Images

class HARTestDataset(Dataset):
    def __init__(self, csv_path, img_dir, class_to_idx, transform=None):
        self.data = pd.read_csv(csv_path)
        self.img_dir = img_dir
        self.transform = transform

        # Reverse map for predictions
        self.idx_to_class = {v: k for k, v in class_to_idx.items()}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        filename = self.data.iloc[idx]["filename"]
        img_path = os.path.join(self.img_dir, filename)

        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)

        return img, filename



# Main Test Function

def run_test():
    base_dir = "/home/ubuntu/HAR/Human Action Recognition"
    test_csv = os.path.join(base_dir, "Testing_set.csv")
    test_img_dir = os.path.join(base_dir, "test")

    # Load training CSV to recover label ordering
    train_csv_path = os.path.join(base_dir, "Training_set.csv")
    train_df = pd.read_csv(train_csv_path)
    classes = sorted(train_df["label"].unique())
    class_to_idx = {c: i for i, c in enumerate(classes)}

    num_classes = len(classes)


    # Preprocessing (NO augmentation)

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    dataset = HARTestDataset(test_csv, test_img_dir, class_to_idx, transform)
    loader = DataLoader(dataset, batch_size=32, shuffle=False)

    # Load trained model

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = BaselineCNN(num_classes=num_classes).to(device)
    model.load_state_dict(torch.load("best_cnn.pth", map_location=device))
    model.eval()

    predictions = []


    # Inference Loop

    with torch.no_grad():
        for imgs, filenames in tqdm(loader):
            imgs = imgs.to(device)
            outputs = model(imgs)
            preds = outputs.argmax(1).cpu().numpy()

            for fname, p in zip(filenames, preds):
                pred_label = classes[p]
                predictions.append((fname, pred_label))


    # Save submission.csv

    submission_df = pd.DataFrame(predictions, columns=["filename", "label"])
    submission_df.to_csv("submission.csv", index=False)
    print("\n Submission saved as submission.csv")
    print(submission_df.head())


if __name__ == "__main__":
    run_test()
