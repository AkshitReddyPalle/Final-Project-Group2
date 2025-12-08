# Human Action Recognition using EfficientNet-B1

This folder contains the full pipeline for a Human Action Recognition (HAR) model using EfficientNet-B1 and the Kaggle HAR dataset. The workflow has four steps:

1. Download the dataset (Kaggle API)
2. Run EDA (eda.py)
3. Train the model (train.py)
4. Test the model with TTA (test.py)

------------------------------------------------------------
**STEP 1 — DOWNLOAD THE DATASET USING KAGGLE API**

Dataset:
https://www.kaggle.com/datasets/meetnagadia/human-action-recognition-har-dataset

1. Get your Kaggle API key:
- Log into Kaggle → Account → API → Create New API Token
- This downloads kaggle.json

2. Configure Kaggle:
pip install kaggle
mkdir -p ~/.kaggle
mv kaggle.json ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json

3. Download the dataset:
mkdir -p Data
cd Data
kaggle datasets download -d meetnagadia/human-action-recognition-har-dataset
unzip human-action-recognition-har-dataset.zip

Only Training_set.csv and train are used for training.

------------------------------------------------------------
**STEP 2 — Run eda.py**

Command:
python3 eda.py --data_dir "Data"

What eda.py does:
- Loads Training_set.csv
- Computes class distribution of all 15 actions
- Saves CSV + bar plot
- Checks overlapping filenames
- Outputs saved in outputs_eda

Purpose:
Provides a quick overview of dataset balance and structure.

------------------------------------------------------------
**STEP 3 — Run train.py**

Command:
python3 train.py --data_dir "Data"

What train.py does:
- Splits dataset into train / val / test (70/15/15)
- Loads EfficientNet-B1 with a custom dropout classification head

Preprocessing used during training:
- Strong augmentation: random crops, flips, rotation, affine transforms, perspective distortion, color jitter
- Mixup augmentation (alpha=0.2)
- Normalization using ImageNet mean/std
- Validation & test images use simple resize + center crop + normalization

These steps help the model generalize better and prevent overfitting.

Training process:
- Uses label smoothing and weight decay
- Option for gradual backbone unfreezing
- Tracks training and validation accuracy/loss
- Saves the best model to outputs/efficientnet_b1_best.pth
- Also saves:
  training_log_efficientnet_b1.csv
  results_efficientnet_b1.csv
  confusion_matrix_efficientnet_b1.csv

------------------------------------------------------------
**STEP 4 — run test.py**

Command:
python3 test.py --data_dir "Data"

What test.py does:
- Loads the best checkpoint
- Applies Test-Time Augmentation (original + flipped)
- Averages predictions for higher accuracy
- Computes final test accuracy and loss
- Saves results to outputs_tta


