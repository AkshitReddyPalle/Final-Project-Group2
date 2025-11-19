import os
import shutil
import random
import pandas as pd

base_dir = r"C:\Users\smdkh\HAR-ResNet50\data"

# Flat image folders from Kaggle
train_flat_dir = os.path.join(base_dir, "train")
test_flat_dir  = os.path.join(base_dir, "test")

# Where we'll build class folders (same dirs, but with subfolders)
val_dir = os.path.join(base_dir, "val")

train_csv = os.path.join(base_dir, "Training_set.csv")
test_csv  = os.path.join(base_dir, "Testing_set.csv")

val_ratio = 0.15
random.seed(42)

# -------------------------
# 1. Organize TRAIN into class folders + create VAL
# -------------------------
print("Reading Training_set.csv ...")
df_train = pd.read_csv(train_csv)

img_col   = df_train.columns[0]   # first column = image name
label_col = df_train.columns[1]   # second column = label

labels = df_train[label_col].unique()
print("Found classes:", labels)

for label in labels:
    # All rows for this class
    rows = df_train[df_train[label_col] == label]
    img_names = rows[img_col].tolist()
    random.shuffle(img_names)

    n_total = len(img_names)
    n_val   = int(val_ratio * n_total)

    val_imgs   = set(img_names[:n_val])
    train_imgs = set(img_names[n_val:])

    # Class folders
    train_class_dir = os.path.join(train_flat_dir, label)
    val_class_dir   = os.path.join(val_dir, label)

    os.makedirs(train_class_dir, exist_ok=True)
    os.makedirs(val_class_dir,   exist_ok=True)

    # Put images into train/<label>
    for img in train_imgs:
        src = os.path.join(train_flat_dir, img)
        if not os.path.exists(src):
            continue
        dst = os.path.join(train_class_dir, img)
        if not os.path.exists(dst):
            shutil.copy(src, dst)

    # Put images into val/<label>
    for img in val_imgs:
        src = os.path.join(train_flat_dir, img)
        if not os.path.exists(src):
            continue
        dst = os.path.join(val_class_dir, img)
        if not os.path.exists(dst):
            shutil.copy(src, dst)

print("Organized TRAIN into class folders + created VAL.")

# -------------------------
# 2. Organize TEST into class folders
# -------------------------
print("Reading Testing_set.csv ...")
df_test = pd.read_csv(test_csv)

img_col_t   = df_test.columns[0]
label_col_t = df_test.columns[1]

test_labels = df_test[label_col_t].unique()
print("Found test classes:", test_labels)

for label in test_labels:
    rows = df_test[df_test[label_col_t] == label]
    img_names = rows[img_col_t].tolist()

    test_class_dir = os.path.join(test_flat_dir, label)
    os.makedirs(test_class_dir, exist_ok=True)

    for img in img_names:
        src = os.path.join(test_flat_dir, img)
        if not os.path.exists(src):
            continue
        dst = os.path.join(test_class_dir, img)
        if not os.path.exists(dst):
            shutil.copy(src, dst)

print("Organized TEST into class folders.")
print("✅ Finished preparing train/val/test splits with class folders.")
