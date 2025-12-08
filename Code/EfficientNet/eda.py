import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

def parse_args():
    parser = argparse.ArgumentParser(description="EDA for HAR dataset")

    parser.add_argument(
        "--data_dir",
        type=str,
        default="/home/ubuntu/DLGP/Data",
        help="Base data directory that contains train/, test/, Training_set.csv, Testing_set.csv",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs_eda",
        help="Directory to save EDA figures and summary tables.",
    )

    return parser.parse_args()

def main(args):
    data_dir = Path(args.data_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_csv = data_dir / "Training_set.csv"
    test_csv = data_dir / "Testing_set.csv"

    if not train_csv.exists():
        raise FileNotFoundError(f"Training_set.csv not found at {train_csv}")
    if not test_csv.exists():
        raise FileNotFoundError(f"Testing_set.csv not found at {test_csv}")

    # Load CSVs
    df_train = pd.read_csv(train_csv)
    df_test = pd.read_csv(test_csv)

    print("=== Basic Info ===")
    print("Train shape:", df_train.shape)
    print("Test shape:", df_test.shape)
    print("\nTrain columns:", df_train.columns.tolist())
    print("Test columns:", df_test.columns.tolist())

    # Unique labels
    train_labels = sorted(df_train["label"].unique())
    print("\nNumber of classes:", len(train_labels))
    print("Classes:", train_labels)

    # Class distribution in training set
    train_counts = df_train["label"].value_counts().sort_index()
    print("\nClass distribution (train):")
    print(train_counts)

    # Save class distribution as CSV
    train_counts.to_csv(out_dir / "class_distribution_train.csv", header=["count"])

    # Plot class distribution
    plt.figure(figsize=(10, 5))
    train_counts.plot(kind="bar")
    plt.title("Class Distribution - Training Set")
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(out_dir / "class_distribution_train.png")
    plt.close()

    # Check for filename overlap between train and test (sanity check)
    train_files = set(df_train["filename"].tolist())
    test_files = set(df_test["filename"].tolist())
    overlap = train_files.intersection(test_files)
    print(f"\nFilename overlap between train and test: {len(overlap)} images")
    with open(out_dir / "filename_overlap.txt", "w") as f:
        f.write(f"Number of overlapping filenames: {len(overlap)}\n")
        if len(overlap) > 0:
            f.write("Examples:\n")
            for i, name in enumerate(sorted(list(overlap))[:50]):
                f.write(name + "\n")

    print(f"\nEDA complete. Outputs saved to: {out_dir}")

if __name__ == "__main__":
    args = parse_args()
    main(args)
