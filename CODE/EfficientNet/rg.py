# eda.py
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# -------------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"

TRAIN_LOG_FILE = OUTPUT_DIR / "training_log_efficientnet_b1.csv"
CONF_MAT_FILE = OUTPUT_DIR / "confusion_matrix_efficientnet_b1.csv"
RESULTS_FILE   = OUTPUT_DIR / "results_efficientnet_b1.csv"

FIG_DIR = OUTPUT_DIR / "figures"
FIG_DIR.mkdir(exist_ok=True, parents=True)


# -------------------------------------------------------------------
# 1. TRAINING CURVES (LOSS + ACCURACY)
# -------------------------------------------------------------------
def plot_training_curves():
    if not TRAIN_LOG_FILE.exists():
        print(f"[WARN] Training log not found: {TRAIN_LOG_FILE}")
        return

    df = pd.read_csv(TRAIN_LOG_FILE)

    # epoch column
    epoch_col = "epoch"
    if epoch_col not in df.columns:
        df[epoch_col] = np.arange(1, len(df) + 1)

    # ----- Loss curve -----
    plt.figure(figsize=(7, 5))
    if "train_loss" in df.columns:
        plt.plot(df[epoch_col], df["train_loss"], label="Train loss", marker="o")
    if "val_loss" in df.columns:
        plt.plot(df[epoch_col], df["val_loss"], label="Val loss", marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    loss_path = FIG_DIR / "loss_curve_efficientnet_b1.png"
    plt.savefig(loss_path, dpi=300)
    plt.close()
    print(f"[INFO] Saved loss curve to {loss_path}")

    # ----- Accuracy curve -----
    def pick_acc(col_prefix):
        for name in [f"{col_prefix}_acc", f"{col_prefix}_accuracy"]:
            if name in df.columns:
                return name
        return None

    train_acc_col = pick_acc("train")
    val_acc_col = pick_acc("val")

    if train_acc_col or val_acc_col:
        plt.figure(figsize=(7, 5))
        if train_acc_col:
            plt.plot(df[epoch_col], df[train_acc_col],
                     label="Train accuracy", marker="o")
        if val_acc_col:
            plt.plot(df[epoch_col], df[val_acc_col],
                     label="Val accuracy", marker="o")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("Training vs Validation Accuracy")
        plt.grid(True, linestyle="--", alpha=0.4)
        plt.legend()
        plt.tight_layout()
        acc_path = FIG_DIR / "accuracy_curve_efficientnet_b1.png"
        plt.savefig(acc_path, dpi=300)
        plt.close()
        print(f"[INFO] Saved accuracy curve to {acc_path}")
    else:
        print("[WARN] No accuracy columns found in training log.")


# -------------------------------------------------------------------
# 2. CONFUSION MATRIX HEATMAP
# -------------------------------------------------------------------
def plot_confusion_matrix(normalize=True):
    if not CONF_MAT_FILE.exists():
        print(f"[WARN] Confusion matrix CSV not found: {CONF_MAT_FILE}")
        return

    cm_df = pd.read_csv(CONF_MAT_FILE, index_col=0)
    labels = cm_df.index.tolist()

    cm = cm_df.to_numpy().astype(float)

    if normalize:
        cm_norm = cm / cm.sum(axis=1, keepdims=True)
        data = cm_norm
        title = "Confusion Matrix (row-normalized)"
        fname = "confusion_matrix_normalized_efficientnet_b1.png"
        fmt = ".2f"
    else:
        # counts (still float array, but we format as integer-like)
        data = cm
        title = "Confusion Matrix (counts)"
        fname = "confusion_matrix_counts_efficientnet_b1.png"
        fmt = ".0f"  # <-- changed from "d" to ".0f"

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        data,
        xticklabels=labels,
        yticklabels=labels,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        cbar=True,
        linewidths=0.5,
        square=True,
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    cm_path = FIG_DIR / fname
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"[INFO] Saved confusion matrix to {cm_path}")


# -------------------------------------------------------------------
# 3. PER-CLASS F1 SCORES (BAR PLOT)
# -------------------------------------------------------------------
def plot_class_f1():
    if not RESULTS_FILE.exists():
        print(f"[WARN] Classification results CSV not found: {RESULTS_FILE}")
        return

    df = pd.read_csv(RESULTS_FILE)

    if "f1-score" not in df.columns:
        print("[WARN] 'f1-score' column not found in results file.")
        return

    if "support" in df.columns:
        df = df[df["support"] > 0]

    # Sometimes the class name column is unnamed or has a weird label
    if "class" not in df.columns:
        df.rename(columns={df.columns[0]: "class"}, inplace=True)

    # Drop rows like accuracy / macro avg / weighted avg
    df = df[~df["class"].str.contains("avg|accuracy", case=False, regex=True)]

    plt.figure(figsize=(10, 4))
    sns.barplot(data=df, x="class", y="f1-score")
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("F1-score")
    plt.xlabel("Class")
    plt.title("Per-class F1-scores (EfficientNet-B1)")
    plt.tight_layout()
    f1_path = FIG_DIR / "per_class_f1_efficientnet_b1.png"
    plt.savefig(f1_path, dpi=300)
    plt.close()
    print(f"[INFO] Saved per-class F1 bar plot to {f1_path}")


# -------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------
def main():
    print("=== EDA for EfficientNet-B1 Results ===")
    plot_training_curves()
    plot_confusion_matrix(normalize=True)
    plot_confusion_matrix(normalize=False)
    plot_class_f1()
    print("Done. Check the figures folder:", FIG_DIR)


if __name__ == "__main__":
    main()

