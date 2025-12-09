# DATS-6303-PROJECTGROUP-2
**Course:** DATS 6303 – DEEP LEARNING 

**Project:** Human Action Recognition

# Overview

This project aims to develop a deep learning–based Human Action Recognition (HAR) system that classifies human activities from still RGB images. The system identifies 15 distinct action categories such as walking, clapping, cycling, and jumping. We explore multiple computer vision architectures, including a Custom CNN, VGG16, ResNet50, DCAM-Net, InceptionV3, and EfficientNet/ViT, to evaluate their performance on the HAR image dataset. The pipeline includes image preprocessing, data augmentation, model training, and evaluation using accuracy, precision, recall, and F1-score. The final phase will focus on model optimization and deployment for real-time activity recognition.


# Components

## Training Summary
Model: **Inception v3** (finetuned)

We trained for 15 epochs and monitored accuracy, precision, recall, and F1 on both train and validation splits.

- Best **validation F1**: **0.8364**
- Best **validation accuracy**: **0.8365**
- Best epoch: **Epoch 14**

The best model checkpoint is saved when validation F1 improves.

## Test Split Results (Inception v3)

Evaluated on a held-out test split of **1260 samples** (15 classes × 84 each):

- **Accuracy**: **0.8262** (82.62%)
- **Macro Precision**: **0.8290**
- **Macro Recall**: **0.8262**
- **Macro F1-score**: **0.8266**

Per-class performance is generally strong, with especially high F1 for:

- **cycling** (≈0.98)
- **eating** (≈0.90)
- **sleeping** (≈0.90)
- **running, fighting, hugging, laughing** (≈0.83–0.89)

The confusion matrix and detailed per-class classification report are generated in the test script and summarize how the model confuses similar actions (e.g., sitting vs using_laptop, listening_to_music vs texting).

## Files

- `train.py` – Training loop for Inception v3 with logging and model checkpointing.
- `test.py` – Loads the best checkpoint and computes test metrics (accuracy, precision, recall, F1) and confusion matrix.
- `Dataset.py` – Custom dataset class and transforms for the 15-class action dataset.

# Credits 

**Group Members:** Akshit Reddy Palle, Ritu Patel, Dhatrija Sukasi, Ayush Meshram, Shaik Mohammed Mujahid Khalandar

**University:** George Washington University

**Instructor:** Dr. Amir Jafari
