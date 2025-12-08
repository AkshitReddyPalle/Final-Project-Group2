# DATS-6303-PROJECTGROUP-2
**Course:** DATS 6303 – DEEP LEARNING 

**Project:** Human Action Recognition

# Overview

This project aims to develop a deep learning–based Human Action Recognition (HAR) system that classifies human activities from still RGB images. The system identifies 15 distinct action categories such as walking, clapping, cycling, and jumping. We explore multiple computer vision architectures, including a Custom CNN, VGG16, ResNet50, DCAM-Net, InceptionV3, and EfficientNet/ViT, to evaluate their performance on the HAR image dataset. The pipeline includes image preprocessing, data augmentation, model training, and evaluation using accuracy, precision, recall, and F1-score. The final phase will focus on model optimization and deployment for real-time activity recognition.

## GitHub Directory Structure

```
├── CODE/                             # Source code for models and Streamlit demo
├── Final-Group-Presentation/         # Slides and presentation materials
├── Final-Group-Project-Report/       # Comprehensive final project documentation
├── Group-Proposal/                   # Initial project proposal and scope
├── Individual-Project-Report/        # Individual contributions and reports
├── README.md                         # Project overview (this file)
├── LICENCE                           # MIT LICENSE

```

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


```
# CNN Baseline Model for Human Action Recognition (HAR)

## 1. Model Overview  
This project implements a **baseline Convolutional Neural Network (CNN)** for the Human Action Recognition (HAR) image classification task involving **15 action classes**.

The goal was to develop a strong, well-optimized baseline before comparing with advanced deep models such as **ResNet50, InceptionV3, EfficientNet, Vision Transformer (ViT), and DCAM-Net**.

The baseline CNN was repeatedly improved using:

- Extensive data augmentation  
- RandAugment  
- Random Erasing  
- Label smoothing  
- Class-balanced weighted sampling  
- Cosine learning-rate scheduling with warmup  
- A 5-stage deep CNN architecture with BatchNorm + Dropout  

**Final Baseline Performance:**  
 **~0.50 Validation F1-Score**  
(Compared to the initial naive ~0.12–0.20 F1)

---

## 2. Dataset Summary  

Dataset: **Human Action Recognition (Kaggle)**  
• 12,000+ labeled images  
• 15 human activity classes:

> calling, clapping, cycling, dancing, drinking, eating, fighting, hugging, laughing, listening_to_music, running, sitting, sleeping, texting, using_laptop

### Dataset Files  
train/ → labeled images  
test/ → unlabeled images  
Training_set.csv → filename → label mapping  
Testing_set.csv → image order for submission  

### Dataset Challenges  
The dataset contains heavy variation in:

- Pose  
- Illumination  
- Background clutter  
- Zoom  
- Subject scale  

This makes static-image action recognition a **non-trivial** classification task.

---

## 3. Baseline CNN Architecture  

The final CNN contains **5 convolutional blocks**:

Conv2D → BatchNorm → ReLU → MaxPool

### Feature Extractor

Block 1 : 3 → 32  (Downsample 1/2)  
Block 2 : 32 → 64 (Downsample 1/4)  
Block 3 : 64 → 128 (Downsample 1/8)  
Block 4 : 128 → 256 (Downsample 1/16)  
Block 5 : 256 → 512 (Downsample 1/32)

Final output passes through:  
AdaptiveAvgPool2d((1,1))

### Classifier

Flatten  
Linear(512 → 256)  
ReLU  
Dropout(0.5)  
Linear(256 → 15)

**Total Parameters:** ~7.4M

---

## 4. Training Strategy  

### ✔ Strong Data Augmentation
- RandomResizedCrop(256, scale=(0.75,1.0))  
- RandAugment(2 ops, magnitude=9)  
- RandomHorizontalFlip  
- RandomErasing (p=0.25)  

### ✔ Label Smoothing: 0.1  
### ✔ Class-Balanced Weighted Sampling  
### ✔ Warmup + Cosine Annealing LR Scheduler  
### ✔ Best-Model Saving  

---

## 5. Training Results  

Final Model Performance:

Best Validation F1: **0.4981**  
Best Validation Accuracy: **~0.5095**  
Final Train F1: **0.5031**  
Final Train Accuracy: **0.5157**

### F1 Score Progression  
Epoch 01 → 0.12  
Epoch 10 → 0.28  
Epoch 20 → 0.39  
Epoch 30 → 0.46  
Epoch 40 → 0.49  

---

## 6. Confusion Matrix & Class Behavior (Interpretation)

### Performs well on:
- cycling  
- using_laptop  
- dancing  
- calling  

### Struggles on:
- hugging vs laughing  
- sitting vs using_laptop  
- drinking vs eating  
- sleeping vs sitting  

---

## 7. Importance of the CNN Baseline  

This baseline model:

- Establishes a fair comparison point for advanced models  
- Shows the benefit of augmentation & regularization  
- Provides a competitive non-pretrained benchmark (~50% F1)  

---

## 8. Key Lessons Learned  

- Augmentations significantly boost performance  
- Weighted sampling fixes imbalance  
- Label smoothing stabilizes training  
- Warmup + cosine LR improves gradient flow  
- CNNs struggle with fine-grained human poses  

---

## 9. Conclusion  

The final CNN baseline achieved:

 **0.498 F1-score (Best Validation)**  

This serves as a strong benchmark for comparison against:

- ResNet50  
- InceptionV3  
- EfficientNet  
- Vision Transformer  
- DCAM-Net  

Given dataset difficulty + no pretrained features, the performance is **highly competitive**.
```
 given the complexity of the dataset and the lack of pretrained
features.
