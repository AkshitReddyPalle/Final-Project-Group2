# DATS-6303-PROJECTGROUP-2
**Course:** DATS 6303 – DEEP LEARNING 

**Project:** Human Action Recognition

# Overview

This project aims to develop a deep learning–based Human Action Recognition (HAR) system that classifies human activities from still RGB images. The system identifies 15 distinct action categories such as walking, clapping, cycling, and jumping. We explore multiple computer vision architectures, including a Custom CNN, VGG16, ResNet50, DCAM-Net, InceptionV3, and EfficientNet/ViT, to evaluate their performance on the HAR image dataset. The pipeline includes image preprocessing, data augmentation, model training, and evaluation using accuracy, precision, recall, and F1-score. The final phase will focus on model optimization and deployment for real-time activity recognition.

# GitHub Directory Structure

# Components

# Credits 

**Group Members:** Akshit Reddy Palle, Ritu Patel, Dhatrija Sukasi, Ayush Meshram, Shaik Mohammed Mujahid Khalandar

**University:** George Washington University

**Instructor:** Dr. Amir Jafari



**CNN Baseline Model**
1. Model Overview

This work implements a baseline Convolutional Neural Network (CNN) for the Human Action
Recognition (HAR) dataset containing 15 human activity classes.
The objective is to establish a strong, optimized baseline before comparing with advanced
models such as ResNet50, InceptionV3, EfficientNet, ViT, and DCAM-Net.
The baseline CNN was progressively improved using:
● Extensive data augmentation
● RandAugment
● Random Erasing
● Label smoothing
● Class-balanced weighted sampling
● Cosine LR scheduling with warmup
● 5-stage deep CNN architecture with BatchNorm + Dropout
The final improvement achieved ~0.50 validation F1-score, a major enhancement over the
naive CNN baseline (~0.12–0.20 F1 in first attempts).

**2. Dataset Summary**

The dataset comes from Kaggle (“Human Action Recognition”), containing 12,000+ labeled
images across 15 classes:
calling, clapping, cycling, dancing, drinking,
eating, fighting, hugging, laughing,
listening_to_music, running, sitting,
sleeping, texting, using_laptop

Dataset Files:
● train/ → labeled images (Training_
set.csv contains label mapping)
● test/ → unlabeled images (Testing_
set.csv defines submission order)
● Training_
set.csv → (filename, label)
● Testing_
set.csv → expected test predictions order
Images have large variance in:
● pose
● illumination
● zoom level
● background clutter
● subject scale
This makes HAR from still images a non-trivial task.

**3. Baseline CNN Architecture**

The final baseline CNN consists of 5 convolutional blocks, each containing:
Conv2D → BatchNorm → ReLU → MaxPool
Feature Extractor
Block Channels Output Downsample
Block 1 3 → 32 1/2
Block 2 32 → 64 1/4
Block 3 64 → 128 1/8
Block 4 128 → 256 1/16
Block 5 256 → 512 1/32
Final representation → AdaptiveAvgPool2d((1,1))
Classifier
Flatten
Linear(512 → 256)
ReLU
Dropout(0.5)
Linear(256 → 15)
Total parameters: ~7.4M

**4. Training Strategy**

Strong Data Augmentation
Used to prevent overfitting and improve generalization:
● RandomResizedCrop(256, scale=(0.75,1.0))
● RandAugment (2 ops, magnitude 9)
● RandomHorizontalFlip
● RandomErasing (p=0.25)
● Label Smoothing (0.1)
Prevents overconfidence and stabilizes gradients.

Class-Balanced Weighted Sampling
The dataset is imbalanced across 15 classes.
A weighted random sampler ensures each class appears with equal probability during training.
Warmup + Cosine Annealing Scheduler
● Warmup: 3 epochs (stabilizes early training)
● Cosine decay for 40 epochs
No Early Stopping
The model saves only the best epoch (best validation F1).

**5. Training Results**

Final Model Performance
Across 40 epochs:

Metric Value
Best Validation F1 0.4981
Best Validation Accuracy ~0.5095
Final Train F1 0.5031
Final Train Accuracy 0.5157

📈 Progression Snapshot
Early epochs:
Epoch 01 → Val F1 ≈ 0.12
Epoch 10 → Val F1 ≈ 0.28
Epoch 20 → Val F1 ≈ 0.39
Epoch 30 → Val F1 ≈ 0.46
Epoch 40 → Val F1 ≈ 0.49

**6. Confusion Matrix & Class Behavior (Interpretation)**

Although not computed here, based on F1 trends:
Model performs well on high-contrast, distinct classes
cycling
using_
laptop
dancing
calling
Model struggles on visually similar poses
hugging vs laughing
sitting vs using_
laptop
drinking vs eating
sleeping vs sitting
This is expected because CNNs must infer fine-grained body pose from still images without
temporal cues.

**7. Why CNN Baseline Is Important**
CNN baseline:
● Establishes a solid comparison for deep transfer learning methods
● Shows the impact of architectural depth, augmentation, and regularization
● Demonstrates improvements from:
○ BatchNorm
○ Dropout
○ Weighted sampling
○ Learning rate schedules
○ Despite being "baseline,
" it reaches almost 50% F1, setting a meaningful benchmark for the
teammates' models.

**8. Key Lessons Learned**

● Strong augmentations dramatically improve CNN generalization.
● Balanced sampling is essential for imbalanced action datasets.
● Label smoothing prevents overconfident misclassifications.
● Warmup + cosine LR improves stability and final metrics.
● Custom CNNs struggle with fine-grained human actions compared to pretrained models.

**9. Conclusion**

The custom CNN baseline achieved a Best Validation F1 of 0.498, representing a significant
improvement from simpler CNN versions (~0.20 F1).
This baseline establishes a fair starting point for comparing transfer learning models like:
● ResNet50
● InceptionV3
● EfficientNet
● Vision Transformer
● DCAM-Net

The optimized CNN is competitive given the complexity of the dataset and the lack of pretrained
features.
