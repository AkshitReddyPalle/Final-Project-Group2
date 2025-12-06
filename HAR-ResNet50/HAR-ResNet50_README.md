# Human Activity Recognition using Fine-Tuned ResNet-50

This project performs **Human Activity Recognition (HAR)** using a **transfer-learning-based ResNet-50** model pretrained on ImageNet.  
The model classifies **15 human activities** from RGB images, including:

> calling, clapping, cycling, dancing, drinking, eating, fighting, hugging,  
> laughing, listening_to_music, running, sitting, sleeping, texting, using_laptop

---

## 📂 Project Structure

```
HAR-ResNet50/
│
├── code/
│   ├── create_val_split.py      # Create train/val folders using CSV
│   ├── train_resnet50.py        # Final ResNet-50 training script
│
├── data/ (⚠ excluded in GitHub due to size)
│   ├── train/
│   ├── val/
│
├── model/ (⚠ excluded in GitHub due to size)
│   └── best_resnet50.pth
│
├── notebook/
│   └── HAR_ResNet50_Finetuned.ipynb
│
├── HAR-ResNet50_README.md
└── requirements.txt
```

> ⚠ The dataset & trained model are not uploaded due to size limitations  
> ✔ They can be recreated using the provided scripts

---

## 📊 Dataset Information

- Source: **Kaggle — Human Action Recognition Dataset**
- **15 balanced action classes**
- Images include real-world variations in background, pose & lighting
- Ground truth labels via CSV
- Train/Val split = **85% / 15%**
- Split created by: `create_val_split.py`
- **Seed = 42** for reproducibility

---

## 🧠 Model Architecture & Hyperparameters

| Component | Setting |
|----------|---------|
| Base Model | ResNet-50 pretrained on ImageNet |
| Trainable Layers | Layer4 + Fully Connected |
| Input Size | 224 × 224 |
| Batch Size | 32 |
| Epochs | 15 |
| Loss Function | CrossEntropyLoss |
| Optimizer | Adam |
| Scheduler | StepLR (step_size=5, gamma=0.1) |
| Learning Rate | 1e-4 |
| Device | Auto CUDA if available |

---

## 🚀 Training Instructions (in Google Colab)

### 1️⃣ Mount Drive & navigate to project directory
```python
from google.colab import drive
drive.mount('/content/drive')
%cd /content/drive/MyDrive/HAR-ResNet50/code
```

### 2️⃣ Create train/val split
```bash
python create_val_split.py
```

### 3️⃣ Train the model
```bash
python train_resnet50.py \
  --data_dir ../data \
  --save_dir ../model \
  --epochs 15 \
  --device auto
```

> ⚡ GPU required for reasonable training speed

---

## 🏆 Results

| Metric | Score |
|--------|------|
| **Best Validation Accuracy** | **0.8196 (~82%)** |
| **Macro F1-Score** | **0.82** |
| **Classes** | 15 |

📌 Example output:
```
=== Best Validation Accuracy: 0.8196 ===
macro avg f1-score: 0.82
weighted avg f1-score: 0.82
```

Model evaluation includes:
- **Classification Report** (Precision / Recall / F1 per class)
- **Confusion Matrix**

---

## 🔮 Future Enhancements

- Fine-tune earlier ResNet blocks for additional performance
- Stronger data augmentation for confusing static poses
- Explore **3D CNN / ConvLSTM** for video-temporal learning
- Real-time deployment on edge devices

---

## 👤 Author

**Shaik Mohammad Mujahid Khalandar**  
Final Project — Group 2  
The George Washington University  

---

## 🙌 Acknowledgements

- **PyTorch Team** — Pretrained ResNet-50 weights  
- **Kaggle** — Human Action Recognition Dataset  

---
