# HAR-ResNet50: Human Activity Recognition using ResNet50

This project implements **Human Activity Recognition (HAR)** using a **fine-tuned ResNet-50** deep learning model.  
The dataset contains 15 human activity classes such as running, hugging, sitting, dancing, drinking, etc.

---

## 🚀 Project Structure

```
HAR-ResNet50/
│
├── code/
│   ├── create_val_split.py
│   ├── train_resnet50.py
│
├── data/
│   ├── train/
│   ├── val/
│
├── model/
│   └── resnet50_har_best_finetuned.pth
│
├── notebook/
│   └── HAR_ResNet50_Finetuned.ipynb
│
├── har_raw.zip
└── requirements_1.txt
```

---

## 🔧 Training Instructions

### 1️⃣ Prepare Dataset  
Upload the dataset ZIP (`har_raw.zip`) into Colab:

```python
!unzip har_raw.zip -d har_raw
```

Then build train/val split:

```python
python code/create_val_split.py
```

---

## 2️⃣ Train the ResNet‑50 Model

```bash
python code/train_resnet50.py --epochs 15 --lr 0.0001 --batch_size 32
```

Automatically detects GPU (CUDA).

---

## 🔒 Reproducibility (Seed Fixing)

```python
import random, numpy as np, torch

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
```

This is integrated in `train_resnet50.py`.

---

## 🧪 Best Model Performance

- **Validation Accuracy:** ~0.81  
- Stable results across all classes  
- Saved model: `resnet50_har_best_finetuned.pth`

---

## 📁 How to Use in GitHub

Commit this structure:

```
HAR-ResNet50/
    code/
    data/
    model/
    notebook/
    README.md
```

---

## 🧑‍💻 Author  
Shaik Mohammad Mujahid Khalandar  
Final-Project-Group2

---

## 🎯 Final Notes  
Your project is **complete, reproducible, and ready for submission**.
