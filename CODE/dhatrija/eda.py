import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------- Read Data ----------------
epoch_metrics = pd.read_csv('epoch_metrics.csv')
test_report = pd.read_csv('test_class_report.csv', index_col=0)
test_cm = pd.read_csv('test_confusion_matrix.csv', index_col=0)

# ---------------- Epoch Table ----------------
print("\nEpoch Metrics Table:\n")
print(epoch_metrics.to_string(index=False))

# ---------------- Training vs Validation Accuracy Curve ----------------
plt.figure(figsize=(8,5))
plt.plot(epoch_metrics['epoch'], epoch_metrics['train_acc'], label='Train Accuracy')
plt.plot(epoch_metrics['epoch'], epoch_metrics['val_acc'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training vs Validation Accuracy')
plt.legend()
plt.grid(True)
plt.savefig('accuracy_curve.png', dpi=150)
plt.close()

# ---------------- Training vs Validation F1 Curve ----------------
plt.figure(figsize=(8,5))
plt.plot(epoch_metrics['epoch'], epoch_metrics['train_f1'], label='Train F1')
plt.plot(epoch_metrics['epoch'], epoch_metrics['val_f1'], label='Validation F1')
plt.xlabel('Epoch')
plt.ylabel('Macro F1 Score')
plt.title('Training vs Validation F1 Score')
plt.legend()
plt.grid(True)
plt.savefig('f1_curve.png', dpi=150)
plt.close()

# ---------------- Test Confusion Matrix ----------------
plt.figure(figsize=(10,8))
sns.heatmap(test_cm, annot=True, fmt='d', cmap='Blues')
plt.title('Test Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.savefig('test_confusion_matrix.png', dpi=150)
plt.close()

# ---------------- Per-Class Metrics ----------------
print("\nTest Classification Report:\n")
print(test_report.to_string())

# ---------------- Optional: Per-Class F1 Bar Plot ----------------
plt.figure(figsize=(10,5))
sns.barplot(x=test_report.index[:-3], y=test_report['f1-score'][:-3])
plt.xticks(rotation=45)
plt.ylabel('F1-Score')
plt.title('Per-Class F1 Score (Test Split)')
plt.grid(True)
plt.savefig('per_class_f1.png', dpi=150)
plt.close()

print("\nEDA plots saved: accuracy_curve.png, f1_curve.png, test_confusion_matrix.png, per_class_f1.png")
