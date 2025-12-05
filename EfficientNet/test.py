import argparse
from pathlib import Path
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm
from train import create_dataloaders, get_model

def parse_args():
    parser = argparse.ArgumentParser(description="Test model with Test-Time Augmentation (TTA)")

    parser.add_argument(
        "--data_dir",
        type=str,
        default="/home/ubuntu/DLGP/Data",
        help="Base data directory that contains train/ and Training_set.csv",
    )

    parser.add_argument(
        "--model_name",
        type=str,
        default="efficientnet_b1",
        help="Model architecture name (must match training).",
    )

    parser.add_argument(
        "--img_size",
        type=int,
        default=240,
        help="Image size used during training.",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size for testing.",
    )

    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.15,
        help="Validation ratio used in training (to reproduce same split).",
    )

    parser.add_argument(
        "--test_ratio",
        type=float,
        default=0.15,
        help="Test ratio used in training (to reproduce same split).",
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of DataLoader workers.",
    )

    parser.add_argument(
        "--ckpt_path",
        type=str,
        default=None,
        help="Path to checkpoint (.pth). If None, will use outputs/{model_name}_best.pth",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs_tta",
        help="Directory to save TTA test metrics.",
    )

    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU even if CUDA is available.",
    )

    return parser.parse_args()

def eval_with_tta(model, dataloader, criterion, device):
    """
    Evaluate model on test loader using simple TTA:
    For each batch:
      - run model on original images
      - run model on horizontally flipped images
      - average logits
    """
    model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    all_labels, all_preds = [], []

    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Test (TTA)", leave=False):
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Original predictions
            outputs = model(inputs)

            # Horizontally flipped inputs
            inputs_flipped = torch.flip(inputs, dims=[3])  # flip along width
            outputs_flipped = model(inputs_flipped)

            # Average logits from original + flipped
            outputs_ens = (outputs + outputs_flipped) / 2.0

            loss = criterion(outputs_ens, labels)

            loss_sum += loss.item() * inputs.size(0)
            preds = outputs_ens.argmax(dim=1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

            all_labels.extend(labels.cpu().tolist())
            all_preds.extend(preds.cpu().tolist())

    test_loss = loss_sum / total
    test_acc = correct / total
    return test_loss, test_acc, all_labels, all_preds

def main(args):
    # Output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"Using device: {device}")

    # Data (recreate the same train/val/test split as training)
    dataloaders, class_names, idx_to_label = create_dataloaders(
        data_dir=args.data_dir,
        img_size=args.img_size,
        batch_size=args.batch_size,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        num_workers=args.num_workers,
    )

    print("\nClass names:")
    print(class_names)
    print(f"Test set size: {len(dataloaders['test'].dataset)} samples")

    num_classes = len(class_names)

    # Model
    if args.ckpt_path is None:
        ckpt_path = Path("outputs") / f"{args.model_name}_best.pth"
    else:
        ckpt_path = Path(args.ckpt_path)

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    print(f"\nLoading checkpoint from: {ckpt_path}")
    model = get_model(
        model_name=args.model_name,
        num_classes=num_classes,
        pretrained=False,  # we are loading our trained weights
    )
    model = model.to(device)

    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Loaded checkpoint (epoch={checkpoint.get('epoch', 'N/A')}, val_acc={checkpoint.get('val_acc', 'N/A')})")

    # Criterion (label smoothing does not matter for eval; use plain CE)
    criterion = nn.CrossEntropyLoss()

    # Evaluate with TTA
    test_loss, test_acc, all_labels, all_preds = eval_with_tta(
        model, dataloaders["test"], criterion, device
    )

    print(f"\n[TTA] Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")

    # Classification report & confusion matrix
    report = classification_report(
        all_labels,
        all_preds,
        target_names=class_names,
        output_dict=True,
    )
    cm = confusion_matrix(all_labels, all_preds)

    # Save metrics
    results_df = pd.DataFrame(report).transpose()
    results_path = output_dir / f"results_{args.model_name}_tta.csv"
    results_df.to_csv(results_path)
    print(f"Classification report (TTA) saved to: {results_path}")

    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    cm_path = output_dir / f"confusion_matrix_{args.model_name}_tta.csv"
    cm_df.to_csv(cm_path)
    print(f"Confusion matrix (TTA) saved to: {cm_path}")

    print("\nDone (TTA evaluation).")

if __name__ == "__main__":
    args = parse_args()
    main(args)
