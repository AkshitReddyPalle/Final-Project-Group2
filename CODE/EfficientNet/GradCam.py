import argparse
from pathlib import Path

import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from torchvision import transforms
import matplotlib.cm as cm

# Import helpers from your training script
from train import get_model, get_transforms, DEFAULT_MODEL_NAME, DEFAULT_IMG_SIZE

# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_DIR = Path(__file__).resolve().parent
CKPT_PATH = BASE_DIR / "outputs" / "efficientnet_b1_best.pth"

# Fallback class names if not stored in checkpoint
DEFAULT_CLASS_NAMES = [
    "calling", "clapping", "cycling", "dancing", "drinking",
    "eating", "fighting", "hugging", "laughing", "listening_to_music",
    "running", "sitting", "sleeping", "texting", "using_laptop",
]


# ---------------------------------------------------------------------
# GRADCAM IMPLEMENTATION
# ---------------------------------------------------------------------
class GradCAM:
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.target_layer = target_layer

        self.feature_maps = None
        self.gradients = None

        # Forward hook: save feature maps
        def forward_hook(module, input, output):
            self.feature_maps = output.detach()

        # Backward hook: save gradients
        def backward_hook(module, grad_input, grad_output):
            # grad_output is a tuple; take gradients w.r.t. the feature maps
            self.gradients = grad_output[0].detach()

        target_layer.register_forward_hook(forward_hook)
        target_layer.register_full_backward_hook(backward_hook)

    def __call__(self, x: torch.Tensor, class_idx: int | None = None):
        """
        x: input tensor of shape (1, C, H, W)
        Returns:
            cam (H, W) numpy array in [0, 1]
            pred_idx (int) predicted class index
            probs (1, num_classes) tensor of probabilities
        """
        self.model.zero_grad()

        logits = self.model(x)
        probs = F.softmax(logits, dim=1)

        if class_idx is None:
            class_idx = int(torch.argmax(probs, dim=1).item())

        score = logits[:, class_idx]
        score.backward()

        # feature_maps: (B, C, H, W)
        # gradients:    (B, C, H, W)
        gradients = self.gradients           # (1, C, H, W)
        feature_maps = self.feature_maps     # (1, C, H, W)

        # Global-average-pool the gradients over spatial dims
        weights = gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)

        # Weighted sum of feature maps
        cam = (weights * feature_maps).sum(dim=1, keepdim=True)  # (1, 1, H, W)
        cam = torch.relu(cam)

        # Normalize CAM to [0, 1]
        cam = cam.squeeze().cpu().numpy()
        cam -= cam.min()
        if cam.max() > 0:
            cam /= cam.max()

        return cam, class_idx, probs.detach()


# ---------------------------------------------------------------------
# UTILITIES
# ---------------------------------------------------------------------
def load_model_and_classes():
    if not CKPT_PATH.exists():
        raise FileNotFoundError(f"Checkpoint not found: {CKPT_PATH}")

    checkpoint = torch.load(CKPT_PATH, map_location=DEVICE)

    class_names = checkpoint.get("class_names", DEFAULT_CLASS_NAMES)
    num_classes = len(class_names)

    model = get_model(
        model_name=DEFAULT_MODEL_NAME,  # e.g. "efficientnet_b1"
        num_classes=num_classes,
        pretrained=False,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()

    return model, class_names


def get_eval_transform():
    """Use the same eval transform as in train.py."""
    _, eval_t = get_transforms(DEFAULT_IMG_SIZE)
    return eval_t


def preprocess_image(img: Image.Image) -> torch.Tensor:
    if img.mode != "RGB":
        img = img.convert("RGB")
    transform = get_eval_transform()
    x = transform(img).unsqueeze(0)  # (1, C, H, W)
    return x.to(DEVICE)


def overlay_cam_on_image(img: Image.Image, cam: np.ndarray, alpha: float = 0.5) -> Image.Image:
    """
    img: PIL RGB image
    cam: (H, W) array in [0, 1]
    alpha: blending factor for heatmap
    """
    cam_uint8 = np.uint8(cam * 255)

    # Resize CAM to image size
    cam_img = Image.fromarray(cam_uint8).resize(img.size, resample=Image.BILINEAR)
    cam_norm = np.array(cam_img, dtype=np.float32) / 255.0

    # Apply color map (jet)
    colormap = cm.get_cmap("jet")
    heatmap = colormap(cam_norm)[..., :3]  # (H, W, 3), RGB in [0,1]
    heatmap = np.uint8(heatmap * 255)
    heatmap_pil = Image.fromarray(heatmap)

    img_rgb = img.convert("RGB")
    blended = Image.blend(img_rgb, heatmap_pil, alpha=alpha)
    return blended


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------
def run_gradcam(image_path: Path, out_path: Path | None = None):
    # Load model
    model, class_names = load_model_and_classes()

    # Pick a target layer for EfficientNet
    if hasattr(model, "features"):
        target_layer = model.features[-1]
    elif hasattr(model, "blocks"):
        target_layer = model.blocks[-1]
    else:
        # Fallback: second-to-last child module
        target_layer = list(model.children())[-2]

    gradcam = GradCAM(model, target_layer)

    # Load and preprocess image
    img = Image.open(image_path)
    x = preprocess_image(img)

    # Run Grad-CAM
    cam, pred_idx, probs = gradcam(x)
    prob = float(probs[0, pred_idx].item())
    pred_label = class_names[pred_idx] if pred_idx < len(class_names) else str(pred_idx)

    print(f"Top-1 prediction: {pred_label} (prob = {prob:.4f}, class_idx = {pred_idx})")

    # Overlay and save
    overlay = overlay_cam_on_image(img, cam, alpha=0.5)

    if out_path is None:
        out_path = image_path.with_name(image_path.stem + "_gradcam.jpg")

    overlay.save(out_path)
    print(f"Grad-CAM overlay saved to: {out_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Grad-CAM for EfficientNet-B1 HAR model")
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to input image (e.g., from HAR test set)",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Optional path to save Grad-CAM overlay image",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    image_path = Path(args.image).expanduser().resolve()
    out_path = Path(args.out).expanduser().resolve() if args.out is not None else None

    run_gradcam(image_path, out_path)


if __name__ == "__main__":
    main()

