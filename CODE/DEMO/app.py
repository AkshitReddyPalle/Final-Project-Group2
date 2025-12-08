import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from PIL import Image
from torchvision import transforms, models
import streamlit as st

# Import helpers from your EfficientNet training script
from train import get_model, get_transforms, DEFAULT_MODEL_NAME, DEFAULT_IMG_SIZE




BASE_DIR = Path(".").resolve()
OUTPUT_DIR = BASE_DIR / "outputs"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


DEFAULT_CLASS_NAMES = [
    "calling", "clapping", "cycling", "dancing", "drinking",
    "eating", "fighting", "hugging", "laughing", "listening_to_music",
    "running", "sitting", "sleeping", "texting", "using_laptop",
]

CHECKPOINT_FILES = {
    "EfficientNet-B1": "efficientnet_b1_best.pth",
    "VGG16":           "best_vgg16.pth",
    "ResNet-50":       "best_resnet50.pth",
    "InceptionV3":     "best_inception_v3.pth",
    "Custom CNN":      "best_cnn.pth",
}

BASE_MODEL_LABELS = list(CHECKPOINT_FILES.keys())
MODEL_OPTIONS = BASE_MODEL_LABELS + ["All models"]


# BaselineCNN definition

class BaselineCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        def block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, 3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2)
            )

        self.features = nn.Sequential(
            block(3, 32),
            block(32, 64),
            block(64, 128),
            block(128, 256),
            block(256, 512)
        )

        self.avg = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avg(x)
        return self.classifier(x)

# PREPROCESSING

@st.cache_resource
def get_eval_transform():
    """
    Use the same eval transform as in train.py
    (Resize/CenterCrop/ToTensor/Normalize).
    """
    _, eval_t = get_transforms(DEFAULT_IMG_SIZE)
    return eval_t


def preprocess_image(img: Image.Image) -> torch.Tensor:
    """Convert uploaded PIL image into model input tensor."""
    if img.mode != "RGB":
        img = img.convert("RGB")
    transform = get_eval_transform()
    x = transform(img).unsqueeze(0)  # (1, C, H, W)
    return x.to(device)


# MODEL BUILDERS FOR TORCHVISION MODELS

def build_vgg16(num_classes: int):
    m = models.vgg16(weights=None)
    in_features = m.classifier[6].in_features
    m.classifier[6] = nn.Linear(in_features, num_classes)
    return m


def build_resnet50(num_classes: int):
    m = models.resnet50(weights=None)
    in_features = m.fc.in_features
    m.fc = nn.Linear(in_features, num_classes)
    return m


def build_inception_v3(num_classes: int):
    # aux_logits=True as in typical training; we will ignore AuxLogits weights
    m = models.inception_v3(weights=None, aux_logits=True)
    in_features = m.fc.in_features
    m.fc = nn.Linear(in_features, num_classes)
    return m

# MODEL LOADING

@st.cache_resource(show_spinner=True)
def load_model(model_label: str):
    """
    Generic loader that:
    - Uses train.py/get_model for EfficientNet-B1.
    - Uses torchvision architectures for VGG16/ResNet50/InceptionV3.
    - Uses BaselineCNN for Custom CNN, handling both full-model and state_dict.
    """

    ckpt_name = CHECKPOINT_FILES[model_label]
    ckpt_path = OUTPUT_DIR / ckpt_name

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")


    checkpoint = torch.load(
        ckpt_path,
        map_location=device,
        weights_only=False
    )


    class_names = DEFAULT_CLASS_NAMES
    if isinstance(checkpoint, dict) and "class_names" in checkpoint:
        class_names = checkpoint["class_names"]

    num_classes = len(class_names)

    # EfficientNet path
    if model_label == "EfficientNet-B1":
        if not (isinstance(checkpoint, dict) and "model_state_dict" in checkpoint):
            raise ValueError("EfficientNet checkpoint must contain 'model_state_dict'.")
        model = get_model(
            model_name=DEFAULT_MODEL_NAME,
            num_classes=num_classes,
            pretrained=False,
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        model.eval()
        return model, class_names

    #  VGG/ResNet/Inception
    if model_label == "VGG16":
        model = build_vgg16(num_classes)
    elif model_label == "ResNet-50":
        model = build_resnet50(num_classes)
    elif model_label == "InceptionV3":
        model = build_inception_v3(num_classes)
    elif model_label == "Custom CNN":


        #BaselineCNN
        if isinstance(checkpoint, BaselineCNN):
            model = checkpoint
            model.to(device)
            model.eval()
            return model, class_names

        # dict
        if isinstance(checkpoint, dict) and isinstance(checkpoint.get("model", None), BaselineCNN):
            model = checkpoint["model"]
            model.to(device)
            model.eval()
            return model, class_names

        # state_dict
        model = BaselineCNN(num_classes)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        return model, class_names

    else:
        raise ValueError(f"Unknown model label: {model_label}")


    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint


    if model_label == "InceptionV3":
        filtered_state_dict = {}
        for k, v in state_dict.items():
            if not k.startswith("AuxLogits."):
                filtered_state_dict[k] = v
        state_dict = filtered_state_dict

    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    return model, class_names


def predict_single(img: Image.Image, model_label: str, top_k: int = 5):
    """Run selected model on a single image and return top-k predictions."""
    model, class_names = load_model(model_label)
    x = preprocess_image(img)

    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1)[0]

    top_k = min(top_k, len(class_names))
    values, indices = torch.topk(probs, k=top_k)

    results = []
    for p, idx in zip(values.tolist(), indices.tolist()):
        results.append(
            {"class": class_names[idx], "probability": float(p)}
        )
    return results


# STREAMLIT UI

def main():
    st.set_page_config(page_title="HAR – Multi-Model Demo", layout="centered")

    st.title("Human Action Recognition – Multi-Model Demo")
    st.write(
        "Upload an image from the HAR dataset and choose a model "
        "or run **all models** to compare their predictions."
    )

    # Sidebar
    st.sidebar.header("Settings")

    model_label = st.sidebar.selectbox(
        "Select model", MODEL_OPTIONS, index=0
    )

    top_k = st.sidebar.slider(
        "Show top-k predictions", min_value=1, max_value=10, value=5
    )

    st.sidebar.write(f"Running on device: **{device}**")

    uploaded_file = st.file_uploader(
        "Upload an image (JPG/PNG)", type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded image", use_column_width=True)

        if st.button("Classify action"):
            if model_label != "All models":
                # Single model mode
                with st.spinner(f"Running {model_label}..."):
                    results = predict_single(image, model_label=model_label, top_k=top_k)

                top1 = results[0]
                st.success(
                    f"**Predicted action ({model_label}):** "
                    f"`{top1['class']}` "
                    f"(probability: {top1['probability']:.3f})"
                )

                st.write("Top-k predictions:")
                st.table({
                    "Class": [r["class"] for r in results],
                    "Probability": [f"{r['probability']:.3f}" for r in results],
                })
            else:
                # All models mode
                st.subheader("Comparison across all models")
                rows = []
                with st.spinner("Running all models..."):
                    for m_label in BASE_MODEL_LABELS:
                        try:
                            results = predict_single(image, model_label=m_label, top_k=top_k)
                            top1 = results[0]
                            rows.append({
                                "Model": m_label,
                                "Top-1 Class": top1["class"],
                                "Top-1 Prob": f"{top1['probability']:.3f}",
                            })
                        except Exception as e:
                            rows.append({
                                "Model": m_label,
                                "Top-1 Class": f"ERROR: {type(e).__name__}",
                                "Top-1 Prob": "-",
                            })

                st.table(rows)

    else:
        st.info("Upload a sample image to get started.")


if __name__ == "__main__":
    main()

