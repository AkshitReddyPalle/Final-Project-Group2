import torch
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from train import BaselineCNN  # Import your model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------- GradCAM Class ---------------- #
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output.detach()

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def __call__(self, input_tensor, class_idx=None):
        output = self.model(input_tensor)

        if class_idx is None:
            class_idx = torch.argmax(output)

        self.model.zero_grad()
        output[0, class_idx].backward()

        gradients = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (gradients * self.activations).sum(dim=1).squeeze()

        cam = np.maximum(cam.cpu(), 0)
        cam = cam / cam.max()

        return cam.cpu().numpy()

# ---------------- Load Model ---------------- #
def load_model():
    base_dir = "/home/ubuntu/HAR/Human Action Recognition"
    train_csv = base_dir + "/Training_set.csv"

    import pandas as pd
    classes = sorted(pd.read_csv(train_csv)["label"].unique())

    model = BaselineCNN(num_classes=len(classes)).to(device)
    model.load_state_dict(torch.load("best_cnn.pth", map_location=device))
    model.eval()

    return model, classes

# ------------ GradCAM Visualization ------------ #
def visualize_gradcam(image_path):
    model, classes = load_model()
    target_layer = model.features[-1]  # Last convolution layer
    gradcam = GradCAM(model, target_layer)

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    img = Image.open(image_path).convert("RGB")
    input_tensor = transform(img).unsqueeze(0).to(device)

    cam = gradcam(input_tensor)
    pred = torch.argmax(model(input_tensor)).item()
    pred_label = classes[pred]

    # Resize heatmap to image size
    cam = cv2.resize(cam, (256, 256))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)

    # Convert PIL → cv2 for blending
    img_cv = cv2.cvtColor(np.array(img.resize((256,256))), cv2.COLOR_RGB2BGR)
    overlay = cv2.addWeighted(img_cv, 0.5, heatmap, 0.5, 0)

    # Display
    plt.figure(figsize=(8, 8))
    plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    plt.title(f"Predicted: {pred_label}", fontsize=14)
    plt.axis("off")
    plt.show()

if __name__ == "__main__":
    # Random test image from folder
    import random

    test_folder = "/home/ubuntu/HAR/Human Action Recognition/test"
    img_file = random.choice(os.listdir(test_folder))
    test_image = os.path.join(test_folder, img_file)

    visualize_gradcam(test_image)

