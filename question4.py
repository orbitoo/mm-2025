import os
import warnings
import torch
from PIL import Image

from dann import DANN
from grad_cam import GradCAM, plot_cam
from train import data_transform

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "./dann_model.pth"
NUM_CLASSES = 4
PLOT_DIR = "./grad_cam_results"
SPECTROGRAM_DIR = "./spectrograms/target/unknown"
os.makedirs(PLOT_DIR, exist_ok=True)

if __name__ == "__main__":
    # ignore UserWarning
    warnings.filterwarnings("ignore", category=UserWarning)
    print("Loading model...")
    model = DANN(num_classes=NUM_CLASSES).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()
    target_layer = model.feature_extractor[8]
    grad_cam = GradCAM(model, target_layer)
    class_names = ["B", "IR", "OR", "N"]
    images_to_analyze = [
        ("A_4.png", None),
        ("B_104.png", None),
        ("C_150.png", None),
        ("D_200.png", None),
        ("E_270.png", None),
        ("F_350.png", None),
        ("G_400.png", None),
        ("H_440.png", None),
        ("I_500.png", None),
        ("J_600.png", None),
        ("K_630.png", None),
        ("L_700.png", None),
        ("M_750.png", None),
        ("N_800.png", None),
        ("O_880.png", None),
        ("P_965.png", None),
    ]
    for img_name, class_idx in images_to_analyze:
        img_path = os.path.join(SPECTROGRAM_DIR, img_name)
        img = Image.open(img_path).convert("RGB")
        input_tensor = data_transform(img).unsqueeze(0).to(DEVICE)  # type: ignore
        heatmap = grad_cam.generate_heatmap(input_tensor, class_idx)
        filename = os.path.basename(img_path)
        save_path = os.path.join(PLOT_DIR, f"cam_{filename}")
        plot_cam(heatmap, img_path, save_path)
