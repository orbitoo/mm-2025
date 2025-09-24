import torch
import torch.nn.functional as F
import cv2
import numpy as np


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.feature_maps = None
        self.gradients = None
        self.target_layer.register_forward_hook(self._save_feature_maps)

    def _save_feature_maps(self, module, input, output):
        self.feature_maps = output

    def _save_gradients(self, grad):
        self.gradients = grad

    def generate_heatmap(self, input_tensor, class_idx):
        self.model.eval()
        _, class_output, _ = self.model(input_tensor)
        self.feature_maps.register_hook(self._save_gradients)  # type: ignore
        if class_idx is None:
            class_idx = torch.argmax(class_output).item()
        self.model.zero_grad()
        one_hot = torch.zeros_like(class_output)
        one_hot[0][class_idx] = 1  # type: ignore
        class_output.backward(gradient=one_hot, retain_graph=True)

        if self.gradients is None or self.feature_maps is None:
            raise ValueError("Gradients or feature maps have not been captured.")
        weights = F.adaptive_avg_pool2d(self.gradients, 1)
        heatmap = self.feature_maps * weights
        heatmap = torch.sum(heatmap, dim=1, keepdim=True)
        heatmap = F.relu(heatmap)
        heatmap = F.interpolate(
            heatmap,
            size=(input_tensor.size(2), input_tensor.size(3)),
            mode="bilinear",
            align_corners=False,
        )
        heatmap = heatmap.squeeze().detach().cpu().numpy()
        heatmap = (heatmap - np.min(heatmap)) / (
            np.max(heatmap) - np.min(heatmap) + 1e-8
        )
        return heatmap


def plot_cam(heatmap, original_image_path, save_path):
    img = cv2.imread(original_image_path)
    if img is None:
        raise ValueError(f"Image at path {original_image_path} could not be loaded.")
    heatmap = np.uint8(255 * heatmap)
    heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)  # type: ignore
    heatmap_colored = cv2.resize(heatmap_colored, (img.shape[1], img.shape[0]))
    superimposed_img = heatmap_colored * 0.4 + img
    superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)
    cv2.imwrite(save_path, superimposed_img)
