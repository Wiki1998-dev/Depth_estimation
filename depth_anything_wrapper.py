# depth_anything_wrapper.py

import os
import sys
import torch
import numpy as np
import torchvision.transforms as T
from PIL import Image

# Add Depth Anything repo to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'Depth-Anything-V2'))

from dpt.models import DPTDepthModel

class DepthAnythingWrapper:
    def __init__(self, model_path: str, device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = DPTDepthModel(
            path=model_path,
            backbone="vitl",
            non_negative=True,
            enable_attention_hooks=False
        ).to(self.device)
        self.model.eval()

        self.transform = T.Compose([
            T.Resize((518, 518)),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5],
                        std=[0.5, 0.5, 0.5])
        ])

    def predict(self, image: Image.Image) -> np.ndarray:
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            prediction = self.model(input_tensor)
            depth = prediction.squeeze().cpu().numpy()
            depth_normalized = (depth - depth.min()) / (depth.max() - depth.min())

        return depth_normalized
