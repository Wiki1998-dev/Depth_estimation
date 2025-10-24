import torch
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from torchvision.transforms import Compose, Resize, ToTensor, Normalize

print(f"Using device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")

from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation

# ------------------------------
# Config
# ------------------------------
IMAGE_PATH = "ChatGPT Image Apr 15, 2025, 02_31_45 PM.png"


# ------------------------------
# Setup device
# ------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------------------
# Load MiDaS for Depth Estimation
# ------------------------------
midas = torch.hub.load("intel-isl/MiDaS", "DPT_Large", trust_repo=True)
midas.to(device).eval()

# Depth preprocessing
transform_midas = Compose([
    Resize(384),
    ToTensor(),
    Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# ------------------------------
# Load SegFormer (ADE20K-trained)
# ------------------------------
feature_extractor = SegformerFeatureExtractor.from_pretrained("nvidia/segformer-b5-finetuned-ade-640-640")
segformer = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b5-finetuned-ade-640-640").to(device).eval()

# ------------------------------
# Load image
# ------------------------------
original_image = Image.open(IMAGE_PATH).convert("RGB")
image_np = np.array(original_image)

# ------------------------------
# Depth Prediction
# ------------------------------
input_midas = transform_midas(original_image).unsqueeze(0).to(device)
with torch.no_grad():
    depth = midas(input_midas)
    depth_resized = torch.nn.functional.interpolate(
        depth.unsqueeze(1),
        size=original_image.size[::-1],
        mode="bicubic",
        align_corners=False
    ).squeeze()
depth_map = depth_resized.cpu().numpy()
depth_normalized = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())
depth_colormap = (depth_normalized * 255).astype(np.uint8)
depth_colormap = cv2.applyColorMap(depth_colormap, cv2.COLORMAP_INFERNO)

# ------------------------------
# Segmentation Prediction
# ------------------------------
inputs = feature_extractor(images=original_image, return_tensors="pt").to(device)
with torch.no_grad():
    outputs = segformer(**inputs)
seg_logits = outputs.logits
seg_mask = torch.argmax(seg_logits, dim=1)[0].cpu().numpy()

# ------------------------------
# Colorize Segmentation Mask
# ------------------------------
# Generate random colormap
num_classes = segformer.config.num_labels
np.random.seed(42)
colors = np.random.randint(0, 255, size=(num_classes, 3), dtype=np.uint8)

color_mask = colors[seg_mask]
color_mask_resized = cv2.resize(color_mask, original_image.size)

# ------------------------------
# Fusion
# ------------------------------
fusion = cv2.addWeighted(depth_colormap, 0.6, color_mask_resized, 0.4, 0)

# ------------------------------
# Legend
# ------------------------------
unique_classes = np.unique(seg_mask)
from matplotlib.patches import Patch

legend_labels = segformer.config.id2label
legend_patches = [Patch(facecolor=np.array(colors[c])/255.0, label=legend_labels[c]) for c in unique_classes]

# ------------------------------
# Display
# ------------------------------
plt.figure(figsize=(20, 10))

plt.subplot(1, 4, 1)
plt.imshow(original_image)
plt.axis('off')
plt.title("Original")

plt.subplot(1, 4, 2)
plt.imshow(depth_colormap[..., ::-1])  # BGR to RGB
plt.axis('off')
plt.title("Depth Map")

plt.subplot(1, 4, 3)
plt.imshow(color_mask_resized)
plt.axis('off')
plt.title("SegFormer Segmentation")
plt.legend(handles=legend_patches, bbox_to_anchor=(1.05, 0.5), loc='center left', borderaxespad=0.)

plt.subplot(1, 4, 4)
plt.imshow(fusion[..., ::-1])  # BGR to RGB
plt.axis('off')
plt.title("Fused Depth + Segmentation")

plt.tight_layout()
plt.show()


import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib.patches import Patch
from transformers import (
    AutoImageProcessor,
    AutoModelForDepthEstimation,
    SegformerFeatureExtractor,
    SegformerForSemanticSegmentation
)

# ----------------------------------------
# Load Image
# ----------------------------------------
IMAGE_PATH = "360_F_287986158_2Tz2w7QKcgmbpecZZzveGUdN9RNPB3c4.jpg"
original_image = Image.open(IMAGE_PATH).convert("RGB")

# ----------------------------------------
# Setup Device
# ----------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ----------------------------------------
# Load Depth Anything V2 - Large Version for Max Accuracy
# ----------------------------------------
depth_processor = AutoImageProcessor.from_pretrained("LiheYoung/depth-anything-large-hf")
depth_model = AutoModelForDepthEstimation.from_pretrained("LiheYoung/depth-anything-large-hf").to(device).eval()

depth_inputs = depth_processor(images=original_image, return_tensors="pt").to(device)

with torch.no_grad():
    depth_outputs = depth_model(**depth_inputs)

depth_post = depth_processor.post_process_depth_estimation(
    depth_outputs,
    target_sizes=[(original_image.height, original_image.width)]
)

depth_map = depth_post[0]["predicted_depth"]
depth_np = depth_map.squeeze().cpu().numpy()
depth_normalized = (depth_np - depth_np.min()) / (depth_np.max() - depth_np.min())
depth_colormap = cv2.applyColorMap((depth_normalized * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)

# ----------------------------------------
# Load SegFormer (Segformer-B5 ADE20K) for High Accuracy Semantic Segmentation
# ----------------------------------------
segformer_processor = SegformerFeatureExtractor.from_pretrained("nvidia/segformer-b5-finetuned-ade-640-640")
segformer = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b5-finetuned-ade-640-640").to(device).eval()

seg_inputs = segformer_processor(images=original_image, return_tensors="pt").to(device)

with torch.no_grad():
    seg_outputs = segformer(**seg_inputs)

seg_mask = torch.argmax(seg_outputs.logits, dim=1)[0].cpu().numpy()

# ----------------------------------------
# Colorize Segmentation and Fuse with Depth
# ----------------------------------------
num_classes = segformer.config.num_labels
np.random.seed(42)
colors = np.random.randint(0, 255, size=(num_classes, 3), dtype=np.uint8)

color_mask = colors[seg_mask]
color_mask = cv2.resize(color_mask, original_image.size)
fusion = cv2.addWeighted(depth_colormap, 0.6, color_mask, 0.4, 0)

# ----------------------------------------
# Build Segmentation Legend
# ----------------------------------------
unique_classes = np.unique(seg_mask)
legend_patches = [
    Patch(facecolor=np.array(colors[c]) / 255.0, label=segformer.config.id2label[c])
    for c in unique_classes if c in segformer.config.id2label
]

# ----------------------------------------
# Visualization
# ----------------------------------------
plt.figure(figsize=(20, 10))

plt.subplot(1, 4, 1)
plt.imshow(original_image)
plt.axis('off')
plt.title("Original Image")

plt.subplot(1, 4, 2)
plt.imshow(depth_colormap[..., ::-1])  # BGR to RGB
plt.axis('off')
plt.title("Depth Anything V2 - Large")

plt.subplot(1, 4, 3)
plt.imshow(color_mask)
plt.axis('off')
plt.title("SegFormer-B5 Segmentation")
plt.legend(handles=legend_patches, bbox_to_anchor=(1.05, 0.5), loc='center left', borderaxespad=0.)

plt.subplot(1, 4, 4)
plt.imshow(fusion[..., ::-1])
plt.axis('off')
plt.title("Fused Depth + Segmentation")

plt.tight_layout()
plt.show()