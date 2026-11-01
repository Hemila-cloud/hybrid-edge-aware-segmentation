import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================

IMAGE_DIR = r"C:\Users\Hemila Saravanan\Downloads\Hybrid_Rooftop_Solar\images_roof_centered_geotiff"

MASK_DIR = r"C:\Users\Hemila Saravanan\Downloads\Hybrid_Rooftop_Solar\masks_segments_reviewed"


# ============================================================
# FILE TO INSPECT
# ============================================================

image_name = "10"


# ============================================================
# PATHS
# ============================================================

image_path = os.path.join(
    IMAGE_DIR,
    image_name + ".tif"
)

mask_path = os.path.join(
    MASK_DIR,
    image_name + ".png"
)


# ============================================================
# READ IMAGE AND MASK
# ============================================================

image = cv2.imread(
    image_path,
    cv2.IMREAD_COLOR
)

mask = cv2.imread(
    mask_path,
    cv2.IMREAD_GRAYSCALE
)


# ============================================================
# CHECK
# ============================================================

if image is None:
    print("ERROR: Image could not be loaded.")
    exit()

if mask is None:
    print("ERROR: Mask could not be loaded.")
    exit()


# Convert BGR → RGB
image_rgb = cv2.cvtColor(
    image,
    cv2.COLOR_BGR2RGB
)


# ============================================================
# DISPLAY
# ============================================================

plt.figure(figsize=(12, 5))


# Original image
plt.subplot(1, 2, 1)

plt.imshow(image_rgb)

plt.title("Original Image - 10")

plt.axis("off")


# Mask
plt.subplot(1, 2, 2)

plt.imshow(
    mask,
    cmap="tab10"
)

plt.title("RID Mask - 10")

plt.colorbar(
    fraction=0.046,
    pad=0.04
)

plt.axis("off")


plt.tight_layout()

plt.show()