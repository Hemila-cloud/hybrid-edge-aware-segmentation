import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================

MASK_DIR = r"C:\Users\Hemila Saravanan\Downloads\Hybrid_Rooftop_Solar\masks_segments_reviewed"


# ============================================================
# SELECT MASK
# ============================================================

mask_file = "10.png"

mask_path = os.path.join(
    MASK_DIR,
    mask_file
)


# ============================================================
# READ MASK
# ============================================================

mask = cv2.imread(
    mask_path,
    cv2.IMREAD_GRAYSCALE
)


if mask is None:

    print("ERROR: Mask could not be loaded.")
    exit()


# ============================================================
# CREATE FIXED COLOUR MAP
# ============================================================

np.random.seed(42)

class_colors = np.random.randint(
    0,
    256,
    size=(18, 3),
    dtype=np.uint8
)


# ============================================================
# CREATE COLOUR IMAGE
# ============================================================

colored_mask = np.zeros(
    (mask.shape[0], mask.shape[1], 3),
    dtype=np.uint8
)


for class_id in range(18):

    colored_mask[mask == class_id] = class_colors[class_id]


# Convert BGR → RGB
colored_mask_rgb = cv2.cvtColor(
    colored_mask,
    cv2.COLOR_BGR2RGB
)


# ============================================================
# DISPLAY
# ============================================================

plt.figure(figsize=(12, 6))

plt.imshow(colored_mask_rgb)

plt.title(
    "RID Mask Classes - 10.png"
)

plt.axis("off")
print("\nClass bounding boxes:")

for class_id in range(18):

    ys, xs = np.where(mask == class_id)

    if len(xs) == 0:
        continue

    x_min = xs.min()
    x_max = xs.max()
    y_min = ys.min()
    y_max = ys.max()

    print(
        f"Class {class_id:2d}: "
        f"pixels={len(xs):6d}, "
        f"bbox=({x_min},{y_min}) -> ({x_max},{y_max})"
    )

plt.show()


# ============================================================
# DISPLAY CLASS VALUES PRESENT
# ============================================================

print("=" * 60)
print("MASK CLASS VISUALIZATION")
print("=" * 60)

print("Mask:", mask_file)

print("Classes present:")

print(np.unique(mask))