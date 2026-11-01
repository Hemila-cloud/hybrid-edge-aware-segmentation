import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# STEP 8B - MASK CLASS VISUALIZATION
# ============================================================

MASK_DIR = r"C:\Users\Hemila Saravanan\Downloads\Hybrid_Rooftop_Solar\masks_segments_reviewed"

MASK_FILE = "10.png"


# ============================================================
# READ MASK
# ============================================================

mask_path = os.path.join(
    MASK_DIR,
    MASK_FILE
)

mask = cv2.imread(
    mask_path,
    cv2.IMREAD_GRAYSCALE
)


if mask is None:
    raise FileNotFoundError(
        f"Could not read mask: {mask_path}"
    )


# ============================================================
# FIND CLASSES
# ============================================================

classes = np.unique(mask)


print("=" * 60)
print("STEP 8B - MASK CLASS VISUALIZATION")
print("=" * 60)

print("Mask:", MASK_FILE)

print("Classes present:")

print(classes)


# ============================================================
# CREATE FIGURE
# ============================================================

fig, axes = plt.subplots(
    3,
    6,
    figsize=(18, 10)
)


axes = axes.flatten()


# ============================================================
# DISPLAY EACH CLASS
# ============================================================

for class_value in range(18):

    class_mask = (
        mask == class_value
    ).astype(np.uint8) * 255


    axes[class_value].imshow(
        class_mask,
        cmap="gray"
    )

    axes[class_value].set_title(
        f"Class {class_value}"
    )

    axes[class_value].axis("off")


# ============================================================
# TITLE
# ============================================================

plt.suptitle(
    "RID Mask - Individual Class Visualization",
    fontsize=16
)


plt.tight_layout()


# ============================================================
# SAVE
# ============================================================

OUTPUT_DIR = os.path.join(
    "outputs",
    "class_analysis"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


output_path = os.path.join(
    OUTPUT_DIR,
    "mask_10_all_classes.png"
)


plt.savefig(
    output_path,
    dpi=200,
    bbox_inches="tight"
)


plt.show()


print("\nVisualization saved to:")

print(output_path)


print("\n✓ STEP 8B COMPLETED")