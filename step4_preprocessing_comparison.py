import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# STEP 4A - PREPROCESSING COMPARISON
# ============================================================

IMAGE_DIR = r"C:\Users\Hemila Saravanan\Downloads\Hybrid_Rooftop_Solar\images_roof_centered_geotiff"

OUTPUT_DIR = r"C:\Users\Hemila Saravanan\Downloads\Hybrid_Rooftop_Solar\outputs\preprocessing"


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# SELECT IMAGE
# ============================================================

image_file = "10.tif"

image_path = os.path.join(
    IMAGE_DIR,
    image_file
)


# ============================================================
# READ IMAGE
# ============================================================

image = cv2.imread(
    image_path,
    cv2.IMREAD_COLOR
)


if image is None:

    print("ERROR: Image could not be loaded.")

    exit()


# Convert BGR → RGB
image_rgb = cv2.cvtColor(
    image,
    cv2.COLOR_BGR2RGB
)


# ============================================================
# 1. GAUSSIAN FILTER
# ============================================================

gaussian = cv2.GaussianBlur(
    image_rgb,
    (5, 5),
    0
)


# ============================================================
# 2. MEDIAN FILTER
# ============================================================

median = cv2.medianBlur(
    image_rgb,
    5
)


# ============================================================
# 3. CLAHE
# ============================================================

# Convert RGB → LAB
lab = cv2.cvtColor(
    image_rgb,
    cv2.COLOR_RGB2LAB
)

L, A, B = cv2.split(lab)


clahe = cv2.createCLAHE(
    clipLimit=2.0,
    tileGridSize=(8, 8)
)


L_enhanced = clahe.apply(L)


lab_enhanced = cv2.merge(
    [L_enhanced, A, B]
)


clahe_image = cv2.cvtColor(
    lab_enhanced,
    cv2.COLOR_LAB2RGB
)


# ============================================================
# 4. NORMALIZATION
# ============================================================

normalized = cv2.normalize(
    image_rgb,
    None,
    0,
    255,
    cv2.NORM_MINMAX
)


normalized = normalized.astype(
    np.uint8
)


# ============================================================
# DISPLAY
# ============================================================

plt.figure(
    figsize=(15, 10)
)


plt.subplot(2, 3, 1)

plt.imshow(image_rgb)

plt.title("Original")

plt.axis("off")


plt.subplot(2, 3, 2)

plt.imshow(gaussian)

plt.title("Gaussian Filter")

plt.axis("off")


plt.subplot(2, 3, 3)

plt.imshow(median)

plt.title("Median Filter")

plt.axis("off")


plt.subplot(2, 3, 4)

plt.imshow(clahe_image)

plt.title("CLAHE")

plt.axis("off")


plt.subplot(2, 3, 5)

plt.imshow(normalized)

plt.title("Normalization")

plt.axis("off")


plt.tight_layout()


# ============================================================
# SAVE COMPARISON
# ============================================================

output_path = os.path.join(
    OUTPUT_DIR,
    "preprocessing_comparison_10.png"
)


plt.savefig(
    output_path,
    dpi=200,
    bbox_inches="tight"
)


plt.show()


print("=" * 60)
print("STEP 4A - PREPROCESSING COMPARISON")
print("=" * 60)

print("Image:", image_file)

print("\nGenerated:")
print("Gaussian filtered image")
print(" Median filtered image")
print(" CLAHE enhanced image")
print(" Normalized image")

print("\nComparison saved to:")

print(output_path)