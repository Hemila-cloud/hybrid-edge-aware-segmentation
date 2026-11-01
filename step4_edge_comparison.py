import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# STEP 4B - EDGE PRESERVATION COMPARISON
# ============================================================

IMAGE_DIR = r"C:\Users\Hemila Saravanan\Downloads\Hybrid_Rooftop_Solar\images_roof_centered_geotiff"

OUTPUT_DIR = r"C:\Users\Hemila Saravanan\Downloads\Hybrid_Rooftop_Solar\outputs\preprocessing"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# LOAD IMAGE
# ============================================================

image_file = "10.tif"

image_path = os.path.join(
    IMAGE_DIR,
    image_file
)

image = cv2.imread(
    image_path,
    cv2.IMREAD_COLOR
)

if image is None:
    print("ERROR: Image could not be loaded.")
    exit()


# ============================================================
# CONVERT TO GRAYSCALE
# ============================================================

gray = cv2.cvtColor(
    image,
    cv2.COLOR_BGR2GRAY
)


# ============================================================
# GAUSSIAN
# ============================================================

gaussian = cv2.GaussianBlur(
    gray,
    (5, 5),
    0
)


# ============================================================
# MEDIAN
# ============================================================

median = cv2.medianBlur(
    gray,
    5
)


# ============================================================
# CLAHE
# ============================================================

clahe = cv2.createCLAHE(
    clipLimit=2.0,
    tileGridSize=(8, 8)
)

clahe_image = clahe.apply(gray)


# ============================================================
# CANNY EDGE DETECTION
# ============================================================

edges_original = cv2.Canny(
    gray,
    50,
    150
)

edges_gaussian = cv2.Canny(
    gaussian,
    50,
    150
)

edges_median = cv2.Canny(
    median,
    50,
    150
)

edges_clahe = cv2.Canny(
    clahe_image,
    50,
    150
)


# ============================================================
# CALCULATE EDGE PIXELS
# ============================================================

edge_counts = {

    "Original": np.count_nonzero(
        edges_original
    ),

    "Gaussian": np.count_nonzero(
        edges_gaussian
    ),

    "Median": np.count_nonzero(
        edges_median
    ),

    "CLAHE": np.count_nonzero(
        edges_clahe
    )
}


# ============================================================
# DISPLAY
# ============================================================

plt.figure(figsize=(12, 10))


plt.subplot(2, 2, 1)

plt.imshow(
    edges_original,
    cmap="gray"
)

plt.title("Original + Canny")

plt.axis("off")


plt.subplot(2, 2, 2)

plt.imshow(
    edges_gaussian,
    cmap="gray"
)

plt.title("Gaussian + Canny")

plt.axis("off")


plt.subplot(2, 2, 3)

plt.imshow(
    edges_median,
    cmap="gray"
)

plt.title("Median + Canny")

plt.axis("off")


plt.subplot(2, 2, 4)

plt.imshow(
    edges_clahe,
    cmap="gray"
)

plt.title("CLAHE + Canny")

plt.axis("off")


plt.tight_layout()


# ============================================================
# SAVE FIGURE
# ============================================================

output_path = os.path.join(
    OUTPUT_DIR,
    "edge_comparison_10.png"
)

plt.savefig(
    output_path,
    dpi=200,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# PRINT RESULTS
# ============================================================

print("=" * 60)
print("STEP 4B - EDGE PRESERVATION")
print("=" * 60)

print("\nNumber of detected edge pixels:")

for method, count in edge_counts.items():

    print(
        f"{method:<12}: {count}"
    )

print("\nComparison saved to:")

print(output_path)