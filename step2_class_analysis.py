import os
import cv2
import numpy as np


# ============================================================
# STEP 2 - RID MASK CLASS ANALYSIS
# ============================================================

IMAGE_DIR = r"C:\Users\Hemila Saravanan\Downloads\Hybrid_Rooftop_Solar\images_roof_centered_geotiff"

MASK_DIR = r"C:\Users\Hemila Saravanan\Downloads\Hybrid_Rooftop_Solar\masks_segments_reviewed"


# ============================================================
# GET MASK FILES
# ============================================================

mask_files = sorted([
    file
    for file in os.listdir(MASK_DIR)
    if file.lower().endswith(".png")
])


print("=" * 60)
print("STEP 2 - RID MASK CLASS ANALYSIS")
print("=" * 60)

print("Number of masks:", len(mask_files))


# ============================================================
# INITIALIZE CLASS COUNTS
# ============================================================

class_pixel_counts = {
    class_id: 0
    for class_id in range(18)
}


# ============================================================
# ANALYZE ALL MASKS
# ============================================================

for index, mask_file in enumerate(mask_files):

    mask_path = os.path.join(
        MASK_DIR,
        mask_file
    )

    mask = cv2.imread(
        mask_path,
        cv2.IMREAD_GRAYSCALE
    )

    if mask is None:
        print("Could not read:", mask_file)
        continue

    unique_values, counts = np.unique(
        mask,
        return_counts=True
    )

    for value, count in zip(unique_values, counts):

        value = int(value)

        if value in class_pixel_counts:

            class_pixel_counts[value] += int(count)


# ============================================================
# TOTAL PIXELS
# ============================================================

total_pixels = sum(class_pixel_counts.values())


# ============================================================
# DISPLAY RESULTS
# ============================================================

print("\n" + "=" * 60)
print("CLASS-WISE PIXEL DISTRIBUTION")
print("=" * 60)

print(
    f"{'Class':<10}"
    f"{'Pixels':<18}"
    f"{'Percentage':<12}"
)

print("-" * 40)


for class_id in range(18):

    pixels = class_pixel_counts[class_id]

    percentage = (
        pixels / total_pixels * 100
        if total_pixels > 0
        else 0
    )

    print(
        f"{class_id:<10}"
        f"{pixels:<18}"
        f"{percentage:.2f}%"
    )


# ============================================================
# CLASSES FOUND
# ============================================================

classes_found = [
    class_id
    for class_id in range(18)
    if class_pixel_counts[class_id] > 0
]


print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)

print("Classes found:")

print(classes_found)

print("\nNumber of classes found:")

print(len(classes_found))