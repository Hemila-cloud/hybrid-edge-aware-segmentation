import os
import cv2
import numpy as np


# ============================================================
# STEP 8A - CLASS PRESENCE ANALYSIS
# ============================================================

MASK_DIR = r"C:\Users\Hemila Saravanan\Downloads\Hybrid_Rooftop_Solar\masks_segments_reviewed"


# ============================================================
# GET MASK FILES
# ============================================================

mask_files = sorted([
    file
    for file in os.listdir(MASK_DIR)
    if file.lower().endswith(".png")
])


print("=" * 65)
print("STEP 8A - RID MASK CLASS PRESENCE ANALYSIS")
print("=" * 65)

print(
    "Number of masks:",
    len(mask_files)
)


# ============================================================
# CLASS RANGE
# ============================================================

classes = range(18)


# ============================================================
# STORAGE
# ============================================================

class_image_count = {
    c: 0 for c in classes
}

class_total_pixels = {
    c: 0 for c in classes
}

class_min_pixels = {
    c: float("inf") for c in classes
}

class_max_pixels = {
    c: 0 for c in classes
}


# ============================================================
# PROCESS ALL MASKS
# ============================================================

for index, mask_file in enumerate(
    mask_files,
    start=1
):

    mask_path = os.path.join(
        MASK_DIR,
        mask_file
    )


    mask = cv2.imread(
        mask_path,
        cv2.IMREAD_GRAYSCALE
    )


    if mask is None:

        print(
            "[ERROR] Could not read:",
            mask_file
        )

        continue


    for c in classes:

        count = np.count_nonzero(
            mask == c
        )


        if count > 0:

            class_image_count[c] += 1


        class_total_pixels[c] += count


        class_min_pixels[c] = min(
            class_min_pixels[c],
            count
        )


        class_max_pixels[c] = max(
            class_max_pixels[c],
            count
        )


    if (
        index % 100 == 0
        or index == len(mask_files)
    ):

        print(
            f"Processed {index}/{len(mask_files)} masks"
        )


# ============================================================
# RESULTS
# ============================================================

print("\n" + "=" * 65)
print("CLASS PRESENCE RESULTS")
print("=" * 65)

print(
    f"{'Class':<10}"
    f"{'Images':<12}"
    f"{'Total Pixels':<18}"
    f"{'Avg Pixels/Image':<20}"
    f"{'Min':<12}"
    f"{'Max':<12}"
)

print("-" * 84)


for c in classes:

    total = class_total_pixels[c]

    average = (
        total / len(mask_files)
    )


    if class_min_pixels[c] == float("inf"):

        minimum = 0

    else:

        minimum = class_min_pixels[c]


    maximum = class_max_pixels[c]


    print(
        f"{c:<10}"
        f"{class_image_count[c]:<12}"
        f"{total:<18}"
        f"{average:<20.2f}"
        f"{minimum:<12}"
        f"{maximum:<12}"
    )


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 65)
print("SUMMARY")
print("=" * 65)

print(
    "Classes found:"
)

found_classes = [
    c for c in classes
    if class_image_count[c] > 0
]

print(
    found_classes
)

print(
    "\n✓ CLASS PRESENCE ANALYSIS COMPLETED"
)
