import os
import cv2
import numpy as np


# ============================================================
# STEP 1 - RID DATASET INSPECTION
# ============================================================

# ------------------------------------------------------------
# 1. PATHS
# ------------------------------------------------------------

IMAGE_DIR = r"C:\Users\Hemila Saravanan\Downloads\Hybrid_Rooftop_Solar\images_roof_centered_geotiff"

MASK_DIR = r"C:\Users\Hemila Saravanan\Downloads\Hybrid_Rooftop_Solar\masks_segments_reviewed"


# ------------------------------------------------------------
# 2. FILE EXTENSIONS
# ------------------------------------------------------------

IMAGE_EXTENSIONS = (".tif", ".tiff")
MASK_EXTENSIONS = (".png",)


# ------------------------------------------------------------
# 3. GET IMAGE FILES
# ------------------------------------------------------------

image_files = sorted([
    file
    for file in os.listdir(IMAGE_DIR)
    if file.lower().endswith(IMAGE_EXTENSIONS)
])


# ------------------------------------------------------------
# 4. GET MASK FILES
# ------------------------------------------------------------

mask_files = sorted([
    file
    for file in os.listdir(MASK_DIR)
    if file.lower().endswith(MASK_EXTENSIONS)
])


# ============================================================
# STEP 1A - DATASET COUNT
# ============================================================

print("=" * 60)
print("STEP 1A - DATASET COUNT")
print("=" * 60)

print("Number of images :", len(image_files))
print("Number of masks  :", len(mask_files))


# ============================================================
# STEP 1B - IMAGE-MASK MATCHING
# ============================================================

image_names = {
    os.path.splitext(file)[0]
    for file in image_files
}

mask_names = {
    os.path.splitext(file)[0]
    for file in mask_files
}


images_without_masks = image_names - mask_names
masks_without_images = mask_names - image_names


print("\n" + "=" * 60)
print("STEP 1B - IMAGE-MASK MATCHING")
print("=" * 60)

print("Images without masks :", len(images_without_masks))
print("Masks without images :", len(masks_without_images))


if len(images_without_masks) == 0 and len(masks_without_images) == 0:

    print("\n✓ PERFECT MATCH")
    print("Every image has a corresponding mask.")

else:

    print("\n⚠ MATCHING PROBLEM FOUND")


# ============================================================
# STEP 1C - IMAGE AND MASK PROPERTIES
# ============================================================

print("\n" + "=" * 60)
print("STEP 1C - IMAGE AND MASK PROPERTIES")
print("=" * 60)


# Select first image
first_image_file = image_files[0]

# Remove image extension
image_name = os.path.splitext(first_image_file)[0]


# Find corresponding mask
corresponding_mask = None

for mask_file in mask_files:

    mask_name = os.path.splitext(mask_file)[0]

    if mask_name == image_name:

        corresponding_mask = mask_file
        break


# Safety check
if corresponding_mask is None:

    print("\nERROR: Corresponding mask not found.")

    exit()


# ------------------------------------------------------------
# Read image and mask
# ------------------------------------------------------------

image_path = os.path.join(
    IMAGE_DIR,
    first_image_file
)

mask_path = os.path.join(
    MASK_DIR,
    corresponding_mask
)


# Read TIFF image
image = cv2.imread(
    image_path,
    cv2.IMREAD_UNCHANGED
)


# Read PNG mask
mask = cv2.imread(
    mask_path,
    cv2.IMREAD_UNCHANGED
)


# ------------------------------------------------------------
# Display image information
# ------------------------------------------------------------

print("\nIMAGE")

print("Filename :", first_image_file)

print("Shape    :", image.shape)

print("Datatype :", image.dtype)


# ------------------------------------------------------------
# Display mask information
# ------------------------------------------------------------

print("\nMASK")

print("Filename :", corresponding_mask)

print("Shape    :", mask.shape)

print("Datatype :", mask.dtype)


# ============================================================
# STEP 1D - UNIQUE MASK VALUES
# ============================================================

unique_values = np.unique(mask)


print("\n" + "=" * 60)
print("STEP 1D - MASK PIXEL VALUES")
print("=" * 60)

print("Unique pixel values in mask:")

print(unique_values)

print("\nNumber of unique mask values:")

print(len(unique_values))


# ------------------------------------------------------------
# Pixel distribution
# ------------------------------------------------------------

print("\nMask pixel distribution:")

for value in unique_values:

    count = np.sum(mask == value)

    percentage = (count / mask.size) * 100

    print(
        f"Value {value}: "
        f"{count} pixels "
        f"({percentage:.2f}%)"
    )


# ============================================================
# STEP 1E - IMAGE-MASK DIMENSION CHECK
# ============================================================

print("\n" + "=" * 60)
print("STEP 1E - IMAGE-MASK DIMENSION CHECK")
print("=" * 60)


dimension_mismatches = []


for image_file in image_files:

    image_name = os.path.splitext(image_file)[0]

    # Find corresponding mask
    mask_file = None

    for current_mask in mask_files:

        mask_name = os.path.splitext(current_mask)[0]

        if mask_name == image_name:

            mask_file = current_mask
            break


    if mask_file is None:

        continue


    image_path = os.path.join(
        IMAGE_DIR,
        image_file
    )

    mask_path = os.path.join(
        MASK_DIR,
        mask_file
    )


    img = cv2.imread(
        image_path,
        cv2.IMREAD_UNCHANGED
    )

    msk = cv2.imread(
        mask_path,
        cv2.IMREAD_UNCHANGED
    )


    if img is None or msk is None:

        continue


    # Compare height and width
    if img.shape[:2] != msk.shape[:2]:

        dimension_mismatches.append(
            (
                image_file,
                img.shape,
                mask_file,
                msk.shape
            )
        )


print(
    "Images with dimension mismatch :",
    len(dimension_mismatches)
)


if len(dimension_mismatches) == 0:

    print("✓ All image-mask dimensions match.")

else:

    print("\nMismatched files:")

    for item in dimension_mismatches[:20]:

        print(
            f"Image: {item[0]} {item[1]} "
            f"| Mask: {item[2]} {item[3]}"
        )

# ============================================================
# STEP 1G - ALL MASK CLASSES IN DATASET
# ============================================================

print("\n" + "=" * 60)
print("STEP 1G - ALL MASK CLASSES")
print("=" * 60)

all_values = set()

for mask_file in mask_files:

    mask_path = os.path.join(
        MASK_DIR,
        mask_file
    )

    mask = cv2.imread(
        mask_path,
        cv2.IMREAD_UNCHANGED
    )

    if mask is None:
        continue

    values = np.unique(mask)

    for value in values:
        all_values.add(int(value))


print("All unique mask values across 1880 masks:")

print(sorted(all_values))

print("\nTotal number of classes/values:")

print(len(all_values))
# ============================================================
# STEP 1 COMPLETED
# ============================================================

print("\n" + "=" * 60)
print("STEP 1 DATASET INSPECTION COMPLETED")
print("=" * 60)