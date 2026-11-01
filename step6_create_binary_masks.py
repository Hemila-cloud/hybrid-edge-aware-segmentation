import os
import cv2
import numpy as np


# ============================================================
# STEP 6A - CREATE BINARY REFERENCE MASKS
# ============================================================

MASK_DIR = r"C:\Users\Hemila Saravanan\Downloads\Hybrid_Rooftop_Solar\masks_segments_reviewed"

OUTPUT_DIR = r"C:\Users\Hemila Saravanan\Downloads\Hybrid_Rooftop_Solar\outputs\binary_masks"


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# GET ALL MASKS
# ============================================================

mask_files = sorted([
    file
    for file in os.listdir(MASK_DIR)
    if file.lower().endswith(".png")
])


print("=" * 60)
print("STEP 6A - BINARY REFERENCE MASK CREATION")
print("=" * 60)

print(
    "Total masks found:",
    len(mask_files)
)


# ============================================================
# PROCESS MASKS
# ============================================================

successful = 0
failed = 0


for index, mask_file in enumerate(
    mask_files,
    start=1
):

    mask_path = os.path.join(
        MASK_DIR,
        mask_file
    )


    # --------------------------------------------------------
    # READ ORIGINAL MASK
    # --------------------------------------------------------

    mask = cv2.imread(
        mask_path,
        cv2.IMREAD_GRAYSCALE
    )


    if mask is None:

        print(
            "[ERROR] Could not read:",
            mask_file
        )

        failed += 1

        continue


    # --------------------------------------------------------
    # CREATE BINARY MASK
    #
    # Class 17 = background
    # Classes 0-16 = foreground
    # --------------------------------------------------------

    binary_mask = np.where(
        mask == 17,
        0,
        255
    ).astype(
        np.uint8
    )


    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    output_path = os.path.join(
        OUTPUT_DIR,
        mask_file
    )


    cv2.imwrite(
        output_path,
        binary_mask
    )


    successful += 1


    # --------------------------------------------------------
    # PROGRESS
    # --------------------------------------------------------

    if (
        index % 100 == 0
        or index == len(mask_files)
    ):

        print(
            f"Processed {index}/{len(mask_files)} masks"
        )


# ============================================================
# FINAL RESULTS
# ============================================================

print("\n" + "=" * 60)
print("BINARY MASK CREATION COMPLETED")
print("=" * 60)

print(
    "Successful:",
    successful
)

print(
    "Failed:",
    failed
)


if successful == 1880 and failed == 0:

    print(
        "\n ALL 1880 BINARY MASKS CREATED SUCCESSFULLY"
    )

else:

    print(
        "\n CHECK THE RESULTS"
    )


print("\nOutput folder:")

print(OUTPUT_DIR)