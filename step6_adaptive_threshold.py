import os
import cv2
import numpy as np


# ============================================================
# STEP 6B - ADAPTIVE THRESHOLDING(considers uneven lighting)
# ============================================================

IMAGE_DIR = r"C:\Users\Hemila Saravanan\Downloads\Hybrid_Rooftop_Solar\images_roof_centered_geotiff"

OUTPUT_DIR = r"C:\Users\Hemila Saravanan\Downloads\Hybrid_Rooftop_Solar\outputs\thresholds"


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# GET ALL IMAGES
# ============================================================

image_files = sorted([
    file
    for file in os.listdir(IMAGE_DIR)
    if file.lower().endswith(
        (".tif", ".tiff")
    )
])


print("=" * 60)
print("STEP 6B - ADAPTIVE THRESHOLDING")
print("=" * 60)

print(
    "Total images found:",
    len(image_files)
)


# ============================================================
# PARAMETERS
# ============================================================

MEDIAN_KERNEL = 5

ADAPTIVE_BLOCK_SIZE = 31

ADAPTIVE_C = 5


# ============================================================
# PROCESSING
# ============================================================

successful = 0
failed = 0

total_foreground_pixels = 0


for index, image_file in enumerate(
    image_files,
    start=1
):

    image_path = os.path.join(
        IMAGE_DIR,
        image_file
    )


    # --------------------------------------------------------
    # READ IMAGE
    # --------------------------------------------------------

    image = cv2.imread(
        image_path,
        cv2.IMREAD_COLOR
    )


    if image is None:

        print(
            "[ERROR] Could not read:",
            image_file
        )

        failed += 1

        continue


    # --------------------------------------------------------
    # GRAYSCALE
    # --------------------------------------------------------

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )


    # --------------------------------------------------------
    # MEDIAN FILTER
    # --------------------------------------------------------

    median = cv2.medianBlur(
        gray,
        MEDIAN_KERNEL
    )


    # --------------------------------------------------------
    # ADAPTIVE THRESHOLD
    # --------------------------------------------------------

    threshold = cv2.adaptiveThreshold(
        median,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        ADAPTIVE_BLOCK_SIZE,
        ADAPTIVE_C
    )


    # --------------------------------------------------------
    # MORPHOLOGICAL CLEANUP
    # --------------------------------------------------------

    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (3, 3)
    )


    threshold = cv2.morphologyEx(
        threshold,
        cv2.MORPH_OPEN,
        kernel,
        iterations=1
    )


    threshold = cv2.morphologyEx(
        threshold,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=1
    )


    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    output_name = (
        os.path.splitext(image_file)[0]
        + ".png"
    )


    output_path = os.path.join(
        OUTPUT_DIR,
        output_name
    )


    cv2.imwrite(
        output_path,
        threshold
    )


    # --------------------------------------------------------
    # STATISTICS
    # --------------------------------------------------------

    foreground_pixels = np.count_nonzero(
        threshold
    )


    total_foreground_pixels += (
        foreground_pixels
    )


    successful += 1


    # --------------------------------------------------------
    # PROGRESS
    # --------------------------------------------------------

    if (
        index % 100 == 0
        or index == len(image_files)
    ):

        print(
            f"Processed {index}/{len(image_files)} images"
        )


# ============================================================
# FINAL RESULTS
# ============================================================

print("\n" + "=" * 60)
print("ADAPTIVE THRESHOLDING COMPLETED")
print("=" * 60)

print(
    "Successful:",
    successful
)

print(
    "Failed:",
    failed
)

if successful > 0:

    print(
        "Average foreground pixels/image:",
        round(
            total_foreground_pixels / successful,
            2
        )
    )


if successful == 1880 and failed == 0:

    print(
        "\n ALL 1880 IMAGES PROCESSED SUCCESSFULLY"
    )

else:

    print(
        "\nSOME IMAGES FAILED"
    )


print("\nOutput folder:")

print(OUTPUT_DIR)