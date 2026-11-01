import os
import cv2
import numpy as np


# ============================================================
# STEP 5B - BATCH EDGE EXTRACTION
# Processes ALL 1880 RID images
# ============================================================

IMAGE_DIR = r"C:\Users\Hemila Saravanan\Downloads\Hybrid_Rooftop_Solar\images_roof_centered_geotiff"

OUTPUT_DIR = r"C:\Users\Hemila Saravanan\Downloads\Hybrid_Rooftop_Solar\outputs\edges"


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
    if file.lower().endswith((".tif", ".tiff"))
])


print("=" * 60)
print("STEP 5B - BATCH EDGE EXTRACTION")
print("=" * 60)

print("Total images found:", len(image_files))


# ============================================================
# PARAMETERS
# ============================================================

MEDIAN_KERNEL = 5

CANNY_LOW = 50
CANNY_HIGH = 150

MORPH_KERNEL_SIZE = 3

MIN_COMPONENT_SIZE = 30


# ============================================================
# PROCESS ALL IMAGES
# ============================================================

successful = 0
failed = 0

total_canny_pixels = 0
total_cleaned_pixels = 0


for index, image_file in enumerate(image_files, start=1):

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
            f"[ERROR] Could not read: {image_file}"
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
    # CANNY
    # --------------------------------------------------------

    edges = cv2.Canny(
        median,
        CANNY_LOW,
        CANNY_HIGH
    )


    # --------------------------------------------------------
    # MORPHOLOGICAL CLOSING
    # --------------------------------------------------------

    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (
            MORPH_KERNEL_SIZE,
            MORPH_KERNEL_SIZE
        )
    )


    closed_edges = cv2.morphologyEx(
        edges,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=1
    )


    # --------------------------------------------------------
    # REMOVE SMALL COMPONENTS
    # --------------------------------------------------------

    num_labels, labels, stats, centroids = (
        cv2.connectedComponentsWithStats(
            closed_edges,
            connectivity=8
        )
    )


    clean_edges = np.zeros_like(
        closed_edges
    )


    for component in range(
        1,
        num_labels
    ):

        area = stats[
            component,
            cv2.CC_STAT_AREA
        ]

        if area >= MIN_COMPONENT_SIZE:

            clean_edges[
                labels == component
            ] = 255


    # --------------------------------------------------------
    # SAVE CLEAN EDGE MAP
    # --------------------------------------------------------

    output_name = os.path.splitext(
        image_file
    )[0] + ".png"


    output_path = os.path.join(
        OUTPUT_DIR,
        output_name
    )


    cv2.imwrite(
        output_path,
        clean_edges
    )


    # --------------------------------------------------------
    # STATISTICS
    # --------------------------------------------------------

    canny_pixels = np.count_nonzero(
        edges
    )

    cleaned_pixels = np.count_nonzero(
        clean_edges
    )


    total_canny_pixels += canny_pixels

    total_cleaned_pixels += cleaned_pixels

    successful += 1


    # --------------------------------------------------------
    # PROGRESS
    # --------------------------------------------------------

    if index % 100 == 0 or index == len(image_files):

        print(
            f"Processed {index}/{len(image_files)} images"
        )


# ============================================================
# FINAL RESULTS
# ============================================================

print("\n" + "=" * 60)
print("BATCH PROCESSING COMPLETED")
print("=" * 60)

print(
    "Successful:",
    successful
)

print(
    "Failed:",
    failed
)

print(
    "Total Canny edge pixels:",
    total_canny_pixels
)

print(
    "Total cleaned edge pixels:",
    total_cleaned_pixels
)

print(
    "Average Canny edge pixels/image:",
    round(
        total_canny_pixels / successful,
        2
    )
)

print(
    "Average cleaned edge pixels/image:",
    round(
        total_cleaned_pixels / successful,
        2
    )
)

print("\nOutput folder:")

print(OUTPUT_DIR)


if successful == 1880 and failed == 0:

    print(
        "\n ALL 1880 IMAGES PROCESSED SUCCESSFULLY"
    )

else:

    print(
        "\n Some images could not be processed."
    )