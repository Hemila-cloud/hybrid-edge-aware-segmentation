# ============================================================
# STEP 17 - SHADOW DETECTION
# Conservative shadow detection using intensity + morphology
# ============================================================

import os
import cv2
import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

BASE_DIR = r"C:\Users\Hemila Saravanan\Downloads\Hybrid_Rooftop_Solar"

STEP14_DIR = os.path.join(
    BASE_DIR,
    "outputs",
    "step14_roof_refinement"
)

MASK_DIR = os.path.join(
    STEP14_DIR,
    "masks"
)

IMAGE_DIR = os.path.join(
    BASE_DIR,
    "images_roof_centered_geotiff"
)

STEP16_CSV = os.path.join(
    BASE_DIR,
    "outputs",
    "step16_roof_orientation",
    "roof_area_orientation.csv"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "outputs",
    "step17_shadow_detection"
)

SHADOW_MASK_DIR = os.path.join(
    OUTPUT_DIR,
    "shadow_masks"
)

OVERLAY_DIR = os.path.join(
    OUTPUT_DIR,
    "overlays"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

os.makedirs(
    SHADOW_MASK_DIR,
    exist_ok=True
)

os.makedirs(
    OVERLAY_DIR,
    exist_ok=True
)


# ============================================================
# PARAMETERS
# ============================================================

# Dark pixel threshold
SHADOW_THRESHOLD = 75

# Minimum shadow component
MIN_SHADOW_AREA = 100

# Morphological cleanup
MORPH_KERNEL_SIZE = 5

# Maximum shadow percentage allowed
# This prevents extreme dark regions from becoming shadows.
MAX_SHADOW_PERCENTAGE = 40.0


# ============================================================
# START
# ============================================================

print("=" * 70)
print("STEP 17 - SHADOW DETECTION")
print("=" * 70)


# ============================================================
# LOAD STEP 16 DATA
# ============================================================

orientation_df = pd.read_csv(
    STEP16_CSV
)

orientation_df["image"] = (
    orientation_df["image"]
    .astype(str)
    .str.strip()
)


print(
    "Orientation records:",
    len(orientation_df)
)


# ============================================================
# FIND ROOF MASKS
# ============================================================

mask_files = sorted([
    f
    for f in os.listdir(MASK_DIR)
    if f.lower().endswith(".png")
])


print(
    "Roof masks:",
    len(mask_files)
)


# ============================================================
# RESULTS
# ============================================================

results = []


processed = 0
missing_images = 0


# ============================================================
# PROCESS EACH IMAGE
# ============================================================

for index, mask_filename in enumerate(
    mask_files,
    start=1
):

    # ========================================================
    # IMAGE ID
    # ========================================================

    stem = os.path.splitext(
        mask_filename
    )[0]

    image_id = stem


    # ========================================================
    # ROOF MASK
    # ========================================================

    mask_path = os.path.join(
        MASK_DIR,
        mask_filename
    )


    roof_mask = cv2.imread(
        mask_path,
        cv2.IMREAD_GRAYSCALE
    )


    if roof_mask is None:

        print(
            "[WARNING] Cannot read roof mask:",
            mask_filename
        )

        continue


    # ========================================================
    # FIND ORIGINAL GEOTIFF
    # ========================================================

    image_path = None


    for ext in [
        ".tif",
        ".tiff",
        ".TIF",
        ".TIFF"
    ]:

        candidate = os.path.join(
            IMAGE_DIR,
            stem + ext
        )


        if os.path.exists(
            candidate
        ):

            image_path = candidate
            break


    if image_path is None:

        print(
            "[WARNING] Original image missing:",
            stem
        )

        missing_images += 1
        continue


    # ========================================================
    # READ IMAGE
    # ========================================================

    image = cv2.imread(
        image_path,
        cv2.IMREAD_COLOR
    )


    if image is None:

        continue


    # ========================================================
    # RESIZE MASK IF NECESSARY
    # ========================================================

    if image.shape[:2] != roof_mask.shape:

        roof_mask = cv2.resize(
            roof_mask,
            (
                image.shape[1],
                image.shape[0]
            ),
            interpolation=cv2.INTER_NEAREST
        )


    # ========================================================
    # GRAYSCALE
    # ========================================================

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )


    # ========================================================
    # LIGHT SMOOTHING
    # ========================================================

    blurred = cv2.GaussianBlur(
        gray,
        (5, 5),
        0
    )


    # ========================================================
    # DARK PIXEL DETECTION
    # ========================================================

    dark_mask = np.zeros_like(
        gray,
        dtype=np.uint8
    )


    dark_mask[
        blurred < SHADOW_THRESHOLD
    ] = 255


    # ========================================================
    # MORPHOLOGICAL CLEANUP
    # ========================================================

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (
            MORPH_KERNEL_SIZE,
            MORPH_KERNEL_SIZE
        )
    )


    dark_mask = cv2.morphologyEx(
        dark_mask,
        cv2.MORPH_OPEN,
        kernel
    )


    dark_mask = cv2.morphologyEx(
        dark_mask,
        cv2.MORPH_CLOSE,
        kernel
    )


    # ========================================================
    # IMPORTANT:
    # SEARCH SHADOWS INSIDE / NEAR ROOF REGION
    #
    # Dilate roof mask slightly so shadows immediately
    # outside roofs can also be detected.
    # ========================================================

    roof_binary = np.zeros_like(
        roof_mask
    )

    roof_binary[
        roof_mask > 0
    ] = 255


    shadow_search_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (15, 15)
    )


    roof_search_region = cv2.dilate(
        roof_binary,
        shadow_search_kernel,
        iterations=1
    )


    # ========================================================
    # LIMIT SHADOW DETECTION
    # ========================================================

    shadow_mask = cv2.bitwise_and(
        dark_mask,
        roof_search_region
    )


    # ========================================================
    # REMOVE VERY SMALL COMPONENTS
    # ========================================================

    num_labels, labels, stats, centroids = (
        cv2.connectedComponentsWithStats(
            shadow_mask,
            connectivity=8
        )
    )


    cleaned_shadow = np.zeros_like(
        shadow_mask
    )


    for component_id in range(
        1,
        num_labels
    ):

        area = stats[
            component_id,
            cv2.CC_STAT_AREA
        ]


        if area >= MIN_SHADOW_AREA:

            cleaned_shadow[
                labels == component_id
            ] = 255


    shadow_mask = cleaned_shadow


    # ========================================================
    # SHADOW PIXELS
    # ========================================================

    shadow_pixels = int(
        np.count_nonzero(
            shadow_mask
        )
    )


    roof_pixels = int(
        np.count_nonzero(
            roof_binary
        )
    )


    # ========================================================
    # SHADOW % OF ROOF
    # ========================================================

    if roof_pixels > 0:

        shadow_percentage = (
            shadow_pixels /
            roof_pixels
        ) * 100

    else:

        shadow_percentage = 0


    # ========================================================
    # SAFETY CHECK
    #
    # If shadow detection becomes excessive, remove it.
    # ========================================================

    if (
        shadow_percentage >
        MAX_SHADOW_PERCENTAGE
    ):

        shadow_mask = np.zeros_like(
            shadow_mask
        )

        shadow_pixels = 0

        shadow_percentage = 0


    # ========================================================
    # SHADOW AREA
    #
    # Area will be calculated from Step 15 pixel area.
    # ========================================================

    matching_area = orientation_df[
        orientation_df["image"] == image_id
    ]


    if len(matching_area) > 0:

        pixel_area_m2 = float(
            matching_area.iloc[0][
                "pixel_area_m2"
            ]
        )

    else:

        pixel_area_m2 = np.nan


    if not np.isnan(
        pixel_area_m2
    ):

        shadow_area_m2 = (
            shadow_pixels *
            pixel_area_m2
        )

    else:

        shadow_area_m2 = np.nan


    # ========================================================
    # CREATE OVERLAY
    # ========================================================

    overlay = image.copy()


    # Shadow pixels shown in red
    shadow_pixels_bool = (
        shadow_mask > 0
    )


    # Blend shadow regions
    red_layer = np.zeros_like(
        image
    )

    red_layer[:, :] = (
        0,
        0,
        255
    )


    blended = cv2.addWeighted(
        image,
        0.65,
        red_layer,
        0.35,
        0
    )


    overlay[
        shadow_pixels_bool
    ] = blended[
        shadow_pixels_bool
    ]


    # ========================================================
    # DRAW ROOF BOUNDARY
    # ========================================================

    roof_contours, _ = cv2.findContours(
        roof_binary,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )


    cv2.drawContours(
        overlay,
        roof_contours,
        -1,
        (0, 255, 0),
        2
    )


    # ========================================================
    # TEXT
    # ========================================================

    cv2.putText(
        overlay,
        f"Shadow: {shadow_percentage:.1f}%",
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )


    cv2.putText(
        overlay,
        f"Shadow area: {shadow_area_m2:.2f} m2",
        (10, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.60,
        (255, 255, 255),
        2
    )


    # ========================================================
    # SAVE SHADOW MASK
    # ========================================================

    shadow_path = os.path.join(
        SHADOW_MASK_DIR,
        stem + ".png"
    )


    cv2.imwrite(
        shadow_path,
        shadow_mask
    )


    # ========================================================
    # SAVE OVERLAY
    # ========================================================

    overlay_path = os.path.join(
        OVERLAY_DIR,
        stem + ".png"
    )


    cv2.imwrite(
        overlay_path,
        overlay
    )


    # ========================================================
    # SAVE RESULT
    # ========================================================

    results.append({

        "image":
            image_id,

        "roof_pixels":
            roof_pixels,

        "shadow_pixels":
            shadow_pixels,

        "shadow_percentage":
            shadow_percentage,

        "pixel_area_m2":
            pixel_area_m2,

        "shadow_area_m2":
            shadow_area_m2
    })


    processed += 1


    # ========================================================
    # PROGRESS
    # ========================================================

    if (
        index % 100 == 0
        or
        index == len(mask_files)
    ):

        print(
            f"Processed "
            f"{index}/{len(mask_files)}"
        )


# ============================================================
# DATAFRAME
# ============================================================

shadow_df = pd.DataFrame(
    results
)


# ============================================================
# SAVE SHADOW CSV
# ============================================================

shadow_csv = os.path.join(
    OUTPUT_DIR,
    "shadow_detection.csv"
)


shadow_df.to_csv(
    shadow_csv,
    index=False
)


# ============================================================
# MERGE WITH PREVIOUS RESULTS
# ============================================================

orientation_df["image"] = (
    orientation_df["image"]
    .astype(str)
    .str.strip()
)

shadow_df["image"] = (
    shadow_df["image"]
    .astype(str)
    .str.strip()
)


combined_df = pd.merge(
    orientation_df,
    shadow_df,
    on="image",
    how="left"
)


combined_csv = os.path.join(
    OUTPUT_DIR,
    "roof_area_orientation_shadow.csv"
)


combined_df.to_csv(
    combined_csv,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

if len(shadow_df) > 0:

    mean_shadow = float(
        shadow_df[
            "shadow_percentage"
        ].mean()
    )


    total_shadow_area = float(
        shadow_df[
            "shadow_area_m2"
        ].sum()
    )

else:

    mean_shadow = 0
    total_shadow_area = 0


# ============================================================
# FINAL REPORT
# ============================================================

print("\n" + "=" * 70)
print("STEP 17 COMPLETED")
print("=" * 70)

print(
    "Images processed:",
    processed
)

print(
    "Missing images:",
    missing_images
)

print(
    "Mean shadow percentage:",
    round(
        mean_shadow,
        2
    ),
    "%"
)

print(
    "Total estimated shadow area:",
    round(
        total_shadow_area,
        2
    ),
    "m²"
)

print(
    "\nShadow masks:"
)

print(
    SHADOW_MASK_DIR
)

print(
    "\nShadow overlays:"
)

print(
    OVERLAY_DIR
)

print(
    "\nShadow CSV:"
)

print(
    shadow_csv
)

print(
    "\nCombined CSV:"
)

print(
    combined_csv
)