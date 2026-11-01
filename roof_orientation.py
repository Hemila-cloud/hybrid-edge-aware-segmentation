# ============================================================
# STEP 16 - ROOF ORIENTATION ESTIMATION
# PCA-BASED ORIENTATION
# ============================================================

import os
import cv2
import numpy as np
import pandas as pd
import math


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

STEP15_CSV = os.path.join(
    BASE_DIR,
    "outputs",
    "step15_roof_area",
    "roof_area_estimation_corrected.csv"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "outputs",
    "step16_roof_orientation"
)

OUTPUT_OVERLAY_DIR = os.path.join(
    OUTPUT_DIR,
    "overlays"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

os.makedirs(
    OUTPUT_OVERLAY_DIR,
    exist_ok=True
)


# ============================================================
# START
# ============================================================

print("=" * 70)
print("STEP 16 - ROOF ORIENTATION ESTIMATION")
print("=" * 70)


# ============================================================
# LOAD STEP 15 AREA DATA
# ============================================================

area_df = pd.read_csv(
    STEP15_CSV
)

print(
    "Area records:",
    len(area_df)
)


# ============================================================
# FIND MASKS
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


# ============================================================
# PROCESS EACH ROOF MASK
# ============================================================

for index, mask_filename in enumerate(
    mask_files,
    start=1
):

    stem = os.path.splitext(
        mask_filename
    )[0]


    # ========================================================
    # MASK
    # ========================================================

    mask_path = os.path.join(
        MASK_DIR,
        mask_filename
    )


    mask = cv2.imread(
        mask_path,
        cv2.IMREAD_GRAYSCALE
    )


    if mask is None:

        continue


    # ========================================================
    # BINARY MASK
    # ========================================================

    binary = (
        mask > 0
    ).astype(
        np.uint8
    ) * 255


    # ========================================================
    # FIND ROOF COMPONENTS
    # ========================================================

    contours, _ = cv2.findContours(
        binary,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE
    )


    # Remove tiny regions
    contours = [
        c
        for c in contours
        if cv2.contourArea(c) >= 100
    ]


    # ========================================================
    # NO ROOF
    # ========================================================

    if len(contours) == 0:

        results.append({

            "image":
                stem,

            "orientation_deg":
                np.nan,

            "orientation_cardinal":
                "Unknown",

            "orientation_confidence":
                0,

            "roof_components":
                0,

            "largest_component_area_pixels":
                0
        })

        continue


    # ========================================================
    # LARGEST ROOF COMPONENT
    # ========================================================

    largest_contour = max(
        contours,
        key=cv2.contourArea
    )


    largest_area = cv2.contourArea(
        largest_contour
    )


    # ========================================================
    # GET CONTOUR POINTS
    # ========================================================

    points = (
        largest_contour
        .reshape(-1, 2)
        .astype(np.float64)
    )


    # ========================================================
    # PCA
    # ========================================================

    mean = np.mean(
        points,
        axis=0
    )


    centered = (
        points -
        mean
    )


    covariance = np.cov(
        centered.T
    )


    eigenvalues, eigenvectors = np.linalg.eigh(
        covariance
    )


    # Largest eigenvalue = principal direction
    principal_vector = eigenvectors[
        :,
        np.argmax(eigenvalues)
    ]


    vx = principal_vector[0]
    vy = principal_vector[1]


    # ========================================================
    # IMAGE COORDINATES
    #
    # x -> East
    # y -> South
    #
    # Convert to compass orientation.
    # ========================================================

    angle_image = math.degrees(
        math.atan2(
            -vy,
            vx
        )
    )


    angle_image = (
        angle_image + 360
    ) % 360


    # ========================================================
    # ROOF ORIENTATION IS BIDIRECTIONAL
    #
    # 0° and 180° represent the same roof axis.
    # Convert to 0-180°.
    # ========================================================

    orientation = (
        angle_image % 180
    )


    # ========================================================
    # CARDINAL DIRECTION
    # ========================================================

    if (
        orientation >= 157.5
        or
        orientation < 22.5
    ):

        cardinal = "East-West"

    elif (
        orientation >= 22.5
        and
        orientation < 67.5
    ):

        cardinal = "Northeast-Southwest"

    elif (
        orientation >= 67.5
        and
        orientation < 112.5
    ):

        cardinal = "North-South"

    else:

        cardinal = "Northwest-Southeast"


    # ========================================================
    # PCA CONFIDENCE
    #
    # Ratio of dominant eigenvalue to total variance.
    # Higher = stronger directional structure.
    # ========================================================

    total_variance = np.sum(
        eigenvalues
    )


    if total_variance > 0:

        confidence = (
            max(eigenvalues) /
            total_variance
        )

    else:

        confidence = 0


    # ========================================================
    # MIN AREA RECTANGLE
    #
    # Additional orientation estimate for comparison.
    # ========================================================

    rect = cv2.minAreaRect(
        largest_contour
    )


    rect_angle = rect[2]

    rect_width = rect[1][0]
    rect_height = rect[1][1]


    if rect_width < rect_height:

        rect_orientation = (
            rect_angle + 90
        )

    else:

        rect_orientation = rect_angle


    rect_orientation = (
        rect_orientation % 180
    )


    # ========================================================
    # CREATE OVERLAY
    # ========================================================

    overlay = cv2.cvtColor(
        binary,
        cv2.COLOR_GRAY2BGR
    )


    # Make background black and roof green
    overlay[
        binary > 0
    ] = (
        0,
        180,
        0
    )


    # ========================================================
    # DRAW PCA AXIS
    # ========================================================

    cx = int(mean[0])
    cy = int(mean[1])


    # Long axis length
    axis_length = int(
        max(
            mask.shape[0],
            mask.shape[1]
        ) * 0.20
    )


    x_end = int(
        cx +
        vx *
        axis_length
    )


    y_end = int(
        cy +
        vy *
        axis_length
    )


    x_start = int(
        cx -
        vx *
        axis_length
    )


    y_start = int(
        cy -
        vy *
        axis_length
    )


    cv2.arrowedLine(
        overlay,
        (
            x_start,
            y_start
        ),
        (
            x_end,
            y_end
        ),
        (0, 0, 255),
        2,
        tipLength=0.08
    )


    # ========================================================
    # WRITE TEXT
    # ========================================================

    cv2.putText(
        overlay,
        f"Orientation: {orientation:.1f} deg",
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )


    cv2.putText(
        overlay,
        f"Direction: {cardinal}",
        (10, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )


    cv2.putText(
        overlay,
        f"Confidence: {confidence:.2f}",
        (10, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )


    # ========================================================
    # SAVE OVERLAY
    # ========================================================

    overlay_path = os.path.join(
        OUTPUT_OVERLAY_DIR,
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
            stem,

        "orientation_deg":
            orientation,

        "orientation_cardinal":
            cardinal,

        "orientation_confidence":
            confidence,

        "min_area_rect_orientation_deg":
            rect_orientation,

        "roof_components":
            len(contours),

        "largest_component_area_pixels":
            largest_area,

        "principal_eigenvalue":
            max(eigenvalues),

        "secondary_eigenvalue":
            min(eigenvalues)
    })


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
# CREATE DATAFRAME
# ============================================================

orientation_df = pd.DataFrame(
    results
)


# ============================================================
# SAVE CSV
# ============================================================

output_csv = os.path.join(
    OUTPUT_DIR,
    "roof_orientation.csv"
)


orientation_df.to_csv(
    output_csv,
    index=False
)


# ============================================================
# MERGE WITH AREA DATA
# FIX: MAKE IMAGE IDENTIFIER STRING IN BOTH DATAFRAMES
# ============================================================

area_df["image"] = (
    area_df["image"]
    .astype(str)
    .str.strip()
)

orientation_df["image"] = (
    orientation_df["image"]
    .astype(str)
    .str.strip()
)


# ============================================================
# MERGE
# ============================================================

merged_df = pd.merge(
    area_df,
    orientation_df,
    on="image",
    how="left"
)


merged_csv = os.path.join(
    OUTPUT_DIR,
    "roof_area_orientation.csv"
)


merged_df.to_csv(
    merged_csv,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

valid_orientations = (
    orientation_df[
        orientation_df[
            "orientation_deg"
        ].notna()
    ]
)


print("\n" + "=" * 70)
print("STEP 16 COMPLETED")
print("=" * 70)

print(
    "Images processed:",
    len(orientation_df)
)

print(
    "Valid orientations:",
    len(valid_orientations)
)

if len(valid_orientations) > 0:

    print(
        "Mean orientation:",
        round(
            valid_orientations[
                "orientation_deg"
            ].mean(),
            2
        ),
        "degrees"
    )


print(
    "\nOrientation CSV:"
)

print(
    output_csv
)

print(
    "\nMerged area + orientation CSV:"
)

print(
    merged_csv
)

print(
    "\nOrientation overlays:"
)

print(
    OUTPUT_OVERLAY_DIR
)