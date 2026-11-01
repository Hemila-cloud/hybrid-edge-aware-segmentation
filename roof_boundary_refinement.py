# ============================================================
# STEP 14 - ROOF BOUNDARY REFINEMENT
# ============================================================

import os
import cv2
import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

BASE_DIR = r"C:\Users\Hemila Saravanan\Downloads\Hybrid_Rooftop_Solar"

STEP13_DIR = os.path.join(
    BASE_DIR,
    "outputs",
    "step13_quality_filtered"
)

STEP13_MASK_DIR = os.path.join(
    STEP13_DIR,
    "masks"
)

STEP13_OVERLAY_DIR = os.path.join(
    STEP13_DIR,
    "overlays"
)

STEP13_CSV = os.path.join(
    STEP13_DIR,
    "quality_filtered_candidates.csv"
)

IMAGE_DIR = os.path.join(
    BASE_DIR,
    "images_roof_centered_geotiff"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "outputs",
    "step14_roof_refinement"
)

REFINED_MASK_DIR = os.path.join(
    OUTPUT_DIR,
    "masks"
)

REFINED_OVERLAY_DIR = os.path.join(
    OUTPUT_DIR,
    "overlays"
)

os.makedirs(
    REFINED_MASK_DIR,
    exist_ok=True
)

os.makedirs(
    REFINED_OVERLAY_DIR,
    exist_ok=True
)


# ============================================================
# PARAMETERS
# ============================================================

# Small morphological cleanup
OPEN_KERNEL = 3
OPEN_ITERATIONS = 1

CLOSE_KERNEL = 3
CLOSE_ITERATIONS = 1

# Remove tiny connected regions
MIN_COMPONENT_AREA = 100

# Small padding around each candidate
ROI_PADDING = 3


# ============================================================
# START
# ============================================================

print("=" * 70)
print("STEP 14 - ROOF BOUNDARY REFINEMENT")
print("=" * 70)


# ============================================================
# LOAD STEP 13 CSV
# ============================================================

df = pd.read_csv(
    STEP13_CSV
)

print(
    "Step 13 candidates:",
    len(df)
)


# ============================================================
# IMAGE LIST
# ============================================================

images = sorted(
    df["image"].unique()
)

print(
    "Images containing roof candidates:",
    len(images)
)


# ============================================================
# MORPHOLOGICAL KERNELS
# ============================================================

open_kernel = cv2.getStructuringElement(
    cv2.MORPH_ELLIPSE,
    (
        OPEN_KERNEL,
        OPEN_KERNEL
    )
)

close_kernel = cv2.getStructuringElement(
    cv2.MORPH_ELLIPSE,
    (
        CLOSE_KERNEL,
        CLOSE_KERNEL
    )
)


# ============================================================
# SUMMARY
# ============================================================

summary = []


# ============================================================
# PROCESS EACH IMAGE
# ============================================================

for index, filename in enumerate(
    images,
    start=1
):


    # ========================================================
    # IMAGE STEM
    # ========================================================

    stem = os.path.splitext(
        filename
    )[0]


    # ========================================================
    # STEP 13 MASK
    # ========================================================

    mask_path = os.path.join(
        STEP13_MASK_DIR,
        stem + ".png"
    )


    if not os.path.exists(
        mask_path
    ):

        print(
            "[WARNING] Step 13 mask missing:",
            filename
        )

        continue


    mask = cv2.imread(
        mask_path,
        cv2.IMREAD_GRAYSCALE
    )


    if mask is None:

        print(
            "[WARNING] Cannot read mask:",
            filename
        )

        continue


    height, width = mask.shape


    # ========================================================
    # ORIGINAL IMAGE
    # ========================================================

    original_path = None


    for ext in [
        ".tif",
        ".tiff",
        ".png",
        ".jpg",
        ".jpeg"
    ]:

        candidate_path = os.path.join(
            IMAGE_DIR,
            stem + ext
        )


        if os.path.exists(
            candidate_path
        ):

            original_path = candidate_path
            break


    if original_path is None:

        print(
            "[WARNING] Original image missing:",
            filename
        )

        continue


    original = cv2.imread(
        original_path,
        cv2.IMREAD_COLOR
    )


    if original is None:

        continue


    # ========================================================
    # RESIZE IF NECESSARY
    # ========================================================

    if original.shape[:2] != mask.shape[:2]:

        original = cv2.resize(
            original,
            (
                width,
                height
            ),
            interpolation=cv2.INTER_AREA
        )


    # ========================================================
    # GRAYSCALE
    # ========================================================

    gray = cv2.cvtColor(
        original,
        cv2.COLOR_BGR2GRAY
    )


    # ========================================================
    # SLIGHT BLUR
    #
    # Reduces small image noise while preserving
    # roof boundaries.
    # ========================================================

    gray_blur = cv2.GaussianBlur(
        gray,
        (5, 5),
        0
    )


    # ========================================================
    # EDGE DETECTION
    # ========================================================

    edges = cv2.Canny(
        gray_blur,
        50,
        150
    )


    # ========================================================
    # CREATE EMPTY REFINED MASK
    # ========================================================

    refined_mask = np.zeros_like(
        mask
    )


    # ========================================================
    # FIND STEP 13 REGIONS
    # ========================================================

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )


    kept_regions = 0
    original_regions = len(contours)


    # ========================================================
    # PROCESS EACH ROOF REGION
    # ========================================================

    for contour in contours:


        area = cv2.contourArea(
            contour
        )


        if area < MIN_COMPONENT_AREA:

            continue


        # ----------------------------------------------------
        # BOUNDING BOX
        # ----------------------------------------------------

        x, y, w, h = cv2.boundingRect(
            contour
        )


        # Add small padding
        x1 = max(
            0,
            x - ROI_PADDING
        )

        y1 = max(
            0,
            y - ROI_PADDING
        )

        x2 = min(
            width,
            x + w + ROI_PADDING
        )

        y2 = min(
            height,
            y + h + ROI_PADDING
        )


        # ----------------------------------------------------
        # CREATE LOCAL MASK
        # ----------------------------------------------------

        local_mask = np.zeros(
            (
                y2 - y1,
                x2 - x1
            ),
            dtype=np.uint8
        )


        shifted_contour = (
            contour -
            np.array(
                [
                    [x1, y1]
                ]
            )
        )


        cv2.drawContours(
            local_mask,
            [shifted_contour],
            -1,
            255,
            thickness=cv2.FILLED
        )


        # ----------------------------------------------------
        # LOCAL EDGES
        # ----------------------------------------------------

        local_edges = edges[
            y1:y2,
            x1:x2
        ]


        # ----------------------------------------------------
        # EDGE-DENSITY CHECK
        #
        # This is NOT used to delete the region.
        # It only helps determine whether a region has
        # meaningful internal boundaries.
        # ----------------------------------------------------

        region_pixels = (
            local_mask > 0
        )


        if np.sum(region_pixels) == 0:

            continue


        edge_pixels = np.sum(
            local_edges[
                region_pixels
            ] > 0
        )


        edge_density = (
            edge_pixels /
            float(
                np.sum(
                    region_pixels
                )
            )
        )


        # ----------------------------------------------------
        # REFINEMENT STRATEGY
        #
        # Keep the original candidate as the main region.
        # This prevents valid roofs from disappearing.
        # ----------------------------------------------------

        refined_local = local_mask.copy()


        # ----------------------------------------------------
        # MORPHOLOGICAL CLEANUP
        # ----------------------------------------------------

        refined_local = cv2.morphologyEx(
            refined_local,
            cv2.MORPH_OPEN,
            open_kernel,
            iterations=OPEN_ITERATIONS
        )


        refined_local = cv2.morphologyEx(
            refined_local,
            cv2.MORPH_CLOSE,
            close_kernel,
            iterations=CLOSE_ITERATIONS
        )


        # ----------------------------------------------------
        # REMOVE TINY LOCAL COMPONENTS
        # ----------------------------------------------------

        n_labels, local_labels, stats, centroids = (
            cv2.connectedComponentsWithStats(
                refined_local,
                connectivity=8
            )
        )


        cleaned_local = np.zeros_like(
            refined_local
        )


        for component_id in range(
            1,
            n_labels
        ):

            component_area = stats[
                component_id,
                cv2.CC_STAT_AREA
            ]


            if (
                component_area >=
                MIN_COMPONENT_AREA
            ):

                cleaned_local[
                    local_labels ==
                    component_id
                ] = 255


        # ----------------------------------------------------
        # IF CLEANUP DESTROYED REGION,
        # KEEP ORIGINAL CANDIDATE
        # ----------------------------------------------------

        if np.sum(cleaned_local) < (
            0.50 *
            np.sum(local_mask)
        ):

            cleaned_local = local_mask


        # ----------------------------------------------------
        # PUT LOCAL RESULT BACK
        # ----------------------------------------------------

        refined_mask[
            y1:y2,
            x1:x2
        ] = np.maximum(
            refined_mask[
                y1:y2,
                x1:x2
            ],
            cleaned_local
        )


        kept_regions += 1


    # ========================================================
    # FINAL GLOBAL CLEANUP
    # ========================================================

    refined_mask = cv2.morphologyEx(
        refined_mask,
        cv2.MORPH_CLOSE,
        close_kernel,
        iterations=1
    )


    # ========================================================
    # REMOVE VERY SMALL COMPONENTS
    # ========================================================

    n_labels, labels, stats, centroids = (
        cv2.connectedComponentsWithStats(
            refined_mask,
            connectivity=8
        )
    )


    final_mask = np.zeros_like(
        refined_mask
    )


    for component_id in range(
        1,
        n_labels
    ):

        component_area = stats[
            component_id,
            cv2.CC_STAT_AREA
        ]


        if (
            component_area >=
            MIN_COMPONENT_AREA
        ):

            final_mask[
                labels ==
                component_id
            ] = 255


    # ========================================================
    # SAVE REFINED MASK
    # ========================================================

    output_mask = os.path.join(
        REFINED_MASK_DIR,
        stem + ".png"
    )


    cv2.imwrite(
        output_mask,
        final_mask
    )


    # ========================================================
    # CREATE OVERLAY
    # ========================================================

    overlay = original.copy()


    mask_pixels = (
        final_mask > 0
    )


    # Semi-transparent green region
    green_layer = np.zeros_like(
        original
    )

    green_layer[:, :] = (
        0,
        255,
        0
    )


    blended = cv2.addWeighted(
        original,
        0.70,
        green_layer,
        0.30,
        0
    )


    overlay[
        mask_pixels
    ] = blended[
        mask_pixels
    ]


    # ========================================================
    # DRAW FINAL ROOF BOUNDARIES
    # ========================================================

    final_contours, _ = cv2.findContours(
        final_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )


    cv2.drawContours(
        overlay,
        final_contours,
        -1,
        (0, 255, 0),
        2
    )


    # ========================================================
    # SAVE OVERLAY
    # ========================================================

    output_overlay = os.path.join(
        REFINED_OVERLAY_DIR,
        stem + ".png"
    )


    cv2.imwrite(
        output_overlay,
        overlay
    )


    # ========================================================
    # STATISTICS
    # ========================================================

    roof_pixels = int(
        np.sum(
            final_mask > 0
        )
    )


    total_pixels = (
        height *
        width
    )


    roof_percentage = (
        roof_pixels /
        float(total_pixels)
    ) * 100


    summary.append({

        "image": filename,

        "step13_regions":
            original_regions,

        "refined_regions":
            len(final_contours),

        "roof_pixels":
            roof_pixels,

        "roof_percentage":
            roof_percentage
    })


    # ========================================================
    # PROGRESS
    # ========================================================

    if (
        index % 100 == 0
        or index == len(images)
    ):

        print(
            f"Processed "
            f"{index}/{len(images)}"
        )


# ============================================================
# SAVE SUMMARY
# ============================================================

summary_df = pd.DataFrame(
    summary
)


summary_file = os.path.join(
    OUTPUT_DIR,
    "step14_refinement_summary.csv"
)


summary_df.to_csv(
    summary_file,
    index=False
)


# ============================================================
# FINAL REPORT
# ============================================================

print("\n" + "=" * 70)
print("STEP 14 COMPLETED")
print("=" * 70)

print(
    "Images processed:",
    len(summary_df)
)

print(
    "Total Step 13 candidates:",
    len(df)
)

print(
    "Total refined roof regions:",
    summary_df[
        "refined_regions"
    ].sum()
)

print(
    "\nRefined masks:",
    REFINED_MASK_DIR
)

print(
    "\nRefined overlays:",
    REFINED_OVERLAY_DIR
)

print(
    "\nSummary:",
    summary_file
)