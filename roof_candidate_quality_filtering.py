# ============================================================
# STEP 13 - ROOF CANDIDATE QUALITY / FALSE-POSITIVE FILTERING
# ============================================================

import os
import cv2
import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

BASE_DIR = r"C:\Users\Hemila Saravanan\Downloads\Hybrid_Rooftop_Solar"

STEP12_DIR = os.path.join(
    BASE_DIR,
    "outputs",
    "final_roof_candidates_v3"
)

MASK_DIR = os.path.join(
    STEP12_DIR,
    "masks"
)

OVERLAY_DIR = os.path.join(
    STEP12_DIR,
    "overlays"
)

PREDICTION_FILE = os.path.join(
    STEP12_DIR,
    "final_candidate_predictions.csv"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "outputs",
    "step13_quality_filtered"
)

OUTPUT_MASK_DIR = os.path.join(
    OUTPUT_DIR,
    "masks"
)

OUTPUT_OVERLAY_DIR = os.path.join(
    OUTPUT_DIR,
    "overlays"
)

os.makedirs(
    OUTPUT_MASK_DIR,
    exist_ok=True
)

os.makedirs(
    OUTPUT_OVERLAY_DIR,
    exist_ok=True
)


# ============================================================
# PARAMETERS
# ============================================================

# Conservative filtering
MIN_AREA = 150
MAX_AREA = 500000

MIN_SOLIDITY = 0.45

MIN_ASPECT_RATIO = 0.15
MAX_ASPECT_RATIO = 6.0

MIN_RECTANGULARITY = 0.15

# Very thin regions are usually not roofs
MIN_WIDTH = 8
MIN_HEIGHT = 8


# ============================================================
# START
# ============================================================

print("=" * 70)
print("STEP 13 - ROOF CANDIDATE QUALITY FILTERING")
print("=" * 70)


# ============================================================
# LOAD STEP 12 PREDICTIONS
# ============================================================

df = pd.read_csv(
    PREDICTION_FILE
)

print(
    "Total Step 12 candidates:",
    len(df)
)


# ============================================================
# CHECK REQUIRED COLUMNS
# ============================================================

required_columns = [
    "image",
    "area",
    "perimeter",
    "width",
    "height",
    "aspect_ratio",
    "solidity"
]

missing = [
    col
    for col in required_columns
    if col not in df.columns
]

if missing:

    raise ValueError(
        f"Missing columns: {missing}"
    )


# ============================================================
# CALCULATE RECTANGULARITY
# ============================================================

df["rectangularity"] = (
    df["area"] /
    (
        df["width"] *
        df["height"]
    ).replace(
        0,
        np.nan
    )
)

df["rectangularity"] = (
    df["rectangularity"]
    .fillna(0)
)


# ============================================================
# ONLY WORK WITH STEP 12 ROOF PREDICTIONS
# ============================================================

if "prediction" in df.columns:

    candidates = df[
        df["prediction"] == 1
    ].copy()

else:

    candidates = df.copy()


print(
    "Step 12 roof candidates:",
    len(candidates)
)


# ============================================================
# APPLY QUALITY FILTERS
# ============================================================

conditions = (

    # Area
    (candidates["area"] >= MIN_AREA) &

    (candidates["area"] <= MAX_AREA) &

    # Width / height
    (candidates["width"] >= MIN_WIDTH) &

    (candidates["height"] >= MIN_HEIGHT) &

    # Aspect ratio
    (candidates["aspect_ratio"] >= MIN_ASPECT_RATIO) &

    (candidates["aspect_ratio"] <= MAX_ASPECT_RATIO) &

    # Solidity
    (candidates["solidity"] >= MIN_SOLIDITY) &

    # Rectangularity
    (candidates["rectangularity"] >= MIN_RECTANGULARITY)
)


filtered = candidates[
    conditions
].copy()


# ============================================================
# ADD FILTER STATUS
# ============================================================

candidates["quality_pass"] = False

candidates.loc[
    filtered.index,
    "quality_pass"
] = True


# ============================================================
# PRINT RESULTS
# ============================================================

print("\n" + "=" * 70)
print("QUALITY FILTER RESULTS")
print("=" * 70)

print(
    "Before filtering:",
    len(candidates)
)

print(
    "After filtering:",
    len(filtered)
)

print(
    "Removed:",
    len(candidates) - len(filtered)
)


if len(candidates) > 0:

    retention = (
        len(filtered) /
        len(candidates)
    ) * 100

    print(
        "Retention:",
        round(retention, 2),
        "%"
    )


# ============================================================
# SAVE CANDIDATE TABLE
# ============================================================

all_candidates_file = os.path.join(
    OUTPUT_DIR,
    "all_step13_candidates.csv"
)

candidates.to_csv(
    all_candidates_file,
    index=False
)


filtered_file = os.path.join(
    OUTPUT_DIR,
    "quality_filtered_candidates.csv"
)

filtered.to_csv(
    filtered_file,
    index=False
)


# ============================================================
# PROCESS IMAGES
# ============================================================

images = sorted(
    filtered["image"].unique()
)

print(
    "\nImages containing filtered roofs:",
    len(images)
)


summary = []


# ============================================================
# CREATE FILTERED MASKS
# ============================================================

for index, filename in enumerate(
    images,
    start=1
):

    # --------------------------------------------------------
    # READ STEP 12 MASK
    # --------------------------------------------------------

    stem = os.path.splitext(
        filename
    )[0]

    mask_path = os.path.join(
        MASK_DIR,
        stem + ".png"
    )

    if not os.path.exists(
        mask_path
    ):

        print(
            "[WARNING] Mask missing:",
            filename
        )

        continue


    mask = cv2.imread(
        mask_path,
        cv2.IMREAD_GRAYSCALE
    )


    if mask is None:

        continue


    # --------------------------------------------------------
    # CREATE EMPTY MASK
    # --------------------------------------------------------

    final_mask = np.zeros_like(
        mask
    )


    # --------------------------------------------------------
    # GET FILTERED CANDIDATES
    # --------------------------------------------------------

    image_candidates = filtered[
        filtered["image"] == filename
    ]


    # --------------------------------------------------------
    # STEP 12 MASK CONTAINS ALL SELECTED REGIONS.
    #
    # We identify connected components and retain only
    # components that correspond to acceptable candidates.
    # --------------------------------------------------------

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask,
        connectivity=8
    )


    # --------------------------------------------------------
    # KEEP COMPONENTS BASED ON AREA
    # --------------------------------------------------------

    kept_components = 0


    for component_id in range(
        1,
        num_labels
    ):

        component_area = stats[
            component_id,
            cv2.CC_STAT_AREA
        ]

        component_width = stats[
            component_id,
            cv2.CC_STAT_WIDTH
        ]

        component_height = stats[
            component_id,
            cv2.CC_STAT_HEIGHT
        ]


        # Conservative component filtering
        if component_area < MIN_AREA:
            continue

        if component_area > MAX_AREA:
            continue

        if component_width < MIN_WIDTH:
            continue

        if component_height < MIN_HEIGHT:
            continue


        component_aspect = (
            component_width /
            float(component_height)
        )


        if (
            component_aspect <
            MIN_ASPECT_RATIO
            or
            component_aspect >
            MAX_ASPECT_RATIO
        ):
            continue


        final_mask[
            labels == component_id
        ] = 255


        kept_components += 1


    # --------------------------------------------------------
    # SAVE FILTERED MASK
    # --------------------------------------------------------

    output_mask = os.path.join(
        OUTPUT_MASK_DIR,
        stem + ".png"
    )

    cv2.imwrite(
        output_mask,
        final_mask
    )


    # ========================================================
    # CREATE OVERLAY
    # ========================================================

    # Find original image
    original_path = None

    image_dir = os.path.join(
        BASE_DIR,
        "images_roof_centered_geotiff"
    )


    for ext in [
        ".tif",
        ".tiff",
        ".png",
        ".jpg",
        ".jpeg"
    ]:

        candidate_path = os.path.join(
            image_dir,
            stem + ext
        )

        if os.path.exists(
            candidate_path
        ):

            original_path = candidate_path
            break


    if original_path is None:

        continue


    original = cv2.imread(
        original_path,
        cv2.IMREAD_COLOR
    )


    if original is None:

        continue


    # Match dimensions
    if original.shape[:2] != mask.shape[:2]:

        original = cv2.resize(
            original,
            (
                mask.shape[1],
                mask.shape[0]
            ),
            interpolation=cv2.INTER_AREA
        )


    # --------------------------------------------------------
    # GREEN MASK OVERLAY
    # --------------------------------------------------------

    overlay = original.copy()


    mask_pixels = final_mask > 0


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


    # --------------------------------------------------------
    # DRAW FINAL BOUNDARIES
    # --------------------------------------------------------

    contours, _ = cv2.findContours(
        final_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )


    cv2.drawContours(
        overlay,
        contours,
        -1,
        (0, 255, 0),
        2
    )


    # --------------------------------------------------------
    # SAVE OVERLAY
    # --------------------------------------------------------

    output_overlay = os.path.join(
        OUTPUT_OVERLAY_DIR,
        stem + ".png"
    )


    cv2.imwrite(
        output_overlay,
        overlay
    )


    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    roof_pixels = int(
        np.sum(
            final_mask > 0
        )
    )


    total_pixels = (
        final_mask.shape[0] *
        final_mask.shape[1]
    )


    roof_percentage = (
        roof_pixels /
        total_pixels
    ) * 100


    summary.append({

        "image": filename,

        "step12_candidates": len(
            candidates[
                candidates["image"] == filename
            ]
        ),

        "step13_candidates": len(
            image_candidates
        ),

        "connected_components_kept":
            kept_components,

        "roof_pixels":
            roof_pixels,

        "roof_percentage":
            roof_percentage
    })


    # --------------------------------------------------------
    # PROGRESS
    # --------------------------------------------------------

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
    "step13_summary.csv"
)


summary_df.to_csv(
    summary_file,
    index=False
)


# ============================================================
# FINAL REPORT
# ============================================================

print("\n" + "=" * 70)
print("STEP 13 COMPLETED")
print("=" * 70)

print(
    "Step 12 candidates:",
    len(candidates)
)

print(
    "Quality-filtered candidates:",
    len(filtered)
)

print(
    "Candidates removed:",
    len(candidates) - len(filtered)
)

print(
    "Images processed:",
    len(summary_df)
)

print(
    "\nFiltered masks:",
    OUTPUT_MASK_DIR
)

print(
    "\nFiltered overlays:",
    OUTPUT_OVERLAY_DIR
)

print(
    "\nCandidate CSV:",
    filtered_file
)

print(
    "\nSummary:",
    summary_file
)