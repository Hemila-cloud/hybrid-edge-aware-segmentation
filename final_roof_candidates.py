import os
import cv2
import joblib
import numpy as np
import pandas as pd


# ============================================================
# STEP 12 - FINAL ROOF MASK GENERATION
# ROBUST CONTOUR MATCHING VERSION
# ============================================================

BASE_DIR = r"C:\Users\Hemila Saravanan\Downloads\Hybrid_Rooftop_Solar"

EDGE_DIR = os.path.join(
    BASE_DIR,
    "outputs",
    "edges"
)

FEATURE_FILE = os.path.join(
    BASE_DIR,
    "outputs",
    "roof_candidate_features",
    "roof_candidate_final_labels.csv"
)

MODEL_FILE = os.path.join(
    BASE_DIR,
    "outputs",
    "roof_candidate_model",
    "roof_candidate_random_forest.pkl"
)

IMAGE_DIR = os.path.join(
    BASE_DIR,
    "images_roof_centered_geotiff"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "outputs",
    "final_roof_candidates_v3"
)

MASK_DIR = os.path.join(
    OUTPUT_DIR,
    "masks"
)

OVERLAY_DIR = os.path.join(
    OUTPUT_DIR,
    "overlays"
)

os.makedirs(
    MASK_DIR,
    exist_ok=True
)

os.makedirs(
    OVERLAY_DIR,
    exist_ok=True
)


# ============================================================
# MODEL FEATURES
# ============================================================

FEATURES = [
    "area",
    "perimeter",
    "width",
    "height",
    "aspect_ratio",
    "solidity",
    "center_x",
    "center_y",
    "edge_density",
    "texture"
]


# ============================================================
# PARAMETERS
# ============================================================

MIN_ROOF_PROBABILITY = 0.50

CLOSE_KERNEL_SIZE = 3
CLOSE_ITERATIONS = 1

MIN_CONTOUR_AREA = 50


# ============================================================
# START
# ============================================================

print("=" * 70)
print("STEP 12 - FINAL ROOF MASK GENERATION")
print("=" * 70)


# ============================================================
# LOAD CANDIDATE DATA
# ============================================================

df = pd.read_csv(
    FEATURE_FILE
)

print(
    "Total candidates:",
    len(df)
)


# ============================================================
# CHECK FEATURES
# ============================================================

missing = [
    f for f in FEATURES
    if f not in df.columns
]

if missing:

    raise ValueError(
        f"Missing model features: {missing}"
    )


# ============================================================
# LOAD MODEL
# ============================================================

model = joblib.load(
    MODEL_FILE
)

print(
    "Random Forest loaded."
)


# ============================================================
# PREDICT ALL CANDIDATES
# ============================================================

X = df[
    FEATURES
].copy()


df["prediction"] = model.predict(
    X
)


df["roof_probability"] = (
    model.predict_proba(X)[:, 1]
)


# ============================================================
# SELECT ROOF CANDIDATES
# ============================================================

roof_df = df[
    (df["prediction"] == 1) &
    (df["roof_probability"] >= MIN_ROOF_PROBABILITY)
].copy()


print("\n" + "=" * 70)
print("ROOF PREDICTION")
print("=" * 70)

print(
    "Total candidates:",
    len(df)
)

print(
    "Predicted roof candidates:",
    len(roof_df)
)

print(
    "Minimum probability:",
    MIN_ROOF_PROBABILITY
)


# ============================================================
# SAVE PREDICTIONS
# ============================================================

prediction_path = os.path.join(
    OUTPUT_DIR,
    "final_candidate_predictions.csv"
)

df.to_csv(
    prediction_path,
    index=False
)


# ============================================================
# IMAGE LIST
# ============================================================

unique_images = sorted(
    df["image"].unique()
)


print(
    "\nImages:",
    len(unique_images)
)


# ============================================================
# SUMMARY
# ============================================================

summary = []

total_selected = 0
total_matched = 0
total_unmatched = 0


# ============================================================
# PROCESS ALL IMAGES
# ============================================================

for image_index, filename in enumerate(
    unique_images,
    start=1
):


    # ========================================================
    # EDGE IMAGE
    # ========================================================

    edge_path = os.path.join(
        EDGE_DIR,
        filename
    )


    if not os.path.exists(
        edge_path
    ):

        print(
            "[WARNING] Edge missing:",
            filename
        )

        continue


    edge = cv2.imread(
        edge_path,
        cv2.IMREAD_GRAYSCALE
    )


    if edge is None:

        print(
            "[WARNING] Cannot read:",
            filename
        )

        continue


    height, width = edge.shape


    # ========================================================
    # ORIGINAL IMAGE
    # ========================================================

    stem = os.path.splitext(
        filename
    )[0]


    original_path = None


    for ext in [
        ".tif",
        ".tiff",
        ".png",
        ".jpg",
        ".jpeg"
    ]:

        p = os.path.join(
            IMAGE_DIR,
            stem + ext
        )


        if os.path.exists(p):

            original_path = p
            break


    if original_path is None:

        print(
            "[WARNING] Original missing:",
            filename
        )

        continue


    original = cv2.imread(
        original_path,
        cv2.IMREAD_COLOR
    )


    if original is None:

        continue


    if original.shape[:2] != edge.shape[:2]:

        original = cv2.resize(
            original,
            (
                width,
                height
            ),
            interpolation=cv2.INTER_AREA
        )


    # ========================================================
    # RECREATE EDGE PROCESSING
    # ========================================================

    binary = np.where(
        edge > 0,
        255,
        0
    ).astype(
        np.uint8
    )


    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (
            CLOSE_KERNEL_SIZE,
            CLOSE_KERNEL_SIZE
        )
    )


    closed = cv2.morphologyEx(
        binary,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=CLOSE_ITERATIONS
    )


    # ========================================================
    # FIND CONTOURS
    # ========================================================

    contours, hierarchy = cv2.findContours(
        closed,
        cv2.RETR_LIST,
        cv2.CHAIN_APPROX_SIMPLE
    )


    # ========================================================
    # CREATE CONTOUR INFORMATION
    # ========================================================

    contour_info = []


    for contour_id, contour in enumerate(
        contours
    ):

        area = cv2.contourArea(
            contour
        )


        if area < MIN_CONTOUR_AREA:

            continue


        perimeter = cv2.arcLength(
            contour,
            True
        )


        x, y, w, h = cv2.boundingRect(
            contour
        )


        if h > 0:

            aspect_ratio = (
                w / float(h)
            )

        else:

            aspect_ratio = 0


        if w > 0 and h > 0:

            rectangularity = (
                area /
                float(w * h)
            )

        else:

            rectangularity = 0


        hull = cv2.convexHull(
            contour
        )


        hull_area = cv2.contourArea(
            hull
        )


        if hull_area > 0:

            solidity = (
                area /
                float(hull_area)
            )

        else:

            solidity = 0


        moments = cv2.moments(
            contour
        )


        if moments["m00"] != 0:

            cx = (
                moments["m10"] /
                moments["m00"]
            )

            cy = (
                moments["m01"] /
                moments["m00"]
            )

        else:

            cx = x + w / 2
            cy = y + h / 2


        contour_info.append({

            "id": contour_id,

            "contour": contour,

            "area": area,

            "perimeter": perimeter,

            "x": x,

            "y": y,

            "width": w,

            "height": h,

            "aspect_ratio": aspect_ratio,

            "rectangularity": rectangularity,

            "solidity": solidity,

            "center_x": cx,

            "center_y": cy
        })


    # ========================================================
    # ROOF CANDIDATES FOR THIS IMAGE
    # ========================================================

    image_roofs = roof_df[
        roof_df["image"] == filename
    ]


    # ========================================================
    # EMPTY FINAL MASK
    # ========================================================

    final_mask = np.zeros(
        (height, width),
        dtype=np.uint8
    )


    overlay = original.copy()


    matched_here = 0
    unmatched_here = 0


    # ========================================================
    # MATCH EACH PREDICTED CANDIDATE
    # ========================================================

    for _, row in image_roofs.iterrows():

        target_area = float(
            row["area"]
        )

        target_perimeter = float(
            row["perimeter"]
        )

        target_x = int(
            row["x"]
        )

        target_y = int(
            row["y"]
        )

        target_w = int(
            row["width"]
        )

        target_h = int(
            row["height"]
        )


        # ----------------------------------------------------
        # FIRST TRY:
        # exact bounding-box match
        # ----------------------------------------------------

        exact_matches = []


        for info in contour_info:

            if (
                info["x"] == target_x
                and
                info["y"] == target_y
                and
                info["width"] == target_w
                and
                info["height"] == target_h
            ):

                exact_matches.append(
                    info
                )


        # ----------------------------------------------------
        # IF MULTIPLE / NO EXACT MATCH
        # USE DISTANCE
        # ----------------------------------------------------

        if len(exact_matches) == 1:

            best = exact_matches[0]

        else:

            candidates = (
                exact_matches
                if len(exact_matches) > 0
                else contour_info
            )


            best = None
            best_score = float("inf")


            for info in candidates:

                # Relative area difference
                area_diff = abs(
                    info["area"] -
                    target_area
                ) / max(
                    target_area,
                    1
                )


                # Relative perimeter difference
                perimeter_diff = abs(
                    info["perimeter"] -
                    target_perimeter
                ) / max(
                    target_perimeter,
                    1
                )


                # Bounding-box difference
                box_diff = (
                    abs(
                        info["x"] -
                        target_x
                    )
                    +
                    abs(
                        info["y"] -
                        target_y
                    )
                    +
                    abs(
                        info["width"] -
                        target_w
                    )
                    +
                    abs(
                        info["height"] -
                        target_h
                    )
                )


                score = (
                    area_diff * 5
                    +
                    perimeter_diff * 3
                    +
                    box_diff / 100.0
                )


                if score < best_score:

                    best_score = score
                    best = info


        # ----------------------------------------------------
        # NO MATCH
        # ----------------------------------------------------

        if best is None:

            unmatched_here += 1
            continue


        matched_here += 1


        contour = best[
            "contour"
        ]


        # ====================================================
        # FILL ACTUAL CONTOUR
        # ====================================================

        cv2.drawContours(
            final_mask,
            [contour],
            -1,
            255,
            thickness=cv2.FILLED
        )


        # ====================================================
        # DRAW ACTUAL CONTOUR ON OVERLAY
        # ====================================================

        cv2.drawContours(
            overlay,
            [contour],
            -1,
            (0, 255, 0),
            thickness=2
        )


    # ========================================================
    # NO AGGRESSIVE CLEANUP
    #
    # Do NOT remove components here.
    # ========================================================


    # ========================================================
    # SAVE MASK
    # ========================================================

    mask_path = os.path.join(
        MASK_DIR,
        stem + ".png"
    )


    cv2.imwrite(
        mask_path,
        final_mask
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
    # STATISTICS
    # ========================================================

    roof_pixels = int(
        np.sum(
            final_mask > 0
        )
    )


    roof_percentage = (
        roof_pixels /
        float(height * width)
    ) * 100


    summary.append({

        "image": filename,

        "predicted_candidates": len(
            image_roofs
        ),

        "matched_contours": matched_here,

        "unmatched_candidates": unmatched_here,

        "roof_pixels": roof_pixels,

        "roof_percentage": roof_percentage
    })


    total_selected += len(
        image_roofs
    )

    total_matched += matched_here

    total_unmatched += unmatched_here


    # ========================================================
    # PROGRESS
    # ========================================================

    if (
        image_index % 100 == 0
        or image_index == len(unique_images)
    ):

        print(
            f"Processed "
            f"{image_index}/{len(unique_images)}"
        )


# ============================================================
# SAVE SUMMARY
# ============================================================

summary_df = pd.DataFrame(
    summary
)


summary_path = os.path.join(
    OUTPUT_DIR,
    "final_roof_summary.csv"
)


summary_df.to_csv(
    summary_path,
    index=False
)


# ============================================================
# FINAL REPORT
# ============================================================

print("\n" + "=" * 70)
print("STEP 12 COMPLETED")
print("=" * 70)

print(
    "Images processed:",
    len(summary_df)
)

print(
    "Predicted candidates:",
    total_selected
)

print(
    "Matched contours:",
    total_matched
)

print(
    "Unmatched candidates:",
    total_unmatched
)

if total_selected > 0:

    print(
        "Matching rate:",
        round(
            total_matched /
            total_selected *
            100,
            2
        ),
        "%"
    )


print(
    "\nMasks:",
    MASK_DIR
)

print(
    "\nOverlays:",
    OVERLAY_DIR
)

print(
    "\nSummary:",
    summary_path
)