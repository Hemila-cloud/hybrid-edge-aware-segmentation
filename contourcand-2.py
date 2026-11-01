import os
import cv2
import numpy as np
import pandas as pd


# ============================================================
# STEP 7 - CONTOUR CANDIDATE GENERATION
# ============================================================

EDGE_DIR = r"C:\Users\Hemila Saravanan\Downloads\Hybrid_Rooftop_Solar\outputs\edges"

IMAGE_DIR = r"C:\Users\Hemila Saravanan\Downloads\Hybrid_Rooftop_Solar\images_roof_centered_geotiff"

OUTPUT_DIR = r"C:\Users\Hemila Saravanan\Downloads\Hybrid_Rooftop_Solar\outputs\contour_candidates_v3"

OUTLINE_DIR = os.path.join(
    OUTPUT_DIR,
    "contour_outlines"
)

OVERLAY_DIR = os.path.join(
    OUTPUT_DIR,
    "contour_overlays"
)

os.makedirs(OUTLINE_DIR, exist_ok=True)
os.makedirs(OVERLAY_DIR, exist_ok=True)


# ============================================================
# PARAMETERS
# ============================================================

MIN_CONTOUR_AREA = 100

CLOSE_KERNEL_SIZE = 3
CLOSE_ITERATIONS = 1

VALID_EXTENSIONS = (
    ".png",
    ".jpg",
    ".jpeg",
    ".tif",
    ".tiff"
)


# ============================================================
# FILES
# ============================================================

files = sorted([
    f for f in os.listdir(EDGE_DIR)
    if f.lower().endswith(VALID_EXTENSIONS)
])


print("=" * 70)
print("STEP 7 - CONTOUR CANDIDATE GENERATION")
print("=" * 70)

print("Images found:", len(files))


all_features = []

total_contours = 0
accepted_contours = 0


# ============================================================
# PROCESS EACH IMAGE
# ============================================================

for index, filename in enumerate(files, start=1):

    edge_path = os.path.join(
        EDGE_DIR,
        filename
    )

    edge = cv2.imread(
        edge_path,
        cv2.IMREAD_GRAYSCALE
    )

    if edge is None:
        print("[ERROR] Edge image:", filename)
        continue


    # --------------------------------------------------------
    # READ ORIGINAL IMAGE
    # --------------------------------------------------------

    # --------------------------------------------------------
# FIND ORIGINAL IMAGE USING SAME FILE STEM
# --------------------------------------------------------

base_name = os.path.splitext(filename)[0]

original_path = None

for ext in [".tif", ".tiff", ".png", ".jpg", ".jpeg"]:

    candidate_path = os.path.join(
        IMAGE_DIR,
        base_name + ext
    )

    if os.path.exists(candidate_path):

        original_path = candidate_path
        break


if original_path is None:

    print(
        "[WARNING] Original image not found:",
        base_name
    )

    # Use edge image as fallback
    original = cv2.cvtColor(
        edge,
        cv2.COLOR_GRAY2BGR
    )

else:

    original = cv2.imread(
        original_path,
        cv2.IMREAD_COLOR
    )

    if original is None:

        print(
            "[WARNING] Could not read:",
            original_path
        )

        original = cv2.cvtColor(
            edge,
            cv2.COLOR_GRAY2BGR
        )

    if original is None:
        print("[WARNING] Original image not found:", filename)

        # Create blank overlay if original unavailable
        original = cv2.cvtColor(
            edge,
            cv2.COLOR_GRAY2BGR
        )


    # --------------------------------------------------------
    # RESIZE ORIGINAL IF NECESSARY
    # --------------------------------------------------------

    if original.shape[:2] != edge.shape[:2]:

        original = cv2.resize(
            original,
            (
                edge.shape[1],
                edge.shape[0]
            ),
            interpolation=cv2.INTER_AREA
        )


    # --------------------------------------------------------
    # BINARY EDGE
    # --------------------------------------------------------

    binary = np.where(
        edge > 0,
        255,
        0
    ).astype(np.uint8)


    # --------------------------------------------------------
    # LIGHT CLOSING
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # FIND CONTOURS
    # --------------------------------------------------------

    contours, hierarchy = cv2.findContours(
        closed,
        cv2.RETR_LIST,
        cv2.CHAIN_APPROX_SIMPLE
    )


    total_contours += len(contours)


    # --------------------------------------------------------
    # OUTPUT IMAGES
    # --------------------------------------------------------

    outline = np.zeros_like(
        binary
    )

    overlay = original.copy()


    # --------------------------------------------------------
    # PROCESS EACH CONTOUR
    # --------------------------------------------------------

    for contour_id, contour in enumerate(contours):

        area = cv2.contourArea(
            contour
        )


        # Only remove extremely tiny contours
        if area < MIN_CONTOUR_AREA:
            continue


        accepted_contours += 1


        perimeter = cv2.arcLength(
            contour,
            True
        )


        x, y, w, h = cv2.boundingRect(
            contour
        )


        # ----------------------------------------------------
        # ASPECT RATIO
        # ----------------------------------------------------

        if h > 0:
            aspect_ratio = w / float(h)
        else:
            aspect_ratio = 0


        # ----------------------------------------------------
        # EXTENT
        # ----------------------------------------------------

        if w > 0 and h > 0:
            extent = area / float(w * h)
        else:
            extent = 0


        # ----------------------------------------------------
        # SOLIDITY
        # ----------------------------------------------------

        hull = cv2.convexHull(
            contour
        )

        hull_area = cv2.contourArea(
            hull
        )

        if hull_area > 0:
            solidity = area / float(hull_area)
        else:
            solidity = 0


        # ----------------------------------------------------
        # CENTER
        # ----------------------------------------------------

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

            cx = 0
            cy = 0


        # ----------------------------------------------------
        # STORE FEATURES
        # ----------------------------------------------------

        all_features.append({

            "image": filename,

            "contour_id": contour_id,

            "area": area,

            "perimeter": perimeter,

            "x": x,

            "y": y,

            "width": w,

            "height": h,

            "aspect_ratio": aspect_ratio,

            "extent": extent,

            "solidity": solidity,

            "center_x": cx,

            "center_y": cy
        })


        # ----------------------------------------------------
        # DRAW ONLY OUTLINE
        # ----------------------------------------------------

        cv2.drawContours(
            outline,
            [contour],
            -1,
            255,
            thickness=1
        )


        # ----------------------------------------------------
        # DRAW ON ORIGINAL IMAGE
        #
        # This allows us to see where contours occur
        # relative to the actual aerial image.
        # ----------------------------------------------------

        cv2.drawContours(
            overlay,
            [contour],
            -1,
            (0, 255, 0),
            thickness=1
        )


    # --------------------------------------------------------
    # SAVE OUTLINE
    # --------------------------------------------------------

    outline_path = os.path.join(
        OUTLINE_DIR,
        filename
    )

    cv2.imwrite(
        outline_path,
        outline
    )


    # --------------------------------------------------------
    # SAVE OVERLAY
    # --------------------------------------------------------

    overlay_path = os.path.join(
        OVERLAY_DIR,
        filename
    )

    cv2.imwrite(
        overlay_path,
        overlay
    )


    if (
        index % 100 == 0
        or index == len(files)
    ):

        print(
            f"Processed {index}/{len(files)}"
        )


# ============================================================
# SAVE CONTOUR FEATURES
# ============================================================

features_df = pd.DataFrame(
    all_features
)

CSV_PATH = os.path.join(
    OUTPUT_DIR,
    "contour_features.csv"
)

features_df.to_csv(
    CSV_PATH,
    index=False
)


# ============================================================
# FINAL REPORT
# ============================================================

print("\n" + "=" * 70)
print("STEP 7 COMPLETED")
print("=" * 70)

print(
    "Images processed:",
    len(files)
)

print(
    "Total contours:",
    total_contours
)

print(
    "Accepted contours:",
    accepted_contours
)

print(
    "Contour feature rows:",
    len(features_df)
)

print("\nSaved:")
print(OUTLINE_DIR)
print(OVERLAY_DIR)
print(CSV_PATH)