import os
import cv2
import numpy as np
import pandas as pd


# ============================================================
# STEP 10B - IMPROVED GT ROOF CANDIDATE LABELING
# ============================================================

BASE_DIR = r"C:\Users\Hemila Saravanan\Downloads\Hybrid_Rooftop_Solar"

INPUT_CSV = os.path.join(
    BASE_DIR,
    "outputs",
    "roof_candidate_features",
    "roof_candidate_actual_contour_labeled.csv"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "outputs",
    "roof_candidate_features"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# PARAMETERS
# ============================================================

# Ignore extremely tiny contour regions
MIN_CANDIDATE_AREA = 300

# Minimum IoU with GT to call it a roof
MIN_IOU = 0.10

# Minimum fraction of GT covered
MIN_GT_COVERAGE = 0.05


# ============================================================
# LOAD
# ============================================================

df = pd.read_csv(
    INPUT_CSV
)


print("=" * 70)
print("STEP 10B - IMPROVED ROOF CANDIDATE LABELING")
print("=" * 70)

print(
    "Original candidates:",
    len(df)
)


# ============================================================
# REMOVE VERY SMALL CANDIDATES
# ============================================================

df = df[
    df["area"] >= MIN_CANDIDATE_AREA
].copy()


print(
    "After minimum area filtering:",
    len(df)
)


# ============================================================
# LABEL
# ============================================================

df["label"] = np.where(

    (
        df["actual_contour_iou"]
        >= MIN_IOU
    )
    &
    (
        df["gt_coverage"]
        >= MIN_GT_COVERAGE
    ),

    "ROOF",

    "NON_ROOF"
)


# ============================================================
# REPORT
# ============================================================

print("\n" + "=" * 70)
print("NEW LABEL DISTRIBUTION")
print("=" * 70)

print(
    df["label"].value_counts()
)


total = len(df)


for label, count in (
    df["label"]
    .value_counts()
    .items()
):

    print(
        f"{label:10s}: "
        f"{count:8d} "
        f"({100 * count / total:.2f}%)"
    )


# ============================================================
# ROOF / NON-ROOF
# ============================================================

roof_df = df[
    df["label"] == "ROOF"
]

nonroof_df = df[
    df["label"] == "NON_ROOF"
]


print("\nRoof candidates:")
print(
    len(roof_df)
)


print("\nNon-roof candidates:")
print(
    len(nonroof_df)
)


# ============================================================
# ROOF STATISTICS
# ============================================================

print("\n" + "=" * 70)
print("ROOF CANDIDATE STATISTICS")
print("=" * 70)

if len(roof_df) > 0:

    print(
        roof_df[
            [
                "area",
                "gt_overlap",
                "gt_coverage",
                "actual_contour_iou"
            ]
        ].describe()
    )


# ============================================================
# SAVE
# ============================================================

OUTPUT_CSV = os.path.join(
    OUTPUT_DIR,
    "roof_candidate_final_labels.csv"
)

df.to_csv(
    OUTPUT_CSV,
    index=False
)


print("\nSaved:")
print(OUTPUT_CSV)