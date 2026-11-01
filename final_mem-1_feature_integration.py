# ============================================================
# STEP 18 - FINAL MEMBER-1 FEATURE INTEGRATION
# ============================================================
#
# Combines:
#   Step 15 -> Roof area
#   Step 16 -> Roof orientation
#   Step 17 -> Shadow detection
#
# FINAL OUTPUT:
#   One CSV containing all extracted rooftop features
#
# This is the FINAL FEATURE TABLE for Member 1.
# ============================================================

import os
import pandas as pd
import numpy as np


# ============================================================
# BASE DIRECTORY
# ============================================================

BASE_DIR = r"C:\Users\Hemila Saravanan\Downloads\Hybrid_Rooftop_Solar"


# ============================================================
# INPUT FILES
# ============================================================

STEP15_CSV = os.path.join(
    BASE_DIR,
    "outputs",
    "step15_roof_area",
    "roof_area_estimation_corrected.csv"
)

STEP16_CSV = os.path.join(
    BASE_DIR,
    "outputs",
    "step16_roof_orientation",
    "roof_area_orientation.csv"
)

STEP17_CSV = os.path.join(
    BASE_DIR,
    "outputs",
    "step17_shadow_detection",
    "roof_area_orientation_shadow.csv"
)


# ============================================================
# OUTPUT DIRECTORY
# ============================================================

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "outputs",
    "step18_member1_final_features"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# OUTPUT FILES
# ============================================================

FINAL_CSV = os.path.join(
    OUTPUT_DIR,
    "member1_final_rooftop_features.csv"
)

SUMMARY_CSV = os.path.join(
    OUTPUT_DIR,
    "member1_feature_summary.csv"
)


# ============================================================
# START
# ============================================================

print("=" * 70)
print("STEP 18 - FINAL MEMBER-1 FEATURE INTEGRATION")
print("=" * 70)


# ============================================================
# CHECK FILES
# ============================================================

required_files = {

    "Step 15":
        STEP15_CSV,

    "Step 16":
        STEP16_CSV,

    "Step 17":
        STEP17_CSV
}


for name, path in required_files.items():

    if not os.path.exists(path):

        raise FileNotFoundError(
            f"{name} file not found:\n{path}"
        )


# ============================================================
# LOAD DATA
# ============================================================

area_df = pd.read_csv(
    STEP15_CSV
)

orientation_df = pd.read_csv(
    STEP16_CSV
)

shadow_df = pd.read_csv(
    STEP17_CSV
)


print("\nInput records:")

print(
    "Step 15:",
    len(area_df)
)

print(
    "Step 16:",
    len(orientation_df)
)

print(
    "Step 17:",
    len(shadow_df)
)


# ============================================================
# NORMALIZE IMAGE IDENTIFIER
# ============================================================

def normalize_image_id(value):

    value = str(
        value
    ).strip()

    # Remove file extension
    value = os.path.splitext(
        value
    )[0]

    # Convert numeric IDs consistently
    try:

        value = str(
            int(
                float(value)
            )
        )

    except:

        pass

    return value


area_df["image"] = (
    area_df["image"]
    .apply(normalize_image_id)
)

orientation_df["image"] = (
    orientation_df["image"]
    .apply(normalize_image_id)
)

shadow_df["image"] = (
    shadow_df["image"]
    .apply(normalize_image_id)
)


# ============================================================
# REMOVE DUPLICATES
# ============================================================

area_df = area_df.drop_duplicates(
    subset=["image"]
)

orientation_df = orientation_df.drop_duplicates(
    subset=["image"]
)

shadow_df = shadow_df.drop_duplicates(
    subset=["image"]
)


# ============================================================
# SELECT RELEVANT STEP 15 FEATURES
# ============================================================

area_columns = [

    "image",

    "pixel_width_m",

    "pixel_height_m",

    "pixel_area_m2",

    "image_area_m2",

    "roof_pixels",

    "roof_area_m2",

    "roof_area_hectares",

    "roof_percentage",

    "roof_region_count",

    "largest_region_m2",

    "average_region_m2"
]


area_columns = [
    c
    for c in area_columns
    if c in area_df.columns
]


area_features = area_df[
    area_columns
].copy()


# ============================================================
# SELECT RELEVANT STEP 16 FEATURES
# ============================================================

orientation_columns = [

    "image",

    "orientation_deg",

    "orientation_cardinal",

    "orientation_confidence",

    "min_area_rect_orientation_deg",

    "roof_components",

    "largest_component_area_pixels",

    "principal_eigenvalue",

    "secondary_eigenvalue"
]


orientation_columns = [
    c
    for c in orientation_columns
    if c in orientation_df.columns
]


orientation_features = orientation_df[
    orientation_columns
].copy()


# ============================================================
# SELECT RELEVANT STEP 17 FEATURES
# ============================================================

shadow_columns = [

    "image",

    "shadow_pixels",

    "shadow_percentage",

    "shadow_area_m2"
]


shadow_columns = [
    c
    for c in shadow_columns
    if c in shadow_df.columns
]


shadow_features = shadow_df[
    shadow_columns
].copy()


# ============================================================
# MERGE STEP 15 + STEP 16
# ============================================================

final_df = pd.merge(
    area_features,
    orientation_features,
    on="image",
    how="outer"
)


# ============================================================
# MERGE STEP 17
# ============================================================

final_df = pd.merge(
    final_df,
    shadow_features,
    on="image",
    how="outer"
)


# ============================================================
# SORT BY IMAGE NUMBER
# ============================================================

def numeric_sort_key(value):

    try:

        return int(value)

    except:

        return 999999999


final_df["_sort_key"] = (
    final_df["image"]
    .apply(numeric_sort_key)
)

final_df = final_df.sort_values(
    "_sort_key"
)

final_df = final_df.drop(
    columns=["_sort_key"]
)

final_df = final_df.reset_index(
    drop=True
)


# ============================================================
# CALCULATE ADDITIONAL MEMBER-1 FEATURES
# ============================================================

# ------------------------------------------------------------
# NON-SHADOW ROOF AREA
# ------------------------------------------------------------

if (
    "roof_area_m2" in final_df.columns
    and
    "shadow_area_m2" in final_df.columns
):

    final_df[
        "non_shadow_roof_area_m2"
    ] = (
        final_df["roof_area_m2"]
        -
        final_df["shadow_area_m2"]
    )


    # Prevent negative values
    final_df[
        "non_shadow_roof_area_m2"
    ] = final_df[
        "non_shadow_roof_area_m2"
    ].clip(
        lower=0
    )


# ------------------------------------------------------------
# USABLE ROOF PERCENTAGE
# ------------------------------------------------------------

if (
    "roof_percentage" in final_df.columns
    and
    "shadow_percentage" in final_df.columns
):

    final_df[
        "usable_roof_percentage"
    ] = (
        final_df["roof_percentage"]
        *
        (
            1 -
            final_df["shadow_percentage"]
            / 100
        )
    )


# ------------------------------------------------------------
# ROOF-TO-SHADOW RATIO
# ------------------------------------------------------------

if (
    "roof_area_m2" in final_df.columns
    and
    "shadow_area_m2" in final_df.columns
):

    final_df[
        "roof_to_shadow_ratio"
    ] = (
        final_df["roof_area_m2"]
        /
        (
            final_df["shadow_area_m2"]
            + 1e-6
        )
    )


# ============================================================
# CLEAN NUMERIC VALUES
# ============================================================

numeric_columns = final_df.select_dtypes(
    include=[np.number]
).columns


for column in numeric_columns:

    final_df[column] = pd.to_numeric(
        final_df[column],
        errors="coerce"
    )


# ============================================================
# CHECK MISSING VALUES
# ============================================================

missing_counts = (
    final_df.isna()
    .sum()
    .reset_index()
)

missing_counts.columns = [
    "feature",
    "missing_values"
]


missing_counts[
    "missing_percentage"
] = (
    missing_counts["missing_values"]
    /
    len(final_df)
) * 100


# ============================================================
# SAVE MISSING VALUE SUMMARY
# ============================================================

missing_path = os.path.join(
    OUTPUT_DIR,
    "missing_value_summary.csv"
)

missing_counts.to_csv(
    missing_path,
    index=False
)


# ============================================================
# SAVE FINAL FEATURE TABLE
# ============================================================

final_df.to_csv(
    FINAL_CSV,
    index=False
)


# ============================================================
# FEATURE SUMMARY
# ============================================================

summary_rows = []


for column in final_df.columns:

    if column == "image":
        continue


    if pd.api.types.is_numeric_dtype(
        final_df[column]
    ):

        series = final_df[column].dropna()


        if len(series) > 0:

            summary_rows.append({

                "feature":
                    column,

                "data_type":
                    str(
                        final_df[
                            column
                        ].dtype
                    ),

                "valid_values":
                    len(series),

                "missing_values":
                    final_df[
                        column
                    ].isna().sum(),

                "minimum":
                    series.min(),

                "maximum":
                    series.max(),

                "mean":
                    series.mean(),

                "median":
                    series.median(),

                "standard_deviation":
                    series.std()
            })

        else:

            summary_rows.append({

                "feature":
                    column,

                "data_type":
                    str(
                        final_df[
                            column
                        ].dtype
                    ),

                "valid_values":
                    0,

                "missing_values":
                    len(final_df),

                "minimum":
                    np.nan,

                "maximum":
                    np.nan,

                "mean":
                    np.nan,

                "median":
                    np.nan,

                "standard_deviation":
                    np.nan
            })


    else:

        summary_rows.append({

            "feature":
                column,

            "data_type":
                str(
                    final_df[
                        column
                    ].dtype
                ),

            "valid_values":
                final_df[
                    column
                ].notna().sum(),

            "missing_values":
                final_df[
                    column
                ].isna().sum(),

            "minimum":
                np.nan,

            "maximum":
                np.nan,

            "mean":
                np.nan,

            "median":
                np.nan,

            "standard_deviation":
                np.nan
        })


summary_df = pd.DataFrame(
    summary_rows
)


summary_df.to_csv(
    SUMMARY_CSV,
    index=False
)


# ============================================================
# FINAL REPORT
# ============================================================

print("\n" + "=" * 70)
print("STEP 18 COMPLETED")
print("=" * 70)

print(
    "Final images:",
    len(final_df)
)

print(
    "Final features:",
    len(final_df.columns)
)

print(
    "\nFeature names:"
)

for i, column in enumerate(
    final_df.columns,
    start=1
):

    print(
        f"{i:02d}. {column}"
    )


# ============================================================
# IMPORTANT STATISTICS
# ============================================================

print("\n" + "-" * 70)

if "roof_area_m2" in final_df.columns:

    print(
        "Total roof area:",
        round(
            final_df[
                "roof_area_m2"
            ].sum(),
            2
        ),
        "m²"
    )


if "shadow_area_m2" in final_df.columns:

    print(
        "Total shadow area:",
        round(
            final_df[
                "shadow_area_m2"
            ].sum(),
            2
        ),
        "m²"
    )


if "usable_roof_percentage" in final_df.columns:

    print(
        "Mean usable roof percentage:",
        round(
            final_df[
                "usable_roof_percentage"
            ].mean(),
            2
        ),
        "%"
    )


if "orientation_deg" in final_df.columns:

    print(
        "Mean roof orientation:",
        round(
            final_df[
                "orientation_deg"
            ].mean(),
            2
        ),
        "degrees"
    )


if "shadow_percentage" in final_df.columns:

    print(
        "Mean shadow percentage:",
        round(
            final_df[
                "shadow_percentage"
            ].mean(),
            2
        ),
        "%"
    )


# ============================================================
# OUTPUT PATHS
# ============================================================

print("\n" + "-" * 70)

print(
    "FINAL MEMBER-1 FEATURE CSV:"
)

print(
    FINAL_CSV
)

print(
    "\nFEATURE SUMMARY:"
)

print(
    SUMMARY_CSV
)

print(
    "\nMISSING VALUE REPORT:"
)

print(
    missing_path
)

print("\n" + "=" * 70)