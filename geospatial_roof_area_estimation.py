# ============================================================
# STEP 15 - CORRECTED GEOSPATIAL ROOF AREA ESTIMATION
# EPSG:4326 -> METRES USING GEODESIC DISTANCE
# ============================================================

import os
import cv2
import numpy as np
import pandas as pd
import rasterio

from pyproj import Geod


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

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "outputs",
    "step15_roof_area"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# GEODESIC MODEL
# ============================================================

geod = Geod(
    ellps="WGS84"
)


# ============================================================
# START
# ============================================================

print("=" * 70)
print("STEP 15 - CORRECTED GEOSPATIAL ROOF AREA ESTIMATION")
print("=" * 70)

print(
    "CRS handling: EPSG:4326"
)

print(
    "Area conversion: latitude/longitude -> metres"
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
    "\nStep 14 masks found:",
    len(mask_files)
)


# ============================================================
# RESULTS
# ============================================================

results = []

processed = 0
missing_images = 0
invalid_geotiff = 0


# ============================================================
# PROCESS EACH MASK
# ============================================================

for index, mask_filename in enumerate(
    mask_files,
    start=1
):

    # ========================================================
    # IMAGE STEM
    # ========================================================

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

        print(
            "[WARNING] Cannot read mask:",
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
            "[WARNING] GeoTIFF missing:",
            stem
        )

        missing_images += 1
        continue


    # ========================================================
    # READ GEOTIFF
    # ========================================================

    try:

        with rasterio.open(
            image_path
        ) as src:

            image_width = src.width
            image_height = src.height

            transform = src.transform

            crs = src.crs

            bounds = src.bounds


    except Exception as e:

        print(
            "[WARNING] Cannot read GeoTIFF:",
            stem,
            e
        )

        invalid_geotiff += 1
        continue


    # ========================================================
    # PIXEL SIZE IN DEGREES
    # ========================================================

    pixel_lon_deg = abs(
        transform.a
    )

    pixel_lat_deg = abs(
        transform.e
    )


    # ========================================================
    # IMAGE CENTER
    # ========================================================

    center_lon = (
        bounds.left +
        bounds.right
    ) / 2.0


    center_lat = (
        bounds.bottom +
        bounds.top
    ) / 2.0


    # ========================================================
    # CONVERT ONE PIXEL WIDTH TO METRES
    #
    # Longitude distance at image latitude
    # ========================================================

    lon1 = center_lon
    lon2 = (
        center_lon +
        pixel_lon_deg
    )

    lat = center_lat


    _, _, pixel_width_m = geod.inv(
        lon1,
        lat,
        lon2,
        lat
    )


    # ========================================================
    # CONVERT ONE PIXEL HEIGHT TO METRES
    #
    # Latitude distance
    # ========================================================

    lat1 = center_lat
    lat2 = (
        center_lat +
        pixel_lat_deg
    )


    _, _, pixel_height_m = geod.inv(
        center_lon,
        lat1,
        center_lon,
        lat2
    )


    pixel_width_m = abs(
        pixel_width_m
    )

    pixel_height_m = abs(
        pixel_height_m
    )


    # ========================================================
    # PIXEL AREA IN M²
    # ========================================================

    pixel_area_m2 = (
        pixel_width_m *
        pixel_height_m
    )


    # ========================================================
    # CHECK MASK SIZE
    # ========================================================

    mask_height, mask_width = (
        mask.shape
    )


    if (
        mask_width != image_width
        or
        mask_height != image_height
    ):

        print(
            "[WARNING] Size mismatch:",
            stem,
            "Mask:",
            mask.shape,
            "GeoTIFF:",
            (
                image_height,
                image_width
            )
        )


        mask = cv2.resize(
            mask,
            (
                image_width,
                image_height
            ),
            interpolation=cv2.INTER_NEAREST
        )


        mask_height, mask_width = (
            mask.shape
        )


    # ========================================================
    # ROOF PIXELS
    # ========================================================

    roof_pixels = int(
        np.count_nonzero(
            mask > 0
        )
    )


    total_pixels = (
        mask_height *
        mask_width
    )


    # ========================================================
    # ROOF AREA
    # ========================================================

    roof_area_m2 = (
        roof_pixels *
        pixel_area_m2
    )


    roof_area_hectares = (
        roof_area_m2 /
        10000.0
    )


    # ========================================================
    # IMAGE GROUND AREA
    # ========================================================

    image_area_m2 = (
        total_pixels *
        pixel_area_m2
    )


    # ========================================================
    # ROOF COVERAGE
    # ========================================================

    if image_area_m2 > 0:

        roof_percentage = (
            roof_area_m2 /
            image_area_m2
        ) * 100

    else:

        roof_percentage = 0


    # ========================================================
    # CONNECTED ROOF REGIONS
    # ========================================================

    num_labels, labels, stats, centroids = (
        cv2.connectedComponentsWithStats(
            mask,
            connectivity=8
        )
    )


    region_count = 0

    region_areas = []


    for component_id in range(
        1,
        num_labels
    ):

        component_pixels = stats[
            component_id,
            cv2.CC_STAT_AREA
        ]


        if component_pixels > 0:

            region_count += 1


            component_area_m2 = (
                component_pixels *
                pixel_area_m2
            )


            region_areas.append(
                component_area_m2
            )


    # ========================================================
    # LARGEST REGION
    # ========================================================

    if region_areas:

        largest_region_m2 = max(
            region_areas
        )

    else:

        largest_region_m2 = 0


    # ========================================================
    # AVERAGE REGION
    # ========================================================

    if region_areas:

        average_region_m2 = float(
            np.mean(
                region_areas
            )
        )

    else:

        average_region_m2 = 0


    # ========================================================
    # SAVE RESULT
    # ========================================================

    results.append({

        "image": stem,

        "mask_file":
            mask_filename,

        "crs":
            str(crs),

        "center_latitude":
            center_lat,

        "center_longitude":
            center_lon,

        "pixel_width_degrees":
            pixel_lon_deg,

        "pixel_height_degrees":
            pixel_lat_deg,

        "pixel_width_m":
            pixel_width_m,

        "pixel_height_m":
            pixel_height_m,

        "pixel_area_m2":
            pixel_area_m2,

        "image_width_pixels":
            image_width,

        "image_height_pixels":
            image_height,

        "total_image_pixels":
            total_pixels,

        "image_area_m2":
            image_area_m2,

        "roof_pixels":
            roof_pixels,

        "roof_area_m2":
            roof_area_m2,

        "roof_area_hectares":
            roof_area_hectares,

        "roof_percentage":
            roof_percentage,

        "roof_region_count":
            region_count,

        "largest_region_m2":
            largest_region_m2,

        "average_region_m2":
            average_region_m2
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

area_df = pd.DataFrame(
    results
)


# ============================================================
# SAVE CSV
# ============================================================

csv_path = os.path.join(
    OUTPUT_DIR,
    "roof_area_estimation_corrected.csv"
)


area_df.to_csv(
    csv_path,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

if len(area_df) > 0:

    total_roof_area = float(
        area_df[
            "roof_area_m2"
        ].sum()
    )


    mean_roof_area = float(
        area_df[
            "roof_area_m2"
        ].mean()
    )


    min_roof_area = float(
        area_df[
            "roof_area_m2"
        ].min()
    )


    max_roof_area = float(
        area_df[
            "roof_area_m2"
        ].max()
    )


    mean_coverage = float(
        area_df[
            "roof_percentage"
        ].mean()
    )


    mean_pixel_area = float(
        area_df[
            "pixel_area_m2"
        ].mean()
    )


    mean_pixel_width = float(
        area_df[
            "pixel_width_m"
        ].mean()
    )


    mean_pixel_height = float(
        area_df[
            "pixel_height_m"
        ].mean()
    )

else:

    total_roof_area = 0
    mean_roof_area = 0
    min_roof_area = 0
    max_roof_area = 0
    mean_coverage = 0
    mean_pixel_area = 0
    mean_pixel_width = 0
    mean_pixel_height = 0


# ============================================================
# FINAL REPORT
# ============================================================

print("\n" + "=" * 70)
print("STEP 15 CORRECTED - COMPLETED")
print("=" * 70)

print(
    "Masks found:",
    len(mask_files)
)

print(
    "Images processed:",
    processed
)

print(
    "Missing GeoTIFFs:",
    missing_images
)

print(
    "Invalid GeoTIFFs:",
    invalid_geotiff
)

print("\n" + "-" * 70)

print(
    "Average pixel width:",
    round(
        mean_pixel_width,
        4
    ),
    "m"
)

print(
    "Average pixel height:",
    round(
        mean_pixel_height,
        4
    ),
    "m"
)

print(
    "Average pixel area:",
    round(
        mean_pixel_area,
        8
    ),
    "m²"
)

print("\n" + "-" * 70)

print(
    "Total estimated roof area:",
    round(
        total_roof_area,
        2
    ),
    "m²"
)

print(
    "Mean roof area per image:",
    round(
        mean_roof_area,
        2
    ),
    "m²"
)

print(
    "Minimum roof area:",
    round(
        min_roof_area,
        2
    ),
    "m²"
)

print(
    "Maximum roof area:",
    round(
        max_roof_area,
        2
    ),
    "m²"
)

print(
    "Mean roof coverage:",
    round(
        mean_coverage,
        2
    ),
    "%"
)

print("\nOutput CSV:")

print(
    csv_path
)