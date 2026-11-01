import os
import cv2


# ============================================================
# STEP 4B - BATCH GAUSSIAN FILTERING
# ============================================================

IMAGE_DIR = r"C:\Users\Hemila Saravanan\Downloads\Hybrid_Rooftop_Solar\images_roof_centered_geotiff"

OUTPUT_DIR = r"C:\Users\Hemila Saravanan\Downloads\Hybrid_Rooftop_Solar\outputs\preprocessing\gaussian"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ------------------------------------------------------------
# SUPPORTED IMAGE FORMATS
# ------------------------------------------------------------

valid_extensions = (".tif", ".tiff")


# ------------------------------------------------------------
# GET ALL IMAGES
# ------------------------------------------------------------

image_files = sorted([
    f for f in os.listdir(IMAGE_DIR)
    if f.lower().endswith(valid_extensions)
])


print("=" * 60)
print("STEP 4B - BATCH GAUSSIAN FILTERING")
print("=" * 60)

print("Total images found:", len(image_files))


# ------------------------------------------------------------
# PROCESS ALL IMAGES
# ------------------------------------------------------------

successful = 0
failed = 0

for i, filename in enumerate(image_files, start=1):

    input_path = os.path.join(
        IMAGE_DIR,
        filename
    )

    # Read image
    image = cv2.imread(
        input_path,
        cv2.IMREAD_COLOR
    )

    if image is None:
        print("FAILED:", filename)
        failed += 1
        continue

    # --------------------------------------------------------
    # APPLY GAUSSIAN FILTER
    # --------------------------------------------------------

    gaussian = cv2.GaussianBlur(
        image,
        (5, 5),
        0
    )

    # --------------------------------------------------------
    # OUTPUT NAME
    # --------------------------------------------------------

    output_name = os.path.splitext(filename)[0] + ".png"

    output_path = os.path.join(
        OUTPUT_DIR,
        output_name
    )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    success = cv2.imwrite(
        output_path,
        gaussian
    )

    if success:
        successful += 1
    else:
        failed += 1
        print("FAILED TO SAVE:", filename)

    # Progress every 100 images
    if i % 100 == 0 or i == len(image_files):
        print(
            f"Processed {i}/{len(image_files)} images"
        )


# ============================================================
# FINAL RESULTS
# ============================================================

print("\n" + "=" * 60)
print("BATCH GAUSSIAN FILTERING COMPLETED")
print("=" * 60)

print("Successful:", successful)
print("Failed:", failed)

print("\nOutput folder:")
print(OUTPUT_DIR)

if successful == len(image_files):
    print("\n ALL 1880 IMAGES PROCESSED SUCCESSFULLY")
else:
    print("\n Some images failed.")