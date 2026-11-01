import os
import cv2
import numpy as np


# ============================================================
# STEP 4F - BATCH NORMALIZATION
# ============================================================

IMAGE_DIR = r"C:\Users\Hemila Saravanan\Downloads\Hybrid_Rooftop_Solar\images_roof_centered_geotiff"

OUTPUT_DIR = r"C:\Users\Hemila Saravanan\Downloads\Hybrid_Rooftop_Solar\outputs\preprocessing\normalization"

os.makedirs(OUTPUT_DIR, exist_ok=True)


valid_extensions = (".tif", ".tiff")


# ------------------------------------------------------------
# GET ALL IMAGES
# ------------------------------------------------------------

image_files = sorted([
    f
    for f in os.listdir(IMAGE_DIR)
    if f.lower().endswith(valid_extensions)
])


print("=" * 60)
print("STEP 4F - BATCH IMAGE NORMALIZATION")
print("=" * 60)

print("Total images found:", len(image_files))


successful = 0
failed = 0


# ------------------------------------------------------------
# PROCESS ALL IMAGES
# ------------------------------------------------------------

for i, filename in enumerate(image_files, start=1):

    input_path = os.path.join(
        IMAGE_DIR,
        filename
    )

    image = cv2.imread(
        input_path,
        cv2.IMREAD_COLOR
    )

    if image is None:
        print("FAILED TO READ:", filename)
        failed += 1
        continue

    # --------------------------------------------------------
    # Convert uint8 0-255 → float32 0-1
    # --------------------------------------------------------

    normalized = image.astype(
        np.float32
    ) / 255.0

    # --------------------------------------------------------
    # SAVE ACTUAL 0-1 ARRAY
    # --------------------------------------------------------

    output_name = (
        os.path.splitext(filename)[0]
        + ".npy"
    )

    output_path = os.path.join(
        OUTPUT_DIR,
        output_name
    )

    try:

        np.save(
            output_path,
            normalized
        )

        successful += 1

    except Exception as e:

        print(
            "FAILED TO SAVE:",
            filename,
            "|",
            e
        )

        failed += 1

    if i % 100 == 0 or i == len(image_files):

        print(
            f"Processed {i}/{len(image_files)} images"
        )


# ============================================================
# RESULTS
# ============================================================

print("\n" + "=" * 60)
print("BATCH NORMALIZATION COMPLETED")
print("=" * 60)

print("Successful:", successful)
print("Failed:", failed)

print("\nOutput folder:")
print(OUTPUT_DIR)

if successful == len(image_files):
    print("\n✓ ALL 1880 IMAGES NORMALIZED SUCCESSFULLY")
else:
    print("\n⚠ Some images failed.")


# ============================================================
# VERIFY ONE NORMALIZED IMAGE
# ============================================================

if successful > 0:

    first_file = os.path.splitext(
        image_files[0]
    )[0] + ".npy"

    first_path = os.path.join(
        OUTPUT_DIR,
        first_file
    )

    sample = np.load(first_path)

    print("\n" + "=" * 60)
    print("NORMALIZATION VERIFICATION")
    print("=" * 60)

    print("Sample file:", first_file)
    print("Shape:", sample.shape)
    print("Datatype:", sample.dtype)
    print("Minimum value:", sample.min())
    print("Maximum value:", sample.max())
    print("Mean value:", sample.mean())

    if sample.min() >= 0.0 and sample.max() <= 1.0:
        print(" Pixel values correctly normalized to 0-1")
    else:
        print(" Normalization range is incorrect")
