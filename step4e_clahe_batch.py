import os
import cv2


# ============================================================
# STEP 4E - BATCH CLAHE
# ============================================================

IMAGE_DIR = r"C:\Users\Hemila Saravanan\Downloads\Hybrid_Rooftop_Solar\images_roof_centered_geotiff"

OUTPUT_DIR = r"C:\Users\Hemila Saravanan\Downloads\Hybrid_Rooftop_Solar\outputs\preprocessing\clahe"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ------------------------------------------------------------
# SETTINGS
# ------------------------------------------------------------

valid_extensions = (".tif", ".tiff")

# CLAHE parameters
clip_limit = 2.0
tile_grid_size = (8, 8)


# ------------------------------------------------------------
# GET ALL IMAGES
# ------------------------------------------------------------

image_files = sorted([
    f
    for f in os.listdir(IMAGE_DIR)
    if f.lower().endswith(valid_extensions)
])


print("=" * 60)
print("STEP 4E - BATCH CLAHE")
print("=" * 60)

print("Total images found:", len(image_files))


successful = 0
failed = 0


# ------------------------------------------------------------
# PROCESS ALL IMAGES
# ------------------------------------------------------------

for i, filename in enumerate(image_files, start=1):

    input_path = os.path.join(IMAGE_DIR, filename)

    image = cv2.imread(
        input_path,
        cv2.IMREAD_COLOR
    )

    if image is None:
        print("FAILED TO READ:", filename)
        failed += 1
        continue

    # --------------------------------------------------------
    # Convert BGR → LAB
    # --------------------------------------------------------

    lab = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2LAB
    )

    # Separate LAB channels
    l_channel, a_channel, b_channel = cv2.split(lab)

    # --------------------------------------------------------
    # Apply CLAHE only to L (brightness) channel
    # --------------------------------------------------------

    clahe = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=tile_grid_size
    )

    enhanced_l = clahe.apply(l_channel)

    # --------------------------------------------------------
    # Merge channels
    # --------------------------------------------------------

    enhanced_lab = cv2.merge([
        enhanced_l,
        a_channel,
        b_channel
    ])

    # LAB → BGR
    enhanced = cv2.cvtColor(
        enhanced_lab,
        cv2.COLOR_LAB2BGR
    )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    output_name = (
        os.path.splitext(filename)[0]
        + ".png"
    )

    output_path = os.path.join(
        OUTPUT_DIR,
        output_name
    )

    if cv2.imwrite(output_path, enhanced):
        successful += 1
    else:
        failed += 1
        print("FAILED TO SAVE:", filename)

    if i % 100 == 0 or i == len(image_files):
        print(
            f"Processed {i}/{len(image_files)} images"
        )


# ============================================================
# RESULTS
# ============================================================

print("\n" + "=" * 60)
print("BATCH CLAHE COMPLETED")
print("=" * 60)

print("Successful:", successful)
print("Failed:", failed)

print("\nOutput folder:")
print(OUTPUT_DIR)

if successful == len(image_files):
    print("\n ALL 1880 IMAGES PROCESSED SUCCESSFULLY")
else:
    print("\nSome images failed.")