import os
import csv
import random


# ============================================================
# STEP 3 - TRAIN / VALIDATION / TEST SPLIT
# ============================================================

BASE_DIR = r"C:\Users\Hemila Saravanan\Downloads\Hybrid_Rooftop_Solar"

INDEX_FILE = os.path.join(
    BASE_DIR,
    "outputs",
    "dataset_index.csv"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "outputs",
    "splits"
)


# ============================================================
# SETTINGS
# ============================================================

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

RANDOM_SEED = 42


# ============================================================
# CHECK RATIOS
# ============================================================

if TRAIN_RATIO + VAL_RATIO + TEST_RATIO != 1.0:

    raise ValueError(
        "Train + Validation + Test ratios must equal 1.0"
    )


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# LOAD DATASET INDEX
# ============================================================

dataset_pairs = []

with open(
    INDEX_FILE,
    "r",
    encoding="utf-8"
) as file:

    reader = csv.DictReader(file)

    for row in reader:

        dataset_pairs.append({
            "image": row["image"],
            "mask": row["mask"]
        })


print("=" * 60)
print("STEP 3 - DATASET SPLITTING")
print("=" * 60)

print("Total pairs:", len(dataset_pairs))


# ============================================================
# SHUFFLE
# ============================================================

random.seed(RANDOM_SEED)

random.shuffle(dataset_pairs)


# ============================================================
# CALCULATE SPLIT SIZES
# ============================================================

total = len(dataset_pairs)

train_count = int(total * TRAIN_RATIO)

val_count = int(total * VAL_RATIO)

test_count = total - train_count - val_count


# ============================================================
# CREATE SPLITS
# ============================================================

train_data = dataset_pairs[
    :train_count
]

val_data = dataset_pairs[
    train_count:train_count + val_count
]

test_data = dataset_pairs[
    train_count + val_count:
]


# ============================================================
# SAVE FUNCTION
# ============================================================

def save_split(data, filename):

    output_file = os.path.join(
        OUTPUT_DIR,
        filename
    )

    with open(
        output_file,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.writer(file)

        writer.writerow([
            "image",
            "mask"
        ])

        for item in data:

            writer.writerow([
                item["image"],
                item["mask"]
            ])


# ============================================================
# SAVE SPLITS
# ============================================================

save_split(
    train_data,
    "train.csv"
)

save_split(
    val_data,
    "validation.csv"
)

save_split(
    test_data,
    "test.csv"
)


# ============================================================
# DISPLAY RESULTS
# ============================================================

print("\nSplit results:")

print(
    "Training   :",
    len(train_data)
)

print(
    "Validation :",
    len(val_data)
)

print(
    "Testing    :",
    len(test_data)
)

print(
    "Total      :",
    len(train_data)
    + len(val_data)
    + len(test_data)
)


# ============================================================
# CHECK DUPLICATES
# ============================================================

train_names = {
    item["image"]
    for item in train_data
}

val_names = {
    item["image"]
    for item in val_data
}

test_names = {
    item["image"]
    for item in test_data
}


train_val_overlap = train_names & val_names

train_test_overlap = train_names & test_names

val_test_overlap = val_names & test_names


print("\nOverlap checks:")

print(
    "Train ∩ Validation :",
    len(train_val_overlap)
)

print(
    "Train ∩ Test       :",
    len(train_test_overlap)
)

print(
    "Validation ∩ Test  :",
    len(val_test_overlap)
)


# ============================================================
# FINAL CHECK
# ============================================================

if (
    len(train_data) == 1316
    and len(val_data) == 282
    and len(test_data) == 282
    and len(train_val_overlap) == 0
    and len(train_test_overlap) == 0
    and len(val_test_overlap) == 0
):

    print("\n✓ DATASET SPLIT SUCCESSFUL")

else:

    print("\n PLEASE CHECK THE SPLIT")


print("\nFiles saved in:")

print(OUTPUT_DIR)