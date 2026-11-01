import os
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import GroupShuffleSplit
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)


# ============================================================
# STEP 11 - ROOF / NON-ROOF CANDIDATE CLASSIFICATION
# ============================================================

BASE_DIR = r"C:\Users\Hemila Saravanan\Downloads\Hybrid_Rooftop_Solar"

INPUT_CSV = os.path.join(
    BASE_DIR,
    "outputs",
    "roof_candidate_features",
    "roof_candidate_final_labels.csv"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "outputs",
    "roof_candidate_model"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


print("=" * 70)
print("STEP 11 - ROOF CANDIDATE CLASSIFICATION")
print("=" * 70)


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(
    INPUT_CSV
)

print(
    "Total candidates:",
    len(df)
)


# ============================================================
# LABELS
# ============================================================

df = df[
    df["label"].isin(
        ["ROOF", "NON_ROOF"]
    )
].copy()


df["target"] = (
    df["label"] == "ROOF"
).astype(int)


print("\nClass distribution:")

print(
    df["label"].value_counts()
)


# ============================================================
# SHOW AVAILABLE COLUMNS
# ============================================================

print("\nAvailable columns:")

print(
    df.columns.tolist()
)


# ============================================================
# FEATURES
#
# ONLY USE FEATURES THAT WERE ACTUALLY CREATED IN STEP 9.
#
# GT-derived columns are deliberately excluded.
# ============================================================

POSSIBLE_FEATURES = [

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


FEATURES = [
    feature
    for feature in POSSIBLE_FEATURES
    if feature in df.columns
]


print("\nFeatures used:")

for feature in FEATURES:
    print(
        " -",
        feature
    )


# ============================================================
# SAFETY CHECK
# ============================================================

if len(FEATURES) < 3:

    raise ValueError(
        "Too few usable features found. "
        "Check the Step 9 feature CSV."
    )


# ============================================================
# IMPORTANT:
# REMOVE GT-DERIVED FEATURES
# ============================================================

GT_FEATURES = [

    "gt_overlap",
    "gt_coverage",
    "actual_contour_iou"
]


# ============================================================
# CLEAN
# ============================================================

df[FEATURES] = (
    df[FEATURES]
    .replace(
        [np.inf, -np.inf],
        np.nan
    )
)


df = df.dropna(
    subset=FEATURES + ["target"]
).copy()


# ============================================================
# X / Y
# ============================================================

X = df[FEATURES]

y = df["target"]


# ============================================================
# GROUP SPLIT BY IMAGE
#
# All contours belonging to one image stay in the same
# train/test group.
# ============================================================

groups = df["image"]


splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42
)


train_idx, test_idx = next(
    splitter.split(
        X,
        y,
        groups=groups
    )
)


X_train = X.iloc[
    train_idx
]

X_test = X.iloc[
    test_idx
]

y_train = y.iloc[
    train_idx
]

y_test = y.iloc[
    test_idx
]


print("\n" + "=" * 70)
print("DATA SPLIT")
print("=" * 70)

print(
    "Training candidates:",
    len(X_train)
)

print(
    "Testing candidates:",
    len(X_test)
)

print(
    "Training images:",
    df.iloc[
        train_idx
    ]["image"].nunique()
)

print(
    "Testing images:",
    df.iloc[
        test_idx
    ]["image"].nunique()
)


# ============================================================
# RANDOM FOREST
# ============================================================

model = RandomForestClassifier(

    n_estimators=300,

    max_depth=18,

    min_samples_leaf=3,

    class_weight="balanced",

    n_jobs=-1,

    random_state=42
)


# ============================================================
# TRAIN
# ============================================================

print("\nTraining Random Forest...")

model.fit(
    X_train,
    y_train
)

print("Training completed.")


# ============================================================
# PREDICT
# ============================================================

y_pred = model.predict(
    X_test
)

y_probability = (
    model.predict_proba(
        X_test
    )[:, 1]
)


# ============================================================
# METRICS
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0
)

auc = roc_auc_score(
    y_test,
    y_probability
)


print("\n" + "=" * 70)
print("MODEL RESULTS")
print("=" * 70)

print(
    f"Accuracy : {accuracy:.4f}"
)

print(
    f"Precision: {precision:.4f}"
)

print(
    f"Recall   : {recall:.4f}"
)

print(
    f"F1 Score : {f1:.4f}"
)

print(
    f"ROC-AUC  : {auc:.4f}"
)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=[
            "NON_ROOF",
            "ROOF"
        ],
        zero_division=0
    )
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

print("\nConfusion Matrix:")

print(
    confusion_matrix(
        y_test,
        y_pred
    )
)


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

importance = pd.DataFrame({

    "feature": FEATURES,

    "importance":
        model.feature_importances_

})


importance = (
    importance
    .sort_values(
        "importance",
        ascending=False
    )
)


print("\n" + "=" * 70)
print("FEATURE IMPORTANCE")
print("=" * 70)

print(
    importance.to_string(
        index=False
    )
)


# ============================================================
# SAVE FEATURE IMPORTANCE
# ============================================================

importance_path = os.path.join(
    OUTPUT_DIR,
    "feature_importance.csv"
)

importance.to_csv(
    importance_path,
    index=False
)


# ============================================================
# SAVE MODEL
# ============================================================

model_path = os.path.join(
    OUTPUT_DIR,
    "roof_candidate_random_forest.pkl"
)

joblib.dump(
    model,
    model_path
)


# ============================================================
# SAVE TEST PREDICTIONS
# ============================================================

test_results = df.iloc[
    test_idx
].copy()


test_results[
    "predicted_label"
] = np.where(
    y_pred == 1,
    "ROOF",
    "NON_ROOF"
)


test_results[
    "roof_probability"
] = y_probability


test_path = os.path.join(
    OUTPUT_DIR,
    "test_predictions.csv"
)

test_results.to_csv(
    test_path,
    index=False
)


# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 70)
print("STEP 11 COMPLETED")
print("=" * 70)

print(
    "Features used:",
    FEATURES
)

print(
    "Model saved:",
    model_path
)

print(
    "Feature importance:",
    importance_path
)

print(
    "Test predictions:",
    test_path
)