import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


CSV_PATH = "data/processed/keypoints.csv"


# ============================================================
# Convert 42 features back into 21 (x, y) landmarks
# ============================================================

def get_landmarks(row):

    values = row.astype(float).values

    landmarks = values.reshape(21, 2)

    return landmarks


# ============================================================
# Calculate distance between two landmarks
# ============================================================

def distance(p1, p2):

    return np.linalg.norm(p1 - p2)


# ============================================================
# Calculate angle ABC
# B = middle point
# ============================================================

def angle(a, b, c):

    ba = a - b
    bc = c - b

    denominator = (
        np.linalg.norm(ba) *
        np.linalg.norm(bc)
    )

    if denominator == 0:
        return 0.0

    cosine = np.dot(ba, bc) / denominator

    cosine = np.clip(
        cosine,
        -1.0,
        1.0
    )

    return np.arccos(cosine)


# ============================================================
# Create geometric features
# ============================================================

def create_geometric_features(row):

    landmarks = get_landmarks(row)

    # --------------------------------------------------------
    # Landmark indices
    #
    # 0  = Wrist
    #
    # Thumb:
    # 1,2,3,4
    #
    # Index:
    # 5,6,7,8
    #
    # Middle:
    # 9,10,11,12
    #
    # Ring:
    # 13,14,15,16
    #
    # Pinky:
    # 17,18,19,20
    # --------------------------------------------------------

    features = []

    # ========================================================
    # 1. Thumb tip to fingertip distances
    # ========================================================

    thumb_tip = landmarks[4]

    fingertip_indices = [
        8,   # Index
        12,  # Middle
        16,  # Ring
        20   # Pinky
    ]

    for index in fingertip_indices:

        features.append(
            distance(
                thumb_tip,
                landmarks[index]
            )
        )

    # ========================================================
    # 2. Distances between neighboring fingertips
    # ========================================================

    fingertip_pairs = [
        (8, 12),    # Index - Middle
        (12, 16),   # Middle - Ring
        (16, 20),   # Ring - Pinky
        (8, 16),    # Index - Ring
        (8, 20),    # Index - Pinky
        (12, 20),   # Middle - Pinky
    ]

    for a, b in fingertip_pairs:

        features.append(
            distance(
                landmarks[a],
                landmarks[b]
            )
        )

    # ========================================================
    # 3. Thumb relationship with important finger joints
    # ========================================================

    thumb_points = [2, 3]

    finger_points = [
        6,   # Index MCP
        10,  # Middle MCP
        14,  # Ring MCP
        18   # Pinky MCP
    ]

    for thumb_index in thumb_points:

        for finger_index in finger_points:

            features.append(
                distance(
                    landmarks[thumb_index],
                    landmarks[finger_index]
                )
            )

    # ========================================================
    # 4. Finger joint angles
    # ========================================================

    angle_triplets = [

        # Index
        (5, 6, 7),
        (6, 7, 8),

        # Middle
        (9, 10, 11),
        (10, 11, 12),

        # Ring
        (13, 14, 15),
        (14, 15, 16),

        # Pinky
        (17, 18, 19),
        (18, 19, 20),

        # Thumb
        (1, 2, 3),
        (2, 3, 4),
    ]

    for a, b, c in angle_triplets:

        features.append(
            angle(
                landmarks[a],
                landmarks[b],
                landmarks[c]
            )
        )

    return np.array(features)


# ============================================================
# Main experiment
# ============================================================

def main():

    print("\n========== M/N Geometric Feature Experiment ==========\n")

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    dataframe = pd.read_csv(
        CSV_PATH,
        header=None
    )

    # Only M and N
    dataframe = dataframe[
        dataframe[0].isin(["M", "N"])
    ]

    X_raw = dataframe.iloc[:, 1:]
    y = dataframe.iloc[:, 0]

    print(
        f"M samples: {sum(y == 'M')}"
    )

    print(
        f"N samples: {sum(y == 'N')}"
    )

    print(
        f"Original features: {X_raw.shape[1]}"
    )

    # --------------------------------------------------------
    # Create geometric features
    # --------------------------------------------------------

    geometric_features = np.array([
        create_geometric_features(
            row
        )
        for _, row in X_raw.iterrows()
    ])

    print(
        f"Geometric features: "
        f"{geometric_features.shape[1]}"
    )

    # --------------------------------------------------------
    # Combine original + geometric
    # --------------------------------------------------------

    X_original = X_raw.values.astype(
        np.float32
    )

    X_combined = np.hstack([
        X_original,
        geometric_features
    ])

    print(
        f"Combined features: "
        f"{X_combined.shape[1]}"
    )

    # --------------------------------------------------------
    # Encode labels
    # M = 0
    # N = 1
    # --------------------------------------------------------

    y_encoded = (
        y == "N"
    ).astype(int).values

    # --------------------------------------------------------
    # Same train/test split
    # --------------------------------------------------------

    (
        X_original_train,
        X_original_test,
        y_train,
        y_test
    ) = train_test_split(

        X_original,
        y_encoded,

        test_size=0.20,

        random_state=42,

        stratify=y_encoded
    )

    (
        X_combined_train,
        X_combined_test,
        _,
        _
    ) = train_test_split(

        X_combined,
        y_encoded,

        test_size=0.20,

        random_state=42,

        stratify=y_encoded
    )

    # ========================================================
    # MODEL 1
    # Original 42 features
    # ========================================================

    print(
        "\n========== BASELINE ==========\n"
    )

    baseline_model = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        n_jobs=-1
    )

    baseline_model.fit(
        X_original_train,
        y_train
    )

    baseline_predictions = (
        baseline_model.predict(
            X_original_test
        )
    )

    baseline_accuracy = accuracy_score(
        y_test,
        baseline_predictions
    )

    print(
        f"Accuracy: "
        f"{baseline_accuracy * 100:.2f}%"
    )

    print(
        classification_report(
            y_test,
            baseline_predictions,
            target_names=["M", "N"]
        )
    )

    # ========================================================
    # MODEL 2
    # Original + geometric features
    # ========================================================

    print(
        "\n========== GEOMETRIC FEATURES ==========\n"
    )

    geometric_model = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        n_jobs=-1
    )

    geometric_model.fit(
        X_combined_train,
        y_train
    )

    geometric_predictions = (
        geometric_model.predict(
            X_combined_test
        )
    )

    geometric_accuracy = accuracy_score(
        y_test,
        geometric_predictions
    )

    print(
        f"Accuracy: "
        f"{geometric_accuracy * 100:.2f}%"
    )

    print(
        classification_report(
            y_test,
            geometric_predictions,
            target_names=["M", "N"]
        )
    )

    # ========================================================
    # Comparison
    # ========================================================

    print(
        "\n========== COMPARISON ==========\n"
    )

    print(
        f"Original 42 features : "
        f"{baseline_accuracy * 100:.2f}%"
    )

    print(
        f"With geometric data  : "
        f"{geometric_accuracy * 100:.2f}%"
    )

    improvement = (
        geometric_accuracy -
        baseline_accuracy
    ) * 100

    print(
        f"Improvement           : "
        f"{improvement:+.2f}%"
    )

    print(
        "\n============================================\n"
    )


if __name__ == "__main__":
    main()