import os

import numpy as np
import pandas as pd


INPUT_PATH = "data/processed/keypoints.csv"
OUTPUT_PATH = "data/processed/keypoints_geometric.csv"


# ============================================================
# Distance between two landmarks
# ============================================================

def distance(p1, p2):

    return np.linalg.norm(p1 - p2)


# ============================================================
# Angle ABC
# B = middle landmark
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
# Create 28 geometric features
# ============================================================

def create_geometric_features(row):

    # --------------------------------------------------------
    # 42 values = 21 landmarks × (x, y)
    # --------------------------------------------------------

    values = row.astype(float).values

    landmarks = values.reshape(21, 2)

    features = []

    # ========================================================
    # 1. Thumb tip → fingertip distances
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
    # 2. Fingertip-to-fingertip distances
    # ========================================================

    fingertip_pairs = [
        (8, 12),
        (12, 16),
        (16, 20),
        (8, 16),
        (8, 20),
        (12, 20),
    ]

    for a, b in fingertip_pairs:

        features.append(
            distance(
                landmarks[a],
                landmarks[b]
            )
        )

    # ========================================================
    # 3. Thumb joint → finger base distances
    # ========================================================

    thumb_points = [
        2,
        3
    ]

    finger_points = [
        6,   # Index
        10,  # Middle
        14,  # Ring
        18   # Pinky
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

    return features


# ============================================================
# Main
# ============================================================

def main():

    print(
        "\n========== Creating Geometric Dataset ==========\n"
    )

    # --------------------------------------------------------
    # Load original dataset
    # --------------------------------------------------------

    dataframe = pd.read_csv(
        INPUT_PATH,
        header=None
    )

    print(
        f"Original dataset shape: "
        f"{dataframe.shape}"
    )

    # --------------------------------------------------------
    # Separate labels and 42 features
    # --------------------------------------------------------

    labels = dataframe.iloc[:, 0]

    original_features = dataframe.iloc[:, 1:]

    # --------------------------------------------------------
    # Verify feature count
    # --------------------------------------------------------

    if original_features.shape[1] != 42:

        raise ValueError(
            "Expected 42 features per sample, "
            f"but found {original_features.shape[1]}"
        )

    # --------------------------------------------------------
    # Create geometric features
    # --------------------------------------------------------

    print(
        "Generating geometric features..."
    )

    geometric_features = np.array([

        create_geometric_features(row)

        for _, row in original_features.iterrows()

    ])

    print(
        f"Geometric features generated: "
        f"{geometric_features.shape[1]}"
    )

    # --------------------------------------------------------
    # Combine original + geometric
    # --------------------------------------------------------

    combined_features = np.hstack([

        original_features.values.astype(
            np.float32
        ),

        geometric_features.astype(
            np.float32
        )

    ])

    print(
        f"Combined feature count: "
        f"{combined_features.shape[1]}"
    )

    # --------------------------------------------------------
    # Create final dataframe
    # --------------------------------------------------------

    output_dataframe = pd.DataFrame(
        np.column_stack([
            labels.values,
            combined_features
        ])
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    os.makedirs(
        os.path.dirname(OUTPUT_PATH),
        exist_ok=True
    )

    output_dataframe.to_csv(
        OUTPUT_PATH,
        header=False,
        index=False
    )

    # --------------------------------------------------------
    # Verify
    # --------------------------------------------------------

    print(
        f"\nSaved geometric dataset to:"
        f"\n{OUTPUT_PATH}"
    )

    print(
        f"\nNew dataset shape:"
        f"\n{output_dataframe.shape}"
    )

    print(
        "\nExpected:"
        "\nFeatures per sample = 70"
    )

    print(
        "\n================================================\n"
    )


if __name__ == "__main__":
    main()