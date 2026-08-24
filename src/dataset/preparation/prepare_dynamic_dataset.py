import os
import numpy as np
from src.core.landmark_processor import LandmarkProcessor


BASE_DIR = "data/processed/dynamic_sequences"
OUTPUT_DIR = "data/processed/dynamic"

LABELS = ["J", "Z"]
HANDS = ["LEFT", "RIGHT"]

TARGET_FRAMES = 40


def resample_sequence(sequence, target_frames):

    original_frames = sequence.shape[0]

    if original_frames == target_frames:
        return sequence

    old_indices = np.linspace(
        0,
        original_frames - 1,
        original_frames
    )

    new_indices = np.linspace(
        0,
        original_frames - 1,
        target_frames
    )

    resampled = np.zeros(
        (target_frames, sequence.shape[1]),
        dtype=np.float32
    )

    for feature_index in range(sequence.shape[1]):

        resampled[:, feature_index] = np.interp(
            new_indices,
            old_indices,
            sequence[:, feature_index]
        )

    return resampled


def load_sequences():

    sequences = []
    labels = []

    processor = LandmarkProcessor()

    print("\n========== Loading Dynamic Dataset ==========\n")

    for label in LABELS:

        for hand in HANDS:

            hand_dir = os.path.join(
                BASE_DIR,
                label,
                hand
            )

            files = sorted(
                file
                for file in os.listdir(hand_dir)
                if file.endswith(".npz")
            )

            print(
                f"{label} - {hand}: "
                f"{len(files)} sequences"
            )

            for file in files:

                file_path = os.path.join(
                    hand_dir,
                    file
                )

                data = np.load(
                    file_path,
                    allow_pickle=True
                )

                sequence = data["sequence"]

                if sequence.ndim != 2:
                    raise ValueError(
                        f"Invalid shape in {file}: "
                        f"{sequence.shape}"
                    )

                    if sequence.shape[1] != 42:

                        raise ValueError(
                            f"Invalid feature count in {file}: "
                            f"{sequence.shape[1]}"
                        )

                # ==========================================
                # Convert 42 features → 70 features
                # ==========================================

                geometric_sequence = []

                for frame in sequence:

                    features_70 = (
                        processor.add_geometric_features(
                            frame
                        )
                    )

                    geometric_sequence.append(
                        features_70
                    )

                sequence = np.asarray(
                    geometric_sequence,
                    dtype=np.float32
                )

                # ==========================================
                # Resample to 40 frames
                # ==========================================

                sequence = resample_sequence(
                    sequence,
                    TARGET_FRAMES
                )

                sequences.append(sequence)
                labels.append(label)

    return (
        np.asarray(sequences, dtype=np.float32),
        np.asarray(labels)
    )


def save_dataset(X, y):

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    output_path = os.path.join(
        OUTPUT_DIR,
        "dynamic_dataset.npz"
    )

    np.savez_compressed(
        output_path,
        X=X,
        y=y
    )

    return output_path


def main():

    X, y = load_sequences()

    print("\n========== Processed Dataset ==========\n")

    print("X shape :", X.shape)
    print("y shape :", y.shape)

    print("\nClass distribution:")

    for label in LABELS:

        count = np.sum(y == label)

        print(
            f"{label}: {count}"
        )

    output_path = save_dataset(
        X,
        y
    )

    print(
        f"\nSaved dataset to:"
    )

    print(output_path)

    print(
        "\n========================================\n"
    )


if __name__ == "__main__":
    main()