import os
import numpy as np

from sklearn.model_selection import train_test_split


INPUT_PATH = "data/processed/dynamic/dynamic_dataset.npz"

OUTPUT_DIR = "data/processed/dynamic/split"

TEST_SIZE = 0.10
VALIDATION_SIZE = 0.10

RANDOM_STATE = 42


def main():

    # ==========================================
    # Load processed dataset
    # ==========================================

    data = np.load(INPUT_PATH)

    X = data["X"]
    y = data["y"]

    print("\n========== Loaded Dataset ==========\n")

    print("X shape :", X.shape)
    print("y shape :", y.shape)

    # ==========================================
    # First split:
    #
    # 90% → train + validation
    # 10% → test
    # ==========================================

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    # ==========================================
    # Second split:
    #
    # From the remaining 90%:
    # 10/90 = 11.11%
    #
    # This gives approximately:
    # 80% train
    # 10% validation
    # ==========================================

    validation_ratio = (
        VALIDATION_SIZE /
        (1 - TEST_SIZE)
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=validation_ratio,
        random_state=RANDOM_STATE,
        stratify=y_train_val
    )

    # ==========================================
    # Create output directory
    # ==========================================

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    # ==========================================
    # Save splits
    # ==========================================

    np.save(
        os.path.join(
            OUTPUT_DIR,
            "X_train.npy"
        ),
        X_train
    )

    np.save(
        os.path.join(
            OUTPUT_DIR,
            "y_train.npy"
        ),
        y_train
    )

    np.save(
        os.path.join(
            OUTPUT_DIR,
            "X_val.npy"
        ),
        X_val
    )

    np.save(
        os.path.join(
            OUTPUT_DIR,
            "y_val.npy"
        ),
        y_val
    )

    np.save(
        os.path.join(
            OUTPUT_DIR,
            "X_test.npy"
        ),
        X_test
    )

    np.save(
        os.path.join(
            OUTPUT_DIR,
            "y_test.npy"
        ),
        y_test
    )

    # ==========================================
    # Display results
    # ==========================================

    print("\n========== Dataset Split ==========\n")

    print(
        f"Training   : {X_train.shape}"
    )

    print(
        f"Validation : {X_val.shape}"
    )

    print(
        f"Testing    : {X_test.shape}"
    )

    print("\n========== Class Distribution ==========\n")

    print(
        f"Training   → "
        f"J: {np.sum(y_train == 'J')} | "
        f"Z: {np.sum(y_train == 'Z')}"
    )

    print(
        f"Validation → "
        f"J: {np.sum(y_val == 'J')} | "
        f"Z: {np.sum(y_val == 'Z')}"
    )

    print(
        f"Testing    → "
        f"J: {np.sum(y_test == 'J')} | "
        f"Z: {np.sum(y_test == 'Z')}"
    )

    print(
        f"\nSaved splits to:\n{OUTPUT_DIR}"
    )

    print(
        "\n====================================\n"
    )


if __name__ == "__main__":
    main()