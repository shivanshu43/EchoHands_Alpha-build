import numpy as np

from src.core.dynamic_predictor import DynamicPredictor


DATASET_PATH = (
    "data/processed/"
    "dynamic/"
    "dynamic_dataset.npz"
)


def main():

    predictor = DynamicPredictor()

    print(
        "\n========== Dynamic Predictor Test ==========\n"
    )

    # ==========================================
    # Load processed 70-feature dataset
    # ==========================================

    data = np.load(
        DATASET_PATH,
        allow_pickle=True
    )

    X = data["X"]
    y = data["y"]

    print(
        "Dataset shape:",
        X.shape
    )

    print(
        "Labels shape:",
        y.shape
    )

    # ==========================================
    # Find one J sequence
    # ==========================================

    j_indices = np.where(y == "J")[0]

    if len(j_indices) == 0:

        raise ValueError(
            "No J sequences found."
        )

    j_sequence = X[j_indices[0]]

    print(
        "\nJ sequence shape:",
        j_sequence.shape
    )

    predicted_class, confidence = (
        predictor.predict(
            j_sequence
        )
    )

    print(
        f"J Prediction: "
        f"{predicted_class}"
    )

    print(
        f"J Confidence: "
        f"{confidence * 100:.2f}%"
    )

    # ==========================================
    # Find one Z sequence
    # ==========================================

    z_indices = np.where(y == "Z")[0]

    if len(z_indices) == 0:

        raise ValueError(
            "No Z sequences found."
        )

    z_sequence = X[z_indices[0]]

    print(
        "\nZ sequence shape:",
        z_sequence.shape
    )

    predicted_class, confidence = (
        predictor.predict(
            z_sequence
        )
    )

    print(
        f"Z Prediction: "
        f"{predicted_class}"
    )

    print(
        f"Z Confidence: "
        f"{confidence * 100:.2f}%"
    )

    print(
        "\n============================================\n"
    )


if __name__ == "__main__":

    main()