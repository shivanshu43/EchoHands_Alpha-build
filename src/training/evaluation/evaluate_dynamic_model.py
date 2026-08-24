import os

import numpy as np
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)


MODEL_PATH = "models/dynamic_lstm.keras"

DATA_DIR = "data/processed/dynamic/split"

ENCODER_PATH = "models/dynamic_label_encoder.npy"


def main():

    print("\n========== Loading Model ==========\n")

    model = tf.keras.models.load_model(
        MODEL_PATH
    )

    classes = np.load(
        ENCODER_PATH,
        allow_pickle=True
    )

    print("Classes:", classes)

    # ==========================================
    # Load test data
    # ==========================================

    X_test = np.load(
        os.path.join(
            DATA_DIR,
            "X_test.npy"
        )
    )

    y_test = np.load(
        os.path.join(
            DATA_DIR,
            "y_test.npy"
        )
    )

    print(
        "Test data:",
        X_test.shape
    )

    # ==========================================
    # Encode test labels
    # ==========================================

    class_to_index = {
        label: index
        for index, label in enumerate(classes)
    }

    y_test_encoded = np.array(
        [
            class_to_index[label]
            for label in y_test
        ]
    )

    # ==========================================
    # Prediction
    # ==========================================

    predictions = model.predict(
        X_test,
        verbose=0
    )

    predicted_indices = np.argmax(
        predictions,
        axis=1
    )

    # ==========================================
    # Accuracy
    # ==========================================

    accuracy = accuracy_score(
        y_test_encoded,
        predicted_indices
    )

    print(
        "\n========== Accuracy ==========\n"
    )

    print(
        f"{accuracy * 100:.2f}%"
    )

    # ==========================================
    # Classification report
    # ==========================================

    print(
        "\n========== Classification Report ==========\n"
    )

    print(
        classification_report(
            y_test_encoded,
            predicted_indices,
            target_names=classes
        )
    )

    # ==========================================
    # Confusion matrix
    # ==========================================

    print(
        "\n========== Confusion Matrix ==========\n"
    )

    print(
        confusion_matrix(
            y_test_encoded,
            predicted_indices
        )
    )

    # ==========================================
    # Individual predictions
    # ==========================================

    print(
        "\n========== Individual Predictions ==========\n"
    )

    for index in range(len(y_test)):

        actual = y_test[index]

        predicted = classes[
            predicted_indices[index]
        ]

        confidence = predictions[index][
            predicted_indices[index]
        ]

        print(
            f"{index + 1:02d}. "
            f"Actual: {actual} | "
            f"Predicted: {predicted} | "
            f"Confidence: "
            f"{confidence * 100:.2f}%"
        )

    print(
        "\n=============================================\n"
    )


if __name__ == "__main__":
    main()