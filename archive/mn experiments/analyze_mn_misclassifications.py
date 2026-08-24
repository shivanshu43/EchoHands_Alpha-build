import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix


CSV_PATH = "data/processed/keypoints_geometric.csv"


def main():

    print("\n========== M/N Misclassification Analysis ==========\n")

    # ========================================================
    # Load M/N data
    # ========================================================

    df = pd.read_csv(
        CSV_PATH,
        header=None
    )

    df = df[
        df[0].isin(["M", "N"])
    ].reset_index(drop=True)

    labels = df.iloc[:, 0]

    X = df.iloc[:, 1:].astype(float).values

    y = (
        labels == "N"
    ).astype(int).values

    # ========================================================
    # Train/test split
    # ========================================================

    (
        X_train,
        X_test,
        y_train,
        y_test,
        index_train,
        index_test
    ) = train_test_split(
        X,
        y,
        np.arange(len(y)),
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    # ========================================================
    # Train model
    # ========================================================

    model = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        n_jobs=-1
    )

    model.fit(
        X_train,
        y_train
    )

    predictions = model.predict(
        X_test
    )

    probabilities = model.predict_proba(
        X_test
    )

    # ========================================================
    # Confusion matrix
    # ========================================================

    print(
        "========== Confusion Matrix ==========\n"
    )

    print(
        confusion_matrix(
            y_test,
            predictions
        )
    )

    # ========================================================
    # Find misclassified samples
    # ========================================================

    wrong = (
        predictions != y_test
    )

    wrong_indices = index_test[wrong]

    wrong_actual = y_test[wrong]

    wrong_predicted = predictions[wrong]

    wrong_confidence = (
        np.max(
            probabilities[wrong],
            axis=1
        )
    )

    # ========================================================
    # Summary
    # ========================================================

    print(
        "\n========== Misclassification Summary ==========\n"
    )

    print(
        f"Total test samples : {len(y_test)}"
    )

    print(
        f"Correct            : {np.sum(~wrong)}"
    )

    print(
        f"Incorrect          : {np.sum(wrong)}"
    )

    print(
        f"M → N              : "
        f"{np.sum((y_test == 0) & (predictions == 1))}"
    )

    print(
        f"N → M              : "
        f"{np.sum((y_test == 1) & (predictions == 0))}"
    )

    # ========================================================
    # Confidence of wrong predictions
    # ========================================================

    print(
        "\n========== Wrong Prediction Confidence ==========\n"
    )

    if len(wrong_confidence) > 0:

        print(
            f"Average confidence : "
            f"{np.mean(wrong_confidence):.4f}"
        )

        print(
            f"Minimum confidence : "
            f"{np.min(wrong_confidence):.4f}"
        )

        print(
            f"Maximum confidence : "
            f"{np.max(wrong_confidence):.4f}"
        )

    # ========================================================
    # Individual errors
    # ========================================================

    print(
        "\n========== Individual Misclassifications ==========\n"
    )

    print(
        "Index | Actual | Predicted | Confidence"
    )

    print(
        "-" * 45
    )

    for i in range(
        len(wrong_indices)
    ):

        actual = (
            "M"
            if wrong_actual[i] == 0
            else "N"
        )

        predicted = (
            "M"
            if wrong_predicted[i] == 0
            else "N"
        )

        print(
            f"{wrong_indices[i]:5d} | "
            f"{actual:6s} | "
            f"{predicted:9s} | "
            f"{wrong_confidence[i]:.4f}"
        )

    print(
        "\n===============================================\n"
    )


if __name__ == "__main__":
    main()