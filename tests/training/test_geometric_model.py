import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)


CSV_PATH = "data/processed/keypoints_geometric.csv"


def main():

    print("\n========== Geometric Model Experiment ==========\n")

    # ========================================================
    # Load dataset
    # ========================================================

    dataframe = pd.read_csv(
        CSV_PATH,
        header=None
    )

    print(
        f"Dataset shape: {dataframe.shape}"
    )

    # ========================================================
    # Separate labels and features
    # ========================================================

    labels = dataframe.iloc[:, 0]

    features = dataframe.iloc[:, 1:]

    print(
        f"Number of classes: "
        f"{labels.nunique()}"
    )

    print(
        f"Number of features: "
        f"{features.shape[1]}"
    )

    # ========================================================
    # Encode labels
    # ========================================================

    encoder = LabelEncoder()

    encoded_labels = encoder.fit_transform(
        labels
    )

    # ========================================================
    # Train/Test split
    #
    # Same random state and 20% test size
    # as your existing Random Forest pipeline.
    # ========================================================

    X_train, X_test, y_train, y_test = (
        train_test_split(
            features,
            encoded_labels,
            test_size=0.20,
            random_state=42,
            stratify=encoded_labels,
        )
    )

    print(
        f"\nTraining samples: "
        f"{len(X_train)}"
    )

    print(
        f"Testing samples: "
        f"{len(X_test)}"
    )

    # ========================================================
    # Train Random Forest
    # ========================================================

    print(
        "\n========== Training ==========\n"
    )

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(
        X_train,
        y_train
    )

    # ========================================================
    # Predictions
    # ========================================================

    predictions = model.predict(
        X_test
    )

    # ========================================================
    # Accuracy
    # ========================================================

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    print(
        "\n========== Accuracy ==========\n"
    )

    print(
        f"{accuracy * 100:.2f}%"
    )

    # ========================================================
    # Classification report
    # ========================================================

    print(
        "\n========== Classification Report ==========\n"
    )

    print(
        classification_report(
            y_test,
            predictions,
            target_names=encoder.classes_,
        )
    )

    # ========================================================
    # Confusion matrix
    # ========================================================

    print(
        "\n========== Confusion Matrix ==========\n"
    )

    print(
        confusion_matrix(
            y_test,
            predictions,
        )
    )

    print(
        "\n============================================\n"
    )


if __name__ == "__main__":
    main()