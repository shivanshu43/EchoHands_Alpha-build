import joblib

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

from src.dataset.preparation.dataset_loader import load_dataset


MODEL_PATH = "models/random_forest.pkl"


def main():

    X_train, X_test, y_train, y_test, encoder = load_dataset()

    model = joblib.load(MODEL_PATH)

    predictions = model.predict(X_test)

    print("\n========== Accuracy ==========\n")

    print(f"{accuracy_score(y_test, predictions) * 100:.2f}%")

    print("\n========== Classification Report ==========\n")

    print(
        classification_report(
            y_test,
            predictions,
            target_names=encoder.classes_,
        )
    )

    print("\n========== Confusion Matrix ==========\n")

    print(
        confusion_matrix(
            y_test,
            predictions,
        )
    )


if __name__ == "__main__":
    main()