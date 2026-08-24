import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from src.dataset.preparation.dataset_loader import load_dataset


MODEL_PATH = "models/random_forest.pkl"
ENCODER_PATH = "models/label_encoder.pkl"


def main():

    X_train, X_test, y_train, y_test, encoder = load_dataset()

    print("\nTraining Random Forest...\n")

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    print(f"Accuracy : {accuracy * 100:.2f}%")

    joblib.dump(model, MODEL_PATH)
    joblib.dump(encoder, ENCODER_PATH)

    print("\nModel Saved Successfully!")
    print(f"Model  : {MODEL_PATH}")
    print(f"Encoder: {ENCODER_PATH}")


if __name__ == "__main__":
    main()