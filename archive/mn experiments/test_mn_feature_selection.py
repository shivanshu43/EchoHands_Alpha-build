import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


CSV_PATH = "data/processed/keypoints_geometric.csv"


def main():

    print("\n========== M/N Feature Selection Experiment ==========\n")

    # ==========================================
    # Load dataset
    # ==========================================

    df = pd.read_csv(
        CSV_PATH,
        header=None
    )

    df = df[df[0].isin(["M", "N"])]

    X = df.iloc[:, 1:].astype(float).values
    y = (df[0].values == "N").astype(int)

    print("M samples:", np.sum(y == 0))
    print("N samples:", np.sum(y == 1))
    print("Total features:", X.shape[1])

    # ==========================================
    # Train/Test split
    # ==========================================

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    # ==========================================
    # Feature groups
    # ==========================================

    feature_groups = {

        "Original 42":
            list(range(42)),

        "Top 5":
            [65, 48, 63, 51, 67],

        "Top 10":
            [65, 48, 63, 51, 67,
             45, 44, 68, 58, 61],

        "Top 15":
            [65, 48, 63, 51, 67,
             45, 44, 68, 58, 61,
             43, 33, 59, 60, 57],

        "Top 20":
            [65, 48, 63, 51, 67,
             45, 44, 68, 58, 61,
             43, 33, 59, 60, 57,
             50, 62, 25, 69, 35],

        "All 70":
            list(range(70))
    }

    # ==========================================
    # Run experiments
    # ==========================================

    results = []

    for name, indices in feature_groups.items():

        print("\n" + "=" * 50)
        print(name)
        print("=" * 50)

        X_train_selected = X_train[:, indices]
        X_test_selected = X_test[:, indices]

        model = RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            n_jobs=-1
        )

        model.fit(
            X_train_selected,
            y_train
        )

        predictions = model.predict(
            X_test_selected
        )

        accuracy = accuracy_score(
            y_test,
            predictions
        )

        print(
            f"Features used : {len(indices)}"
        )

        print(
            f"Accuracy      : "
            f"{accuracy * 100:.2f}%"
        )

        print(
            classification_report(
                y_test,
                predictions,
                target_names=["M", "N"]
            )
        )

        results.append(
            (
                name,
                len(indices),
                accuracy
            )
        )

    # ==========================================
    # Comparison
    # ==========================================

    print("\n\n========== FINAL COMPARISON ==========\n")

    for name, count, accuracy in results:

        print(
            f"{name:12} | "
            f"{count:2} features | "
            f"{accuracy * 100:.2f}%"
        )

    print(
        "\n=====================================\n"
    )


if __name__ == "__main__":
    main()