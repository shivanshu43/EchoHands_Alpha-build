import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score


CSV_PATH = "data/processed/keypoints_geometric.csv"


def main():

    print("\n========== M/N 5-Fold Cross Validation ==========\n")

    # ==========================================
    # Load M/N data
    # ==========================================

    df = pd.read_csv(
        CSV_PATH,
        header=None
    )

    df = df[df[0].isin(["M", "N"])]

    X = df.iloc[:, 1:].astype(float).values

    y = (
        df[0].values == "N"
    ).astype(int)

    print("M samples:", np.sum(y == 0))
    print("N samples:", np.sum(y == 1))
    print("Features :", X.shape[1])

    # ==========================================
    # 5-Fold Stratified Cross Validation
    # ==========================================

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    )

    scores = cross_val_score(
        model,
        X,
        y,
        cv=cv,
        scoring="accuracy"
    )

    # ==========================================
    # Results
    # ==========================================

    print("\n========== Fold Results ==========\n")

    for i, score in enumerate(scores, 1):

        print(
            f"Fold {i}: "
            f"{score * 100:.2f}%"
        )

    print("\n========== Final Result ==========\n")

    print(
        f"Mean Accuracy : "
        f"{scores.mean() * 100:.2f}%"
    )

    print(
        f"Std Deviation : "
        f"{scores.std() * 100:.2f}%"
    )

    print(
        f"Minimum       : "
        f"{scores.min() * 100:.2f}%"
    )

    print(
        f"Maximum       : "
        f"{scores.max() * 100:.2f}%"
    )

    print(
        "\n==================================\n"
    )


if __name__ == "__main__":
    main()