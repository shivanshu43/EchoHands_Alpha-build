import pandas as pd
import numpy as np


CSV_PATH = "data/processed/keypoints_geometric.csv"


def main():

    print("\n========== M/N Geometry Analysis ==========\n")

    df = pd.read_csv(
        CSV_PATH,
        header=None
    )

    df = df[df[0].isin(["M", "N"])]

    X = df.iloc[:, 1:].astype(float)

    m = X[df[0].values == "M"]
    n = X[df[0].values == "N"]

    print("M samples:", len(m))
    print("N samples:", len(n))
    print("Features :", X.shape[1])

    # ==========================================
    # Compare every feature
    # ==========================================

    results = []

    for i in range(X.shape[1]):

        m_mean = m.iloc[:, i].mean()
        n_mean = n.iloc[:, i].mean()

        m_std = m.iloc[:, i].std()
        n_std = n.iloc[:, i].std()

        pooled_std = np.sqrt(
            (m_std ** 2 + n_std ** 2) / 2
        )

        if pooled_std == 0:
            separation = 0
        else:
            separation = abs(
                m_mean - n_mean
            ) / pooled_std

        results.append(
            (
                i,
                m_mean,
                n_mean,
                abs(m_mean - n_mean),
                separation
            )
        )

    results.sort(
        key=lambda x: x[4],
        reverse=True
    )

    # ==========================================
    # Display
    # ==========================================

    print("\n========== Top 15 Geometric Differences ==========\n")

    print(
        "Feature | M Mean | N Mean | Difference | Separation"
    )

    print("-" * 60)

    for (
        feature,
        m_mean,
        n_mean,
        difference,
        separation
    ) in results[:15]:

        print(
            f"{feature:7} | "
            f"{m_mean:7.3f} | "
            f"{n_mean:7.3f} | "
            f"{difference:10.3f} | "
            f"{separation:10.3f}"
        )

    print(
        "\n===============================================\n"
    )


if __name__ == "__main__":
    main()