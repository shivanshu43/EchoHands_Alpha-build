import numpy as np
import pandas as pd


CSV_PATH = "data/processed/keypoints_geometric.csv"


def main():

    print("\n========== M/N Feature Analysis ==========\n")

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    df = pd.read_csv(
        CSV_PATH,
        header=None
    )

    df = df[
        df[0].isin(["M", "N"])
    ]

    labels = df.iloc[:, 0]

    X = df.iloc[:, 1:].astype(float)

    m = X[labels == "M"]
    n = X[labels == "N"]

    print(f"M samples: {len(m)}")
    print(f"N samples: {len(n)}")
    print(f"Features: {X.shape[1]}")

    # --------------------------------------------------------
    # Compare means
    # --------------------------------------------------------

    m_mean = m.mean()
    n_mean = n.mean()

    difference = np.abs(
        m_mean - n_mean
    )

    # --------------------------------------------------------
    # Standardized separation
    #
    # Larger value = better separation
    # --------------------------------------------------------

    m_std = m.std()
    n_std = n.std()

    pooled_std = np.sqrt(
        (m_std ** 2 + n_std ** 2) / 2
    )

    pooled_std = pooled_std.replace(
        0,
        1e-8
    )

    separation = (
        difference /
        pooled_std
    )

    # --------------------------------------------------------
    # Sort features
    # --------------------------------------------------------

    ranked = np.argsort(
        separation.values
    )[::-1]

    print(
        "\n========== Top 20 Separating Features ==========\n"
    )

    for rank, index in enumerate(
        ranked[:20],
        start=1
    ):

        print(
            f"{rank:2d}. "
            f"Feature {index:2d} | "
            f"M mean = {m_mean.iloc[index]: .4f} | "
            f"N mean = {n_mean.iloc[index]: .4f} | "
            f"Difference = {difference.iloc[index]: .4f} | "
            f"Separation = {separation.iloc[index]: .4f}"
        )

    # --------------------------------------------------------
    # Compare average feature distributions
    # --------------------------------------------------------

    print(
        "\n========== Most Overlapping Features ==========\n"
    )

    overlap_rank = np.argsort(
        separation.values
    )

    for rank, index in enumerate(
        overlap_rank[:10],
        start=1
    ):

        print(
            f"{rank:2d}. "
            f"Feature {index:2d} | "
            f"M mean = {m_mean.iloc[index]: .4f} | "
            f"N mean = {n_mean.iloc[index]: .4f} | "
            f"Difference = {difference.iloc[index]: .4f} | "
            f"Separation = {separation.iloc[index]: .4f}"
        )

    print(
        "\n============================================\n"
    )


if __name__ == "__main__":
    main()