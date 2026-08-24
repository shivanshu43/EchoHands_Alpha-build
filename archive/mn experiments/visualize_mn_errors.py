import cv2
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


CSV_PATH = "data/processed/keypoints_geometric.csv"


# ============================================================
# Draw one hand from the original 42 landmark features
# ============================================================

def draw_hand(frame, features, title, actual, predicted):

    # First 42 values are the original landmark coordinates
    landmarks = np.array(
        features[:42],
        dtype=np.float32
    ).reshape(21, 2)

    height, width = frame.shape[:2]

    # Convert normalized coordinates to pixels
    points = []

    for x, y in landmarks:

        px = int(
            (x + 1.5) / 3.0 * width
        )

        py = int(
            (y + 1.5) / 3.0 * height
        )

        px = max(0, min(width - 1, px))
        py = max(0, min(height - 1, py))

        points.append((px, py))

    # MediaPipe hand connections
    connections = [

        (0, 1),
        (1, 2),
        (2, 3),
        (3, 4),

        (0, 5),
        (5, 6),
        (6, 7),
        (7, 8),

        (0, 9),
        (9, 10),
        (10, 11),
        (11, 12),

        (0, 13),
        (13, 14),
        (14, 15),
        (15, 16),

        (0, 17),
        (17, 18),
        (18, 19),
        (19, 20),

        (5, 9),
        (9, 13),
        (13, 17),
    ]

    for a, b in connections:

        cv2.line(
            frame,
            points[a],
            points[b],
            (255, 255, 255),
            2
        )

    for i, point in enumerate(points):

        cv2.circle(
            frame,
            point,
            5,
            (0, 255, 0),
            -1
        )

        cv2.putText(
            frame,
            str(i),
            (
                point[0] + 5,
                point[1] - 5
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (255, 255, 255),
            1
        )

    # --------------------------------------------------------
    # Information
    # --------------------------------------------------------

    cv2.putText(
        frame,
        title,
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Actual: {actual}",
        (20, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Predicted: {predicted}",
        (20, 105),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    return frame


# ============================================================
# Main
# ============================================================

def main():

    print(
        "\n========== M/N Visual Diagnostic ==========\n"
    )

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Same split as before
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Train model
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Find incorrect predictions
    # --------------------------------------------------------

    wrong_positions = np.where(
        predictions != y_test
    )[0]

    if len(wrong_positions) == 0:

        print(
            "No misclassified samples found."
        )

        return

    print(
        f"Found {len(wrong_positions)} "
        "misclassified samples."
    )

    print(
        "\nControls:"
    )

    print(
        "N = next error"
    )

    print(
        "P = previous error"
    )

    print(
        "Q = quit"
    )

    print(
        "\n============================================\n"
    )

    current = 0

    while True:

        position = wrong_positions[current]

        original_index = index_test[position]

        actual = (
            "M"
            if y_test[position] == 0
            else "N"
        )

        predicted = (
            "M"
            if predictions[position] == 0
            else "N"
        )

        # ----------------------------------------------------
        # Create blank visualization
        # ----------------------------------------------------

        frame = np.zeros(
            (700, 900, 3),
            dtype=np.uint8
        )

        title = (
            f"Error {current + 1} "
            f"/ {len(wrong_positions)} "
            f"(Dataset row {original_index})"
        )

        frame = draw_hand(
            frame,
            X_test[position],
            title,
            actual,
            predicted
        )

        cv2.imshow(
            "M/N Error Diagnostic",
            frame
        )

        key = (
            cv2.waitKey(0)
            & 0xFF
        )

        # ----------------------------------------------------
        # Next
        # ----------------------------------------------------

        if key == ord("n"):

            current = (
                current + 1
            ) % len(wrong_positions)

        # ----------------------------------------------------
        # Previous
        # ----------------------------------------------------

        elif key == ord("p"):

            current = (
                current - 1
            ) % len(wrong_positions)

        # ----------------------------------------------------
        # Quit
        # ----------------------------------------------------

        elif key == ord("q"):

            break

    cv2.destroyAllWindows()

    print(
        "\nDiagnostic closed.\n"
    )


if __name__ == "__main__":
    main()