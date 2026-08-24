import os
import numpy as np


BASE_DIR = "data/processed/dynamic_sequences"

HANDS = ["LEFT", "RIGHT"]
LABELS = ["J", "Z"]


def validate_sequences():

    grand_total = 0
    grand_valid = 0

    print("\n========== Dynamic Dataset Validation ==========\n")

    for label in LABELS:

        label_dir = os.path.join(BASE_DIR, label)

        label_total = 0
        label_valid = 0
        frame_counts = []

        print(f"---------- {label} ----------")

        for hand in HANDS:

            hand_dir = os.path.join(label_dir, hand)

            if not os.path.exists(hand_dir):

                print(f"{hand}: folder not found")
                continue

            files = sorted(
                file
                for file in os.listdir(hand_dir)
                if file.endswith(".npz")
            )

            print(f"{hand}: {len(files)} sequences")

            for file in files:

                file_path = os.path.join(
                    hand_dir,
                    file
                )

                label_total += 1

                try:

                    data = np.load(
                        file_path,
                        allow_pickle=True
                    )

                    sequence = data["sequence"]

                    saved_label = str(data["label"])
                    saved_hand = str(data["hand"])

                    # -----------------------------
                    # Shape validation
                    # -----------------------------

                    if sequence.ndim != 2:

                        print(
                            f"INVALID {file}: "
                            f"expected 2D, got {sequence.shape}"
                        )

                        continue

                    if sequence.shape[1] != 42:

                        print(
                            f"INVALID {file}: "
                            f"expected 42 features, "
                            f"got {sequence.shape[1]}"
                        )

                        continue

                    if sequence.shape[0] < 10:

                        print(
                            f"INVALID {file}: "
                            f"only {sequence.shape[0]} frames"
                        )

                        continue

                    # -----------------------------
                    # Metadata validation
                    # -----------------------------

                    if saved_label != label:

                        print(
                            f"INVALID {file}: "
                            f"label = {saved_label}"
                        )

                        continue

                    if saved_hand != hand:

                        print(
                            f"INVALID {file}: "
                            f"hand = {saved_hand}"
                        )

                        continue

                    # -----------------------------
                    # Data validation
                    # -----------------------------

                    if not np.isfinite(sequence).all():

                        print(
                            f"INVALID {file}: "
                            f"contains NaN or infinity"
                        )

                        continue

                    label_valid += 1
                    frame_counts.append(sequence.shape[0])

                except Exception as error:

                    print(
                        f"INVALID {file}: "
                        f"{error}"
                    )

        print(
            f"Total : {label_total}"
        )

        print(
            f"Valid : {label_valid}"
        )

        print(
            f"Invalid : {label_total - label_valid}"
        )

        if frame_counts:

            print(
                f"Minimum frames : {min(frame_counts)}"
            )

            print(
                f"Maximum frames : {max(frame_counts)}"
            )

            print(
                f"Average frames : "
                f"{sum(frame_counts) / len(frame_counts):.2f}"
            )

        print()

        grand_total += label_total
        grand_valid += label_valid

    print("================================================")

    print(
        f"Total sequences : {grand_total}"
    )

    print(
        f"Valid sequences : {grand_valid}"
    )

    print(
        f"Invalid         : "
        f"{grand_total - grand_valid}"
    )

    print("================================================\n")


if __name__ == "__main__":
    validate_sequences()