import csv
import os
import sys
import time
import cv2

from src.dataset.collection.collector import DatasetCollector
from src.dataset.collection.quality_checker import QualityChecker
from src.dataset.collection.duplicate_detector import DuplicateDetector

from src.core.hand_detector import HandDetector
from src.core.landmark_processor import LandmarkProcessor


CSV_PATH = "data/processed/keypoints.csv"

TARGET_NEW_SAMPLES = 100

SAMPLES_PER_HAND = 50

SAMPLES_PER_VARIATION = 10

VARIATIONS = [
    "Canonical Pose",
    "Slight Wrist Rotation Left",
    "Slight Wrist Rotation Right",
    "Natural Finger Configuration",
    "Slight Palm Tilt",
]

CAPTURE_INTERVAL = 0.25


def get_existing_samples(csv_path, label):

    if not os.path.exists(csv_path):
        return 0

    count = 0

    with open(csv_path, "r", newline="") as file:

        reader = csv.reader(file)

        for row in reader:

            if len(row) > 0 and row[0] == label:
                count += 1

    return count


def get_hand_label(results):

    """
    Returns LEFT or RIGHT according to MediaPipe's
    detected handedness.
    """

    if (
        results is None
        or not results.multi_hand_landmarks
        or not results.multi_handedness
    ):
        return None

    classification = (
        results.multi_handedness[0].classification[0]
    )

    return classification.label.upper()


def collect_label(label):

    existing = get_existing_samples(
        CSV_PATH,
        label
    )

    print()
    print("=" * 60)
    print(f"Targeted Augmentation: {label}")
    print("=" * 60)
    print(f"Existing samples : {existing}")
    print("New samples      : 100")
    print("  LEFT           : 50")
    print("  RIGHT          : 50")
    print(f"Final total      : {existing + 100}")
    print("=" * 60)
    print()

    collector = DatasetCollector()
    detector = HandDetector()
    processor = LandmarkProcessor()
    quality_checker = QualityChecker()

    # Separate duplicate detectors for each hand
    # so the first sample of one hand does not
    # interfere with the other hand.
    duplicate_detectors = {
        "LEFT": DuplicateDetector(),
        "RIGHT": DuplicateDetector()
    }

    current_hand = "LEFT"

    hand_samples = {
        "LEFT": 0,
        "RIGHT": 0
    }

    variation_index = 0
    variation_count = 0

    paused = True

    pause_message = (
        f"Show LEFT hand - "
        f"{VARIATIONS[variation_index]}"
    )

    last_capture_time = time.time()

    os.makedirs(
        os.path.dirname(CSV_PATH),
        exist_ok=True
    )

    csv_file = open(
        CSV_PATH,
        "a",
        newline=""
    )

    writer = csv.writer(csv_file)

    try:

        while True:

            frame = collector.get_frame()

            if frame is None:

                print("\nFailed to capture frame.")
                break

            results = detector.detect(frame)

            features = None

            detected_hand = get_hand_label(results)

            # ======================================
            # Quality Check
            # ======================================

            if quality_checker.is_valid(results):

                # ==================================
                # Hand Check
                # ==================================

                if detected_hand == current_hand:

                    features = processor.extract_features(
                        results
                    )

                    # ==============================
                    # Duplicate Check
                    # ==============================

                    if duplicate_detectors[
                        current_hand
                    ].is_duplicate(features):

                        features = None

                else:

                    features = None

            current_time = time.time()

            # ======================================
            # Save Sample
            # ======================================

            if (
                not paused
                and features is not None
                and current_time - last_capture_time
                >= CAPTURE_INTERVAL
            ):

                row = [label] + features

                writer.writerow(row)

                csv_file.flush()

                hand_samples[current_hand] += 1

                variation_count += 1

                last_capture_time = current_time

                total_new = (
                    hand_samples["LEFT"]
                    + hand_samples["RIGHT"]
                )

                print(
                    f"\r{label}: "
                    f"LEFT {hand_samples['LEFT']}/50 | "
                    f"RIGHT {hand_samples['RIGHT']}/50",
                    end=""
                )

                # ==================================
                # Move to next variation
                # ==================================

                if (
                    variation_count
                    >= SAMPLES_PER_VARIATION
                ):

                    variation_count = 0
                    variation_index += 1

                    if (
                        variation_index
                        >= len(VARIATIONS)
                    ):

                        variation_index = (
                            len(VARIATIONS) - 1
                        )

                    paused = True

                    pause_message = (
                        f"Next: {current_hand} - "
                        f"{VARIATIONS[variation_index]}"
                    )

                # ==================================
                # LEFT → RIGHT
                # ==================================

                if (
                    current_hand == "LEFT"
                    and hand_samples["LEFT"]
                    >= SAMPLES_PER_HAND
                ):

                    current_hand = "RIGHT"

                    variation_index = 0
                    variation_count = 0

                    paused = True

                    pause_message = (
                        f"Switch to RIGHT hand - "
                        f"{VARIATIONS[variation_index]}"
                    )

                # ==================================
                # Finished
                # ==================================

                if (
                    hand_samples["LEFT"]
                    >= SAMPLES_PER_HAND
                    and
                    hand_samples["RIGHT"]
                    >= SAMPLES_PER_HAND
                ):

                    print()
                    print()
                    print(
                        f"Finished targeted "
                        f"augmentation for {label}."
                    )

                    break

            # ======================================
            # Draw landmarks
            # ======================================

            frame = detector.draw(
                frame,
                results
            )

            # ======================================
            # Dashboard
            # ======================================

            cv2.putText(
                frame,
                f"Label: {label}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                f"LEFT: "
                f"{hand_samples['LEFT']}/50",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                f"RIGHT: "
                f"{hand_samples['RIGHT']}/50",
                (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                f"Required: {current_hand}",
                (20, 160),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (255, 180, 0),
                2
            )

            cv2.putText(
                frame,
                f"Variation: "
                f"{VARIATIONS[variation_index]}",
                (20, 200),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 0),
                2
            )

            if detected_hand is not None:

                cv2.putText(
                    frame,
                    f"Detected: {detected_hand}",
                    (20, 240),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (255, 255, 255),
                    2
                )

            if paused:

                cv2.putText(
                    frame,
                    pause_message,
                    (20, 290),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 255, 255),
                    2
                )

                cv2.putText(
                    frame,
                    "Press SPACE to start",
                    (20, 330),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 0, 255),
                    2
                )

            cv2.imshow(
                "Targeted Static Augmentation",
                frame
            )

            key = cv2.waitKey(1) & 0xFF

            # ======================================
            # Quit
            # ======================================

            if key == ord("q"):
                break

            # ======================================
            # Resume
            # ======================================

            if paused and key == ord(" "):

                paused = False

                pause_message = ""

    finally:

        csv_file.close()

        detector.close()

        collector.stop()

        cv2.destroyAllWindows()


def main():

    if len(sys.argv) != 2:

        print(
            
            "Usage: python -m src.dataset.preparation.collect_targeted_augmentation <label>"
            
        )

        return

    label = sys.argv[1].upper()

    if len(label) != 1:

        print("Label must be a single character.")

        return

    collect_label(label)


if __name__ == "__main__":
    main()
