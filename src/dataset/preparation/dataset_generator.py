import csv
import os
import sys
import time
import cv2

from src.dataset.collection.collector import DatasetCollector
from src.dataset.collection.quality_checker import QualityChecker
from src.dataset.collection.duplicate_detector import DuplicateDetector
from src.dataset.collection.variation_manager import VariationManager

from src.core.hand_detector import HandDetector
from src.core.landmark_processor import LandmarkProcessor

CSV_PATH = "data/processed/keypoints_geometric.csv"


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




def main():

    if len(sys.argv) != 2:
        print("Usage: python -m src.training.dataset_generator <label>")
        return

    label = sys.argv[1].upper()

    TARGET_SAMPLES = 300

    SAMPLES_PER_HAND = TARGET_SAMPLES // 2
    SAMPLES_PER_VARIATION = 15

    current_hand = "LEFT"

    paused = False
    pause_message = ""

    sample_count = get_existing_samples(CSV_PATH, label)

    status = "Waiting..."

    CAPTURE_INTERVAL = 0.25

    last_capture_time = time.time()

    os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)

    # ==========================
    # Initialize Modules
    # ==========================

    collector = DatasetCollector()

    detector = HandDetector()

    processor = LandmarkProcessor()

    quality_checker = QualityChecker()

    duplicate_detector = DuplicateDetector()

    variation_manager = VariationManager()

    # ==========================
    # Restore hand & variation
    # when resuming collection
    # ==========================

    if sample_count >= SAMPLES_PER_HAND:

        current_hand = "RIGHT"

        variation_manager.reset()

        right_samples = sample_count - SAMPLES_PER_HAND

        for _ in range(right_samples // SAMPLES_PER_VARIATION):

            variation_manager.next_variation()

    else:

        left_samples = sample_count

        for _ in range(left_samples // SAMPLES_PER_VARIATION):

            variation_manager.next_variation()



    

    csv_file = open(CSV_PATH, "a", newline="")

    writer = csv.writer(csv_file)

    print(f"\nCollecting samples for label '{label}'")
    print(f"Resuming from {sample_count}/{TARGET_SAMPLES}")
    print("Press Q to quit.\n")

    try:

        while True:

            frame = collector.get_frame()

            if frame is None:
                print("Failed to capture frame.")
                break

            # Detect hand
            results = detector.detect(frame)

            # ==========================
            # Quality Check
            # ==========================

            if quality_checker.is_valid(results):

                features = processor.extract_features(results)

                if duplicate_detector.is_duplicate(features):

                    if not paused:
                        status = "Duplicate Sample"

                    features = None

                else:

                    if not paused:
                        status = "Ready"

            else:

                if not paused:
                    status = "No Hand Detected"

                features = None

            current_time = time.time()

            # ==========================
            # Save Sample
            # ==========================

            if (
            not paused
            and features is not None
            and current_time - last_capture_time >= CAPTURE_INTERVAL
        ):

                row = [label] + features

                writer.writerow(row)

                csv_file.flush()

                last_capture_time = current_time

                sample_count += 1

                status = "Sample Saved"

                # ====================================
                # Left Hand
                # ====================================

                if sample_count < SAMPLES_PER_HAND:

                    if (
                        sample_count > 0
                        and sample_count % SAMPLES_PER_VARIATION == 0
                    ):

                        variation_manager.next_variation()

                        paused = True
                        pause_message = "Move to next variation"

                        status = "Waiting..."
                                                            

                # ====================================
                # Switch to Right Hand
                # ====================================

                elif sample_count == SAMPLES_PER_HAND:

                    current_hand = "RIGHT"

                    variation_manager.reset()

                    paused = True
                    pause_message = "Switch to RIGHT Hand"

                    status = "Waiting for RIGHT Hand"

            
                # ====================================
                # Right Hand
                # ====================================

                else:

                    right_samples = sample_count - SAMPLES_PER_HAND

                    if (
                        right_samples > 0
                        and right_samples % SAMPLES_PER_VARIATION == 0
                    ):

                        variation_manager.next_variation()

                        paused = True
                        pause_message = "Move to next variation"

                        status = "Waiting..."
                        

                print(
                    f"\rSamples: {sample_count}/{TARGET_SAMPLES}",
                    end=""
                )

                if sample_count >= TARGET_SAMPLES:

                    print(f"\nFinished collecting '{label}'")

                    break

            # ==========================
            # Draw Landmarks
            # ==========================

            frame = detector.draw(frame, results)

            # ==========================
            # Dashboard
            # ==========================

            cv2.putText(
                frame,
                f"Label : {label}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
            )

            cv2.putText(
                frame,
                f"Samples : {sample_count}/{TARGET_SAMPLES}",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )

            cv2.putText(
                frame,
                f"Variation : {variation_manager.get_current_variation()}",
                (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 0),
                2,
            )

            cv2.putText(
                frame,
                f"Hand : {current_hand}",
                (20, 160),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 180, 0),
                2,
            )

            cv2.putText(
                frame,
                f"Status : {status}",
                (20, 200),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
            )

            if paused:

                cv2.putText(
                    frame,
                    pause_message,
                    (20, 240),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    (0, 255, 255),
                    2,
                )

                cv2.putText(
                    frame,
                    "Press SPACE to continue",
                    (20, 280),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    (0, 0, 255),
                    2,
                )

            cv2.imshow("Dataset Generator", frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                    break

            if paused and key == ord(" "):

                    paused = False

                    pause_message = ""

                    status = "Ready"

    finally:

        csv_file.close()

        detector.close()

        collector.stop()


if __name__ == "__main__":
    main()