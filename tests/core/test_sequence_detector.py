import cv2

from src.core.camera import Camera
from src.core.hand_detector import HandDetector
from src.core.landmark_processor import LandmarkProcessor
from src.core.sequence_detector import SequenceDetector


WINDOW_NAME = "Sequence Detector Test"


def main():

    camera = Camera()
    detector = HandDetector()
    processor = LandmarkProcessor()

    sequence_detector = SequenceDetector()

    try:

        camera.start()

        print("\n========== Sequence Detector Test ==========\n")

        print("Perform J or Z naturally.")
        print("The system should automatically detect:")
        print("START → RECORDING → STOP")
        print("\nPress Q to quit.\n")

        while True:

            frame = camera.get_frame()

            if frame is None:

                break

            results = detector.detect(
                frame
            )

            features = processor.extract_features(
                results
            )

            completed_sequence = (
                sequence_detector.update(
                    features
                )
            )

            # ==================================
            # Display state
            # ==================================

            state = sequence_detector.get_state()

            sequence_length = (
                sequence_detector.get_sequence_length()
            )

            motion = 0.0

            if features is not None:

                motion = (
                    sequence_detector.get_motion(
                        features
                    )
                )

            frame = detector.draw(
                frame,
                results
            )

            cv2.putText(
                frame,
                f"State: {state}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                f"Frames: {sequence_length}",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 0),
                2
            )

            cv2.putText(
                frame,
                f"Motion: {motion:.5f}",
                (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2
            )

            cv2.imshow(
                WINDOW_NAME,
                frame
            )

            if completed_sequence is not None:

                print(
                    "\nGesture detected!"
                )

                print(
                    f"Frames captured: "
                    f"{len(completed_sequence)}"
                )

            if (
                cv2.waitKey(1) & 0xFF
                == ord("q")
            ):

                break

    finally:

        detector.close()
        camera.stop()


if __name__ == "__main__":
    main()