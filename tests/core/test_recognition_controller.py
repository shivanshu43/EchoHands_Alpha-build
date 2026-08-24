import cv2

from src.core.camera import Camera
from src.core.hand_detector import HandDetector
from src.core.landmark_processor import LandmarkProcessor
from src.core.predictor import Predictor
from src.core.dynamic_predictor import DynamicPredictor
from src.core.recognition_controller import RecognitionController


WINDOW_NAME = "Recognition Controller Test"


def main():

    camera = Camera()
    detector = HandDetector()
    processor = LandmarkProcessor()

    static_predictor = Predictor()
    dynamic_predictor = DynamicPredictor()

    controller = RecognitionController(
        static_predictor,
        dynamic_predictor
    )

    last_prediction = "None"
    last_confidence = 0.0

    try:

        camera.start()

        print("\n========== Recognition Controller Test ==========\n")
        print("Test static signs normally.")
        print("Then perform J or Z naturally.")
        print("Hold each completed gesture until NONE appears.")
        print("Press Q to quit.\n")

        while True:

            # ==========================================
            # Capture frame
            # ==========================================

            frame = camera.get_frame()

            if frame is None:
                break

            # ==========================================
            # Detect hand
            # ==========================================

            results = detector.detect(frame)

            features = processor.extract_features(
                results
            )

            # ==========================================
            # Recognition controller
            # ==========================================

            result = controller.update(
                features
            )

            prediction = result["prediction"]
            confidence = result["confidence"]
            mode = result["mode"]
            sequence_complete = result["sequence_complete"]

            # ==========================================
            # Update displayed prediction
            # ==========================================

            if features is None:

                last_prediction = "No Hand Detected"
                last_confidence = 0.0

            elif prediction is not None:

                last_prediction = prediction
                last_confidence = confidence

            # ==========================================
            # Draw landmarks
            # ==========================================

            frame = detector.draw(
                frame,
                results
            )

            # ==========================================
            # Mode
            # ==========================================

            cv2.putText(
                frame,
                f"Mode: {mode}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 0),
                2
            )

            # ==========================================
            # Prediction
            # ==========================================

            cv2.putText(
                frame,
                f"Prediction: {last_prediction}",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

            # ==========================================
            # Confidence
            # ==========================================

            cv2.putText(
                frame,
                f"Confidence: "
                f"{last_confidence * 100:.1f}%",
                (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2
            )

            # ==========================================
            # Sequence length
            # ==========================================

            cv2.putText(
                frame,
                f"Sequence frames: "
                f"{controller.get_sequence_length()}",
                (20, 160),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 180, 0),
                2
            )

            # ==========================================
            # NONE state
            # ==========================================

            if mode == controller.NONE:

                cv2.putText(
                    frame,
                    "NONE - Waiting for next gesture",
                    (20, 200),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2
                )

            # ==========================================
            # Dynamic gesture detected
            # ==========================================

            if sequence_complete:

                cv2.putText(
                    frame,
                    "Dynamic gesture detected!",
                    (20, 240),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )

            # ==========================================
            # Display
            # ==========================================

            cv2.imshow(
                WINDOW_NAME,
                frame
            )

            # ==========================================
            # Quit
            # ==========================================

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:

        detector.close()
        camera.stop()


if __name__ == "__main__":
    main()