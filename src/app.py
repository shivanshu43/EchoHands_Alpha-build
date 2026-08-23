import time
import cv2

from src.core.camera import Camera
from src.core.hand_detector import HandDetector
from src.core.landmark_processor import LandmarkProcessor
from src.core.predictor import Predictor
from src.core.dynamic_predictor import DynamicPredictor
from src.core.recognition_controller import RecognitionController
from src.core.word_builder import WordBuilder

from src.utils.config import WINDOW_NAME


def main():

    # ==========================================
    # Initialize components
    # ==========================================

    camera = Camera()

    detector = HandDetector()

    processor = LandmarkProcessor()

    static_predictor = Predictor()

    dynamic_predictor = DynamicPredictor()

    controller = RecognitionController(
        static_predictor,
        dynamic_predictor
    )

    word_builder = WordBuilder()

    # ==========================================
    # Display variables
    # ==========================================

    last_prediction = "None"

    last_confidence = 0.0

    # ==========================================
    # Gesture emission protection
    #
    # Prevents:
    # H -> HH -> HHH -> HHHHH
    # caused by repeated predictions during
    # controller state transitions.
    # ==========================================

    gesture_consumed = False

    previous_mode = controller.NONE

    # ==========================================
    # Space key tracking
    # ==========================================

    last_space_time = 0.0

    double_space_interval = 0.5

    # ==========================================
    # Start camera
    # ==========================================

    camera.start()

    print("\n========== Sign Language Recognition ==========\n")

    print("Static gestures : A-Y + 0-9")
    print("Dynamic gestures: J / Z")
    print("Press SPACE to add a space.")
    print("Press SPACE twice quickly to clear text.")
    print("Press BACKSPACE to remove last character.")
    print("Press 'Q' to exit.\n")

    try:

        while True:

            # ==========================================
            # Get frame
            # ==========================================

            frame = camera.get_frame()

            if frame is None:

                print("Failed to capture frame.")

                break

            # ==========================================
            # Detect hand
            # ==========================================

            results = detector.detect(
                frame
            )

            # ==========================================
            # Extract features
            # ==========================================

            features = processor.extract_features(
                results
            )

            # ==========================================
            # Recognition Controller
            #
            # Handles:
            # - INITIALIZING state
            # - static recognition
            # - jitter filtering
            # - dynamic gesture detection
            # - gesture locking
            # ==========================================

            result = controller.update(
                features
            )

            prediction = result[
                "prediction"
            ]

            confidence = result[
                "confidence"
            ]

            mode = result[
                "mode"
            ]

            sequence_complete = result[
                "sequence_complete"
            ]

            # ==========================================
            # Reset gesture permission
            #
            # Only when the controller ENTERS NONE.
            #
            # Do NOT reset this continuously while
            # already in NONE, otherwise jitter can
            # cause repeated letters.
            # ==========================================

            if (
                mode == controller.NONE
                and previous_mode != controller.NONE
            ):

                gesture_consumed = False

            # ==========================================
            # Add recognized gesture only once
            #
            # Extra safety layer against:
            #
            # H -> HH -> HHH -> HHHHH
            #
            # The controller should already emit a
            # confirmed gesture once, but this prevents
            # accidental repeated WordBuilder additions.
            # ==========================================

            if (
                prediction is not None
                and not gesture_consumed
            ):

                word_builder.add(
                    prediction
                )

                gesture_consumed = True

            # ==========================================
            # Save current mode
            #
            # Used on the next frame to detect an
            # actual state transition.
            # ==========================================

            previous_mode = mode

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
                (
                    f"Confidence: "
                    f"{last_confidence * 100:.1f}%"
                ),
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
                (
                    f"Sequence frames: "
                    f"{controller.get_sequence_length()}"
                ),
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
            # INITIALIZING state
            #
            # Prevents first-hand-entry duplicate
            # such as:
            #
            # HELLO -> HHELLO
            # ==========================================

            elif mode == controller.INITIALIZING:

                cv2.putText(
                    frame,
                    "Initializing hand...",
                    (20, 200),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 165, 255),
                    2
                )

            # ==========================================
            # Dynamic gesture completed
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
            # Current text
            # ==========================================

            current_text = word_builder.get_text()

            cv2.putText(
                frame,
                f"Text: {current_text}",
                (20, 300),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (255, 255, 255),
                2
            )

            # ==========================================
            # Controls
            # ==========================================

            cv2.putText(
                frame,
                (
                    "[SPACE] Space  "
                    "[DOUBLE SPACE] Clear  "
                    "[BACKSPACE] Delete  "
                    "[Q] Quit"
                ),
                (20, 350),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (200, 200, 200),
                1
            )

            # ==========================================
            # Display
            # ==========================================

            cv2.imshow(
                WINDOW_NAME,
                frame
            )

            # ==========================================
            # Keyboard input
            # ==========================================

            key = cv2.waitKey(1) & 0xFF

            # ==========================================
            # Quit
            # ==========================================

            if key == ord("q"):

                break

            # ==========================================
            # Space / Double Space
            # ==========================================

            elif key == ord(" "):

                current_time = time.time()

                # --------------------------------------
                # Double SPACE
                #
                # Clear entire displayed text.
                # --------------------------------------

                if (
                    last_space_time > 0
                    and (
                        current_time - last_space_time
                        <= double_space_interval
                    )
                ):

                    word_builder.clear()

                    # Prevent a third space from being
                    # interpreted as another double space.
                    last_space_time = 0.0

                # --------------------------------------
                # Single SPACE
                #
                # Add separation for next word.
                # --------------------------------------

                else:

                    word_builder.space()

                    last_space_time = current_time

            # ==========================================
            # Backspace
            # ==========================================

            elif key in [8, 127]:

                word_builder.backspace()

                last_space_time = 0.0

            # ==========================================
            # Any other key resets double-space timing
            # ==========================================

            elif key != 255:

                last_space_time = 0.0

    finally:

        detector.close()

        camera.stop()

        cv2.destroyAllWindows()


if __name__ == "__main__":

    main()