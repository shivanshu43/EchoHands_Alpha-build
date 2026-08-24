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
        dynamic_predictor,

        # Static recognition confidence
        static_confidence_threshold=0.60,
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
    #
    # H -> HH -> HHH -> HHHHH
    #
    # caused by repeated predictions during
    # controller state transitions.
    # ==========================================

    gesture_consumed = False

    previous_mode = controller.NONE

    # ==========================================
    # Hand-entry protection
    #
    # Prevents accidental first gesture
    # duplication when the hand enters:
    #
    # HELLO -> HHELLO
    #
    # Recognition remains blocked until:
    #
    # NO HAND
    #    ↓
    # HAND ENTERS
    #    ↓
    # INITIALIZING
    #    ↓
    # STATIC
    #    ↓
    # RECOGNITION READY
    # ==========================================

    hand_was_present = False

    waiting_for_hand_initialization = False

    recognition_ready = False

    # ==========================================
    # Space key tracking
    # ==========================================

    last_space_time = 0.0

    double_space_interval = 0.5

    # ==========================================
    # Start camera
    # ==========================================

    camera.start()

    print(
        "\n========== Sign Language Recognition ==========\n"
    )

    print("Static gestures : A-Y + 0-9")

    print("Dynamic gestures: J / Z")

    print("Press SPACE to add a space.")

    print(
        "Press SPACE twice quickly to clear text."
    )

    print(
        "Press BACKSPACE to remove last character."
    )

    print("Press 'Q' to exit.\n")

    try:

        while True:

            # ==========================================
            # Get frame
            # ==========================================

            frame = camera.get_frame()

            if frame is None:

                print(
                    "Failed to capture frame."
                )

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

            features = (
                processor.extract_features(
                    results
                )
            )

            # ==========================================
            # Hand entry / exit detection
            # ==========================================

            hand_present = features is not None

            # ------------------------------------------
            # Hand just entered the frame
            # ------------------------------------------

            if (
                hand_present
                and not hand_was_present
            ):

                waiting_for_hand_initialization = True

                recognition_ready = False

                # Block any accidental gesture
                # while the hand is entering.
                gesture_consumed = True

            # ------------------------------------------
            # Hand left the frame
            # ------------------------------------------

            elif (
                not hand_present
                and hand_was_present
            ):

                waiting_for_hand_initialization = False

                recognition_ready = False

                gesture_consumed = False

            # Update hand presence state
            hand_was_present = hand_present

            # ==========================================
            # Recognition Controller
            #
            # Handles:
            # - hand initialization
            # - static recognition
            # - confidence filtering
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
            # Hand initialization completed
            #
            # Enable recognition only when the
            # controller has reached STATIC state
            # after the hand entry.
            # ==========================================
            if (
                waiting_for_hand_initialization
                and mode == controller.STATIC
            ):

                # Hand initialization is complete.
                #
                # The gesture currently being held is treated as the
                # entry/initialization gesture and must not be emitted.
                #
                # This prevents:
                #
                # Hand enters with A
                #     ↓
                # Initialization finishes
                #     ↓
                # A is emitted again
                #     ↓
                # AA

                waiting_for_hand_initialization = False

                recognition_ready = True

                # Consume the gesture that was already present
                # during hand initialization.
                gesture_consumed = True

            # ==========================================
            # Reset gesture permission
            #
            # Only when the controller ENTERS NONE.
            #
            # Do NOT reset continuously while
            # already in NONE, otherwise jitter can
            # cause:
            #
            # H -> HH -> HHH
            # ==========================================

            if (
                recognition_ready
                and mode == controller.NONE
                and previous_mode != controller.NONE
            ):

                gesture_consumed = False

            # ==========================================
            # Add recognized gesture only once
            #
            # Extra protection against:
            #
            # H -> HH -> HHH -> HHHHH
            #
            # Also prevents accidental prediction
            # during hand initialization.
            # ==========================================

            if (
                prediction is not None
                and recognition_ready
                and not waiting_for_hand_initialization
                and not gesture_consumed
            ):

                word_builder.add(
                    prediction
                )

                gesture_consumed = True

                last_prediction = prediction

                last_confidence = confidence

            # ==========================================
            # Update display prediction
            #
            # This can display predictions without
            # necessarily adding them to the word.
            # ==========================================

            elif prediction is not None:

                last_prediction = prediction

                last_confidence = confidence

            # ==========================================
            # Store current mode
            # ==========================================

            previous_mode = mode

            # ==========================================
            # Draw hand landmarks
            # ==========================================

            frame = detector.draw(
                frame,
                results
            )

            # ==========================================
            # Prediction display
            # ==========================================

            cv2.putText(
                frame,
                f"Prediction: {last_prediction}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

            # ==========================================
            # Confidence display
            # ==========================================

            cv2.putText(
                frame,
                (
                    f"Confidence: "
                    f"{last_confidence * 100:.1f}%"
                ),
                (20, 75),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

            # ==========================================
            # Mode display
            # ==========================================

            cv2.putText(
                frame,
                f"Mode: {mode}",
                (20, 110),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

            # ==========================================
            # Dynamic sequence information
            # ==========================================

            sequence_length = (
                controller.get_sequence_length()
            )

            cv2.putText(
                frame,
                (
                    f"Sequence frames: "
                    f"{sequence_length}"
                ),
                (20, 145),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
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
            # Hand has just entered the screen.
            # Recognition is temporarily blocked.
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
            # Recognition ready state
            # ==========================================

            elif (
                recognition_ready
                and mode == controller.STATIC
            ):

                cv2.putText(
                    frame,
                    "Ready for gesture",
                    (20, 200),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
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

            current_text = (
                word_builder.get_text()
            )

            cv2.putText(
                frame,
                f"Text: {current_text}",
                (20, 285),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (255, 255, 0),
                2
            )

            # ==========================================
            # Show frame
            # ==========================================

            cv2.imshow(
                WINDOW_NAME,
                frame
            )

            # ==========================================
            # Keyboard input
            # ==========================================

            key = cv2.waitKey(1) & 0xFF

            # ------------------------------------------
            # Quit
            # ------------------------------------------

            if key in [ord("q"), ord("Q")]:

                break

            # ------------------------------------------
            # SPACE
            # ------------------------------------------

            elif key == 32:

                current_time = time.time()

                # --------------------------------------
                # Double SPACE
                #
                # Current implementation:
                # clears the complete text.
                # --------------------------------------

                if (
                    last_space_time > 0
                    and (
                        current_time
                        - last_space_time
                        <= double_space_interval
                    )
                ):

                    word_builder.clear()

                    # Prevent another consecutive space
                    # from being treated as double-space.
                    last_space_time = 0.0

                # --------------------------------------
                # Single SPACE
                #
                # Separate words.
                # --------------------------------------

                else:

                    word_builder.space()

                    last_space_time = current_time

            # ------------------------------------------
            # BACKSPACE
            # ------------------------------------------

            elif key in [8, 127]:

                word_builder.backspace()

                last_space_time = 0.0

            # ------------------------------------------
            # Any other key resets
            # double-space timing
            # ------------------------------------------

            elif key != 255:

                last_space_time = 0.0

    finally:

        detector.close()

        camera.stop()

        cv2.destroyAllWindows()


if __name__ == "__main__":

    main()