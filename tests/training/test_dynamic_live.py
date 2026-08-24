import cv2
import numpy as np
import tensorflow as tf

from src.core.camera import Camera
from src.core.hand_detector import HandDetector
from src.core.landmark_processor import LandmarkProcessor


MODEL_PATH = "models/dynamic_lstm.keras"
ENCODER_PATH = "models/dynamic_label_encoder.npy"

TARGET_FRAMES = 40

WINDOW_NAME = "Dynamic Gesture Test"


def resample_sequence(sequence, target_frames):

    original_frames = sequence.shape[0]

    if original_frames == target_frames:
        return sequence

    old_indices = np.linspace(
        0,
        original_frames - 1,
        original_frames
    )

    new_indices = np.linspace(
        0,
        original_frames - 1,
        target_frames
    )

    resampled = np.zeros(
        (target_frames, sequence.shape[1]),
        dtype=np.float32
    )

    for feature_index in range(sequence.shape[1]):

        resampled[:, feature_index] = np.interp(
            new_indices,
            old_indices,
            sequence[:, feature_index]
        )

    return resampled


def predict_sequence(model, classes, sequence):

    sequence = resample_sequence(
        sequence,
        TARGET_FRAMES
    )

    sequence = sequence.reshape(
        1,
        TARGET_FRAMES,
        42
    )

    prediction = model.predict(
        sequence,
        verbose=0
    )

    predicted_index = np.argmax(
        prediction[0]
    )

    predicted_class = classes[
        predicted_index
    ]

    confidence = prediction[0][
        predicted_index
    ]

    return predicted_class, confidence


def main():

    print("\nLoading dynamic model...")

    model = tf.keras.models.load_model(
        MODEL_PATH
    )

    classes = np.load(
        ENCODER_PATH,
        allow_pickle=True
    )

    print(
        "Classes:",
        classes
    )

    camera = Camera()
    detector = HandDetector()
    processor = LandmarkProcessor()

    recording = False
    sequence = []

    prediction_text = "Press SPACE to start"

    try:

        camera.start()

        print("\n========== Dynamic Live Test ==========\n")
        print("SPACE → Start recording")
        print("SPACE → Stop recording and predict")
        print("Q     → Quit")
        print("\n")

        while True:

            frame = camera.get_frame()

            if frame is None:

                print(
                    "Failed to capture frame."
                )

                break

            results = detector.detect(
                frame
            )

            features = processor.extract_features(
                results
            )

            # ==================================
            # Collect sequence
            # ==================================

            if recording:

                if features is not None:

                    sequence.append(
                        features
                    )

                status = (
                    f"Recording | "
                    f"Frames: {len(sequence)}"
                )

            else:

                status = "Ready"

            # ==================================
            # Draw landmarks
            # ==================================

            frame = detector.draw(
                frame,
                results
            )

            # ==================================
            # Display prediction
            # ==================================

            cv2.putText(
                frame,
                prediction_text,
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                status,
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2
            )

            if recording:

                cv2.putText(
                    frame,
                    "Press SPACE to stop",
                    (20, 120),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2
                )

            else:

                cv2.putText(
                    frame,
                    "Press SPACE to start",
                    (20, 120),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2
                )

            cv2.imshow(
                WINDOW_NAME,
                frame
            )

            key = cv2.waitKey(1) & 0xFF

            # ==================================
            # SPACE
            # ==================================

            if key == ord(" "):

                if not recording:

                    sequence = []

                    prediction_text = (
                        "Recording..."
                    )

                    recording = True

                else:

                    recording = False

                    # --------------------------
                    # Check sequence
                    # --------------------------

                    if len(sequence) < 5:

                        prediction_text = (
                            "Too few frames"
                        )

                    else:

                        sequence_array = np.asarray(
                            sequence,
                            dtype=np.float32
                        )

                        predicted_class, confidence = (
                            predict_sequence(
                                model,
                                classes,
                                sequence_array
                            )
                        )

                        prediction_text = (
                            f"Prediction: "
                            f"{predicted_class} | "
                            f"Confidence: "
                            f"{confidence * 100:.1f}%"
                        )

            # ==================================
            # Quit
            # ==================================

            elif key == ord("q"):

                break

    finally:

        detector.close()
        camera.stop()


if __name__ == "__main__":
    main()