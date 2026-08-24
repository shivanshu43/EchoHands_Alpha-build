import os
import pickle

import pandas as pd
import tensorflow as tf

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from tensorflow.keras import Sequential, Input
from tensorflow.keras.layers import Dense


CSV_PATH = "data/processed/keypoints.csv"

MODEL_PATH = "models/model.keras"
LABEL_ENCODER_PATH = "models/label_encoder.pkl"


def main():

    # ==========================================
    # Load Dataset
    # ==========================================

    dataset = pd.read_csv(CSV_PATH, header=None)

    print("=" * 60)
    print("First 5 Rows")
    print("=" * 60)
    print(dataset.head())

    # ==========================================
    # Dataset Shape
    # ==========================================

    print("\n" + "=" * 60)
    print("Dataset Shape")
    print("=" * 60)
    print(dataset.shape)

    # ==========================================
    # Dataset Information
    # ==========================================

    print("\n" + "=" * 60)
    print("Dataset Information")
    print("=" * 60)
    dataset.info()

    # ==========================================
    # Missing Values
    # ==========================================

    print("\n" + "=" * 60)
    print("Missing Values")
    print("=" * 60)
    print(dataset.isnull().sum())

    # ==========================================
    # Separate Features and Labels
    # ==========================================

    X = dataset.iloc[:, 1:]
    y = dataset.iloc[:, 0]

    print("\n" + "=" * 60)
    print("Features (X)")
    print("=" * 60)
    print(X.head())

    print("\n" + "=" * 60)
    print("Labels (y)")
    print("=" * 60)
    print(y.head())

    # ==========================================
    # Label Encoding
    # ==========================================

    label_encoder = LabelEncoder()

    y_encoded = label_encoder.fit_transform(y)

    print("\n" + "=" * 60)
    print("Encoded Labels")
    print("=" * 60)
    print(y_encoded[:10])

    # ==========================================
    # Label Mapping
    # ==========================================

    print("\n" + "=" * 60)
    print("Label Mapping")
    print("=" * 60)

    for index, label in enumerate(label_encoder.classes_):
        print(f"{label} -> {index}")

    # ==========================================
    # Check Number of Classes
    # ==========================================

    num_classes = len(label_encoder.classes_)

    print("\n" + "=" * 60)
    print(f"Total Classes : {num_classes}")
    print("=" * 60)

    if num_classes < 2:

        print("\nDataset currently contains only one class.")
        print("Training skipped.")
        print("Collect data for at least one more class.")
        return

    # ==========================================
    # Train-Test Split
    # ==========================================

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_encoded,
        test_size=0.2,
        random_state=42,
        stratify=y_encoded
    )

    print("\n" + "=" * 60)
    print("Training Set Shape")
    print("=" * 60)
    print(X_train.shape)

    print("\n" + "=" * 60)
    print("Testing Set Shape")
    print("=" * 60)
    print(X_test.shape)

    print("\n" + "=" * 60)
    print("Training Labels Shape")
    print("=" * 60)
    print(y_train.shape)

    print("\n" + "=" * 60)
    print("Testing Labels Shape")
    print("=" * 60)
    print(y_test.shape)

    # ==========================================
    # Build Neural Network
    # ==========================================

    model = Sequential([

        Input(shape=(42,)),

        Dense(128, activation="relu"),

        Dense(64, activation="relu"),

        Dense(num_classes, activation="softmax")

    ])

    print("\n" + "=" * 60)
    print("Model Summary")
    print("=" * 60)

    model.summary()

    # ==========================================
    # Compile Model
    # ==========================================

    model.compile(

        optimizer=tf.keras.optimizers.Adam(),

        loss=tf.keras.losses.SparseCategoricalCrossentropy(),

        metrics=[
            tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")
        ]

    )

    print("\n" + "=" * 60)
    print("Model Compiled Successfully")
    print("=" * 60)

    # ==========================================
    # Train Model
    # ==========================================

    history = model.fit(

        X_train,
        y_train,

        epochs=20,

        batch_size=32,

        validation_split=0.2,

        verbose=1

    )

    print("\n" + "=" * 60)
    print("Training Completed")
    print("=" * 60)

    # ==========================================
    # Evaluate Model
    # ==========================================

    loss, accuracy = model.evaluate(
        X_test,
        y_test,
        verbose=0
    )

    print("\n" + "=" * 60)
    print("Test Performance")
    print("=" * 60)
    print(f"Test Loss     : {loss:.4f}")
    print(f"Test Accuracy : {accuracy * 100:.2f}%")

    # ==========================================
    # Save Model
    # ==========================================

    os.makedirs("models", exist_ok=True)

    model.save(MODEL_PATH)

    with open(LABEL_ENCODER_PATH, "wb") as file:
        pickle.dump(label_encoder, file)

    print("\n" + "=" * 60)
    print("Model Saved Successfully")
    print("=" * 60)
    print(f"Model         : {MODEL_PATH}")
    print(f"Label Encoder : {LABEL_ENCODER_PATH}")


if __name__ == "__main__":
    main()