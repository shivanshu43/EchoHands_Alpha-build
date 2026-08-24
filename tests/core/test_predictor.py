from src.core.predictor import Predictor


def main():

    predictor = Predictor()

    print("Model loaded successfully.")
    print("Label encoder loaded successfully.")

    print("Classes:", predictor.label_encoder.classes_)


if __name__ == "__main__":
    main()