from src.core.word_builder import WordBuilder


def main():

    builder = WordBuilder()

    print("\n========== WordBuilder Test ==========\n")

    print("Add H:")
    print(builder.add("H"))

    print("Add E:")
    print(builder.add("E"))

    print("Add L:")
    print(builder.add("L"))

    print("Add L:")
    print(builder.add("L"))

    print("Add O:")
    print(builder.add("O"))

    print("\nCurrent text:")
    print(builder.get_text())

    print("\nAdd space:")
    print(builder.space())

    print("Add W:")
    print(builder.add("W"))

    print("Add O:")
    print(builder.add("O"))

    print("Add R:")
    print(builder.add("R"))

    print("Add L:")
    print(builder.add("L"))

    print("Add D:")
    print(builder.add("D"))

    print("\nCurrent text:")
    print(builder.get_text())

    print("\nBackspace:")
    print(builder.backspace())

    print("Current text:")
    print(builder.get_text())

    print("\nClear:")
    print(builder.clear())

    print("\n======================================\n")


if __name__ == "__main__":
    main()