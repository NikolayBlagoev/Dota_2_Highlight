import easyocr

reader = easyocr.Reader(["en"])


def extract_text(img):
    text_items = reader.readtext(img)
    return text_items


if __name__ == "__main__":
    # pre-processing actually hurts the recognition
    print(extract_text("test_screenshot.png"))
