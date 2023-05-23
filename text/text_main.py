from text_pre import prep_image
from text_ocr import extract_text
from text_post import text_ocr_filter


def text_main(img):
    return text_ocr_filter(extract_text(img))


if __name__ == "__main__":
    img = "test_screenshot.png"
    important_info = text_main(img)
    for info in important_info:
        print(info)
