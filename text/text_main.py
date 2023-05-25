from text_pre import crop_image
from text_ocr import extract_text
from text_post import text_ocr_filter


def text_main(img):
    return text_ocr_filter(extract_text(crop_image(img)))


if __name__ == "__main__":
    import cv2

    src = "test_screenshot.png"
    img = cv2.imread(src)
    important_info = text_main(img)
    print("done")
    for info in important_info:

        print(str(info))
