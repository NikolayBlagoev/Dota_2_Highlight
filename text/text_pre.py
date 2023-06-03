import cv2  # openCV
import numpy as np


def crop_image(img, window=[0.2, 0.8, 0, 0.5]):
    height, width = img.shape[0], img.shape[1]
    return img[
        int(height * window[0]) : int(height * window[1]),
        int(width * window[2]) : int(width * window[3]),
    ]


def prep_image(img, method="Canny"):
    "Call with the img as a string pointing to file."
    readout = cv2.imread(img)
    grayscale = cv2.cvtColor(readout, cv2.COLOR_BGR2GRAY)
    smoothed = cv2.GaussianBlur(grayscale, (3, 3), 0)
    return _edge_detection(smoothed, method)


def _edge_detection(img, method):
    "Relay function to choose between Sobel and Canny edge detection."
    if method == "Sobel":
        return _sobel(img)
    elif method == "Canny":
        return _canny(img)


def _sobel(img):
    return cv2.Scharr(src=img, ddepth=cv2.CV_64F, dx=1, dy=0)


def _canny(img):
    return cv2.Canny(
        image=img, threshold1=0, threshold2=90, L2gradient=True
    )  # threshold2 max 145 for the i, max 125 for !, threshold1 max 110 for !, for l2gradient= true threshold2 max is 90 for !


if __name__ == "__main__":
    test_image = "test_screenshot.png"
    src = prep_image(test_image, method="Canny")
    cv2.imshow("preped test image", src.copy())
    cv2.waitKey()
    kernel = np.ones((3, 3), np.uint8)

    img_erosion = cv2.erode(src, kernel, iterations=1)
    img_dilation = cv2.dilate(src, kernel, iterations=1)

    cv2.imshow("Input", src)
    cv2.imshow("Erosion", img_erosion)
    cv2.imshow("Dilation", img_dilation)

    cv2.waitKey(0)
