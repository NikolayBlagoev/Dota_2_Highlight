from text_pre import crop_image
from text_ocr import extract_text
from text_post import text_ocr_filter
from tqdm import tqdm
import traceback
import logging

logging.basicConfig(
    filename="text.log", filemode="w", format="%(name)s - %(levelname)s - %(message)s"
)


def text_main(img):
    result = text_ocr_filter(extract_text(crop_image(img)))
    return len(result)


def video_processing(video, sample_rate, func):
    # importing the necessary libraries
    import cv2
    import numpy as np

    time_series = np.array([])
    # Creating a VideoCapture object to read the video
    cap = cv2.VideoCapture(video)

    # get height, width and frame count of the video
    width, height = (
        int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    )
    print(width, height)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    print(fps)
    no_of_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    sample_every = fps // sample_rate
    print(sample_every)

    try:
        for i in tqdm(range(no_of_frames)):
            ret, frame = cap.read()
            if not ret:
                break
            if i % sample_every == 0:
                img = frame
                output = func(img)
                time_series = np.append(time_series, output)

    except:
        cv2.imshow("frame {}".format(i), img)
        cv2.waitKey(0)
        error_message = traceback.format_exc()
        logging.error(error_message)
        # Release resources
        cap.release()
    # release the video capture object
    cap.release()

    return time_series


if __name__ == "__main__":
    # import cv2

    # src = "test_screenshot.png"
    # img = cv2.imread(src)
    # important_info = text_main(img)
    # print("done")
    # print(str(important_info))
    import numpy as np

    kill_count_time_series = video_processing(
        "data/custom_game.mp4",
        1,
        text_main,
    )
    print(kill_count_time_series)

    np.savetxt("kills.csv", kill_count_time_series, delimiter=",")
