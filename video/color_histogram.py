from os import path
from tqdm import trange
import cv2
import matplotlib.pyplot as plt
import numpy as np

DATA_DIR    = "data"
RESULTS_DIR = "results"

def color_histograms(video_path: str):
    """
    Compute RGB histograms per frame

    Args:
        video_path: Path to the video to analyse

    Returns:
        List where each entry is a tuple of RGB histograms
    """
    video                   = cv2.VideoCapture(video_path, apiPreference=cv2.CAP_FFMPEG)
    frame_count             = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    histograms: list[tuple] = []
    for _ in trange(0, frame_count, desc="Computing histograms", unit=" frames"):
        _, frame    = video.read()
        b_hist      = cv2.calcHist([frame], [0], None, [256], [0, 256])
        g_hist      = cv2.calcHist([frame], [1], None, [256], [0, 256])
        r_hist      = cv2.calcHist([frame], [2], None, [256], [0, 256])
        histograms.append((r_hist, g_hist, b_hist))
    return histograms


def differences(time_series, compute_average = False):
    """
    Compute differences between elements of time series

    Args:
        time_series: Data to operate on
        compute_average: Whether the average should be computed for all of the data at each time frame

    Returns:
        Differences between successive elements
    """
    diffs = []
    for idx in trange(1, len(time_series), desc="Computing time series differences"):
        absolute_difference = np.absolute(np.subtract(time_series[idx], time_series[idx - 1]))
        diffs.append(absolute_difference)
    
    if compute_average:
        diffs = list(map(lambda datum: np.mean(datum), diffs))
    return diffs


def plot_difference(differences: list, color: str, show = False, file_path: str = None, use_seconds = True):
    """
    Plot given values over frames

    Args:
        differences: Per frame values to be plotted
        color: Color to use for the plot
        show: Display the plot on-screen
        file_path: Path of file to save the plot to
    """
    time_steps = np.arange(1, len(differences) + 1, dtype=np.uint32)
    if use_seconds:
        time_steps  = np.divide(time_steps, int(30))
        x_label     = "Time (seconds)"
    else:
        x_label     = "Frame"
    plt.plot(time_steps, differences, color=color)
    plt.title(f"Mean color histogram differences ({color})")
    plt.xlabel(x_label)
    plt.ylabel("Pixel count")

    if file_path is not None:
        plt.savefig(file_path, bbox_inches="tight")
    if show:
        plt.show()
    plt.clf()


if __name__ == "__main__":
    SOURCE_VIDEO        = path.join(DATA_DIR, "Topson Sniper Assassin - Dota 2 Pro Gameplay [Watch & Learn].mp4")
    GRAPH_FILE_RED      = path.join(RESULTS_DIR, "color_histogram_differences_red.png")
    GRAPH_FILE_GREEN    = path.join(RESULTS_DIR, "color_histogram_differences_green.png")
    GRAPH_FILE_BLUE     = path.join(RESULTS_DIR, "color_histogram_differences_blue.png")

    histograms          = color_histograms(SOURCE_VIDEO)
    red_differences     = differences(list(map(lambda rgb: rgb[0], histograms)), True)
    green_differences   = differences(list(map(lambda rgb: rgb[1], histograms)), True)
    blue_differences    = differences(list(map(lambda rgb: rgb[2], histograms)), True)
    plot_difference(red_differences,    'red',   show=True, file_path=GRAPH_FILE_RED)
    plot_difference(green_differences,  'green', show=True, file_path=GRAPH_FILE_GREEN)
    plot_difference(blue_differences,   'blue',  show=True, file_path=GRAPH_FILE_BLUE)
