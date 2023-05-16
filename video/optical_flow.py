from os import path
from tqdm import tqdm, trange
import cv2
import matplotlib.pyplot as plt
import numpy as np

DATA_DIR                = "data"
RESULTS_DIR             = "results"
PROCESSING_FRAME_LENGTH = 256
PROCESSING_FRAME_SIZE   = (PROCESSING_FRAME_LENGTH, PROCESSING_FRAME_LENGTH)

def optical_flow(video_path: str, vis_path: str = None):
    """
    Compute optical flow for a given video. Based on https://medium.com/@igorirailean/dense-optical-flow-with-python-using-opencv-cb6d9b6abcaf

    Args:
        video_path: Path to the video file to process
        vis_path: Path to save visualisation of optical flow
        
    Returns:
        List of optical flow frames
    """
    # Capture first frame, resize and convert to greyscale
    video       = cv2.VideoCapture(video_path, apiPreference=cv2.CAP_FFMPEG)
    _, frame    = video.read()
    frame       = cv2.resize(frame,
                             dsize=PROCESSING_FRAME_SIZE,
                             interpolation=cv2.INTER_AREA) # Better for shrinking apparently
    frame_grey  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Configure mask and video writer for saving visualisation if needed
    if vis_path is not None:
        mask            = np.zeros_like(frame)  # Mask is same size as final frame
        mask[:, :, 1]   = 255                   # Set image saturation to maximum value as we do not need it
        vis_writer      = cv2.VideoWriter(vis_path,
                                          cv2.VideoWriter_fourcc(*'avc1'),  # H.264 encoding
                                          video.get(cv2.CAP_PROP_FPS),      # Same framerate as source video
                                          PROCESSING_FRAME_SIZE)            # Same size as optical flow frames


    # Process rest of video
    flow_frames = []
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    for frame_idx in trange(1, frame_count, desc="Processing frames", unit=" frames"):
        # Set previous frame
        prev_frame      = frame
        prev_frame_grey = frame_grey

        # Acquire next frame, resize, and convert to greyscale
        _, frame        = video.read()
        frame           = cv2.resize(frame,
                                 dsize=PROCESSING_FRAME_SIZE,
                                 interpolation=cv2.INTER_AREA)
        frame_grey      = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Compute optical flow for frame using Farneback method
        flow = cv2.calcOpticalFlowFarneback(prev_frame_grey, frame_grey, None,
                                            pyr_scale=0.5, levels=5, winsize=11, iterations=5,
                                            poly_n=5, poly_sigma=1.1,
                                            flags=0)
        
        # Write to visualisation file if needed
        if vis_path is not None:
            magnitude, angle    = cv2.cartToPolar(flow[:, :, 0], flow[:, :, 1], angleInDegrees=True)    # Compute the magnitude and angle of the 2D vectors
            mask[:, :, 0]       = (179 / 360) * angle                                                   # Set image hue according to the optical flow direction (map [0, 360] to [0, 179], see: https://codeyarns.com/tech/2014-02-26-carttopolar-in-opencv.html#gsc.tab=0)
            mask[:, :, 2]       = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)               # Set image value according to the optical flow magnitude (normalized)
            flow_rgb            = cv2.cvtColor(mask, cv2.COLOR_HSV2BGR)                                 # Convert HSV to RGB (BGR) color representation
            combined_frame      = cv2.addWeighted(frame, 1, flow_rgb, 2, 0)                             # Combine original frame and optical flow RGB frame
            vis_writer.write(combined_frame)                                                            # Write frame to visualisation video

        flow_frames.append(flow)
    
    # Release reader (and writer) and return computed optical flow frames
    if vis_path is not None:
        vis_writer.release()
    video.release()
    return flow_frames


def flow_intensity(flow_frames: list) -> list[float]:
    """
    Compute the mean intensity of a series of flow frames

    Args:
        flow_frames: A list of optical flow frames

    Returns:
        List where each element is the mean of optical flow vector magnitudes of each input frame
    """
    intensities = []
    for frame in tqdm(flow_frames, desc="Computing frame intensities"):
        rms             = lambda x1, x2: np.sqrt(np.square(x1) + np.square(x2))
        frame_intensity = rms(frame[:, :, 0], frame[:, :, 1])
        intensities.append(np.mean(frame_intensity))
    return intensities


def plot_intensity(intensities: list[float], show = False, file_path: str = None):
    """
    Plot optical flow intensity over time

    Args:
        intensities: List of per-frame intensities as computed by flow_intensity()
        show: Display the plot on-screen
        file_path: Path of file to save the plot to
    """
    frame_counts = np.arange(0, len(intensities))
    plt.plot(frame_counts, intensities)
    plt.title("Optical Flow Mean Intensity")
    plt.xlabel("Frame")
    plt.ylabel("Intensity (Mean RMS of flow vectors)")

    if file_path is not None:
        plt.savefig(file_path, bbox_inches="tight")
    if show:
        plt.show()
    plt.clf()


if __name__ == "__main__":
    SOURCE_VIDEO    = path.join(DATA_DIR, "trim.mp4")
    VIS_FILE        = path.join(RESULTS_DIR, "optical_flow_visualisation.mp4")
    GRAPH_FILE      = path.join(RESULTS_DIR, "optical_flow_intensity.png")

    optical_flow_frames = optical_flow(SOURCE_VIDEO)
    intensities         = flow_intensity(optical_flow_frames)
    plot_intensity(intensities, True, GRAPH_FILE)
