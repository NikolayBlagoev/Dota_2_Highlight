from os import path
from tqdm import tqdm, trange
import cv2
import json
import matplotlib.pyplot as plt
import numpy as np

DATA_DIR                = "data"
RESULTS_DIR             = "results"
PROCESSING_FRAME_LENGTH = 128
PROCESSING_FRAME_SIZE   = (PROCESSING_FRAME_LENGTH, PROCESSING_FRAME_LENGTH)

def optical_flow_stats(video_path: str):
    """
    Compute optical flow statistics for a given video.

    Args:
        video_path: Path to the video file to process
        
    Returns:
        intensities: Mean of optical flow vector magnitudes of each input frame
        angle_means: Mean of optical flow vector angles of each input frame
        angle_stds: Standard deviation of flow vector angles of each input frame
    """
    # Capture first frame, resize and convert to greyscale
    video       = cv2.VideoCapture(video_path, apiPreference=cv2.CAP_FFMPEG)
    _, frame    = video.read()
    frame       = cv2.resize(frame,
                             dsize=PROCESSING_FRAME_SIZE,
                             interpolation=cv2.INTER_AREA) # Better for shrinking apparently
    frame_grey  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Process rest of video
    intensities: list[float]    = []
    angle_means: list[float]    = []
    angle_stds: list[float]     = []
    frame_count                 = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    for _ in trange(1, frame_count, desc="Computing optical flow", unit=" frames"):
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
        
        magnitudes, angles  = cv2.cartToPolar(flow[:, :, 0], flow[:, :, 1], angleInDegrees=True) # Compute the magnitude and angle of the 2D vectors
        angle_mean          = np.mean(angles)
        angle_std           = np.std(angles) 
        angle_means.append(angle_mean)
        angle_stds.append(angle_std)
        intensities.append(np.mean(magnitudes))
    
    # Zero out the magnitudes of frames where the angle standard deviation is less than
    # the mean minus one standard deviation of the angle std computed for ALL frames 
    angle_std_mean  = np.mean(angle_stds)
    angle_std_std   = np.std(angle_stds)
    minus_one_std   = angle_std_mean - angle_std_std
    for frame_idx in trange(len(intensities), desc="Pruning outliers"):
        frame_angle_std         = angle_stds[frame_idx]
        intensities[frame_idx]  = intensities[frame_idx] if frame_angle_std > minus_one_std else 0

    return intensities, angle_means, angle_stds


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
    for _ in trange(1, frame_count, desc="Computing optical flow", unit=" frames"):
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


def flow_intensity(flow_frames: list):
    """
    Compute the mean intensity of a series of flow frames

    Args:
        flow_frames: A list of optical flow frames

    Returns:
        intensities: Mean of optical flow vector magnitudes of each input frame
        angle_means: Mean of optical flow vector angles of each input frame
        angle_stds: Standard deviation of flow vector angles of each input frame
    """
    intensities: list[float] = []
    angle_means: list[float] = []
    angle_stds: list[float]  = []

    # Compute basic outputs
    for frame in tqdm(flow_frames, desc="Computing frame statistics"):
        magnitudes, angles  = cv2.cartToPolar(frame[:, :, 0], frame[:, :, 1], angleInDegrees=True)
        angle_mean          = np.mean(angles)
        angle_std           = np.std(angles) 
        angle_means.append(angle_mean)
        angle_stds.append(angle_std)
        intensities.append(np.mean(magnitudes))

    # Zero out the magnitudes of frames where the angle standard deviation is less than
    # the mean minus one standard deviation of the angle std computed for ALL frames 
    angle_std_mean  = np.mean(angle_stds)
    angle_std_std   = np.std(angle_stds)
    minus_one_std   = angle_std_mean - angle_std_std
    for frame_idx in trange(len(flow_frames), desc="Pruning outliers"):
        frame_angle_std         = angle_stds[frame_idx]
        intensities[frame_idx]  = intensities[frame_idx] if frame_angle_std > minus_one_std else 0

    return intensities, angle_means, angle_stds


def plot_per_frame_values(values: list[float], title:str, y_label:str,
                          start_frame: int = 1, show = False, file_path: str = None,
                          use_seconds = True):
    """
    Plot given values over frames

    Args:
        values: Per frame values to be plotted
        title: Title to display on plot
        y_label: Label of y-axis to display on plot
        start_frame: Number of first frame in the sequence
        show: Display the plot on-screen
        file_path: Path of file to save the plot to
    """
    time_steps = np.arange(start_frame, len(values) + start_frame, dtype=np.uint32)
    if use_seconds:
        time_steps  = np.divide(time_steps, int(30))
        x_label     = "Time (seconds)"
    else:
        x_label     = "Frame"
    plt.plot(time_steps, values)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    if file_path is not None:
        plt.savefig(file_path, bbox_inches="tight")
    if show:
        plt.show()
    plt.clf()


if __name__ == "__main__":
    SOURCE_VIDEO        = path.join(DATA_DIR, "custom game.mp4")
    VIS_FILE            = path.join(RESULTS_DIR, "optical_flow_visualisation.mp4")
    GRAPH_FILE          = path.join(RESULTS_DIR, "optical_flow_intensity.png")
    GRAPH_MEANS_FILE    = path.join(RESULTS_DIR, "optical_flow_intensity_angle_means.png")
    GRAPH_STDS_FILE     = path.join(RESULTS_DIR, "optical_flow_intensity_angle_stds.png")
    DATA_FILE           = path.join(RESULTS_DIR, "optical_flow_data.json")

    # Compute and save results
    intensities, angle_means, angle_stds = optical_flow_stats(SOURCE_VIDEO)

    plot_per_frame_values(intensities, "Optical Flow Mean Intensity", "Intensity (Mean RMS of flow vectors)", show=True, file_path=GRAPH_FILE)
    plot_per_frame_values(angle_means, "Optical Flow Angle Mean", "Angle", show=False, file_path=GRAPH_MEANS_FILE)
    plot_per_frame_values(angle_stds, "Optical Flow Angle Standard Deviation", "Angle", show=False, file_path=GRAPH_STDS_FILE)

    with open(DATA_FILE) as optical_data_file:
        json.dump(intensities)
