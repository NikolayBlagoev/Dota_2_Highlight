# Sound Features
Contains functionality for extracting audio-based features

## Directory Structure
- `loudeval.py`: Computes loudness of an audio file
- `sound_eval.py`: Extracts sound related features. It is performing at a sampling rate of 6 times per second, resulting in 6 overlapping windows. The script can be ran independently for testing purposes by passing a path to an audio file as the first argument, though the functions are intended to be used as part of a pipeline instead. This computes the following on a per window basis, though only the first is used as a feature:
    1. Number of frequencies above a predetermined threshold
    2. Means of RMS of amplitudes of all frequencies
    3. Maximum amplitude RMS
