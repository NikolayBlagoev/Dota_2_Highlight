# Dota_2_Highlight

Data is provided on demand

### Time Series

The time series folder contains time series related functions. 

smoothening_functions.py contains functions which smooth the signal. Of interest are the Kaiser Window smoothening and the linear smoothening.

anomaly_detection.py contains functions for anomaly detection. Of interest are the LOF (local outliaer factor) and lo

polyreg_outliar_mse, which returns a trained linear autoregressive to be used with minimum squared error for predictions.


### Sound

sound_eval.py extracts sound related features. It is performing at a sampling rate of 6 times per second. Currently we use number of frequencies present above a certain threshold in a 1 second window as feature for the pipeline (called zerooorone in the file).

The functions here work with mono sound. You can extract mono audio with:
```
ffmpeg -i in.mp4 -ac 1 out.mp3
```
