# Dota_2_Highlight

*Note*: Data is provided on demand

### Data

The `data` folder is used to store the files which are used by the scripts in the `get_anomalies.ipynb`. Since those files have to be provided on demand, this folder is kept empty and only contains a single `.gitkeep` file.

### Demo Autoencoder

The `demo_autoencoder` folder contains our attempt at implementing an autoencoder which can be used for anomaly detection. Unfortunately, due to time constraints, the implementation is not entirely finished - it still needs some improvements - and therefore we used other methods for anomaly detection in our evaluation pipeline.

### ImageNet Autoencoder

The `imagenet_autoencoder` folder is a submodule of this project. Its contents can be inspected at https://github.com/NikolayBlagoev/imagenet-autoencoder, which is our fork of https://github.com/Horizon2333/imagenet-autoencoder. We use this autoencoder for Single Frame Analysis.

### Sound

The `sound` folder contains scripts which can be used to extract sound-related features.

sound_eval.py extracts sound-related features. It is performing at a sampling rate of 6 times per second. Currently we use number of frequencies present above a certain threshold in a 1 second window as feature for the pipeline (called zerooorone in the file).

The functions here work with mono sound. You can extract mono audio with:
```
ffmpeg -i in.mp4 -ac 1 out.mp3
```

### Text

The `text` folder contains our implemenation of optical character recognition (OCR) which we use to extract text-related features from videos.

### Time Series

The `time_series` folder contains functions related to time series. 

smoothening_functions.py contains functions which smooth the signal. Of interest are the Kaiser Window smoothening and the linear smoothening.

anomaly_detection.py contains functions for anomaly detection. Of interest are the LOF (local outliaer factor) and lo

polyreg_outliar_mse, which returns a trained linear autoregressive to be used with minimum squared error for predictions.

### Video

The `video` folder contains scripts which can be used to extract video-related features from videos.

### Running the evaluation pipeline

*Note*: Running the evaluation pipeline requires also requires requires a CUDA-capable system and Python 3.8 or newer with PyTorch and CUDA installed. For more information, please refer to [this](https://pytorch.org/get-started/locally/) website. Moreover, it is required to have `ffmpeg` installed and its installation directory added to `PATH`. There are different ways to achieve this, depending on your operating system. Therefore, please refer to `ffmpeg`'s [official website](https://ffmpeg.org/) for further instructions. The remainder of this section assumes that those requirements have been met.

In order to run the evaluation pipeline, you need two files - a `.mp4` file, which contains a recording of a DotA 2 game, and a `.txt` file, which contains the start and end times of all highlights from said recording in the required format (per line: mm:ss (*denoting the start time*) - mm:ss (*denoting the end time*)). Let's assume that those files are named `video.mp4` and `labels.txt`, respectively. Then, you can run the evaluation pipeline using following steps:

- Clone this repository using `git clone --recurse-submodules https://github.com/NikolayBlagoev/Dota_2_Highlight.git`;
- Go into the root directory of the repository using `cd Dota_2_Highlight`;
- Move the `video.mp4` file into the root directory of the repository;
- Move the `labels.txt` file into the `data/` directory in the repository;
- Create the directory `tmp/extr5/` inside the root directory of the repository;
- Download this [file](https://drive.google.com/file/d/1Ny7QbtnywqRwYU2aFe2qKIyZOk7R-odY/view?usp=sharing) and move it nito the `imagenet-autoencoder/outputs/` directory in the repository;
- Run the following commands:
```
ffmpeg -i video.mp4 -ac 1 data/audio_mono.mp3
ffmpeg -i video.mp4 -vf scale=256:256 -r 6 tmp/extr5/image-%3d.png
python imagenet-autoencoder/tools/generate_list.py --name highlights --path tmp
python imagenet-autoencoder/train.py --arch resnet50 --train_list list/highlights_list.txt --batch-size 6 --workers 1 --start-epoch 10 --epochs 11 --pth-save-fold outputs 
python imagenet-autoencoder/run_autoencoder.py
mv imagenet-autoencoder/arr.csv data/frame_analysis.csv
```
- In the `Data Loading` section of the `get_anomalies.ipynb` notebook set the variable `audio_file` to `"audio_mono"`;
- In the `Data Loading` section of the `get_anomalies.ipynb` notebook set the variable `frame_analysis_file` to `"frame_analysis"`;
- In the `Data Loading` section of the `get_anomalies.ipynb` notebook set the variable `labels_file` to `"labels"`;
- In the `Data Loading` section of the `get_anomalies.ipynb` notebook set the variables `minutes` and `seconds` to reflect the length of your recording;
- Run all sections of the `get_anomalies.ipynb` notebook.

The outputs of the `Results` section of the `get_anomalies.ipynb` notebook will show the results of the evaluation.
