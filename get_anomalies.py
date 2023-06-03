import os
from sound.sound_eval import eval_sound
import numpy as np
from matplotlib import pyplot as plt
import pydub
from time_series.anomaly_detection import polyreg_outliar_mse, lof
folders = ["data/cust", "data/topsdata"]
data_rms = []
data_bin = []
def eval_linreg(arr, n, linreg):
    arr = np.array(arr).reshape((len(arr),))
    windows = []
    labels = []
    for i in range(len(arr)-n):
        windows.append(arr[i:i+n])
        labels.append(arr[i+n])
    return (linreg.predict(windows) - labels)**2
for dir in folders:
    for filename in os.listdir(dir):
        fil = os.path.join(dir, filename)
        print("getting", fil)
        a = pydub.AudioSegment.from_mp3(fil)
        
        y = np.array(a.get_array_of_samples())
        print(y.shape)
        data_binr, data_rmsr, _ = eval_sound(y,a)
        data_bin.append(data_binr)
        data_rms.append(data_rmsr)
        del a
        del y
a = pydub.AudioSegment.from_mp3("data/custom_game.mp3")
        
y = np.array(a.get_array_of_samples())
data_binr, data_rmsr, _ = eval_sound(y,a)
linreg_rms = polyreg_outliar_mse(data_rms, 15)
linreg_bin = polyreg_outliar_mse(data_bin, 15)
fig, axs = plt.subplots(5)
    
axs[0].plot(eval_linreg(data_binr, 15, linreg_bin))
axs[1].plot(eval_linreg(data_rmsr, 15, linreg_rms))
axs[2].plot(lof(np.array(data_binr), 100))
axs[3].plot(lof(np.array(data_rmsr), 100))

plt.show()
