import pydub
import numpy as np
from scipy.fft import *
from matplotlib import pyplot as plt
from scipy.signal import periodogram, butter, sosfilt
import math 
from sys import argv

N = 256

a = pydub.AudioSegment.from_mp3(argv[1])
y = np.array(a.get_array_of_samples())
if a.channels == 2:
    y = y.reshape((-1, 2))
    y = np.transpose(y)
print(y.shape)
zeroorone = []
sums = []
max_ampls = []
band_stop = butter(5, [300,3000], 'bandstop', fs=a.frame_rate, output='sos', analog = False)
for i in range(4*43*60):
    
    dataToRead = y[int(i * a.frame_rate/4 ) : int((i + 1) * a.frame_rate/4)]
    dataToRead = sosfilt(band_stop, dataToRead)

    # N = len(dataToRead)
    # yf = rfft(dataToRead)
    # xf = rfftfreq(N, 1 / a.frame_rate)
    
    # zf = np.log10(np.abs(yf))
    f, Pxx = periodogram(dataToRead, a.frame_rate)
    A_rms = np.sqrt(Pxx)    
    filter_sum = A_rms[A_rms > 6]
    
        
    zeroorone.append(filter_sum.shape[0])
    sums.append(np.mean(A_rms))
    max_ampls.append(Pxx.max())
    # idx = np.argmax(np.abs(yf))
    # freq = xf[idx]
    # print(freq)
np.savetxt('arr.csv', np.asarray(zeroorone), delimiter=',')
plt.plot(zeroorone)
plt.savefig(f"{argv[2]}/freqs.png")
plt.show()

plt.plot(sums)
plt.savefig(f"{argv[2]}/mean_rms.png")
plt.show()

plt.plot(max_ampls)
plt.savefig(f"{argv[2]}/rms_ampl.png")
plt.show()
# Uncomment these to see the frequency spectrum as a plot
# pyplot.plot(xf, np.abs(yf))
# pyplot.show()
exit()
