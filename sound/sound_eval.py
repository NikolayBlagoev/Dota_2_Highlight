from matplotlib import pyplot as plt
from scipy.fft import *
from scipy.signal import periodogram, butter, sosfilt
from sys import argv
import pydub
import numpy as np

RMS_THRESHOLD = 6


def eval_sound(sound_data, audio_seg):
    """
    Compute sound feature metrics by dividing each second of audio into 6 overlapping windows

    Args:
        sound_data: Array encoding audio data where each entry is one sample
        audio_seg: PyDub AudioSegment object wrapping the data to be analysed

    Returns:
        freq_excitations: List containing number of frequencies above a predetermined threshold
        for each window
        freq_amplitudes: Means of RMS of amplitudes of all frequencies for each window
        max_amplitudes: Maximum amplitude RMS for each window
    """
    freq_excitations    = []
    freq_amplitudes     = []
    max_amplitudes      = []

    # Apply bandstop on audio
    band_stop = butter(5, [300, 3000], 'bandstop', fs=audio_seg.frame_rate, output='sos', analog = False)
    for i in range(6*((sound_data.shape[0]//audio_seg.frame_rate) - 1)):
        dataToRead = sound_data[int(i * audio_seg.frame_rate//6) : int((i + 1) * audio_seg.frame_rate//6)]
        dataToRead = sosfilt(band_stop, dataToRead)

        _, Pxx = periodogram(dataToRead, audio_seg.frame_rate)
        A_rms = np.sqrt(Pxx)  

        # Count frequencies above certain threshold
        filter_sum = A_rms[A_rms > RMS_THRESHOLD]
            
        freq_excitations.append(filter_sum.shape[0])
        freq_amplitudes.append(np.mean(A_rms))
        max_amplitudes.append(Pxx.max())
    return freq_excitations, freq_amplitudes, max_amplitudes


if __name__ == "__main__":
    a = pydub.AudioSegment.from_mp3(argv[1])
    y = np.array(a.get_array_of_samples())
    if a.channels == 2:
        y = y.reshape((-1, 2))
        y = np.transpose(y)
        
    freq_excitations, freq_amplitudes, max_amplitudes = eval_sound(y,a)
    np.savetxt('arr.csv', np.asarray(freq_excitations), delimiter=',')
    plt.plot(freq_excitations)
    plt.savefig(f"{argv[2]}/freqs.png")
    plt.show()

    plt.plot(freq_amplitudes)
    plt.savefig(f"{argv[2]}/mean_rms.png")
    plt.show()

    plt.plot(max_amplitudes)
    plt.savefig(f"{argv[2]}/rms_ampl.png")
    plt.show()
