import numpy as np
from matplotlib import pyplot as plt
from statsmodels.tsa.arima.model import ARIMA


def normalize_mean_cov(signal):
    """
    Normalize given data by substracing mean and dividing by standard deviation

    Args:
        signal: Data to be operated on

    Returns:
        Normalised data
    """
    mean = np.mean(signal)
    std  = np.std(signal)
    return (signal - mean) / std


def normalize_min_max(signal):
    """
    Normalize given data by substracting the minimum and dividing by the difference between minimum and maximum

    Args:
        signal: Data to be operated on

    Returns:
        Normalised data
    """
    return (signal - np.min(signal)) / (np.max(signal) - np.min(signal))


def ARIMA_noise(signal):
    """
    Fit an ARIMA model onto the given data and compute residuals

    Args:
        signal: Data to be operated on

    Returns:
        Values predicted by the ARIMA model
    """
    model   = ARIMA(signal, order=(3,1,3))
    res     = model.fit()
    return np.abs(res.fittedvalues)


def fft_filter(signal, perc):
    """
    Remove all frequencies having an amplitude below a certain threshold

    Args:
        signal: Data to be operated on in the time domain
        perc: Percentage of the normalised amplitude, below which a frequency is considered noise and removed

    Returns:
        Filtered data in the frequency domain
    """
    fft_signal  = np.fft.fft(signal)
    fft_abs     = np.abs(fft_signal)
    th          = perc * (2 * fft_abs[0 : int(len(signal) / 2.)] / len(signal)).max()
    
    fft_tof                     = fft_signal.copy()
    fft_tof_abs                 = np.abs(fft_tof)
    fft_tof_abs                 = 2 * fft_tof_abs/len(signal)
    fft_tof[fft_tof_abs <= th]  = 0
    return fft_tof


def derivative(signal):
    """
    Compute difference of consecutive elements. In effect, a discrete derivative

    Args:
        signal: Data to operate on

    Returns:
        List where each entry is the difference between two succesive elements in the input array
    """
    ret = []
    for i in range(len(signal) - 1):
        ret.append(signal[i + 1] - signal[i])
    return ret


def filter_signal(signal, th):
    """
    Remove all frequencies having an amplitude below a certain threshold

    Args:
        signal: Data to be operated on in the time domain
        th: Percentage of the normalised amplitude, below which a frequency is considered noise and removed

    Returns:
        Filtered data in the time domain
    """
    f_s = fft_filter(signal, th)
    return np.real(np.fft.ifft(f_s))


def linear_denoise(signal, n):
    """
    Convolve a mean filter with the given data
    
    Args:
        signal: Data to be operated on
        n: Size of the mean filter (the filter will be n long and all its entries will be (1 / n))
        
    Returns:
        Filtered data
    """
    return np.convolve(signal, [1 / n for _ in range(n)], 'valid')


if __name__ == "__main__":
    signal      = np.loadtxt("arr.csv", delimiter = ",")
    fig, axs    = plt.subplots(6)
    axs[0].plot(signal)
    axs[1].plot(filter_signal(signal, 0.05))
    axs[2].plot(linear_denoise(signal, 100))
    axs[3].plot(normalize_mean_cov(signal))
    axs[4].plot(derivative(signal))
    axs[5].plot(ARIMA_noise(filter_signal(signal, 0.02)))
    plt.show()