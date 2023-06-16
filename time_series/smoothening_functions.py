import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from tqdm import tqdm

def ewma_f(value, prev_value, alpha):
    """
    Helper function for ewma
    """
    return alpha * value + (1 - alpha) * prev_value

def ewma(arr, alpha):
    """
    Applies EWMA (Exponentially Weighted Moving Average) on input data with constant alpha

    Args:
        arr: Data to operate on
        alpha: EWMA constant

    Returns:
        Smoothed signal
    """
    prev = 0
    ret = []
    for it in arr:
        prev = ewma_f(it, prev, alpha)
        ret.append(prev)
    return ret


def ewma_bias_corrected_f(value, prev_value, alpha, t):
    """
    Helper function for ewma with bias correction
    """
    return (alpha*value + (1-alpha)*prev_value)/(1-alpha**t)


def ewma_bias_corrected(arr, alpha):
    """
    Applies EWMA (Exponentially Weighted Moving Average) with bias correction
    on input data with constant alpha

    Args:
        arr: Data to operate on
        alpha: EWMA constant

    Returns:
        Smoothed signal
    """
    prev = 0
    ret = []
    for t,it in enumerate(arr):
        prev = ewma_bias_corrected_f(it, prev, alpha, t+1)
        ret.append(prev)
    return ret


def power_smooth(arr, n):
    """
    Apply a power smoothing filter (as detailed in https://www.sciencedirect.com/science/article/pii/S2666827022000469)
    to the given data

    Args:
        arr: Data to be operated on
        n: Size of the power filter
        
    Returns:
        Filtered data
    """
    return np.convolve(arr, [2**abs(i - int(n / 2)) for i in range(n)], 'valid')


def llr_smooth(arr, n):
    """
    Applies local linear regression smoothening

    Args:
        arr: Data to be operated on
        n: Size of the linear regression window
        
    Returns:
        Smoothened data
    """
    reg     = LinearRegression()
    ret     = []
    loc     = np.array([i for i in range(n)]).reshape(-1, 1)
    topr    = np.array([int(n/2)]).reshape(1,-1)
    for t, _ in tqdm(enumerate(arr)):
        if t < int(n / 2) or t > len(arr) - int(n / 2):
            continue
        reg.fit(loc, arr[t-int(n/2):t+int(n/2)])
        if t < int(n / 2):
            for i in range(int(n / 2)):
                ret.append(reg.predict([i]))
        ret.append(reg.predict(topr))
    return ret


def kaiser_wind(arr, l, b):
    """
    Convolves a Kaiser Window with the given data (see https://numpy.org/doc/stable/reference/generated/numpy.kaiser.html)

    Args:
        arr: data to operate on
        l: Kaiser window length
        b: Kaiser window constant (beta)

    Returns:
        Smoothed signal
    """
    return np.convolve(arr, np.kaiser(l,b) , 'valid')


if __name__ == "__main__":
    signal      = np.loadtxt("arr.csv", delimiter = ",")
    fig, axs    = plt.subplots(4)
    
    axs[0].plot(signal)
    axs[1].plot(ewma(signal, 0.7))
    axs[2].plot(ewma_bias_corrected(signal, 0.7))
    axs[3].plot(power_smooth(signal, 10))
    plt.show()