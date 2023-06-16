import numpy as np
from matplotlib import pyplot as plt
from sklearn.neighbors import LocalOutlierFactor
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.arima.model import ARIMA


def lof(signal, n: int):
    """
    Compute Local Outlier Factor (LoF) scoring

    Args:
        signal: Data to be scored
        n: Number of neighbours for LoF

    Returns:
        LoF scores
    """
    lof_model = LocalOutlierFactor(n_neighbors = n)
    lof_model.fit_predict(signal.reshape(-1, 1))
    return np.abs(lof_model.negative_outlier_factor_)


def arima_outlier(signal):
    """
    Compute ARIMA residuals

    Args:
        signal: Data to be scored
        
    Returns:
        ARIMA residuals
    """
    model   = ARIMA(signal, order = (15,2,3), enforce_invertibility = False)
    res     = model.fit()
    return np.abs(res.resid)


def pca_outliar(arr, n):
    """
    Compute PCA residuals. See (https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html) for additional details

    Args:
        arr: Data to be processed
        n: Used as n_components parameter for PCA object instantiation. See above link for details

    Returns:
        Mean squared error reconstruction loss
    """
    arr_pca = arr
    pca     = PCA(n_components=n)
    pca.fit(arr_pca)
        
    transformed_pca = pca.transform(arr_pca)
    projected_pca   = pca.inverse_transform(transformed_pca)
    loss            = np.sum((arr_pca - projected_pca) ** 2, axis=1)
    return loss


def polyreg_outliar_mse(arrs, n):
    """
    Compute fitted linear regressor

    Args:
        arrs: 2D array where each row is a time series
        n: Length of window to use for autoregressor

    Returns:
        Fitted linear regression model
    """
    windows = []
    labels  = []
    for arr in arrs:
        arr = np.array(arr).reshape((len(arr), ))
        for i in range(len(arr) - n):
            windows.append(arr[i : i + n])
            labels.append(arr[i + n])
    
    linreg = LinearRegression()
    linreg.fit(windows, labels)
    return linreg

    
if __name__ == "__main__":
    signal  = np.loadtxt("arr.csv", delimiter = ",")
    signal2 = np.loadtxt("arr2.csv", delimiter = ",")
    np_fft  = np.fft.fft(signal2)
    np_fft  = np_fft - 0.05
    
    fig, axs = plt.subplots(3)
    rest = np.fft.ifft(np_fft)
    axs[0].plot(rest)
    axs[1].plot(signal2)
    axs[2].plot(rest-signal2)
    plt.show()
    
    mnmx = MinMaxScaler(feature_range=(-1,1))
    mnmx.fit(signal.reshape(-1, 1))
    signal = mnmx.transform(signal.reshape(-1, 1))
    
    signal2     = np.concatenate((np.zeros((len(signal)-len(signal2),), dtype=int), np.array(signal2)))
    fig, axs    = plt.subplots(4)
    axs[0].plot(signal)
    axs[1].plot(lof(signal, 100))
    axs[2].plot(np.abs(polyreg_outliar_mse(signal, 15)))
    axs[3].plot(np.abs(polyreg_outliar_mse(signal, 150)))
    plt.show()
