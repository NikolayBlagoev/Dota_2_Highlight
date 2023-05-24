import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from tqdm import tqdm
def ewma_f(value, prev_value, alpha):
    return alpha*value + (1-alpha)*prev_value

def ewma(arr, alpha):
    prev = 0
    ret = []
    for it in arr:
        prev = ewma_f(it, prev, alpha)
        ret.append(prev)
    return ret


def ewma_bias_corrected_f(value, prev_value, alpha, t):
    return (alpha*value + (1-alpha)*prev_value)/(1-alpha**t)


def ewma_bias_corrected(arr, alpha):
    prev = 0
    ret = []
    for t,it in enumerate(arr):
        prev = ewma_bias_corrected_f(it, prev, alpha, t+1)
        ret.append(prev)
    return ret

def linear_smooth(arr, n):
    return np.convolve(arr,[1/n for i in range(n)], 'valid')
def power_smooth(arr, n):
    return np.convolve(arr,[2**abs(i-int(n/2)) for i in range(n)], 'valid')

def llr_smooth(arr, n):
    reg = LinearRegression()
    ret = []
    loc = np.array([i for i in range(n)]).reshape(-1, 1)
    topr = np.array([int(n/2)]).reshape(1,-1)
    for t, _ in tqdm(enumerate(arr)):
        if t < int(n/2) or t > len(arr)-int(n/2):
            continue
        reg.fit(loc, arr[t-int(n/2):t+int(n/2)])
        if t < int(n/2):
            for i in range(int(n/2)):
                ret.append(reg.predict([i]))
        ret.append(reg.predict(topr))
    return ret
def derivative(signal):
    ret = []
    for i in range(len(signal)-1):
        ret.append(signal[i+1]-signal[i])
    return ret
if __name__ == "__main__":
    signal = [145, 281, 156, 66, 58, 242, 332, 972, 623, 389, 1299, 191, 1239, 440, 1982, 1479, 1050, 1567, 939, 1266, 364, 1251, 312, 1454, 731, 278, 1272, 320, 1405, 974, 1421, 497, 1169, 775, 269, 1324, 1100, 1775, 442, 1417, 1071, 2276, 1351, 376, 1841, 1085, 2051, 1178, 749, 749, 450, 1415, 137, 1098, 220, 1347, 1265, 591, 845, 1220, 2119, 832, 1251, 475, 1273, 1992, 970, 377, 447, 1476, 72, 1020, 606, 1464, 1364, 1109, 769, 805, 2210, 546, 1447, 334, 1713, 1199, 2057, 1596, 169, 1976, 690, 1382, 1640, 1969, 1773, 259, 1570, 1894, 1705, 875, 67, 1766, 1953, 1729, 339, 1205, 1013, 231, 1707, 651, 220, 1416, 1415, 262, 1330, 69, 983, 513, 183, 1134, 1270, 1271, 1362, 1032, 1024, 864, 1290, 693, 1089, 777, 553, 278, 334, 467, 326, 469, 561, 498, 626, 542, 496, 1693, 1768, 430, 508, 340, 83, 195, 392, 682, 624, 1465, 1408, 1129, 1269, 1018, 755, 246, 85, 81, 30, 30, 147, 207, 202, 99, 40, 87, 144, 28, 183, 515, 1397, 582, 886, 808, 327, 1036, 717, 976, 576, 504, 814, 791, 757, 495, 141, 83, 353, 421, 1798, 1097, 564, 422, 327, 353, 200, 83, 61, 47, 59, 30, 38, 39, 247, 816, 597, 1038, 562, 622, 1005, 272, 494, 172, 432, 78, 1302, 1172, 110, 1473, 66, 1431, 139, 1373, 357, 53, 24, 31, 128, 654, 578, 149, 924, 1714, 1809, 934, 1665, 1055, 1344, 1318, 1412, 1094, 1291, 1019, 783, 1183, 805, 1066, 1126, 435, 539, 1019, 1363, 811, 617, 664, 286, 816, 577, 1370, 1220, 1573, 776, 452, 679, 130, 429, 84, 460, 886, 1647, 2049, 899, 1542, 1349, 1537, 1411, 368, 995, 163, 426, 1473, 1709, 1588, 1631, 427, 69, 729, 342, 554, 383, 1436, 1066, 163, 1056, 163, 479, 578, 248, 256, 58, 66, 694, 777, 854, 461, 1025, 941, 700, 1293, 898, 820, 732, 1186, 1410, 182, 724, 207, 705, 74, 296, 210, 242, 378, 127, 890, 265, 932, 304, 749, 280, 171, 356, 81, 80, 94, 98, 328, 271, 560, 263, 329, 126, 659, 229, 110, 503, 242, 126, 244, 98, 65, 300, 7, 3, 10, 8, 7, 11, 13, 7, 12, 8, 9, 5, 5, 13, 11, 3, 10, 13, 5, 8, 3, 13, 205, 218, 271, 586, 82, 909, 833, 1004, 659, 1187, 1271, 392, 277, 205, 1071, 116, 1277, 345, 969, 1339, 497, 1863, 299, 951, 181, 81, 77, 806, 78, 941, 83, 1138, 199, 962, 418, 114, 100, 73, 46, 35, 35, 13, 934, 67, 1125, 317, 215, 67, 1222, 89, 1026, 771, 1324, 431, 241, 1463, 71, 33, 39, 50, 51, 38, 26, 11, 4, 6, 8, 7, 9, 11, 61, 71, 645, 1262, 210, 150, 47, 9, 6, 80, 37, 64, 11, 8, 21, 4, 3, 7, 4, 7, 4, 7, 3, 2, 94, 448, 126, 92, 94, 238, 938, 85, 391, 294, 66, 70, 63, 1028, 224, 40, 592, 476, 54, 997, 172, 882, 110, 765, 228, 325, 373, 150, 813, 163, 108, 78, 53]
    signal = np.loadtxt("arr.csv", delimiter = ",")
    plt.acorr(derivative(signal, 0.05), maxlags = 40)
    plt.grid(True)
    plt.show()
    fig, axs = plt.subplots(5)
    axs[0].plot(signal)
    axs[1].plot(ewma(signal, 0.7))
    axs[2].plot(ewma_bias_corrected(signal, 0.7))
    axs[3].plot(llr_smooth(signal,10))
    axs[4].plot(power_smooth(signal,10))
    plt.show()