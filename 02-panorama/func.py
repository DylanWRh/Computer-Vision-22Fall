import numpy as np


def zero_padding_2d(img: np.ndarray,
                    padding_size: tuple):
    h, w = img.shape
    hps, wps = padding_size
    res = np.zeros((h+hps*2, w+wps*2))
    res[hps:h+hps, wps:w+wps] = img
    return res


def reflection_padding_2d(img: np.ndarray,
                          padding_size: tuple):
    h, w = img.shape
    hps, wps = padding_size
    res = np.zeros((h+hps*2, w+wps*2))
    res[hps:h+hps, wps:w+wps] = img
    res[:hps, :] = res[hps+hps-1:hps-1:-1, :]
    res[h+hps:, :] = res[h+hps-1:h-1:-1, :]
    res[:, :wps] = res[:, wps+wps-1:wps-1:-1]
    res[:, w+wps:] = res[:, w+wps-1:w-1:-1]
    return res


def cross_correlation_2d(img: np.ndarray,
                         kernel: np.ndarray):
    h, w = img.shape
    padding_size_h, padding_size_w = kernel.shape
    padding_size = (padding_size_h//2, padding_size_w//2)
    img_pad = reflection_padding_2d(img, padding_size)
    hk, wk = kernel.shape
    res = np.zeros(img.shape)
    for i in range(hk):
        for j in range(wk):
            res += img_pad[i:i+h, j:j+w] * kernel[i, j]
    return res


def convolve_2d(img: np.ndarray,
                kernel: np.ndarray):
    return cross_correlation_2d(img, kernel[::-1, ::-1])
