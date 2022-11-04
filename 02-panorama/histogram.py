import numpy as np
import cv2
from harris import gradient_x, gradient_y
from func import reflection_padding_2d


def histogram_of_gradients(img: np.ndarray,
                           pixels: list):
    window_size = 8
    img_pad = reflection_padding_2d(img, (window_size, window_size))
    x_grad = gradient_x(img_pad)
    y_grad = gradient_y(img_pad)
    magnitude = np.sqrt(x_grad ** 2 + y_grad ** 2)
    orientation = np.arctan2(x_grad, y_grad)
    N = len(pixels)
    features = np.zeros((N, 4*8))
    for idx in range(N):
        x, y = pixels[idx]
        zone_mag = magnitude[x:x+2*window_size, y:y+2*window_size]
        zone_ori = orientation[x:x+2*window_size, y:y+2*window_size]
        zone_ori = zone_ori * 180 / np.pi + 180  # range in [0, 360)
        zone_ori //= 45  # range in [0, 7)
        zone_ori = zone_ori.astype(int)
        zone_ori[zone_ori == 8] = 0
        vec = np.zeros(4*8)
        for p in range(2):
            for q in range(2):
                zone_mag_cut = zone_mag[p*window_size:(p+1)*window_size,
                                        q*window_size:(q+1)*window_size]
                zone_ori_cut = zone_ori[p*window_size:(p+1)*window_size,
                                        q*window_size:(q+1)*window_size]
                for i in range(window_size):
                    for j in range(window_size):
                        vec[(p*2+q)*8+zone_ori_cut[i, j]] += zone_mag_cut[i, j]
        vec = vec / np.sqrt(np.sum(vec * vec))
        features[idx] = vec
    return features

