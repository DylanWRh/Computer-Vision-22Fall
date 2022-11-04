import numpy as np
import cv2
from harris import harris_response, corner_selection
from histogram import histogram_of_gradients
import matplotlib.pyplot as plt


def feature_matching(img_1: np.ndarray,
                     img_2: np.ndarray):
    R_1 = harris_response(img_1)
    R_2 = harris_response(img_2)
    pix_1 = corner_selection(R_1)
    pix_2 = corner_selection(R_2)
    N_1 = len(pix_1)
    N_2 = len(pix_2)
    features_1 = histogram_of_gradients(img_1, pix_1)
    features_2 = histogram_of_gradients(img_2, pix_2)
    print(N_1, N_2)
    threshold = 0.8
    threshold_2 = 0.55
    pixels_1 = []
    pixels_2 = []
    for i in range(N_1):
        min_dist = np.inf
        second_min_dist = np.inf
        min_j = -1
        for j in range(N_2):
            vec_1 = features_1[i, :]
            vec_2 = features_2[j, :]
            vec_diff = vec_1 - vec_2
            dist = np.sqrt(np.sum(vec_diff*vec_diff))
            if dist < min_dist:
                min_dist = dist
                min_j = j
            elif dist < second_min_dist:
                second_min_dist = dist
        if min_dist <= threshold and min_dist <= threshold_2*second_min_dist:
            pixels_1.append(pix_1[i])
            pixels_2.append(pix_2[min_j])

    return pixels_1, pixels_2


if __name__ == '__main__':
    IMAGE_PATH_1 = 'images\\1_1.jpg'
    IMAGE_PATH_2 = 'images\\1_2.jpg'
    img_1 = cv2.imread(IMAGE_PATH_1)
    img_2 = cv2.imread(IMAGE_PATH_2)

    img_gray_1 = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
    img_gray_2 = cv2.cvtColor(img_2, cv2.COLOR_BGR2GRAY)

    pixels_1, pixels_2 = feature_matching(img_gray_1, img_gray_2)

    H_1, W_1 = img_gray_1.shape
    H_2, W_2 = img_gray_2.shape
    img = np.zeros((max(H_1, H_2), W_1 + W_2, 3))
    img[:H_1, :W_1, (2, 1, 0)] = img_1 / 255
    img[:H_2, W_1:, (2, 1, 0)] = img_2 / 255
    plt.imshow(img)

    N = len(pixels_1)
    for i in range(N):
        x1, y1 = pixels_1[i]
        x2, y2 = pixels_2[i]
        plt.plot([y1, y2+W_1], [x1, x2])

    plt.show()
