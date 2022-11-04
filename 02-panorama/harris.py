import numpy as np
import cv2
from func import convolve_2d
import matplotlib.pyplot as plt


def gradient_x(img: np.ndarray):
    sobel_kernel_x = np.array([
        [1, 0, -1],
        [2, 0, -2],
        [1, 0, -1]
    ])
    grad_x = convolve_2d(img, sobel_kernel_x)
    return grad_x


def gradient_y(img: np.ndarray):
    sobel_kernel_y = np.array([
        [ 1,  2,  1],
        [ 0,  0,  0],
        [-1, -2, -1]
    ])
    grad_y = convolve_2d(img, sobel_kernel_y)
    return grad_y


def harris_response(img: np.ndarray,
                    alpha: float = 0.04,
                    window_size: int = 3):
    W, H = img.shape
    Ix, Iy = gradient_x(img), gradient_y(img)
    Ixx, Ixy, Iyy = Ix * Ix, Ix * Iy, Iy * Iy
    R = np.zeros((W, H))
    kernel = np.ones((window_size, window_size))
    A = convolve_2d(Ixx, kernel)
    B = convolve_2d(Ixy, kernel)
    C = convolve_2d(Iyy, kernel)
    det = A * C - B ** 2
    trace = A + C
    R = det - alpha * trace ** 2
    return R


def corner_selection(R: np.ndarray,
                     threshold: float = 0.05,
                     min_distance: int = 1):
    W, H = R.shape
    max_R = np.max(R)
    R[R < threshold * max_R] = 0
    x_slots, y_slots = np.where(R > 0)
    for x, y in zip(x_slots, y_slots):
        if R[x, y] == 0:
            continue
        for i in range(max(0, x-min_distance), min(W, x+min_distance+1)):
            for j in range(max(0, y-min_distance), min(H, y+min_distance+1)):
                if R[i, j] < R[x, y]:
                    R[i, j] = 0
    x_slots, y_slots = np.where(R > 0)
    pixels = [(i, j) for i, j in zip(x_slots, y_slots)]
    return pixels


if __name__ == '__main__':
    IMAGE_PATH = 'images\\1_1.jpg'
    img = cv2.imread(IMAGE_PATH)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    H, W = img_gray.shape
    img_ = np.zeros((H, 2*W, 3))

    imgcpy = img.copy()
    R = harris_response(img_gray)
    pixels = corner_selection(R)
    for i, j in pixels:
        imgcpy[i, j] = np.array([0, 0, 255])
    img_[:, :W, (2, 1, 0)] = imgcpy / 255

    imgcpy = img.copy()
    R_cv2 = cv2.cornerHarris(img_gray, blockSize=3, ksize=3, k=0.04)
    pixels_cv2 = corner_selection(R_cv2)
    for i, j in pixels_cv2:
        imgcpy[i, j] = np.array([0, 0, 255])
    img_[:, W:, (2, 1, 0)] = imgcpy / 255

    plt.imshow(img_)
    plt.show()
