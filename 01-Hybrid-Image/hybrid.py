import numpy as np
import cv2
import argparse


def read_file(path):
    return cv2.imread(path, cv2.IMREAD_COLOR)


def save_file(path, img):
    cv2.imwrite(path, img)


def zero_padding(img, padding_size):
    h, w, d = img.shape
    hps, wps = padding_size
    res = np.zeros((h + hps * 2, w + wps * 2, d))
    res[hps:h + hps, wps:w + wps, :] = img
    return res


def cross_correlation_2d(img, kernel):
    h, w, _ = img.shape
    padding_size_h, padding_size_w = kernel.shape
    padding_size = (padding_size_h // 2, padding_size_w // 2)
    img_pad = zero_padding(img, padding_size)
    hk, wk = kernel.shape
    res = np.zeros(img.shape)
    for i in range(hk):
        for j in range(wk):
            res += img_pad[i:i + h,
                           j:j + w, :] * kernel[i, j]
    return res


def convolve_2d(img, kernel):
    return cross_correlation_2d(img, kernel[::-1, ::-1])


def gaussian_blur_kernel_2d(h, w, sigma):
    hc, wc = h // 2, w // 2
    res = np.zeros((h, w))
    sh = 2 * (sigma ** 2)
    sw = 2 * (sigma ** 2)
    for i in range(h):
        for j in range(w):
            x, y = i - hc, j - wc
            res[i, j] = np.exp(-((x ** 2) / sh + (y ** 2) / sw))
    sum_res = np.sum(res)
    res /= sum_res
    return res


def low_pass(img, h, w, sigma):
    kernel = gaussian_blur_kernel_2d(h, w, sigma)
    res = convolve_2d(img, kernel)
    resmax, resmin = np.max(res), np.min(res)
    if resmax != resmin:
        res = (res - resmin) / (resmax - resmin)
    return res


def high_pass(img, h, w, sigma):
    res = img - low_pass(img, h, w, sigma)
    resmax, resmin = np.max(res), np.min(res)
    if resmax != resmin:
        res = (res - resmin) / (resmax - resmin)
    return res


def hybrid_image(left, right, hl, wl, hr, wr, sigmal, sigmar, weight):
    low_left = low_pass(left, hl, wl, sigmal)
    high_right = high_pass(right, hr, wr, sigmar)
    res = low_left * weight + high_right * (1 - weight)
    # save_file(args.left_save, low_left * 255)
    # save_file(args.right_save, high_right * 255)
    save_file(args.hybrid_save, res * 255)


def main():
    left = read_file(args.left)
    right = read_file(args.right)
    left = left / np.max(left)
    right = right / np.max(right)
    hybrid_image(left, right, args.left_gaussian_size_h, args.left_gaussian_size_w,
                 args.right_gaussian_size_h, args.right_gaussian_size_h,
                 args.left_gaussian_sigma, args.right_gaussian_sigma, args.weight)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--left', type=str, default='1_dog.bmp')
    parser.add_argument('--right', type=str, default='1_cat.bmp')
    # parser.add_argument('--left_save', type=str, default='left.png')
    # parser.add_argument('--right_save', type=str, default='right.png')
    parser.add_argument('--hybrid_save', type=str, default='hybrid.png')
    parser.add_argument('--left_gaussian_size_h', type=int, default=15)
    parser.add_argument('--left_gaussian_size_w', type=int, default=15)
    parser.add_argument('--left_gaussian_sigma', type=float, default=15)
    parser.add_argument('--right_gaussian_size_h', type=int, default=13)
    parser.add_argument('--right_gaussian_size_w', type=int, default=13)
    parser.add_argument('--right_gaussian_sigma', type=float, default=13)
    parser.add_argument('--weight', type=float, default=0.5)
    args = parser.parse_args()
    main()
