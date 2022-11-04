import cv2
import numpy as np
from stitching import compute_homo_matrices, compute_bounding_box
from panorama import generate_panorama
import matplotlib.pyplot as plt


def get_est_panos_color(ordered_img_seq: list):
    N = len(ordered_img_seq)
    mid = N // 2

    ordered_img_seq_gray = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in ordered_img_seq]

    homo_matrices = compute_homo_matrices(ordered_img_seq_gray, mid)

    maxx, maxy, minx, miny = compute_bounding_box(ordered_img_seq_gray, homo_matrices, mid)

    est_panos = [np.zeros((maxx, maxy, 3)) for _ in range(N)]
    for x in range(maxx):
        for y in range(maxy):
            for i in range(N):
                point = np.array([x + minx, y + miny, 1])
                des = homo_matrices[i].dot(point)
                des_x = des[0] / des[2]
                des_y = des[1] / des[2]
                H_i, W_i, _ = ordered_img_seq[i].shape
                if not (0 <= des_x < H_i - 1 and 0 <= des_y < W_i - 1):
                    continue

                lx, ly = int(des_x), int(des_y)
                rx, ry = lx + 1, ly + 1
                dx, dy = des_x - lx, des_y - ly
                val = ordered_img_seq[i][lx, ly] * (1 - dx) * (1 - dy) \
                      + ordered_img_seq[i][lx, ry] * (1 - dx) * dy \
                      + ordered_img_seq[i][rx, ly] * dx * (1 - dy) \
                      + ordered_img_seq[i][rx, ry] * dx * dy
                est_panos[i][x, y] = val
    return est_panos


def primitive_pano(ordered_layer_seq: list):
    N = len(ordered_layer_seq)
    panorama = np.zeros(ordered_layer_seq[0].shape)
    for i in range(N):
        panorama[ordered_layer_seq[i] > 0] = ordered_layer_seq[i][ordered_layer_seq[i] > 0]
    return panorama


def panorama_averaging(ordered_layer_seq: list):
    N = len(ordered_layer_seq)
    H, W, _ = ordered_layer_seq[0].shape
    layer_cnt = ordered_layer_seq.copy()
    for t in range(N):
        layer_cnt[t] = layer_cnt[t].sum(axis=2)
        layer_cnt[t][layer_cnt[t] > 0] = 1
    layer_cnt = sum(layer_cnt)
    res_panorama_aver = np.zeros((H, W, 3))
    for i in range(N):
        res_panorama_aver += ordered_layer_seq[i]
    for i in range(H):
        for j in range(W):
            if layer_cnt[i, j]:
                res_panorama_aver[i, j] /= layer_cnt[i, j]

    return res_panorama_aver


def gaussian_filtering(img: np.ndarray):
    blurred_img = cv2.GaussianBlur(img, (5, 5), 13)
    return blurred_img


def poisson_blending(ordered_layer_seq: list):
    N = len(ordered_layer_seq)
    poisson_img = ordered_layer_seq[0].astype(np.uint8)

    H, W, _ = poisson_img.shape

    for i in range(1, N):
        mask = np.ones((H, W, 3), dtype=np.uint8)
        mask[ordered_layer_seq[i-1] == 0] = 255
        mask[ordered_layer_seq[i] == 0] = 255
        mask[mask == 1] = 0
        mask[mask == 255] = 1

        tmp_layer = ordered_layer_seq[i] * mask
        poisson_img[tmp_layer > 0] = tmp_layer[tmp_layer > 0]

        mask = np.ones((H, W, 3), dtype=np.uint8)
        mask[ordered_layer_seq[i] == 0] = 0
        mask[mask == 1] = 255

        x, y, w, h = cv2.boundingRect(mask[:, :, 0])
        cx = x + w // 2
        cy = y + h // 2

        poisson_img = cv2.seamlessClone(ordered_layer_seq[i].astype(np.uint8), poisson_img, mask, (cx, cy), cv2.NORMAL_CLONE)

    return poisson_img


if __name__ == '__main__':
    # ordered_img_seq = []
    # ordered_img_seq.append(cv2.imread('Analyze\\BQ1\\1_1.jpg'))
    # ordered_img_seq.append(cv2.imread('Analyze\\BQ1\\1_2.jpg'))
    # cv2.imwrite('Analyze\\BQ1\\results\\res_stitching_primitive.jpg', generate_panorama_color(ordered_img_seq))

    # panorama_primitive = cv2.imread('Analyze\\BQ1\\results\\res_stitching_primitive.jpg')
    # panorama_blurred = gaussian_filtering(panorama_primitive)
    # H, W, _ = panorama_blurred.shape
    # panorama_blurred = cv2.resize(panorama_blurred, (W // 3, H // 3))
    # cv2.imwrite('Analyze\\BQ1\\results\\res_stitching_gaussian.jpg', panorama_blurred)

    # ordered_img_seq = []
    # for i in range(1,3):
    #     ordered_img_seq.append(cv2.imread(f'Analyze\\Q4\\{i}.jpg'))
    #
    # est_panos = get_est_panos_color(ordered_img_seq)
    # for i in range(len(est_panos)):
    #     cv2.imwrite(f'res_panorama_layer{i}.png', est_panos[i])
    #
    # ordered_layer_seq = []
    # for i in range(2):
    #     ordered_layer_seq.append(cv2.imread(f'res_panorama_layer{i}.png'))
    # pano = np.zeros(ordered_layer_seq[0].shape)
    # for i in range(2):
    #     pano[ordered_layer_seq[i] > 0] = ordered_layer_seq[i][ordered_layer_seq[i] > 0]
    # cv2.imwrite('Analyze\\Q4\\results\\res_stitching_primitive.jpg', pano)
    #
    # ordered_layer_seq = []
    # for i in range(2):
    #     ordered_layer_seq.append(cv2.imread(f'res_panorama_layer{i}.png'))
    # est_pano = panorama_averaging(ordered_layer_seq)
    # cv2.imwrite('Analyze\\Q4\\results\\res_stitching_averaging.jpg', est_pano)
    #
    # ordered_layer_seq = []
    # for i in range(2):
    #     ordered_layer_seq.append(cv2.imread(f'res_panorama_layer{i}.png'))
    # poisson_panorama = poisson_blending(ordered_layer_seq)
    # cv2.imwrite('Analyze\\Q4\\results\\res_stitching_poisson.jpg', poisson_panorama)

    # primitive_panorama = primitive_pano(ordered_layer_seq)
    # cv2.imwrite('res_panorama_color.png', primitive_panorama)

    pass

