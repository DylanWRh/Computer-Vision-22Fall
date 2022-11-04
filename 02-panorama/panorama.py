import numpy as np
import cv2
from stitching import compute_homo_matrices, compute_bounding_box
import time


def generate_panorama_gray(ordered_img_seq: list):
    N = len(ordered_img_seq)
    mid = N // 2

    homo_matrices = compute_homo_matrices(ordered_img_seq, mid)

    maxx, maxy, minx, miny = compute_bounding_box(ordered_img_seq, homo_matrices, mid)

    print('---------------------Generating Images---------------------')
    start_time = time.process_time()
    est_pano = np.zeros((maxx, maxy))
    for x in range(maxx):
        for t in range(100):
            if x == maxx // 100 * t:
                print('[' + '='*(t//2) + '>' + '.'*(50-t//2) + f']  Finished {t}%, time consumption: {time.process_time() - start_time}')
        for y in range(maxy):
            for i in range(N):
                point = np.array([x+minx, y+miny, 1])
                des = homo_matrices[i].dot(point)
                des_x = des[0] / des[2]
                des_y = des[1] / des[2]
                H_i, W_i = ordered_img_seq[i].shape
                if not (0 <= des_x < H_i-1 and 0 <= des_y < W_i-1):
                    continue

                lx, ly = int(des_x), int(des_y)
                rx, ry = lx + 1, ly + 1
                dx, dy = des_x - lx, des_y - ly
                val = ordered_img_seq[i][lx, ly] * (1 - dx) * (1 - dy) \
                      + ordered_img_seq[i][lx, ry] * (1 - dx) * dy \
                      + ordered_img_seq[i][rx, ly] * dx * (1 - dy) \
                      + ordered_img_seq[i][rx, ry] * dx * dy
                est_pano[x, y] = val
    print('[==================================================>]  Finished 100%')
    print('-------------------------Finished-------------------------')
    return est_pano


def generate_panorama(ordered_img_seq: list):
    N = len(ordered_img_seq)
    mid = N // 2

    ordered_img_seq_gray = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in ordered_img_seq]

    homo_matrices = compute_homo_matrices(ordered_img_seq_gray, mid)

    maxx, maxy, minx, miny = compute_bounding_box(ordered_img_seq_gray, homo_matrices, mid)

    print('---------------------Generating Images---------------------')
    start_time = time.process_time()
    est_pano = np.zeros((maxx, maxy, 3))
    print(est_pano.shape)
    for x in range(maxx):
        for t in range(100):
            if x == maxx // 100 * t:
                print('[' + '=' * (t // 2) + '>' + '.' * (
                            50 - t // 2) + f']  Finished {t}%, time consumption: {time.process_time() - start_time}')
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
                est_pano[x, y] = val
    print('[==================================================>]  Finished 100%')
    print('-------------------------Finished-------------------------')
    return est_pano


if __name__ == '__main__':
    ordered_img_seq = []
    for i in range(7):
        ordered_img_seq.append(cv2.imread(f'images\\panoramas\\parrington\\prtn0{i}.jpg'))

    ordered_img_seq_gray = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in ordered_img_seq]

    start_time = time.process_time()

    # est_pano_gray = generate_panorama_gray(ordered_img_seq_gray)
    # cv2.imwrite('res_panorama_gray.jpg', est_pano_gray)

    est_pano = generate_panorama(ordered_img_seq)
    cv2.imwrite('results.png', est_pano)

    end_time = time.process_time()
    print(f'Time consumption: {end_time - start_time}')

