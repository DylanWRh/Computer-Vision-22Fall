import numpy as np
import random
import cv2
from matching import feature_matching
import time


def compute_homography(pixels_1: list,
                       pixels_2: list):
    N = len(pixels_1)
    assert N >= 4
    A = np.zeros((2*N, 9))
    for i in range(N):
        x1, y1 = pixels_1[i]
        x2, y2 = pixels_2[i]

        A[2*i, 0] = x1
        A[2*i, 1] = y1
        A[2*i, 2] = 1
        A[2*i, 6] = -x2 * x1
        A[2*i, 7] = -x2 * y1
        A[2*i, 8] = -x2

        A[2*i+1, 3] = x1
        A[2*i+1, 4] = y1
        A[2*i+1, 5] = 1
        A[2*i+1, 6] = -y2 * x1
        A[2*i+1, 7] = -y2 * y1
        A[2*i+1, 8] = -y2
    U, S, V = np.linalg.svd(A)
    homo_matrix = V[8].reshape((3, 3))
    return homo_matrix


def estimate_error(pixels_1: list,
                   pixels_2: list,
                   homo_matrix: np.ndarray):
    N = len(pixels_1)
    pixels_1_array = np.array(pixels_1).T
    pixels_2_array = np.array(pixels_2).T

    homo_coor = np.ones((3, N))
    homo_coor[:2, :] = pixels_1_array
    homo_coor_dest = homo_matrix.dot(homo_coor)

    coor_dest = np.zeros((2, N))
    for i in range(N):
        coor_dest[:, i] = homo_coor_dest[:2, i] / homo_coor_dest[2, i]

    err = coor_dest - pixels_2_array
    err = np.sqrt(np.sum(err*err, axis=0))
    return err


def align_pair(pixels_1: list,
               pixels_2: list):
    N = len(pixels_1)
    assert N >= 4

    iter = 8000
    threshold = 100
    inlier_Nmin = N * 0.4
    inlier_Nmax = N * 0.95

    est_homo = None
    est_err = np.inf

    random_N = 4
    lst_N = list(range(N))
    for _ in range(iter):
        random.shuffle(lst_N)

        random_index = lst_N[:random_N]
        random_pixels_1 = [pixels_1[i] for i in random_index]
        random_pixels_2 = [pixels_2[i] for i in random_index]

        homo_matrix = compute_homography(random_pixels_1, random_pixels_2)

        test_index = lst_N[random_N:]
        test_pixels_1 = []
        test_pixels_2 = []
        for i in test_index:
            test_pixels_1.append(pixels_1[i])
            test_pixels_2.append(pixels_2[i])

        test_err = estimate_error(test_pixels_1, test_pixels_2, homo_matrix)

        test_index = np.array(test_index)
        new_inlier_index = test_index[test_err < threshold]

        new_inlier_N = len(new_inlier_index)

        if new_inlier_N <= inlier_Nmin:
            continue

        inlier_pixels_1 = random_pixels_1[:]
        for i in new_inlier_index:
            inlier_pixels_1.append(pixels_1[i])
        inlier_pixels_2 = random_pixels_2[:]
        for i in new_inlier_index:
            inlier_pixels_2.append(pixels_2[i])

        better_homo = compute_homography(inlier_pixels_1, inlier_pixels_2)
        better_err = estimate_error(inlier_pixels_1, inlier_pixels_2, better_homo)
        better_err = np.mean(better_err)
        if better_err < est_err:
            est_err = better_err
            est_homo = better_homo

        if len(inlier_pixels_1) >= inlier_Nmax:
            break
    return est_homo


def compute_homo_matrices(ordered_img_seq: list,
                          background_index: int):
    N = len(ordered_img_seq)
    homo_matrices = [None] * N
    homo_matrices[background_index] = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

    # print('-------------------Computing Homo Matrix-------------------')
    # homo_matrix_check = ['□'] * N
    # homo_matrix_check[background_index] = '■'
    # out_str = '['
    # for i in range(N):
    #     out_str += f' {homo_matrix_check[i]} '
    # out_str += ']'
    # print(f'Image index {background_index} finished, ' + out_str)

    for i in range(background_index-1, -1, -1):
        start_time = time.process_time()

        pixels_1, pixels_2 = feature_matching(ordered_img_seq[i+1], ordered_img_seq[i])
        homo_matrix = align_pair(pixels_1, pixels_2)
        homo_matrix = homo_matrices[i + 1].dot(homo_matrix)
        homo_matrix /= np.sqrt(np.sum(homo_matrix ** 2))
        homo_matrices[i] = homo_matrix * 10

        end_time = time.process_time()

        # homo_matrix_check[i] = '■'
        # out_str = '['
        # for j in range(N):
        #     out_str += f' {homo_matrix_check[j]} '
        # out_str += ']'
        # print(f'Image index {i} finished, ' + out_str + f', time consumption: {end_time - start_time}')
    for i in range(background_index+1, N):
        start_time = time.process_time()

        pixels_1, pixels_2 = feature_matching(ordered_img_seq[i - 1], ordered_img_seq[i])
        homo_matrix = align_pair(pixels_1, pixels_2)
        homo_matrix = homo_matrices[i - 1].dot(homo_matrix)
        homo_matrix /= np.sqrt(np.sum(homo_matrix ** 2))
        homo_matrices[i] = homo_matrix

        end_time = time.process_time()

        # homo_matrix_check[i] = '■'
        # out_str = '['
        # for j in range(N):
        #     out_str += f' {homo_matrix_check[j]} '
        # out_str += ']'
        # print(f'Image index {i} finished, ' + out_str + f', time consumption: {end_time - start_time}')
    return homo_matrices


def compute_bounding_box(ordered_img_seq: list,
                         ordered_homo_seq: list,
                         background_index: int):
    N = len(ordered_img_seq)

    # print('--------------------Setting Bounding Box--------------------')
    # bounding_box_check = ['□'] * N

    minx, miny = 0, 0
    maxx, maxy = ordered_img_seq[background_index].shape

    for i in range(N):
        if i == background_index:
            # bounding_box_check[i] = '■'
            # out_str = '['
            # for j in range(N):
            #     out_str += f' {bounding_box_check[j]} '
            # out_str += ']'
            # print(f'Image index {i} finished, ' + out_str)
            continue
        H_i, W_i = ordered_img_seq[i].shape
        homo_inv = np.linalg.inv(ordered_homo_seq[i])
        corner_coor = np.array([[0, 0, 1], [0, W_i-1, 1], [H_i-1, 0, 1], [H_i-1, W_i-1, 1]])
        for j in range(4):
            boarder_homo_coor = homo_inv.dot(corner_coor[j])
            boarder_x = boarder_homo_coor[0] / boarder_homo_coor[2]
            boarder_y = boarder_homo_coor[1] / boarder_homo_coor[2]
            minx = min(minx, boarder_x)
            miny = min(miny, boarder_y)
            maxx = max(maxx, boarder_x)
            maxy = max(maxy, boarder_y)

        # bounding_box_check[i] = '■'
        # out_str = '['
        # for j in range(N):
        #     out_str += f' {bounding_box_check[j]} '
        # out_str += ']'
        # print(f'Image index {i} finished, ' + out_str)

    maxx -= minx
    maxy -= miny
    maxx = int(maxx)
    maxy = int(maxy)
    return maxx, maxy, minx, miny


def stitch_blend_gray(img_1: np.ndarray,
                      img_2: np.ndarray,
                      est_homo: np.ndarray):
    identify_homo_matrix = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    maxx, maxy, minx, miny = compute_bounding_box([img_1, img_2], [identify_homo_matrix, est_homo], 0)
    ox, oy = -int(minx), -int(miny)

    H_1, W_1 = img_1.shape
    H_2, W_2 = img_2.shape

    est_img = np.zeros((maxx, maxy))

    for x in range(maxx):
        for y in range(maxy):
            point = np.array([x+minx, y+miny, 1])
            des = est_homo.dot(point)
            des_x = des[0] / des[2]
            des_y = des[1] / des[2]
            if not (0 <= des_x < H_2-1 and 0 <= des_y < W_2-1):
                continue

            lx, ly = int(des_x), int(des_y)
            rx, ry = lx + 1, ly + 1
            dx, dy = des_x - lx, des_y - ly
            val = img_2[lx, ly] * (1 - dx) * (1 - dy) \
                  + img_2[lx, ry] * (1 - dx) * dy \
                  + img_2[rx, ly] * dx * (1 - dy) \
                  + img_2[rx, ry] * dx * dy
            est_img[x, y] = val

    est_img[ox:ox + H_1, oy:oy + W_1] = img_1

    return est_img


def stitch_blend(img_1: np.ndarray,
                 img_2: np.ndarray,
                 est_homo: np.ndarray):
    identify_homo_matrix = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    maxx, maxy, minx, miny = compute_bounding_box([img_1[:, :, 0], img_2[:, :, 0]],
                                                  [identify_homo_matrix, est_homo], 0)
    ox, oy = -int(minx), -int(miny)

    H_1, W_1, _ = img_1.shape
    H_2, W_2, _ = img_2.shape

    est_img = np.zeros((maxx, maxy, 3))

    for x in range(maxx):
        for y in range(maxy):
            point = np.array([x+minx, y+miny, 1])
            des = est_homo.dot(point)
            des_x = des[0] / des[2]
            des_y = des[1] / des[2]
            if not (0 <= des_x < H_2-1 and 0 <= des_y < W_2-1):
                continue

            lx, ly = int(des_x), int(des_y)
            rx, ry = lx + 1, ly + 1
            dx, dy = des_x - lx, des_y - ly
            val = img_2[lx, ly, :] * (1 - dx) * (1 - dy) \
                  + img_2[lx, ry, :] * (1 - dx) * dy \
                  + img_2[rx, ly, :] * dx * (1 - dy) \
                  + img_2[rx, ry, :] * dx * dy
            est_img[x, y, :] = val

    est_img[ox:ox + H_1, oy:oy + W_1, :] = img_1

    return est_img


if __name__ == '__main__':
    IMAGE_PATH_1 = 'images\\3_1.jpg'
    IMAGE_PATH_2 = 'images\\3_2.jpg'
    img_1 = cv2.imread(IMAGE_PATH_1)
    img_2 = cv2.imread(IMAGE_PATH_2)

    img_gray_1 = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
    img_gray_2 = cv2.cvtColor(img_2, cv2.COLOR_BGR2GRAY)

    homo_matrices = compute_homo_matrices([img_gray_1, img_gray_2], 0)
    homo_matrix = homo_matrices[1]

    # est_img_gray = stitch_blend(img_gray_1, img_gray_2, homo_matrix)
    # cv2.imwrite('res_stitching_gray.png', est_img_gray)

    est_img = stitch_blend(img_1, img_2, homo_matrix)
    cv2.imwrite('results\\blend_3.png', est_img)

