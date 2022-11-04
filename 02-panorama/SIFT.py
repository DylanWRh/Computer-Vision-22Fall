import numpy as np
import cv2
import matplotlib.pyplot as plt


def SIFT_stitching(img_1: np.ndarray,
                   img_2: np.ndarray):
    sift = cv2.SIFT_create()
    kp_1, feature_1 = sift.detectAndCompute(img_1, None)
    kp_2, feature_2 = sift.detectAndCompute(img_2, None)

    kp_1 = np.float32([k.pt for k in kp_1])
    kp_2 = np.float32([k.pt for k in kp_2])

    bf = cv2.BFMatcher()
    match = bf.knnMatch(feature_1, feature_2, k=2)
    good_match = []
    for m1, m2 in match:
        if m1.distance < 0.75 * m2.distance:
            good_match.append((m1.trainIdx, m1.queryIdx))
            # good_match.append([m1])

    # return cv2.drawMatchesKnn(img_1, kp_1, img_2, kp_2, good_match, None, flags=2)

    if len(good_match) > 4:
        points_1 = np.float32([kp_1[i] for (_, i) in good_match])
        points_2 = np.float32([kp_2[i] for (i, _) in good_match])
        homo_matrix = cv2.findHomography(points_2, points_1, cv2.RANSAC, 4.0)[0]
        img_ret = np.zeros((img_1.shape[0]+img_2.shape[0], img_1.shape[1]+img_2.shape[1], 3))
        trans = np.array([[1, 0, 200], [0, 1, 200], [0, 0, 1]], dtype=float)
        img_ret_1 = cv2.warpPerspective(img_2, trans.dot(homo_matrix),
                            (img_1.shape[1]+img_2.shape[1], img_1.shape[0]+img_2.shape[0]))
        img_ret_2 = cv2.warpPerspective(img_1, trans,
                            (img_1.shape[1]+img_2.shape[1], img_1.shape[0]+img_2.shape[0]))
        img_ret[img_ret_1 > 0] = img_ret_1[img_ret_1 > 0]
        img_ret[img_ret_2 > 0] = img_ret_2[img_ret_2 > 0]
        return img_ret
    return None


if __name__ == '__main__':
    IMAGE_PATH_1 = 'Analyze/BQ2/1.jpg'
    IMAGE_PATH_2 = 'Analyze/BQ2/2.jpg'
    img_1 = cv2.imread(IMAGE_PATH_1)
    img_2 = cv2.imread(IMAGE_PATH_2)

    img_ret = SIFT_stitching(img_1, img_2)
    cv2.imwrite('res_stitching_SIFT.png', img_ret)

    # stitcher = cv2.Stitcher_create(cv2.Stitcher_PANORAMA)
    # status, stitched = stitcher.stitch((img_1.transpose((1, 0, 2)), img_2.transpose((1, 0, 2))))
    # cv2.imwrite('res_stitching_cv2.png', stitched.transpose((1, 0, 2)))

