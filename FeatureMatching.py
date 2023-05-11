import os
import cv2
import numpy as np


def matchfeatures(src, tgt, nfeatures=1000, verbose=False):

    orb = cv2.ORB_create(nfeatures=nfeatures, scoreType=cv2.ORB_FAST_SCORE)
    kp1, des1 = orb.detectAndCompute(src, None)
    kp2, des2 = orb.detectAndCompute(tgt, None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)

    matches = sorted(matches, key=lambda x: x.distance)

    common_points = []
    for match in matches:
        x1y1 = kp1[match.queryIdx].pt
        x2y2 = kp2[match.trainIdx].pt
        feature = list(map(int, list(x1y1) + list(x2y2) + [match.distance]))
        common_points.append(feature)

    return np.array(common_points)

