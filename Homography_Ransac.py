import os
import cv2
import random
import numpy as np
from tqdm import tqdm

from itertools import combinations
from FeatureMatching import matchfeatures


def homography(poc):

    A = []
    for x, y, u, v, distance in poc:
        A.append([x, y, 1, 0, 0, 0, -u*x, -u*y, -u])
        A.append([0, 0, 0, x, y, 1, -v*x, -v*y, -v])
    A = np.asarray(A)

    U, S, Vh = np.linalg.svd(A)

    L = Vh[-1, :] / Vh[-1, -1]
    H = L.reshape(3, 3)

    return H


def ransac(poc, n=30, threshold=2, max_iterations=4000):
    assert(len(poc) > n)

    best_score = 0          
    best_inliers = None    
    best_poc = poc[:n]   

    match_pairs = list(combinations(best_poc, 4))


    random.shuffle(match_pairs)

    for matches in tqdm(match_pairs[:max_iterations]):

        H = homography(matches)

        inliers = []
        count = 0

        for feature in best_poc:
            src = np.ones((3, 1))
            tgt = np.ones((3, 1))
            src[:2, 0] = feature[:2]
            tgt[:2, 0] = feature[2:4]

            tgt_hat = H@src

            if tgt_hat[-1, 0] != 0:
                tgt_hat = tgt_hat/tgt_hat[-1, 0]

                if np.linalg.norm(tgt_hat-tgt) < threshold:
                    count += 1
                    inliers.append(feature)

        if count > best_score:
            best_score = count
            best_inliers = inliers

    best_H = homography(best_inliers)

    return best_H
