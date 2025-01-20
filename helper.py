import cv2
import numpy as np

def extract_keypoints(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    kp, des = sift.detectAndCompute(gray, None)
    return kp, des

def match_keypoints(des1, des2):
    bf = cv2.BFMatcher()
    matches = bf.match(des1, des2)
    matches = sorted(matches, key = lambda x:x.distance)

    return matches

def fund_matrix(kp1, kp2, matches):
    pts1 = []
    pts2 = []
    for match in matches:
        pts1.append(kp1[match.queryIdx].pt)
        pts2.append(kp2[match.trainIdx].pt)
    
    pts1 = np.int32(pts1)
    pts2 = np.int32(pts2)

    F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_LMEDS)
    return F, mask, pts1, pts2