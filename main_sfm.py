import cv2
import numpy as np
import matplotlib.pyplot as plt
import open3d
from helper import * 


def main():
    path1 = "./Data/0000.jpg"
    path2 = "./Data/0001.jpg"
    image1 = cv2.imread(path1, cv2.IMREAD_COLOR)
    image2 = cv2.imread(path2, cv2.IMREAD_COLOR)

    if image1 is None or image2 is None:
        print("Image not found")

    kp1, des1 = extract_keypoints(image1)
    kp2, des2 = extract_keypoints(image2)


    if kp1 is None or des1 is None:
        print("No keypoints found")
    if kp2 is None or des2 is None:
        print("No keypoints found")

    matches = match_keypoints(des1, des2)
    if matches is None:
        print("No matches found")

    # Draw matches
    img_matches = cv2.drawMatches(image1, kp1, image2, kp2, matches[:20], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    img_matches_rgb = cv2.cvtColor(img_matches, cv2.COLOR_BGR2RGB)
    plt.imshow(img_matches_rgb)
    plt.show()

    

    #img2 = cv2.drawKeypoints(image, kp, None, flags=0)

   #image_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    #plt.imshow(image_rgb)
   # plt.show()

if __name__ == "__main__":
    main()