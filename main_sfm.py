import cv2
import numpy as np
import matplotlib.pyplot as plt
import open3d
from helper import * 


def main():
    path1 = "./Data/0000.jpg"
    image = cv2.imread(path1, cv2.IMREAD_COLOR)

    if image is None:
        print("Image not found")

    kp, des = extract_keypoints(image)

    if kp is None or des is None:
        print("No keypoints found")

    img2 = cv2.drawKeypoints(image, kp, None, flags=0)
    plt.imshow(img2)
    if image is None:
        print("Image not found")

    #image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
   # plt.imshow(image)
    plt.show()

if __name__ == "__main__":
    main()