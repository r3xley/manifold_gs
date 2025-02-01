import cv2
import numpy as np
import matplotlib.pyplot as plt
import open3d
from helper import * 


def main():
    image_paths = [f"./Data/{i:04d}.jpg" for i in range(10)]  # Adjust the range as needed
    images = [cv2.imread(path, cv2.IMREAD_COLOR) for path in image_paths]

    # Check if all images are loaded
    for i, image in enumerate(images):
        if image is None:
            print(f"Image {image_paths[i]} not found")
            return

    matrix = load_intrinsic_matrix("./Data/intrinsic_matrix.txt")
    print(matrix)
    K = np.array(matrix).reshape(3, 3)
    print(K)

    kp_list = []
    des_list = []
    for image in images:
        kp, des = extract_keypoints(image)
        kp_list.append(kp)
        des_list.append(des)

    # Match features between images
    matches = []
    for i in range(len(images) - 1):
        matches.append(match_keypoints(des_list[i], des_list[i + 1]))

    # Estimate the fundamental matrix



if __name__ == "__main__":
    main()