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
    #plt.imshow(img_matches_rgb)
    #plt.show()

    with open("./Data/intrinsic_matrix.txt", 'r') as file:
        lines = file.readlines()
    
    # Extract the matrix values from the custom format
    matrix_str = ''.join(lines).replace('K = [', '').replace(']', '').replace(';', '').strip()
    #print(matrix_str)
    matrix_values = [float(num) for num in matrix_str.split()]
    #print(matrix_values)
    matrix = np.array(matrix_values).reshape(3, 3)
    #print(matrix)

    E, mask, pts1, pts2 = essential_matrix(kp1, kp2, matches, matrix)
    #print(E)
    #print(mask)
    # Draw epipolar lines
    lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1, 1, 2), 2, E)
    lines1 = lines1.reshape(-1, 3)
    img1_epilines, img2_points = draw_epipolar_lines(image1, image2, lines1, pts1, pts2)

    lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1, 1, 2), 1, E)
    lines2 = lines2.reshape(-1, 3)
    img2_epilines, img1_points = draw_epipolar_lines(image2, image1, lines2, pts2, pts1)

    plt.figure(figsize=(15, 5))
    plt.subplot(121), plt.imshow(img1_epilines)
    plt.subplot(122), plt.imshow(img2_epilines)
    plt.show()

    #plt.imshow(mask)
    #plt.show()


    #img2 = cv2.drawKeypoints(image, kp, None, flags=0)

   #image_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    #plt.imshow(image_rgb)
   # plt.show()

if __name__ == "__main__":
    main()