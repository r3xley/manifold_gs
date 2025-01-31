import cv2
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

def load_intrinsic_matrix(filepath):
    with open(filepath, 'r') as file:
        lines = file.readlines()
    
    # Extract the matrix values from the custom format
    matrix_str = ''.join(lines).replace('K = [', '').replace(']', '').replace(';', '').strip()
    matrix_values = [float(num) for num in matrix_str.split()]
    matrix = np.array(matrix_values).reshape(3, 3)
    return matrix

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

#
#def fund_matrix(kp1, kp2, matches):
#    pts1 = []
#    pts2 = []
#    for match in matches:
#        pts1.append(kp1[match.queryIdx].pt)
#        pts2.append(kp2[match.trainIdx].pt)
#    
#    pts1 = np.int32(pts1)
#    pts2 = np.int32(pts2)
#
#    F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_LMEDS)
#

def essential_matrix(kp1, kp2, matches, K):
    pts1 = []
    pts2 = []
    for match in matches:
        pts1.append(kp1[match.queryIdx].pt)
        pts2.append(kp2[match.trainIdx].pt)

    pts1 = np.int32(pts1)
    pts2 = np.int32(pts2)
    E, mask = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)

    return E, mask, pts1, pts2

def recover_pose(E, K, pts1, pts2):
    _, R, t, mask = cv2.recoverPose(E, pts1, pts2, K)
    return R, t

def draw_epipolar_lines(img1, img2, lines, pts1, pts2):
    ''' img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines '''
    r, c = img1.shape[:2]
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    for r, pt1, pt2 in zip(lines, pts1, pts2):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        x0, y0 = map(int, [0, -r[2] / r[1]])
        x1, y1 = map(int, [c, -(r[2] + r[0] * c) / r[1]])
        img1 = cv2.line(img1, (x0, y0), (x1, y1), color, 1)
        img1 = cv2.circle(img1, tuple(pt1), 5, color, -1)
        img2 = cv2.circle(img2, tuple(pt2), 5, color, -1)
    return img1, img2

def triangulate_points(points1, points2, P1, P2):
    points4D = cv2.triangulatePoints(P1, P2, points1.T, points2.T)
    points3D = points4D / points4D[3]
    return points3D[:3].T

def save_points_to_ply(points, filename):
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    o3d.io.write_point_cloud(filename, point_cloud)

def visualize_points(points):
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    o3d.visualization.draw_geometries([point_cloud])

def plot_points_matplotlib(points):
    if points.size == 0:
        print("No points to plot")
        return

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='r', marker='o')
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    plt.show()