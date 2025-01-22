import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def visualize_ply_file_open3d(ply_file_path):
    # Load the point cloud from the PLY file
    pcd = o3d.io.read_point_cloud(ply_file_path)
    
    # Print some information about the point cloud
    print(pcd)
    print(np.asarray(pcd.points))
    
    # Visualize the point cloud
    o3d.visualization.draw_geometries([pcd])

def visualize_ply_file_matplotlib(ply_file_path):
    # Load the point cloud from the PLY file
    pcd = o3d.io.read_point_cloud(ply_file_path)
    points = np.asarray(pcd.points)
    
    # Check if points are empty
    if points.size == 0:
        print("No points to visualize")
        return
    
    # Plot the points using Matplotlib
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='r', marker='o')
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    plt.show()

if __name__ == "__main__":
    ply_file_path = "output.ply"  # Path to your PLY file
    
    # Visualize using Open3D
    visualize_ply_file_open3d(ply_file_path)
    
    # Visualize using Matplotlib
    visualize_ply_file_matplotlib(ply_file_path)