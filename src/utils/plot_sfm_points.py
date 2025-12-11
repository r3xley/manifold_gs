"""
Plot COLMAP/SFM 3D points
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import sys

# Add project root to path for imports
_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.join(_script_dir, '../..')
sys.path.insert(0, os.path.abspath(_project_root))

from data.load_colmap import load_colmap_points

def plot_sfm_points(scene_root, max_points=10000, save_path=None, show=True):
    """
    Plot COLMAP/SFM 3D points
    
    Args:
        scene_root: Path to scene directory
        max_points: Maximum number of points to load
        save_path: Optional path to save figure
        show: Whether to display the plot
    """
    print(f"Loading COLMAP points from {scene_root}...")
    
    try:
        points, colors, ids, _ = load_colmap_points(scene_root, max_points=max_points)
        print(f"✓ Loaded {len(points)} points")
    except Exception as e:
        print(f"❌ Error loading COLMAP points: {e}")
        return
    
    # Create 3D plot
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot points colored by their RGB values
    # Colors are already normalized to [0, 1]
    ax.scatter(points[:, 0], 
               points[:, 1], 
               points[:, 2], 
               c=colors,  # Use RGB colors from COLMAP
               s=10, 
               alpha=0.6,
               edgecolors='none')
    
    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'COLMAP/SFM 3D Points: {os.path.basename(scene_root)}\n({len(points)} points)')
    
    # Set equal aspect ratio
    max_range = np.array([points[:, 0].max() - points[:, 0].min(),
                          points[:, 1].max() - points[:, 1].min(),
                          points[:, 2].max() - points[:, 2].min()]).max() / 2.0
    mid_x = (points[:, 0].max() + points[:, 0].min()) * 0.5
    mid_y = (points[:, 1].max() + points[:, 1].min()) * 0.5
    mid_z = (points[:, 2].max() + points[:, 2].min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    # Print statistics
    print(f"\nPoint cloud statistics:")
    print(f"  Number of points: {len(points)}")
    print(f"  X range: [{points[:, 0].min():.3f}, {points[:, 0].max():.3f}]")
    print(f"  Y range: [{points[:, 1].min():.3f}, {points[:, 1].max():.3f}]")
    print(f"  Z range: [{points[:, 2].min():.3f}, {points[:, 2].max():.3f}]")
    print(f"  Color range: R=[{colors[:, 0].min():.3f}, {colors[:, 0].max():.3f}], "
          f"G=[{colors[:, 1].min():.3f}, {colors[:, 1].max():.3f}], "
          f"B=[{colors[:, 2].min():.3f}, {colors[:, 2].max():.3f}]")
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\n✓ Saved plot to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()

if __name__ == "__main__":
    scene_root = "data/nerf_llff_data/fern"
    if len(sys.argv) > 1:
        scene_root = sys.argv[1]
    
    max_points = 10000
    if len(sys.argv) > 2:
        max_points = int(sys.argv[2])
    
    save_path = "outputs/sfm_points.png"
    plot_sfm_points(scene_root, max_points=max_points, save_path=save_path, show=True)
