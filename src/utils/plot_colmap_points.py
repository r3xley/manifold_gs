"""
Simple script to load and plot/print COLMAP points
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from data.load_colmap import load_colmap_points

scene_root = "data/nerf_llff_data/fern"

print("Loading COLMAP points...")
try:
    # Load tiny sample
    points_3d, colors, point_ids, _ = load_colmap_points(scene_root, max_points=100)
    print(f"✓ Loaded {len(points_3d)} points\n")
except Exception as e:
    print(f"❌ Error: {e}")
    exit(1)

# Print values
print("="*70)
print("COLMAP POINT VALUES")
print("="*70)
print(f"First 10 points:")
for i in range(min(10, len(points_3d))):
    print(f"  Point {i}: pos=({points_3d[i, 0]:.3f}, {points_3d[i, 1]:.3f}, {points_3d[i, 2]:.3f}), "
          f"color=({colors[i, 0]:.3f}, {colors[i, 1]:.3f}, {colors[i, 2]:.3f})")

print(f"\nAll points summary:")
print(f"  Count: {len(points_3d)}")
print(f"  Position range: X=[{points_3d[:, 0].min():.3f}, {points_3d[:, 0].max():.3f}], "
      f"Y=[{points_3d[:, 1].min():.3f}, {points_3d[:, 1].max():.3f}], "
      f"Z=[{points_3d[:, 2].min():.3f}, {points_3d[:, 2].max():.3f}]")
print(f"  Color range: R=[{colors[:, 0].min():.3f}, {colors[:, 0].max():.3f}], "
      f"G=[{colors[:, 1].min():.3f}, {colors[:, 1].max():.3f}], "
      f"B=[{colors[:, 2].min():.3f}, {colors[:, 2].max():.3f}]")

# Simple plot
print(f"\nCreating plot...")
fig = plt.figure(figsize=(12, 4))

# 3D plot
ax1 = fig.add_subplot(131, projection='3d')
ax1.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2], 
           c=colors, s=50, alpha=0.8)
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Z')
ax1.set_title('3D COLMAP Points')

# XY
ax2 = fig.add_subplot(132)
ax2.scatter(points_3d[:, 0], points_3d[:, 1], c=colors, s=50, alpha=0.8)
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_title('XY Projection')
ax2.grid(True, alpha=0.3)

# XZ
ax3 = fig.add_subplot(133)
ax3.scatter(points_3d[:, 0], points_3d[:, 2], c=colors, s=50, alpha=0.8)
ax3.set_xlabel('X')
ax3.set_ylabel('Z')
ax3.set_title('XZ Projection')
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('outputs/colmap_points.png', dpi=100, bbox_inches='tight')
print("✓ Saved to: outputs/colmap_points.png")
plt.show()

print("\n✅ Done!")
