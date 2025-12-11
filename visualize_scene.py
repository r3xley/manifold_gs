"""
Visualize cameras and manifold Gaussians using matplotlib
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from pathlib import Path
from data.load_lff import LLFFDataset
from src.models.manifold_gaussian import ManifoldGaussian3D


def compute_camera_frustum(K, pose, image_size, near=0.1, far=2.0):
    """
    Compute camera frustum corners in world space.
    LLFF uses OpenGL convention: camera looks down -Z axis in camera space.
    
    Args:
        K: 3x3 intrinsic matrix
        pose: 4x4 camera-to-world transform
        image_size: (width, height)
        near: near plane distance
        far: far plane distance
    
    Returns:
        corners: (8, 3) array of frustum corners in world space
    """
    if isinstance(K, torch.Tensor):
        K = K.cpu().numpy()
    if isinstance(pose, torch.Tensor):
        pose = pose.cpu().numpy()
    
    w, h = image_size[0], image_size[1]
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    
    # Camera position and rotation (pose is camera-to-world)
    cam_pos = pose[:3, 3]
    R_c2w = pose[:3, :3]  # Camera-to-world rotation
    
    # Generate frustum corners in camera space
    # OpenGL/LLFF: camera looks down -Z, X right, Y up
    # Image coordinates: X right, Y down
    corners_cam = []
    for z_cam in [-far, -near]:  # Negative Z (camera looks down -Z)
        z_abs = abs(z_cam)
        for x_sign in [-1, 1]:
            for y_sign in [-1, 1]:
                # Image pixel coordinates
                x_pixel = w if x_sign > 0 else 0
                y_pixel = h if y_sign > 0 else 0
                
                # Convert to camera space
                # x_cam = (x_pixel - cx) / fx * z (but z is negative, so use abs)
                x_cam = (x_pixel - cx) / fx * z_abs
                # Y is flipped: image Y increases down, camera Y increases up
                y_cam = -(y_pixel - cy) / fy * z_abs
                corners_cam.append([x_cam, y_cam, z_cam])
    
    corners_cam = np.array(corners_cam)
    
    # Transform to world space: p_world = R_c2w @ p_cam + t
    corners_world = (R_c2w @ corners_cam.T).T + cam_pos[None, :]
    
    return corners_world, cam_pos, R_c2w


def plot_camera_frustum(ax, K, pose, image_size, color='red', alpha=0.3, label=None):
    """
    Plot a camera frustum on a 3D axis.
    
    Args:
        ax: matplotlib 3D axis
        K: 3x3 intrinsic matrix
        pose: 4x4 camera-to-world transform
        image_size: (width, height)
        color: frustum color
        alpha: transparency
        label: optional label for the camera
    """
    corners, cam_pos, R = compute_camera_frustum(K, pose, image_size)
    
    # Define frustum faces (6 faces of a box)
    faces = [
        [0, 1, 3, 2],  # near face
        [4, 5, 7, 6],  # far face
        [0, 1, 5, 4],  # top face
        [2, 3, 7, 6],  # bottom face
        [0, 2, 6, 4],  # left face
        [1, 3, 7, 5],  # right face
    ]
    
    # Plot frustum faces
    for face in faces:
        vertices = corners[face]
        ax.add_collection3d(Poly3DCollection([vertices], alpha=alpha, facecolor=color, edgecolor=color, linewidths=1))
    
    # Plot camera position
    ax.scatter(*cam_pos, c=color, s=100, marker='o', label=label, edgecolors='black', linewidths=1)
    
    # Plot viewing direction (OpenGL/LLFF convention: camera looks down -Z in camera space)
    view_dir_world = -R[:, 2]  # Negative Z axis in camera space (OpenGL convention)
    view_arrow = cam_pos + view_dir_world * 0.5
    ax.plot([cam_pos[0], view_arrow[0]], 
            [cam_pos[1], view_arrow[1]], 
            [cam_pos[2], view_arrow[2]], 
            color=color, linewidth=2, alpha=0.8)


def plot_gaussians(ax, gaussians, max_points=5000, alpha=0.6, scale_factor=1.0):
    """
    Plot manifold Gaussians as colored points.
    
    Args:
        ax: matplotlib 3D axis
        gaussians: ManifoldGaussian3D instance
        max_points: maximum number of points to plot (for performance)
        alpha: point transparency
        scale_factor: scale factor for point sizes
    """
    with torch.no_grad():
        positions = gaussians.pos.cpu().numpy()
        colors = gaussians.get_colors().cpu().numpy()
        opacities = gaussians.get_opacity().cpu().numpy()
        scales = gaussians.get_scales().cpu().numpy()
        
        # Subsample if too many points
        if len(positions) > max_points:
            indices = np.random.choice(len(positions), max_points, replace=False)
            positions = positions[indices]
            colors = colors[indices]
            opacities = opacities[indices]
            scales = scales[indices]
        
        # Compute point sizes based on scales
        point_sizes = np.mean(scales, axis=1) * scale_factor * 100
        
        # Plot points with colors
        scatter = ax.scatter(
            positions[:, 0],
            positions[:, 1],
            positions[:, 2],
            c=colors,
            s=point_sizes,
            alpha=alpha,
            edgecolors='black',
            linewidths=0.5
        )
        
        return scatter


def visualize_scene(dataset, gaussians=None, save_path=None, show=True, scene_root=None):
    """
    Create a 3D visualization of cameras and Gaussians.
    
    Args:
        dataset: LLFFDataset instance
        gaussians: Optional ManifoldGaussian3D instance
        save_path: Optional path to save the figure
        show: Whether to display the figure
        scene_root: Optional path to scene root (for loading poses_bounds.npy)
    """
    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111, projection='3d')
    
    # Load camera poses correctly from poses_bounds.npy (correct method)
    import os
    if scene_root is None:
        scene_root = dataset.scene_root if hasattr(dataset, 'scene_root') else None
    if scene_root is None:
        scene_root = "data/nerf_llff_data/fern"  # fallback
    
    poses_bounds_path = os.path.join(scene_root, "poses_bounds.npy")
    camera_positions = None
    poses_raw = None
    
    if os.path.exists(poses_bounds_path):
        print("Loading camera poses from poses_bounds.npy (correct method)...")
        poses_bounds = np.load(poses_bounds_path)  # (N, 17)
        poses_raw = poses_bounds[:, :15].reshape(-1, 3, 5)  # (N, 3, 5)
        camera_positions = poses_raw[:, :, 3]  # (N, 3) - correct extraction method
        print(f"✓ Loaded {len(camera_positions)} camera positions")
        print(f"  Position range: X=[{camera_positions[:, 0].min():.2f}, {camera_positions[:, 0].max():.2f}], "
              f"Y=[{camera_positions[:, 1].min():.2f}, {camera_positions[:, 1].max():.2f}], "
              f"Z=[{camera_positions[:, 2].min():.2f}, {camera_positions[:, 2].max():.2f}]")
    else:
        print(f"Warning: {poses_bounds_path} not found, using dataset poses")
    
    # Plot cameras
    print("Plotting cameras...")
    camera_colors = plt.cm.tab20(np.linspace(0, 1, len(dataset)))
    
    # Plot camera positions as points (using correct positions from poses_bounds.npy)
    if camera_positions is not None:
        ax.scatter(camera_positions[:, 0], camera_positions[:, 1], camera_positions[:, 2],
                   c='red', s=150, marker='o', 
                   label='Camera centers', edgecolors='black', linewidths=2, alpha=0.9, zorder=10)
    
    # Plot camera frustums using dataset poses (for orientation)
    for i in range(len(dataset)):
        sample = dataset[i]
        K = sample['K']
        pose = sample['pose']
        image_size = (sample['image'].shape[2], sample['image'].shape[1])
        
        color = camera_colors[i][:3]  # RGB only
        label = f'Camera {i}' if i < 5 else None  # Only label first 5
        plot_camera_frustum(ax, K, pose, image_size, color=color, alpha=0.15, label=label)
    
    # Plot Gaussians if provided
    if gaussians is not None:
        print("Plotting Gaussians...")
        plot_gaussians(ax, gaussians, max_points=5000, alpha=0.7, scale_factor=2.0)
    
    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Camera Frustums and Manifold Gaussians')
    
    # Set equal aspect ratio
    # Get bounds from cameras and Gaussians
    all_points = []
    
    # Use camera positions from poses_bounds.npy if available
    if camera_positions is not None:
        all_points.append(camera_positions)
    else:
        # Fallback to dataset poses
        for i in range(len(dataset)):
            sample = dataset[i]
            pose = sample['pose']
            if isinstance(pose, torch.Tensor):
                pose = pose.cpu().numpy()
            cam_pos = pose[:3, 3].reshape(1, 3)
            all_points.append(cam_pos)
    
    if gaussians is not None:
        with torch.no_grad():
            gaussian_pos = gaussians.pos.cpu().numpy()
            all_points.append(gaussian_pos)
    
    # Concatenate all points (all should be 2D arrays now)
    if len(all_points) > 0:
        all_points = np.concatenate(all_points, axis=0)
    else:
        all_points = np.array([[0, 0, 0]])
    
    # Set axis limits with some padding
    center = all_points.mean(axis=0)
    max_range = np.abs(all_points - center).max() * 1.2
    
    ax.set_xlim(center[0] - max_range, center[0] + max_range)
    ax.set_ylim(center[1] - max_range, center[1] + max_range)
    ax.set_zlim(center[2] - max_range, center[2] + max_range)
    
    # Add legend
    if len(dataset) <= 10:
        ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1))
    
    # Add colorbar for Gaussians
    if gaussians is not None:
        # Create a dummy scatter plot for colorbar
        sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, 
                                   norm=plt.Normalize(vmin=0, vmax=1))
        sm.set_array([])
        # Note: Colorbar would show a colormap, but we're using RGB colors
        # So we'll skip the colorbar or create a custom one
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def visualize_2d_projections(dataset, gaussians, num_views=4, save_dir=None):
    """
    Visualize 2D projections of Gaussians from different camera views.
    
    Args:
        dataset: LLFFDataset instance
        gaussians: ManifoldGaussian3D instance
        num_views: number of views to visualize
        save_dir: optional directory to save figures
    """
    from src.redering.renderer import render_gaussians_3d
    
    num_views = min(num_views, len(dataset))
    fig, axes = plt.subplots(2, num_views, figsize=(5*num_views, 10))
    if num_views == 1:
        axes = axes.reshape(2, 1)
    
    for i in range(num_views):
        sample = dataset[i]
        camera = {
            'pose': sample['pose'],
            'K': sample['K'],
            'width': sample['image'].shape[2],
            'height': sample['image'].shape[1]
        }
        
        # Render
        with torch.no_grad():
            rendered = render_gaussians_3d(gaussians, camera)
        
        # Ground truth
        gt = sample['image'].permute(1, 2, 0).cpu().numpy()
        
        # Plot
        axes[0, i].imshow(np.clip(gt, 0, 1))
        axes[0, i].set_title(f'Ground Truth - View {i}')
        axes[0, i].axis('off')
        
        axes[1, i].imshow(np.clip(rendered.cpu().numpy(), 0, 1))
        axes[1, i].set_title(f'Rendered - View {i}')
        axes[1, i].axis('off')
    
    plt.tight_layout()
    
    if save_dir:
        save_path = Path(save_dir) / 'projections.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved projections to {save_path}")
    
    plt.show()


def main():
    """Main function to visualize scene"""
    import argparse
    from pathlib import Path
    
    parser = argparse.ArgumentParser(description='Visualize cameras and Gaussians')
    parser.add_argument('--scene', type=str, default='data/nerf_llff_data/fern',
                       help='Path to scene directory')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to Gaussian checkpoint (optional)')
    parser.add_argument('--downscale', type=float, default=4.0,
                       help='Image downscale factor')
    parser.add_argument('--save', type=str, default=None,
                       help='Path to save visualization')
    parser.add_argument('--no-show', action='store_true',
                       help='Don\'t display the plot')
    parser.add_argument('--projections', action='store_true',
                       help='Also show 2D projections')
    
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}\n")
    
    # Load dataset
    dataset = LLFFDataset(scene_root=args.scene, downscale=args.downscale, device=device)
    print(f"✓ Loaded {len(dataset)} views, resolution {dataset.H}x{dataset.W}\n")
    
    # Load camera positions correctly from poses_bounds.npy (correct method)
    import os
    poses_bounds_path = os.path.join(args.scene, "poses_bounds.npy")
    camera_positions = None
    
    if os.path.exists(poses_bounds_path):
        print("Loading camera positions from poses_bounds.npy (correct method)...")
        poses_bounds = np.load(poses_bounds_path)  # (N, 17)
        poses_raw = poses_bounds[:, :15].reshape(-1, 3, 5)  # (N, 3, 5)
        camera_positions = poses_raw[:, :, 3]  # (N, 3) - correct extraction
        print(f"✓ Loaded {len(camera_positions)} camera positions")
    else:
        print(f"Warning: {poses_bounds_path} not found")
    
    # Load or create Gaussians
    gaussians = None
    if args.checkpoint:
        print(f"Loading Gaussians from {args.checkpoint}...")
        # TODO: Implement checkpoint loading if needed
        # gaussians = load_gaussians(args.checkpoint, device=device)
        print("Checkpoint loading not implemented yet. Creating dummy Gaussians...")
    
    if gaussians is None:
        print("Creating dummy Gaussians for visualization...")
        # Simple initialization for visualization
        n_gaussians = 5000
        gaussians = ManifoldGaussian3D(n_gaussians=n_gaussians, device=device)
        
        # Initialize within scene bounds using correct camera positions
        if camera_positions is not None:
            scene_min = torch.tensor(camera_positions.min(axis=0), device=device)
            scene_max = torch.tensor(camera_positions.max(axis=0), device=device)
        else:
            # Fallback to dataset poses
            camera_positions_list = []
            for i in range(len(dataset)):
                sample = dataset[i]
                pose = sample["pose"]
                if isinstance(pose, torch.Tensor):
                    pose = pose.cpu().numpy()
                camera_positions_list.append(pose[:3, 3])
            camera_positions_array = np.array(camera_positions_list)
            scene_min = torch.tensor(camera_positions_array.min(axis=0), device=device)
            scene_max = torch.tensor(camera_positions_array.max(axis=0), device=device)
        
        with torch.no_grad():
            margin = (scene_max - scene_min) * 0.1
            expanded_min = scene_min - margin
            expanded_max = scene_max + margin
            gaussians.pos.data = torch.rand(n_gaussians, 3, device=device) * (expanded_max - expanded_min) + expanded_min
    
    # Visualize
    visualize_scene(dataset, gaussians, save_path=args.save, show=not args.no_show, scene_root=args.scene)
    
    # Show projections if requested
    if args.projections and gaussians is not None:
        visualize_2d_projections(dataset, gaussians, num_views=4, save_dir=Path(args.save).parent if args.save else None)


if __name__ == "__main__":
    main()

