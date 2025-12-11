"""
Working 3D Gaussian Splatting Pipeline
Clean implementation with visualization
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from data.load_lff import LLFFDataset
from src.models.manifold_gaussian import ManifoldGaussian3D
from src.redering.renderer import render_gaussians_3d


def initialize_gaussians(dataset, n_gaussians=10000, device='cuda', scene_root=None):
    """Initialize Gaussians from COLMAP sparse point cloud or fallback to random placement"""
    import os
    from data.load_colmap import load_colmap_points, transform_colmap_to_llff
    
    # Get scene root path
    if scene_root is None:
        scene_root = dataset.scene_root if hasattr(dataset, 'scene_root') else None
    if scene_root is None:
        scene_root = "data/nerf_llff_data/fern"  # fallback
    
    # Try to load COLMAP points first
    use_colmap = False
    colmap_points = None
    colmap_colors = None
    
    try:
        # Limit COLMAP points to avoid memory issues
        # Use smaller limit for safety - can always clone points later
        max_colmap_points = min(n_gaussians * 2, 50000)  # Cap at 50k for memory safety
        print(f"  Loading up to {max_colmap_points} COLMAP points...")
        colmap_points, colmap_colors, point_ids, images = load_colmap_points(
            scene_root, max_points=max_colmap_points
        )
        
        # Filter out corrupted points (unreasonable values)
        valid_mask = (
            (np.abs(colmap_points[:, 0]) < 100) &
            (np.abs(colmap_points[:, 1]) < 100) &
            (np.abs(colmap_points[:, 2]) < 100) &
            (point_ids < 1e10)
        )
        valid_count = valid_mask.sum()
        
        if valid_count < 10:  # Too few valid points
            print(f"⚠ Only {valid_count} valid COLMAP points found (file appears corrupted)")
            print(f"  COLMAP file corruption detected: point 1+ contain garbage data")
            print(f"  Falling back to random initialization (using poses_bounds.npy)")
            use_colmap = False
        else:
            # Use only valid points
            colmap_points = colmap_points[valid_mask]
            colmap_colors = colmap_colors[valid_mask]
            point_ids = point_ids[valid_mask]
            use_colmap = True
            print(f"✓ COLMAP points loaded: {len(colmap_points)} valid points (filtered from corrupted file)")
    except (FileNotFoundError, MemoryError, Exception) as e:
        print(f"⚠ COLMAP data not found or error loading: {e}")
        print(f"  Falling back to random initialization")
        use_colmap = False
    
    # Load camera positions for scene bounds calculation
    poses_bounds_path = os.path.join(scene_root, "poses_bounds.npy")
    camera_positions = None
    
    if os.path.exists(poses_bounds_path):
        poses_bounds = np.load(poses_bounds_path)  # (N, 17)
        poses = poses_bounds[:, :15].reshape(-1, 3, 5)  # (N, 3, 5)
        camera_positions = poses[:, :, 3]  # (N, 3) - correct extraction method
    else:
        # Fallback to dataset poses
        camera_positions_list = []
        for i in range(len(dataset)):
            sample = dataset[i]
            pose = sample["pose"]
            if isinstance(pose, torch.Tensor):
                pose = pose.cpu().numpy()
            camera_positions_list.append(pose[:3, 3])
        camera_positions = np.array(camera_positions_list)
    
    # Compute scene bounds from camera positions
    pos_min = camera_positions.min(axis=0)
    pos_max = camera_positions.max(axis=0)
    pos_mean = camera_positions.mean(axis=0)
    
    scene_min = torch.tensor(pos_min, device=device)
    scene_max = torch.tensor(pos_max, device=device)
    scene_center = torch.tensor(pos_mean, device=device)
    scene_size = scene_max - scene_min
    
    print(f"Scene bounds:")
    print(f"  X: [{pos_min[0]:.2f}, {pos_max[0]:.2f}] (size: {pos_max[0] - pos_min[0]:.2f})")
    print(f"  Y: [{pos_min[1]:.2f}, {pos_max[1]:.2f}] (size: {pos_max[1] - pos_min[1]:.2f})")
    print(f"  Z: [{pos_min[2]:.2f}, {pos_max[2]:.2f}] (size: {pos_max[2] - pos_min[2]:.2f})")
    print(f"  Center: {pos_mean}")
    
    # Create Gaussians
    gaussians = ManifoldGaussian3D(n_gaussians=n_gaussians, device=device)
    
    # Initialize positions and colors
    with torch.no_grad():
        if use_colmap and colmap_points is not None:
            # Use COLMAP points for initialization
            print(f"\nInitializing from COLMAP points...")
            
            # Transform COLMAP coordinates to LLFF if needed (test first, may not be needed)
            # For now, use as-is and test
            points_3d = colmap_points.copy()
            colors_3d = colmap_colors.copy()
            
            n_colmap = len(points_3d)
            print(f"  COLMAP points: {n_colmap}")
            print(f"  Requested Gaussians: {n_gaussians}")
            
            if n_colmap >= n_gaussians:
                # Subsample COLMAP points
                indices = np.random.choice(n_colmap, n_gaussians, replace=False)
                positions_array = points_3d[indices]
                colors_array = colors_3d[indices]
                print(f"  Subsample: using {n_gaussians} out of {n_colmap} COLMAP points")
            else:
                # Use all COLMAP points, then add more by cloning/perturbing
                positions_array = points_3d.copy()
                colors_array = colors_3d.copy()
                
                # Add more by cloning with small perturbations
                n_needed = n_gaussians - n_colmap
                print(f"  Cloning: adding {n_needed} more Gaussians by perturbing COLMAP points")
                
                # Clone points with small random offsets
                clone_indices = np.random.choice(n_colmap, n_needed, replace=True)
                clone_positions = points_3d[clone_indices] + np.random.randn(n_needed, 3) * 0.01
                clone_colors = colors_3d[clone_indices]
                
                positions_array = np.vstack([positions_array, clone_positions])
                colors_array = np.vstack([colors_array, clone_colors])
            
            gaussians.pos.data = torch.from_numpy(positions_array).float().to(device)
            
            # Convert colors to logit space
            colors_clamped = np.clip(colors_array, 0.01, 0.99)
            colors_logit = np.log(colors_clamped / (1 - colors_clamped))
            gaussians.rgb_logit.data = torch.from_numpy(colors_logit).float().to(device)
            
        else:
            # Fallback: Random initialization in front of cameras
            print(f"\nUsing random initialization (COLMAP not available)...")
            positions_list = []
            
            # Get viewing directions for all cameras
            for i in range(len(dataset)):
                sample = dataset[i]
                pose = sample['pose']
                if isinstance(pose, torch.Tensor):
                    pose = pose.cpu().numpy()
                
                cam_pos_i = pose[:3, 3]
                R_c2w_i = pose[:3, :3]
                # Camera looks along -z in camera space, which is -R[:, 2] in world space
                view_dir_i = -R_c2w_i[:, 2]  # Viewing direction in world space
                
                # Place Gaussians in front of camera
                n_per_camera = n_gaussians // len(dataset) + 1
                for j in range(n_per_camera):
                    # Distance in front
                    dist = np.random.uniform(0.2, 1.0)
                    # Small random offset perpendicular to viewing direction
                    if abs(view_dir_i[0]) < 0.9:
                        perp = np.cross(view_dir_i, [1, 0, 0])
                    else:
                        perp = np.cross(view_dir_i, [0, 1, 0])
                    perp = perp / (np.linalg.norm(perp) + 1e-8)
                    offset = np.random.randn() * 0.05 * perp
                    
                    pos = cam_pos_i + view_dir_i * dist + offset
                    positions_list.append(pos)
            
            # Trim to exact number
            positions_list = positions_list[:n_gaussians]
            positions_array = np.array(positions_list)
            
            gaussians.pos.data = torch.from_numpy(positions_array).float().to(device)
            
            # Initialize colors from nearest image pixels
            colors_list = []
            for idx, pos in enumerate(positions_array):
                color_found = False
                best_color = None
                
                # Try multiple nearby cameras to get a good color sample
                distances = np.linalg.norm(camera_positions - pos, axis=1)
                sorted_cam_indices = np.argsort(distances)
                
                # Try up to 3 nearest cameras
                for cam_idx in sorted_cam_indices[:3]:
                    try:
                        # Get image from that camera
                        sample = dataset[cam_idx]
                        img = sample['image']  # (3, H, W)
                        pose = sample['pose']
                        K = sample['K']
                        
                        # Project Gaussian position to this camera's image
                        if isinstance(pose, torch.Tensor):
                            pose_np = pose.cpu().numpy()
                            K_np = K.cpu().numpy()
                        else:
                            pose_np = pose
                            K_np = K
                        
                        # Transform to camera space
                        w2c = np.linalg.inv(pose_np)
                        pos_hom = np.append(pos, 1.0)
                        pos_cam = w2c[:3, :] @ pos_hom
                        
                        # Check if in front of camera
                        if pos_cam[2] > -0.01:  # Behind camera or too close
                            continue
                        
                        # Project to 2D
                        depth = -pos_cam[2]  # Positive depth (camera looks along -z)
                        depth = max(depth, 0.01)
                        
                        x_2d = (pos_cam[0] / depth) * K_np[0, 0] + K_np[0, 2]
                        y_2d = (pos_cam[1] / depth) * K_np[1, 1] + K_np[1, 2]
                        
                        # Check if projection is within image bounds
                        H, W = img.shape[1], img.shape[2]
                        if x_2d < 0 or x_2d >= W or y_2d < 0 or y_2d >= H:
                            continue
                        
                        # Sample color from image
                        x_pixel = int(np.clip(x_2d, 0, W - 1))
                        y_pixel = int(np.clip(y_2d, 0, H - 1))
                        
                        if isinstance(img, torch.Tensor):
                            color = img[:, y_pixel, x_pixel].cpu().numpy()
                        else:
                            color = img[:, y_pixel, x_pixel]
                        
                        # Check if color is valid (not all zeros or all ones)
                        if np.any(color < 0.01) or np.all(color > 0.99):
                            continue
                        
                        best_color = color
                        color_found = True
                        break
                    except:
                        continue
                
                # If no valid color found, use random color
                if not color_found or best_color is None:
                    color = np.random.rand(3) * 0.5 + 0.25
                else:
                    color = best_color
                
                # Convert to logit space
                color_clamped = np.clip(color, 0.01, 0.99)
                color_logit = np.log(color_clamped / (1 - color_clamped))
                colors_list.append(color_logit)
            
            colors_array = np.array(colors_list)
            gaussians.rgb_logit.data = torch.from_numpy(colors_array).float().to(device)
        
        # Set initial scales based on scene size - make them proportional to scene scale
        avg_scene_size = scene_size.mean().item()
        # Initialize scales to be about 1-2% of scene size
        initial_scale = avg_scene_size * 0.015
        gaussians.log_scales.data = torch.ones(n_gaussians, 3, device=device) * np.log(initial_scale)
    
    print(f"✓ Initialized {n_gaussians} Gaussians")
    print(f"  Position range:")
    print(f"    X: [{gaussians.pos[:, 0].min().item():.2f}, {gaussians.pos[:, 0].max().item():.2f}]")
    print(f"    Y: [{gaussians.pos[:, 1].min().item():.2f}, {gaussians.pos[:, 1].max().item():.2f}]")
    print(f"    Z: [{gaussians.pos[:, 2].min().item():.2f}, {gaussians.pos[:, 2].max().item():.2f}]")
    print(f"  Initial scale: {torch.exp(gaussians.log_scales[0, 0]).item():.4f} (log_scale: {gaussians.log_scales[0, 0].item():.4f})")
    return gaussians


def compute_metrics(gt, pred):
    """Compute PSNR and MSE"""
    mse = ((pred - gt) ** 2).mean()
    psnr = -10 * torch.log10(mse + 1e-10)
    return mse.item(), psnr.item()


def visualize_comparison(gt_img, pred_img, epoch, view_idx, metrics, save_dir):
    """Create comparison visualization"""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    # Ensure images are in [0, 1] range
    gt_img = np.clip(gt_img, 0, 1)
    pred_img = np.clip(pred_img, 0, 1)
    
    # Ground truth
    if len(gt_img.shape) == 3:
        axes[0].imshow(gt_img)
    else:
        axes[0].imshow(gt_img, cmap='gray')
    axes[0].set_title('Ground Truth', fontsize=14)
    axes[0].axis('off')
    
    # Predicted
    if len(pred_img.shape) == 3:
        axes[1].imshow(pred_img)
    else:
        axes[1].imshow(pred_img, cmap='gray')
    axes[1].set_title(f'Predicted (PSNR: {metrics["psnr"]:.2f} dB)', fontsize=14)
    axes[1].axis('off')
    
    # Error map
    if len(gt_img.shape) == 3:
        gt_gray = gt_img.mean(axis=-1)
        pred_gray = pred_img.mean(axis=-1)
    else:
        gt_gray = gt_img
        pred_gray = pred_img
    
    error = np.abs(gt_gray - pred_gray)
    im = axes[2].imshow(error, cmap='hot', vmin=0, vmax=0.3)
    axes[2].set_title('Error Map', fontsize=14)
    axes[2].axis('off')
    plt.colorbar(im, ax=axes[2])
    
    # Difference (signed)
    diff = gt_gray - pred_gray
    im2 = axes[3].imshow(diff, cmap='RdBu', vmin=-0.3, vmax=0.3)
    axes[3].set_title('Difference (GT - Pred)', fontsize=14)
    axes[3].axis('off')
    plt.colorbar(im2, ax=axes[3])
    
    plt.tight_layout()
    plt.savefig(save_dir / f'epoch_{epoch:03d}_view_{view_idx:02d}.png', 
                dpi=150, bbox_inches='tight')
    plt.close()


def plot_training_curves(history, save_path):
    """Plot training metrics"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    epochs = np.arange(len(history['loss']))
    
    ax1.plot(epochs, history['loss'], 'b-', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training Loss', fontsize=14)
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(epochs, history['psnr'], 'r-', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('PSNR (dB)', fontsize=12)
    ax2.set_title('PSNR', fontsize=14)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def train(gaussians, dataset, num_epochs=50, lr=0.0001, device='cuda', output_dir='outputs'):
    """Main training loop"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    optimizer = torch.optim.Adam(gaussians.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.7)
    
    print("="*80)
    print("TRAINING")
    print("="*80)
    print(f"Epochs: {num_epochs} | Gaussians: {gaussians.num} | LR: {lr}")
    print("="*80 + "\n")
    
    history = {'loss': [], 'psnr': []}
    
    for epoch in range(num_epochs):
        epoch_loss = 0
        num_batches = 0
        
        # Shuffle views
        indices = torch.randperm(len(dataset))
        
        for idx in indices:
            optimizer.zero_grad()
            
            # Get data
            sample = dataset[int(idx)]
            img_gt = sample['image'].to(device)  # (3, H, W)
            pose = sample['pose'].to(device)
            K = sample['K'].to(device)
            
            # Render
            camera = {
                'pose': pose,
                'K': K,
                'width': img_gt.shape[2],
                'height': img_gt.shape[1]
            }
            img_pred = render_gaussians_3d(gaussians, camera)  # (H, W, 3)
            img_gt_hwc = img_gt.permute(1, 2, 0)  # (H, W, 3)
            
            # Debug first render
            if epoch == 0 and idx == 0:
                print(f"\nDebug - First render:")
                print(f"  Predicted image range: [{img_pred.min().item():.3f}, {img_pred.max().item():.3f}]")
                print(f"  Predicted image mean: {img_pred.mean().item():.3f}")
                print(f"  Opacity range: [{gaussians.get_opacity().min().item():.3f}, {gaussians.get_opacity().max().item():.3f}]")
                print(f"  Opacity mean: {gaussians.get_opacity().mean().item():.3f}")
                print(f"  Scale range: [{gaussians.get_scales().min().item():.3f}, {gaussians.get_scales().max().item():.3f}]")
                print()
            
            # Loss
            recon_loss = ((img_pred - img_gt_hwc) ** 2).mean()
            # Reduced opacity regularization to prevent driving opacities to zero
            opacity_reg = 0.001 * gaussians.get_opacity().mean()
            scale_reg = 0.0001 * (gaussians.get_scales() ** 2).mean()
            loss = recon_loss + opacity_reg + scale_reg
            
            # Optimize
            loss.backward()
            torch.nn.utils.clip_grad_norm_(gaussians.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
        
        scheduler.step()
        avg_loss = epoch_loss / num_batches
        
        # Evaluate and visualize
        if epoch % 5 == 0 or epoch == 0:
            with torch.no_grad():
                test_sample = dataset[0]
                test_camera = {
                    'pose': test_sample['pose'].to(device),
                    'K': test_sample['K'].to(device),
                    'width': test_sample['image'].shape[2],
                    'height': test_sample['image'].shape[1]
                }
                rendered = render_gaussians_3d(gaussians, test_camera)
                
                gt_img = test_sample['image'].permute(1, 2, 0).cpu().numpy()
                rendered_img = rendered.cpu().numpy()
                
                mse, psnr = compute_metrics(
                    test_sample['image'].permute(1, 2, 0).to(device),
                    rendered
                )
                
                history['loss'].append(avg_loss)
                history['psnr'].append(psnr)
                
                # Visualize
                visualize_comparison(
                    gt_img, rendered_img, epoch, 0,
                    {'psnr': psnr, 'mse': mse},
                    output_dir / 'renders'
                )
                
                print(f"Epoch {epoch:3d}/{num_epochs}: Loss={avg_loss:.5f}, PSNR={psnr:.2f} dB")
    
    # Plot training curves
    plot_training_curves(history, output_dir / 'training_curves.png')
    
    return gaussians, history


def evaluate_all_views(gaussians, dataset, device='cuda', output_dir='outputs'):
    """Evaluate on all views and save results"""
    output_dir = Path(output_dir)
    (output_dir / 'final_renders').mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*80)
    print("FINAL EVALUATION")
    print("="*80)
    
    all_psnrs = []
    
    with torch.no_grad():
        for i in range(len(dataset)):
            sample = dataset[i]
            camera = {
                'pose': sample['pose'].to(device),
                'K': sample['K'].to(device),
                'width': sample['image'].shape[2],
                'height': sample['image'].shape[1]
            }
            
            rendered = render_gaussians_3d(gaussians, camera)
            gt = sample['image'].permute(1, 2, 0).to(device)
            
            mse, psnr = compute_metrics(gt, rendered)
            all_psnrs.append(psnr)
            
            # Save final renders
            gt_img = gt.cpu().numpy()
            rendered_img = rendered.cpu().numpy()
            visualize_comparison(
                gt_img, rendered_img, 999, i,
                {'psnr': psnr, 'mse': mse},
                output_dir / 'final_renders'
            )
            
            if i < 5:  # Print first 5
                print(f"  View {i:2d}: PSNR = {psnr:.2f} dB")
    
    print(f"\n  Mean PSNR: {np.mean(all_psnrs):.2f} dB")
    print(f"  Min PSNR:  {np.min(all_psnrs):.2f} dB")
    print(f"  Max PSNR:  {np.max(all_psnrs):.2f} dB")
    
    return all_psnrs


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}\n")
    
    # Load dataset
    scene = "data/nerf_llff_data/fern"
    ds = LLFFDataset(scene_root=scene, downscale=4.0, device=device)
    print(f"✓ Loaded {len(ds)} views, resolution {ds.H}x{ds.W}\n")
    
    # Initialize Gaussians
    n_gaussians = 10000
    gaussians = initialize_gaussians(ds, n_gaussians=n_gaussians, device=device)
    print()
    
    # Train
    gaussians, history = train(
        gaussians, ds,
        num_epochs=50,
        lr=0.0001,
        device=device,
        output_dir='outputs/fern'
    )
    
    # Final evaluation
    psnrs = evaluate_all_views(gaussians, ds, device=device, output_dir='outputs/fern')
    
    print("\n" + "="*80)
    print("✅ TRAINING COMPLETE!")
    print("="*80)
    print(f"\nResults saved to: outputs/fern/")
    print(f"  - Training renders: outputs/fern/renders/")
    print(f"  - Final renders: outputs/fern/final_renders/")
    print(f"  - Training curves: outputs/fern/training_curves.png")
    print(f"\nMean PSNR: {np.mean(psnrs):.2f} dB")


if __name__ == "__main__":
    main()

