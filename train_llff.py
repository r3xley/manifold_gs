"""
3D Gaussian Splatting Training Pipeline for LLFF Datasets
Simple, extensible implementation with matplotlib visualization
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from data.load_lff import LLFFDataset
from src.models.manifold_gaussian import ManifoldGaussian3D
from src.redering.renderer import render_gaussians_3d
from data.load_colmap import load_colmap_points


def initialize_gaussians_from_colmap(dataset, n_gaussians=10000, device='cuda', scene_root=None):
    """Initialize Gaussians from COLMAP/SFM points"""
    import os
    
    # Get scene root path
    if scene_root is None:
        scene_root = dataset.scene_root if hasattr(dataset, 'scene_root') else None
    if scene_root is None:
        scene_root = "data/nerf_llff_data/fern"  # fallback
    
    print(f"Initializing Gaussians from COLMAP points...")
    
    # Try to load COLMAP points
    use_colmap = False
    colmap_points = None
    colmap_colors = None
    
    try:
        # Reduce memory usage: limit COLMAP points more aggressively
        max_colmap_points = min(n_gaussians, 20000)  # Reduced from 50k to 20k
        print(f"  Loading up to {max_colmap_points} COLMAP points (memory-safe limit)...")
        colmap_points, colmap_colors, point_ids, images = load_colmap_points(
            scene_root, max_points=max_colmap_points
        )
        
        # Filter out corrupted points
        valid_mask = (
            (np.abs(colmap_points[:, 0]) < 100) &
            (np.abs(colmap_points[:, 1]) < 100) &
            (np.abs(colmap_points[:, 2]) < 100) &
            (point_ids < 1e10)
        )
        valid_count = valid_mask.sum()
        
        if valid_count < 10:
            print(f"⚠ Only {valid_count} valid COLMAP points found (file may be corrupted)")
            print(f"  Falling back to random initialization")
            use_colmap = False
        else:
            colmap_points = colmap_points[valid_mask]
            colmap_colors = colmap_colors[valid_mask]
            use_colmap = True
            print(f"✓ COLMAP points loaded: {len(colmap_points)} valid points")
    except (FileNotFoundError, MemoryError, Exception) as e:
        print(f"⚠ COLMAP data not found or error loading: {e}")
        print(f"  Falling back to random initialization")
        use_colmap = False
    
    # Create Gaussians
    gaussians = ManifoldGaussian3D(n_gaussians=n_gaussians, device=device)
    
    with torch.no_grad():
        if use_colmap and colmap_points is not None:
            # Use COLMAP points for initialization
            print(f"\nInitializing from COLMAP points...")
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
            
            # Anisotropic scales for elongated ellipsoids
            log_scales = torch.zeros(n_gaussians, 3, device=device)
            log_scales[:, 0] = 0.8  # Longer axis
            log_scales[:, 1] = 0.2  # Medium axis
            log_scales[:, 2] = -0.2  # Shorter axis
            log_scales += torch.randn(n_gaussians, 3, device=device) * 0.1
            gaussians.log_scales.data = log_scales
            
            # Random rotations
            quats = torch.randn(n_gaussians, 4, device=device) * 0.1
            quats[:, 0] = 1.0
            quats = quats / quats.norm(dim=1, keepdim=True)
            gaussians.quaternions.data = quats
            
            # High initial opacity
            gaussians.logit_opacity.data = torch.ones(n_gaussians, device=device) * 2.0
            
        else:
            # Fallback: Random initialization
            print(f"\nUsing random initialization (COLMAP not available)...")
            # Load camera positions for scene bounds
            poses_bounds_path = os.path.join(scene_root, "poses_bounds.npy")
            if os.path.exists(poses_bounds_path):
                poses_bounds = np.load(poses_bounds_path)
                poses = poses_bounds[:, :15].reshape(-1, 3, 5)
                camera_positions = poses[:, :, 3]
                pos_min = camera_positions.min(axis=0)
                pos_max = camera_positions.max(axis=0)
                pos_mean = camera_positions.mean(axis=0)
                spread = (pos_max - pos_min) * 0.5
                gaussians.pos.data = torch.from_numpy(
                    pos_mean + (np.random.rand(n_gaussians, 3) - 0.5) * spread
                ).float().to(device)
            else:
                # Very basic fallback
                gaussians.pos.data = torch.randn(n_gaussians, 3, device=device) * 0.5
    
    print(f"✓ Initialized {n_gaussians} Gaussians")
    print(f"  Position range:")
    print(f"    X: [{gaussians.pos[:, 0].min().item():.2f}, {gaussians.pos[:, 0].max().item():.2f}]")
    print(f"    Y: [{gaussians.pos[:, 1].min().item():.2f}, {gaussians.pos[:, 1].max().item():.2f}]")
    print(f"    Z: [{gaussians.pos[:, 2].min().item():.2f}, {gaussians.pos[:, 2].max().item():.2f}]")
    
    return gaussians


# Global figure for live display
_live_fig = None
_live_axes = None

def visualize_render(gt_img, pred_img, epoch, view_idx, save_dir, show_live=True):
    """Visualize and save rendered images, optionally showing live"""
    global _live_fig, _live_axes
    
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Prepare images
    gt_display = np.clip(gt_img, 0, 1)
    pred_display = np.clip(pred_img, 0, 1)
    
    # Compute difference
    if len(gt_img.shape) == 3:
        gt_gray = gt_img.mean(axis=-1)
    else:
        gt_gray = gt_img
    if len(pred_img.shape) == 3:
        pred_gray = pred_img.mean(axis=-1)
    else:
        pred_gray = pred_img
    diff = np.abs(gt_gray - pred_gray)
    
    if show_live:
        # Create or update live figure
        if _live_fig is None:
            _live_fig, _live_axes = plt.subplots(1, 3, figsize=(15, 5))
            plt.ion()  # Turn on interactive mode
            plt.show()
        else:
            # Clear axes for update
            for ax in _live_axes:
                ax.clear()
        
        # Ground truth
        _live_axes[0].imshow(gt_display)
        _live_axes[0].set_title(f'Ground Truth (Epoch {epoch})')
        _live_axes[0].axis('off')
        
        # Predicted
        if pred_display.shape[2] == 1 or len(pred_display.shape) == 2:
            _live_axes[1].imshow(pred_display, cmap='gray')
        else:
            _live_axes[1].imshow(pred_display)
        _live_axes[1].set_title(f'Predicted (Epoch {epoch})')
        _live_axes[1].axis('off')
        
        # Difference
        _live_axes[2].imshow(diff, cmap='hot')
        _live_axes[2].set_title('Difference')
        _live_axes[2].axis('off')
        
        plt.tight_layout()
        plt.draw()
        plt.pause(0.01)  # Small pause to update display
    
    # Save to file
    fig_save, axes_save = plt.subplots(1, 3, figsize=(15, 5))
    axes_save[0].imshow(gt_display)
    axes_save[0].set_title('Ground Truth')
    axes_save[0].axis('off')
    
    if pred_display.shape[2] == 1 or len(pred_display.shape) == 2:
        axes_save[1].imshow(pred_display, cmap='gray')
    else:
        axes_save[1].imshow(pred_display)
    axes_save[1].set_title('Predicted')
    axes_save[1].axis('off')
    
    axes_save[2].imshow(diff, cmap='hot')
    axes_save[2].set_title('Difference')
    axes_save[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_dir / f'epoch_{epoch:03d}_view_{view_idx:02d}.png', dpi=100, bbox_inches='tight')
    plt.close(fig_save)


def plot_training_history(history, save_path):
    """Plot training loss and PSNR curves"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    epochs = range(len(history['loss']))
    
    ax1.plot(epochs, history['loss'])
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss')
    ax1.grid(True)
    
    ax2.plot(epochs, history['psnr'])
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('PSNR (dB)')
    ax2.set_title('PSNR')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close()


def visualize_gaussians_opengl_step(gaussians, epoch, max_gaussians=500):
    """Visualize Gaussians in OpenGL during training"""
    try:
        import sys
        import os
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from src.utils.visualize_gaussians_opengl import visualize_gaussians_opengl
        # This will open a window - user can close it to continue training
        print(f"\n  📊 Opening OpenGL visualization (epoch {epoch})...")
        print(f"  Close the window to continue training...")
        visualize_gaussians_opengl(gaussians, max_gaussians=max_gaussians)
    except ImportError as e:
        print(f"  ⚠ OpenGL visualization not available: {e}")
        print(f"  Install with: pip install PyOpenGL pygame")
    except Exception as e:
        print(f"  ⚠ OpenGL visualization failed: {e}")

def train(gaussians, dataset, num_epochs=50, lr=0.0001, device='cuda', output_dir='outputs', 
          visualize_opengl=False, opengl_interval=10, show_images=True):
    """Main training loop with optional OpenGL visualization and live image display"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    optimizer = torch.optim.Adam(gaussians.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.7)
    
    print("="*80)
    print("TRAINING")
    print("="*80)
    print(f"Epochs: {num_epochs} | Gaussians: {gaussians.num} | LR: {lr}")
    if visualize_opengl:
        print(f"OpenGL visualization: Every {opengl_interval} epochs")
    if show_images:
        print(f"Live image display: Enabled (updates every 5 epochs)")
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
            
            # Loss
            recon_loss = ((img_pred - img_gt_hwc) ** 2).mean()
            # Reduced opacity regularization to prevent driving opacities to zero
            opacity_reg = 0.001 * gaussians.get_opacity().mean()
            # Add scale regularization to prevent scales from becoming too small
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
        
        # Evaluate on test view
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
                
                mse = ((rendered - test_sample['image'].permute(1, 2, 0).to(device)) ** 2).mean()
                psnr = -10 * torch.log10(mse + 1e-10)
                
                history['loss'].append(avg_loss)
                history['psnr'].append(psnr.item())
                
                # Visualize (with live display)
                visualize_render(
                    gt_img.clip(0, 1),
                    rendered_img.clip(0, 1),
                    epoch, 0, output_dir / 'renders',
                    show_live=show_images
                )
                
                print(f"Epoch {epoch:3d}/{num_epochs}: Loss={avg_loss:.5f}, PSNR={psnr:.2f} dB")
        
        # OpenGL visualization (optional, every N epochs)
        # Reduce max_gaussians for OpenGL to save memory
        if visualize_opengl and (epoch % opengl_interval == 0 or epoch == num_epochs - 1):
            visualize_gaussians_opengl_step(gaussians, epoch, max_gaussians=300)  # Reduced from 500
    
    return gaussians, history


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}\n")
    
    # Load dataset
    scene = "data/nerf_llff_data/fern"
    # Increase downscale for memory safety (smaller images = less memory)
    downscale = 8.0 if device == 'cpu' else 4.0
    ds = LLFFDataset(scene_root=scene, downscale=downscale, device=device)
    print(f"✓ Loaded {len(ds)} views, resolution {ds.H}x{ds.W} (downscale: {downscale})\n")
    
    # Initialize Gaussians from COLMAP points
    # Reduce n_gaussians for memory safety (especially on CPU)
    n_gaussians = 5000 if device == 'cpu' else 10000
    print(f"Using {n_gaussians} Gaussians (device: {device})")
    gaussians = initialize_gaussians_from_colmap(ds, n_gaussians=n_gaussians, device=device, scene_root=scene)
    print()
    
    # Train with OpenGL visualization and live image display
    gaussians, history = train(
        gaussians, ds,
        num_epochs=50,
        lr=0.0001,
        device=device,
        output_dir='outputs/fern',
        visualize_opengl=True,  # Enable OpenGL visualization
        opengl_interval=10,  # Visualize every 10 epochs
        show_images=True  # Show images as they train
    )
    
    # Final evaluation
    print("\n" + "="*80)
    print("FINAL EVALUATION")
    print("="*80)
    
    with torch.no_grad():
        for i in range(min(3, len(ds))):
            sample = ds[i]
            camera = {
                'pose': sample['pose'].to(device),
                'K': sample['K'].to(device),
                'width': sample['image'].shape[2],
                'height': sample['image'].shape[1]
            }
            
            rendered = render_gaussians_3d(gaussians, camera)
            gt = sample['image'].permute(1, 2, 0).to(device)
            mse = ((rendered - gt) ** 2).mean()
            psnr = -10 * torch.log10(mse + 1e-10)
            
            print(f"  View {i}: PSNR = {psnr:.2f} dB")
            
            # Save final renders
            gt_img = gt.cpu().numpy()
            rendered_img = rendered.cpu().numpy()
            visualize_render(
                gt_img.clip(0, 1),
                rendered_img.clip(0, 1),
                999, i, Path('outputs/fern/final')
            )
    
    # Close live display if it was open
    global _live_fig
    if _live_fig is not None:
        plt.close(_live_fig)
        _live_fig = None
    
    print("\n" + "="*80)
    print("✅ DONE!")
    print("="*80)
    print(f"\nResults saved to: outputs/fern/")


if __name__ == "__main__":
    main()

