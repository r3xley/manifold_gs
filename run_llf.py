# examples/run_llff_rerun.py
import torch
import numpy as np
from data.load_lff import LLFFDataset
from src.models.manifold_gaussian import ManifoldGaussian3D
from src.redering.renderer import render_gaussians_3d
from src.training.train import train_multiview_3d
from src.utils.rerun_viz import (
    init_rerun, 
    log_camera_frustum, 
    log_gaussians_as_ellipsoids, 
    log_gaussian_points
)
import rerun as rr


def initialize_gaussians_from_cameras(dataset, n_gaussians=10000, device='cuda'):
    """Smart initialization: place Gaussians near camera viewing volume"""
    camera_positions = []
    camera_view_directions = []
    
    for i in range(len(dataset)):
        sample = dataset[i]
        pose = sample["pose"]
        if isinstance(pose, torch.Tensor):
            pose = pose.cpu().numpy()
        
        # Camera position (translation part of pose)
        cam_pos = pose[:3, 3]
        camera_positions.append(cam_pos)
        
        # Camera viewing direction: in camera space, camera looks down -Z axis
        # In world space, this is the third column of the rotation matrix
        # For camera-to-world: viewing direction in world = R @ [0, 0, -1]^T = -R[:, 2]
        R = pose[:3, :3]
        # Standard OpenGL/NeRF convention: camera looks down -Z
        view_dir = -R[:, 2]  # Negative Z axis in camera space
        camera_view_directions.append(view_dir)
    
    # Convert to numpy array first to avoid warning, then to tensor
    camera_positions = torch.tensor(np.array(camera_positions), device=device)
    camera_view_directions = torch.tensor(np.array(camera_view_directions), device=device)
    
    print(f"Camera positions:")
    print(f"  X range: [{camera_positions[:, 0].min():.2f}, {camera_positions[:, 0].max():.2f}]")
    print(f"  Y range: [{camera_positions[:, 1].min():.2f}, {camera_positions[:, 1].max():.2f}]")
    print(f"  Z range: [{camera_positions[:, 2].min():.2f}, {camera_positions[:, 2].max():.2f}]")
    
    # Compute average camera center and viewing direction
    center = camera_positions.mean(dim=0)
    avg_view_dir = camera_view_directions.mean(dim=0)
    avg_view_dir = avg_view_dir / (avg_view_dir.norm() + 1e-8)  # Normalize
    
    print(f"Average camera viewing direction: {avg_view_dir.cpu().numpy()}")
    
    # For LLFF datasets, cameras typically look toward the origin
    # Place Gaussians near the origin, but check coordinate system
    # Try placing at origin first
    scene_center = torch.zeros(3, device=device)
    
    # Alternative: place between cameras and origin
    # But based on the viewing direction, cameras look in +Z, so place Gaussians at origin
    # The issue might be coordinate convention - let's try both
    if avg_view_dir[2] > 0:
        # Cameras look in +Z, place Gaussians at origin (should be in front)
        scene_center = torch.zeros(3, device=device)
    else:
        # Cameras look in -Z, place Gaussians behind cameras
        scene_center = center - avg_view_dir * 1.0
    
    # Create Gaussians
    gaussians = ManifoldGaussian3D(n_gaussians=n_gaussians, device=device)
    
    # Initialize positions around scene center
    with torch.no_grad():
        # Place Gaussians in a box around scene center
        # Use a tighter distribution to ensure they're visible
        rand_offset = torch.randn(n_gaussians, 3, device=device)
        gaussians.pos.data = scene_center + rand_offset * 0.3  # Smaller spread
    
    print(f"✓ Initialized {n_gaussians} Gaussians")
    print(f"  Camera center: {center.cpu().numpy()}")
    print(f"  Scene center (Gaussian center): {scene_center.cpu().numpy()}")
    print(f"  Gaussian positions range:")
    print(f"    X: [{gaussians.pos[:, 0].min().item():.2f}, {gaussians.pos[:, 0].max().item():.2f}]")
    print(f"    Y: [{gaussians.pos[:, 1].min().item():.2f}, {gaussians.pos[:, 1].max().item():.2f}]")
    print(f"    Z: [{gaussians.pos[:, 2].min().item():.2f}, {gaussians.pos[:, 2].max().item():.2f}]")
    
    return gaussians


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}\n")
    
    # ========================================================================
    # 1. LOAD DATASET
    # ========================================================================
    scene = "/Users/rexley/Documents/Class/Geometric Methods in ML/3DGS/data/nerf_llff_data/fern"
    ds = LLFFDataset(scene_root=scene, downscale=2.0, device=device)
    print(f"✓ Loaded {len(ds)} views, resolution {ds.H}x{ds.W}\n")
    
    # ========================================================================
    # 2. INITIALIZE GAUSSIANS
    # ========================================================================
    n_gaussians = 10000  # Use more for better quality
    
    gaussians = initialize_gaussians_from_cameras(
        dataset=ds,
        n_gaussians=n_gaussians,
        device=device
    )
    print()
    
    # ========================================================================
    # 3. SETUP RERUN VISUALIZATION
    # ========================================================================
    init_rerun("LLFF 3D Gaussian Splatting Training")
    print("✓ Rerun initialized\n")
    
    # Log all cameras (NO FLIPPING NEEDED - already correct!)
    print("Logging cameras...")
    for i in range(len(ds)):
        sample = ds[i]
        K = sample["K"]
        pose = sample["pose"]
        
        # Convert to numpy if needed
        if isinstance(K, torch.Tensor):
            K = K.cpu().numpy()
        if isinstance(pose, torch.Tensor):
            pose = pose.cpu().numpy()
        
        color = [0.8, 0.2, 0.2] if i == 0 else [0.2, 0.8, 0.2]
        log_camera_frustum(
            f"world/cameras/camera_{i}", 
            K, 
            pose,  # Use original pose - it's already correct!
            image_size=(ds.W, ds.H),
            color=color
        )
    print(f"✓ Logged {len(ds)} cameras\n")
    
    # Log initial Gaussians (gray - use axis-aligned for speed)
    rr.set_time("stage", sequence=0)
    log_gaussian_points("world/gaussians/centers_init", gaussians, radius=0.01)
    log_gaussians_as_ellipsoids(
        "world/gaussians/ellipsoids_init", 
        gaussians, 
        max_objects=min(1000, n_gaussians),
        colors=[150, 150, 150],  # RGB as list of ints (0-255), will be normalized
        oriented=False  # Use axis-aligned for faster initial visualization
    )
    print("✓ Logged initial Gaussians\n")
    
    # ========================================================================
    # 4. TRAIN
    # ========================================================================
    print("="*80)
    print("TRAINING")
    print("="*80)
    print(f"Epochs: 50 | Gaussians: {n_gaussians} | LR: 0.0001")
    print("="*80 + "\n")
    
    # Training loop with visualization
    optimizer = torch.optim.Adam(gaussians.parameters(), lr=0.0001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.7)
    
    num_epochs = 50
    log_every = 5
    
    for epoch in range(num_epochs):
        epoch_loss = 0
        num_batches = 0
        
        # Shuffle views
        indices = torch.randperm(len(ds))
        
        for idx in indices:
            optimizer.zero_grad()
            
            # Get data
            sample = ds[int(idx)]
            img_gt = sample['image'].to(device)
            pose = sample['pose'].to(device)
            K = sample['K'].to(device)
            
            # Render
            camera = {
                'pose': pose,
                'K': K,
                'width': img_gt.shape[2],
                'height': img_gt.shape[1]
            }
            img_pred = render_gaussians_3d(gaussians, camera)
            img_gt_hwc = img_gt.permute(1, 2, 0)
            
            # Debug: check if rendering produced anything
            if epoch == 0 and idx == 0:
                print(f"Debug - First render:")
                print(f"  Image pred range: [{img_pred.min().item():.3f}, {img_pred.max().item():.3f}]")
                print(f"  Image pred mean: {img_pred.mean().item():.3f}")
                print(f"  Opacity range: [{gaussians.get_opacity().min().item():.3f}, {gaussians.get_opacity().max().item():.3f}]")
                print(f"  Opacity mean: {gaussians.get_opacity().mean().item():.3f}")
            
            # Loss
            recon_loss = ((img_pred - img_gt_hwc) ** 2).mean()
            opacity_reg = 0.01 * gaussians.get_opacity().mean()
            loss = recon_loss + opacity_reg
            
            # Optimize
            loss.backward()
            torch.nn.utils.clip_grad_norm_(gaussians.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
        
        scheduler.step()
        avg_loss = epoch_loss / num_batches
        
        # ====================================================================
        # VISUALIZATION (every N epochs)
        # ====================================================================
        if epoch % log_every == 0 or epoch == 0:
            # Evaluate on test view
            with torch.no_grad():
                test_sample = ds[0]
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
                psnr = -10 * torch.log10(mse)
            
            # Log to Rerun
            rr.set_time("epoch", sequence=epoch)
            
            # Update Gaussians visualization (use axis-aligned for speed during training)
            log_gaussians_as_ellipsoids(
                "world/gaussians/ellipsoids_training",
                gaussians,
                max_objects=min(1000, n_gaussians),  # Reduced for performance
                colors=[100, 150, 255],
                oriented=False  # Faster visualization during training
            )
            
            # Log images (ensure they're in the correct format)
            # Rerun expects images as numpy arrays with shape (H, W, 3) and values in [0, 1]
            gt_img_normalized = np.clip(gt_img, 0, 1)
            rendered_img_normalized = np.clip(rendered_img, 0, 1)
            rr.log("renders/ground_truth", rr.Image(gt_img_normalized))
            rr.log("renders/predicted", rr.Image(rendered_img_normalized))
            
            # Log metrics
            rr.log("metrics/loss", rr.Scalars(avg_loss))
            rr.log("metrics/psnr", rr.Scalars(psnr.item()))
            
            print(f"Epoch {epoch:3d}/{num_epochs}: Loss={avg_loss:.5f}, PSNR={psnr:.2f} dB")
    
    # ========================================================================
    # 5. FINAL VISUALIZATION
    # ========================================================================
    print("\n" + "="*80)
    print("FINAL RESULTS")
    print("="*80)
    
    rr.set_time("stage", sequence=999)
    
    # Log final Gaussians (red - use oriented for final visualization)
    log_gaussians_as_ellipsoids(
        "world/gaussians/ellipsoids_final",
        gaussians,
        max_objects=min(2000, n_gaussians),  # Reasonable number for oriented visualization
        colors=[255, 100, 100],
        oriented=True  # Show proper orientation for manifold Gaussians
    )
    
    # Render and log final test views
    print("\nFinal evaluation:")
    for i in range(min(3, len(ds))):
        sample = ds[i]
        camera = {
            'pose': sample['pose'].to(device),
            'K': sample['K'].to(device),
            'width': sample['image'].shape[2],
            'height': sample['image'].shape[1]
        }
        
        with torch.no_grad():
            rendered = render_gaussians_3d(gaussians, camera)
        
        gt = sample['image'].permute(1, 2, 0).to(device)
        mse = ((rendered - gt) ** 2).mean()
        psnr = -10 * torch.log10(mse)
        
        print(f"  View {i}: PSNR = {psnr:.2f} dB")
        
        # Ensure images are normalized to [0, 1]
        gt_np = np.clip(gt.cpu().numpy(), 0, 1)
        rendered_np = np.clip(rendered.cpu().numpy(), 0, 1)
        rr.log(f"final/view_{i}/gt", rr.Image(gt_np))
        rr.log(f"final/view_{i}/rendered", rr.Image(rendered_np))
    
    print("\n" + "="*80)
    print("✅ DONE!")
    print("="*80)
    print("\nCheck Rerun viewer:")
    print("  📹 Cameras (red=main, green=others)")
    print("  ⚫ Initial Gaussians (gray)")
    print("  🔴 Final Gaussians (red)")
    print("  📊 Training curves (loss & PSNR)")
    print("  🖼️  Ground truth vs rendered")
    print("\nUse timeline slider to see training!")
    print("\nPress Ctrl+C to exit...")
    
    # Keep alive
    try:
        import time
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nExiting...")


if __name__ == "__main__":
    main()