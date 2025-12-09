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


def initialize_gaussians(dataset, n_gaussians=10000, device='cuda'):
    """Initialize Gaussians at the scene center"""
    # Get camera positions to determine scene bounds
    camera_positions = []
    for i in range(len(dataset)):
        sample = dataset[i]
        pose = sample["pose"]
        if isinstance(pose, torch.Tensor):
            pose = pose.cpu().numpy()
        camera_positions.append(pose[:3, 3])
    
    camera_positions = np.array(camera_positions)
    center = torch.tensor(camera_positions.mean(axis=0), device=device)
    
    print(f"Scene bounds:")
    print(f"  X: [{camera_positions[:, 0].min():.2f}, {camera_positions[:, 0].max():.2f}]")
    print(f"  Y: [{camera_positions[:, 1].min():.2f}, {camera_positions[:, 1].max():.2f}]")
    print(f"  Z: [{camera_positions[:, 2].min():.2f}, {camera_positions[:, 2].max():.2f}]")
    print(f"  Center: {center.cpu().numpy()}")
    
    # Create Gaussians
    gaussians = ManifoldGaussian3D(n_gaussians=n_gaussians, device=device)
    
    # For LLFF, cameras typically look toward origin
    # Place Gaussians at origin with spread
    scene_center = torch.zeros(3, device=device)
    
    # Initialize positions around origin
    with torch.no_grad():
        rand_offset = torch.randn(n_gaussians, 3, device=device)
        # Use larger spread to ensure coverage
        gaussians.pos.data = scene_center + rand_offset * 1.0
        
        # Also increase initial scales so Gaussians are more visible
        gaussians.log_scales.data = torch.randn(n_gaussians, 3, device=device) * 0.1 - 1.0  # Larger scales
    
    print(f"✓ Initialized {n_gaussians} Gaussians")
    print(f"  Position range:")
    print(f"    X: [{gaussians.pos[:, 0].min().item():.2f}, {gaussians.pos[:, 0].max().item():.2f}]")
    print(f"    Y: [{gaussians.pos[:, 1].min().item():.2f}, {gaussians.pos[:, 1].max().item():.2f}]")
    print(f"    Z: [{gaussians.pos[:, 2].min().item():.2f}, {gaussians.pos[:, 2].max().item():.2f}]")
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

