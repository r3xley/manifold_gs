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


def initialize_gaussians_from_cameras(dataset, n_gaussians=10000, device='cuda'):
    """Initialize Gaussians in the scene volume"""
    camera_positions = []
    
    for i in range(len(dataset)):
        sample = dataset[i]
        pose = sample["pose"]
        if isinstance(pose, torch.Tensor):
            pose = pose.cpu().numpy()
        camera_positions.append(pose[:3, 3])
    
    camera_positions = np.array(camera_positions)
    camera_positions_t = torch.tensor(camera_positions, device=device)
    
    print(f"Camera positions:")
    print(f"  X: [{camera_positions[:, 0].min():.2f}, {camera_positions[:, 0].max():.2f}]")
    print(f"  Y: [{camera_positions[:, 1].min():.2f}, {camera_positions[:, 1].max():.2f}]")
    print(f"  Z: [{camera_positions[:, 2].min():.2f}, {camera_positions[:, 2].max():.2f}]")
    
    # Place Gaussians at scene center
    center = camera_positions_t.mean(dim=0)
    
    # For LLFF, try placing at origin or slightly offset
    # Test a few positions to find where Gaussians are visible
    scene_center = torch.zeros(3, device=device)
    
    # Create Gaussians
    gaussians = ManifoldGaussian3D(n_gaussians=n_gaussians, device=device)
    
    # Initialize positions around scene center
    with torch.no_grad():
        # Place at origin with spread
        rand_offset = torch.randn(n_gaussians, 3, device=device)
        # Use larger spread to cover more of the scene
        gaussians.pos.data = scene_center + rand_offset * 1.0
    
    print(f"✓ Initialized {n_gaussians} Gaussians")
    print(f"  Center: {center.cpu().numpy()}")
    print(f"  Position range:")
    print(f"    X: [{gaussians.pos[:, 0].min().item():.2f}, {gaussians.pos[:, 0].max().item():.2f}]")
    print(f"    Y: [{gaussians.pos[:, 1].min().item():.2f}, {gaussians.pos[:, 1].max().item():.2f}]")
    print(f"    Z: [{gaussians.pos[:, 2].min().item():.2f}, {gaussians.pos[:, 2].max().item():.2f}]")
    
    return gaussians


def visualize_render(gt_img, pred_img, epoch, view_idx, save_dir):
    """Visualize and save rendered images"""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Ground truth
    axes[0].imshow(np.clip(gt_img, 0, 1))
    axes[0].set_title('Ground Truth')
    axes[0].axis('off')
    
    # Predicted
    pred_display = np.clip(pred_img, 0, 1)
    if pred_display.shape[2] == 1 or len(pred_display.shape) == 2:
        axes[1].imshow(pred_display, cmap='gray')
    else:
        axes[1].imshow(pred_display)
    axes[1].set_title('Predicted')
    axes[1].axis('off')
    
    # Difference
    if len(gt_img.shape) == 3:
        gt_gray = gt_img.mean(axis=-1)
    else:
        gt_gray = gt_img
    if len(pred_img.shape) == 3:
        pred_gray = pred_img.mean(axis=-1)
    else:
        pred_gray = pred_img
    diff = np.abs(gt_gray - pred_gray)
    axes[2].imshow(diff, cmap='hot')
    axes[2].set_title('Difference')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_dir / f'epoch_{epoch:03d}_view_{view_idx:02d}.png', dpi=100, bbox_inches='tight')
    plt.close()


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


def train(gaussians, dataset, num_epochs=50, lr=0.0001, device='cuda', output_dir='outputs'):
    """Main training loop"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    optimizer = torch.optim.Adam(gaussians.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.7)
    
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
                
                # Visualize
                visualize_render(
                    gt_img.clip(0, 1),
                    rendered_img.clip(0, 1),
                    epoch, 0, output_dir / 'renders'
                )
                
                print(f"Epoch {epoch:3d}/{num_epochs}: Loss={avg_loss:.5f}, PSNR={psnr:.2f} dB")
    
    return gaussians, history


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}\n")
    
    # Load dataset
    scene = "data/nerf_llff_data/fern"
    ds = LLFFDataset(scene_root=scene, downscale=4.0, device=device)  # Downscale more for speed
    print(f"✓ Loaded {len(ds)} views, resolution {ds.H}x{ds.W}\n")
    
    # Initialize Gaussians
    n_gaussians = 10000
    gaussians = initialize_gaussians_from_cameras(ds, n_gaussians=n_gaussians, device=device)
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
    
    print("\n" + "="*80)
    print("✅ DONE!")
    print("="*80)
    print(f"\nResults saved to: outputs/fern/")


if __name__ == "__main__":
    main()

