"""
Minimal memory training with live image display
Shows rendered images as training progresses
"""
import torch
import numpy as np
from pathlib import Path
from data.load_lff import LLFFDataset
from src.models.manifold_gaussian import ManifoldGaussian3D
from src.redering.renderer import render_gaussians_3d
from train_llff import initialize_gaussians_from_colmap, visualize_render
import matplotlib.pyplot as plt

# Global figure for live display
_live_fig = None
_live_axes = None

def train_minimal(gaussians, dataset, num_epochs=20, lr=0.0001, device='cuda', output_dir='outputs'):
    """Minimal training loop with live image display"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    optimizer = torch.optim.Adam(gaussians.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.8)
    
    print("="*80)
    print("MINIMAL TRAINING WITH LIVE DISPLAY")
    print("="*80)
    print(f"Epochs: {num_epochs} | Gaussians: {gaussians.num} | LR: {lr}")
    print(f"Resolution: {dataset.H}x{dataset.W}")
    print("="*80 + "\n")
    
    history = {'loss': [], 'psnr': []}
    
    # Initialize live display
    global _live_fig, _live_axes
    plt.ion()  # Interactive mode
    _live_fig, _live_axes = plt.subplots(1, 3, figsize=(15, 5))
    _live_fig.suptitle('Training Progress - Close window to stop', fontsize=14)
    
    try:
        for epoch in range(num_epochs):
            epoch_loss = 0
            num_batches = 0
            
            # Shuffle views
            indices = torch.randperm(len(dataset))
            
            for idx in indices:
                optimizer.zero_grad()
                
                # Get data
                sample = dataset[int(idx)]
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
                
                # Loss
                recon_loss = ((img_pred - img_gt_hwc) ** 2).mean()
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
            
            # Show live image every epoch (or every 2 epochs for speed)
            if epoch % 2 == 0 or epoch == 0:
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
                    
                    # Update live display
                    gt_display = np.clip(gt_img, 0, 1)
                    pred_display = np.clip(rendered_img, 0, 1)
                    
                    # Compute difference
                    if len(gt_img.shape) == 3:
                        gt_gray = gt_img.mean(axis=-1)
                    else:
                        gt_gray = gt_img
                    if len(rendered_img.shape) == 3:
                        pred_gray = rendered_img.mean(axis=-1)
                    else:
                        pred_gray = rendered_img
                    diff = np.abs(gt_gray - pred_gray)
                    
                    # Update axes
                    _live_axes[0].clear()
                    _live_axes[0].imshow(gt_display)
                    _live_axes[0].set_title(f'Ground Truth\nEpoch {epoch}/{num_epochs}')
                    _live_axes[0].axis('off')
                    
                    _live_axes[1].clear()
                    _live_axes[1].imshow(pred_display)
                    _live_axes[1].set_title(f'Predicted\nPSNR: {psnr:.2f} dB | Loss: {avg_loss:.5f}')
                    _live_axes[1].axis('off')
                    
                    _live_axes[2].clear()
                    _live_axes[2].imshow(diff, cmap='hot')
                    _live_axes[2].set_title('Difference')
                    _live_axes[2].axis('off')
                    
                    plt.tight_layout()
                    plt.draw()
                    plt.pause(0.1)  # Small pause to update
                    
                    # Save to file every 5 epochs
                    if epoch % 5 == 0:
                        visualize_render(
                            gt_display,
                            pred_display,
                            epoch, 0, output_dir / 'renders',
                            show_live=False  # Don't create another window
                        )
                    
                    print(f"Epoch {epoch:3d}/{num_epochs}: Loss={avg_loss:.5f}, PSNR={psnr:.2f} dB")
            
            # Check if window was closed
            if not plt.get_fignums():
                print("\n⚠ Display window closed by user. Stopping training...")
                break
    
    except KeyboardInterrupt:
        print("\n⚠ Training interrupted by user")
    finally:
        # Close display
        if _live_fig is not None:
            plt.close(_live_fig)
        plt.ioff()  # Turn off interactive mode
    
    return gaussians, history


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}\n")
    
    # MINIMAL MEMORY SETTINGS
    scene = "data/nerf_llff_data/fern"
    downscale = 16.0  # Very small images for memory safety
    n_gaussians = 2000  # Few Gaussians
    num_epochs = 20  # Quick training
    
    print("="*80)
    print("MINIMAL TRAINING WITH LIVE IMAGE DISPLAY")
    print("="*80)
    print(f"Gaussians: {n_gaussians}")
    print(f"Downscale: {downscale}x")
    print(f"Epochs: {num_epochs}")
    print(f"Device: {device}")
    print("="*80 + "\n")
    print("A window will open showing training progress.")
    print("Images update every 2 epochs.")
    print("Close the window or press Ctrl+C to stop training.\n")
    
    # Load dataset
    ds = LLFFDataset(scene_root=scene, downscale=downscale, device=device)
    print(f"✓ Loaded {len(ds)} views, resolution {ds.H}x{ds.W}\n")
    
    # Initialize Gaussians from COLMAP
    gaussians = initialize_gaussians_from_colmap(
        ds, 
        n_gaussians=n_gaussians, 
        device=device, 
        scene_root=scene
    )
    print()
    
    # Train with live display
    gaussians, history = train_minimal(
        gaussians, ds,
        num_epochs=num_epochs,
        lr=0.0001,
        device=device,
        output_dir='outputs/fern_minimal'
    )
    
    print("\n" + "="*80)
    print("✅ TRAINING COMPLETE!")
    print("="*80)
    if history['psnr']:
        print(f"Final PSNR: {history['psnr'][-1]:.2f} dB")
        print(f"Best PSNR: {max(history['psnr']):.2f} dB")
    print(f"\nResults saved to: outputs/fern_minimal/")


if __name__ == "__main__":
    main()
