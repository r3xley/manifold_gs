# src/training/train.py
import torch
import torch.optim as optim
from ..redering.renderer import render_improved, render_gaussians_3d


def train_improved_gaussians(gaussians, target, num_steps=500, lr=0.015):
    """
    Train Gaussians on a single 2D target image.
    
    Args:
        gaussians: ManifoldGaussian3D instance
        target: (H, W) target image
        num_steps: number of optimization steps
        lr: learning rate
    
    Returns:
        losses, recon_losses, reg_losses: training metrics
    """
    optimizer = optim.Adam(gaussians.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=150, gamma=0.5)

    losses, recon_losses, reg_losses = [], [], []

    for step in range(num_steps):
        optimizer.zero_grad()
        
        rendered = render_improved(gaussians, img_size=target.shape[0])
        
        # Reconstruction loss
        recon_loss = ((rendered - target)**2).mean()
        
        # Regularization
        opacities = gaussians.get_opacity()
        opacity_l1 = opacities.mean()
        opacity_entropy = -(opacities * torch.log(opacities + 1e-8) +
                           (1-opacities)*torch.log(1-opacities + 1e-8)).mean()
        scales = gaussians.get_scales()
        scale_reg = (scales**2).mean()
        
        # Total loss
        loss = recon_loss + 0.05*opacity_l1 + 0.01*opacity_entropy + 0.001*scale_reg
        
        # Optimize
        loss.backward()
        torch.nn.utils.clip_grad_norm_(gaussians.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        
        # Track metrics
        losses.append(loss.item())
        recon_losses.append(recon_loss.item())
        reg_losses.append((0.05*opacity_l1 + 0.01*opacity_entropy + 0.001*scale_reg).item())
    
    return losses, recon_losses, reg_losses


def train_multiview_3d(gaussians, dataset, num_epochs=30, batch_size=1, lr=0.0001):
    """
    Train on multi-view dataset (LLFF or NeRF Synthetic).
    NO RERUN LOGGING - Pure training only.
    
    Args:
        gaussians: ManifoldGaussian3D instance
        dataset: LLFFDataset or NerfSyntheticDataset
        num_epochs: number of training epochs
        batch_size: views per batch
        lr: learning rate
    
    Returns:
        gaussians: trained model
        history: dict with training metrics
    """
    device = gaussians.device
    optimizer = torch.optim.Adam(gaussians.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.7)
    
    # Track training history
    history = {
        'epoch_losses': [],
        'test_psnrs': [],
        'learning_rates': []
    }
    
    print(f"Training on {len(dataset)} views for {num_epochs} epochs...")
    
    for epoch in range(num_epochs):
        epoch_loss = 0
        num_batches = 0
        
        # Shuffle dataset
        indices = torch.randperm(len(dataset))
        
        for batch_idx in range(0, len(dataset), batch_size):
            batch_indices = indices[batch_idx:batch_idx + batch_size]
            
            optimizer.zero_grad()
            batch_loss = 0
            
            for idx in batch_indices:
                # Get data
                sample = dataset[int(idx)]
                img_gt = sample['image'].to(device)  # (3, H, W)
                pose = sample['pose'].to(device)  # (4, 4)
                K = sample['K'].to(device)  # (3, 3)
                
                # Prepare camera dict
                H, W = img_gt.shape[1], img_gt.shape[2]
                camera = {
                    'pose': pose,
                    'K': K,
                    'width': W,
                    'height': H
                }
                
                # Render from this camera
                img_pred = render_gaussians_3d(gaussians, camera)  # (H, W, 3)
                
                # Convert ground truth to (H, W, 3)
                img_gt_hwc = img_gt.permute(1, 2, 0)  # (H, W, 3)
                
                # Loss
                recon_loss = ((img_pred - img_gt_hwc) ** 2).mean()
                
                # Regularization
                opacity_reg = 0.01 * gaussians.get_opacity().mean()
                scale_reg = 0.001 * (gaussians.get_scales() ** 2).mean()
                
                loss = recon_loss + opacity_reg + scale_reg
                batch_loss += loss
            
            # Average over batch
            batch_loss = batch_loss / len(batch_indices)
            
            # Backward
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(gaussians.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_loss += batch_loss.item()
            num_batches += 1
        
        scheduler.step()
        
        # Compute average loss
        avg_loss = epoch_loss / num_batches
        
        # Evaluate on test view
        with torch.no_grad():
            test_sample = dataset[0]
            test_camera = {
                'pose': test_sample['pose'].to(device),
                'K': test_sample['K'].to(device),
                'width': test_sample['image'].shape[2],
                'height': test_sample['image'].shape[1]
            }
            
            rendered = render_gaussians_3d(gaussians, test_camera)
            
            mse = ((rendered - test_sample['image'].permute(1, 2, 0).to(device)) ** 2).mean()
            psnr = -10 * torch.log10(mse)
        
        # Track history
        history['epoch_losses'].append(avg_loss)
        history['test_psnrs'].append(psnr.item())
        history['learning_rates'].append(optimizer.param_groups[0]['lr'])
        
        # Print progress
        print(f"Epoch {epoch}/{num_epochs}: Loss={avg_loss:.5f}, Test PSNR={psnr:.2f} dB")
    
    print("\n✅ Training complete!")
    return gaussians, history


def evaluate_on_views(gaussians, dataset, num_views=None):
    """
    Evaluate trained Gaussians on multiple views.
    
    Args:
        gaussians: trained ManifoldGaussian3D
        dataset: dataset to evaluate on
        num_views: number of views to evaluate (None = all)
    
    Returns:
        dict with evaluation metrics
    """
    device = gaussians.device
    
    if num_views is None:
        num_views = len(dataset)
    
    psnrs = []
    mses = []
    
    print(f"Evaluating on {num_views} views...")
    
    with torch.no_grad():
        for i in range(min(num_views, len(dataset))):
            sample = dataset[i]
            
            camera = {
                'pose': sample['pose'].to(device),
                'K': sample['K'].to(device),
                'width': sample['image'].shape[2],
                'height': sample['image'].shape[1]
            }
            
            # Render
            rendered = render_gaussians_3d(gaussians, camera)
            
            # Ground truth
            gt = sample['image'].permute(1, 2, 0).to(device)
            
            # Metrics
            mse = ((rendered - gt) ** 2).mean()
            psnr = -10 * torch.log10(mse)
            
            psnrs.append(psnr.item())
            mses.append(mse.item())
            
            if i % 10 == 0:
                print(f"  View {i}/{num_views}: PSNR = {psnr:.2f} dB")
    
    results = {
        'psnrs': psnrs,
        'mses': mses,
        'mean_psnr': sum(psnrs) / len(psnrs),
        'mean_mse': sum(mses) / len(mses),
        'min_psnr': min(psnrs),
        'max_psnr': max(psnrs)
    }
    
    print(f"\nEvaluation Results:")
    print(f"  Mean PSNR: {results['mean_psnr']:.2f} dB")
    print(f"  Min PSNR:  {results['min_psnr']:.2f} dB")
    print(f"  Max PSNR:  {results['max_psnr']:.2f} dB")
    
    return results