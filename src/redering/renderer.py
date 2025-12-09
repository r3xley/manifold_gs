import torch
import numpy as np
def render_improved(gaussians, img_size=64):
    H, W = img_size, img_size
    xx, yy = torch.meshgrid(
        torch.linspace(-1, 1, W, device=gaussians.device),
        torch.linspace(-1, 1, H, device=gaussians.device),
        indexing='xy'
    )
    pixels = torch.stack([xx.flatten(), yy.flatten()], dim=-1)
    
    positions = gaussians.pos[:, :2]
    covariances = gaussians.get_covariance_matrices()[:, :2, :2]
    opacities = gaussians.get_opacity()
    
    img = torch.zeros(H * W, device=gaussians.device)
    for idx in range(gaussians.num):
        cov_inv = torch.inverse(covariances[idx] + 1e-6 * torch.eye(2, device=gaussians.device))
        diff = pixels - positions[idx]
        mahalanobis = torch.sum(diff @ cov_inv * diff, dim=-1)
        g = torch.exp(-0.5 * mahalanobis)
        img = img + opacities[idx] * g
    
    return torch.clamp(img.reshape(H, W), 0, 1)

def render_gaussians_3d(gaussians, camera_dict, img_size=None):
    """
    Proper 3D Gaussian Splatting renderer with camera projection.
    
    Args:
        gaussians: ManifoldGaussian3D instance
        camera_dict: dict with keys:
            - 'pose': (4, 4) camera-to-world transform
            - 'K': (3, 3) intrinsic matrix
            - 'width': int
            - 'height': int
        img_size: optional (H, W) override
    
    Returns:
        rendered: (H, W, 3) RGB image
    """
    device = gaussians.device
    
    # Get image dimensions
    if img_size is None:
        H, W = int(camera_dict['height']), int(camera_dict['width'])
    else:
        H, W = img_size
    
    # === STEP 1: Get Gaussian parameters ===
    positions_3d = gaussians.pos  # (N, 3)
    covariances_3d = gaussians.get_covariance_matrices()  # (N, 3, 3)
    opacities = gaussians.get_opacity()  # (N,)
    
    # === STEP 2: Transform to camera space ===
    pose = camera_dict['pose']  # camera-to-world (4, 4)
    
    # World-to-camera transform
    if isinstance(pose, torch.Tensor):
        w2c = torch.inverse(pose).to(device)
    else:
        w2c = torch.from_numpy(np.linalg.inv(pose)).float().to(device)
    
    # Transform positions: p_cam = R @ p_world + t
    # positions_3d is (N, 3), w2c[:3, :3] is (3, 3), w2c[:3, 3] is (3,)
    positions_cam = positions_3d @ w2c[:3, :3].T + w2c[:3, 3]  # (N, 3)
    
    # === STEP 3: Visibility culling ===
    # Only render Gaussians in front of camera
    # Standard convention: camera looks down +Z axis, so positive Z = in front
    visible_mask = positions_cam[:, 2] > 0.01
    
    # If no Gaussians in front, try negative Z (some coordinate systems)
    if visible_mask.sum() == 0:
        visible_mask = positions_cam[:, 2] < -0.01
        if visible_mask.sum() > 0:
            # Flip Z coordinates for rendering
            positions_cam[:, 2] = -positions_cam[:, 2]
    
    if visible_mask.sum() == 0:
        # Debug: print camera space Z range
        z_min, z_max = positions_cam[:, 2].min().item(), positions_cam[:, 2].max().item()
        print(f"WARNING: No Gaussians visible! Camera Z range: [{z_min:.3f}, {z_max:.3f}]")
        return torch.zeros(H, W, 3, device=device)
    
    # Filter visible Gaussians
    positions_cam = positions_cam[visible_mask]
    covariances_3d_vis = covariances_3d[visible_mask]
    opacities_vis = opacities[visible_mask]
    
    # === STEP 4: Project 3D covariances to 2D ===
    K = camera_dict['K']
    if isinstance(K, np.ndarray):
        K = torch.from_numpy(K).float().to(device)
    elif not isinstance(K, torch.Tensor):
        K = torch.tensor(K, dtype=torch.float32, device=device)
    else:
        K = K.to(device)
    
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    
    # Perspective projection
    depths = positions_cam[:, 2]  # (N,)
    x_2d = (positions_cam[:, 0] / depths) * fx + cx
    y_2d = (positions_cam[:, 1] / depths) * fy + cy
    positions_2d = torch.stack([x_2d, y_2d], dim=-1)  # (N, 2)
    
    # Project 3D covariance to 2D (Jacobian approximation)
    # J = [[fx/z, 0, -fx*x/z²],
    #      [0, fy/z, -fy*y/z²]]
    N_vis = positions_cam.shape[0]
    J = torch.zeros(N_vis, 2, 3, device=device)
    J[:, 0, 0] = fx / depths
    J[:, 0, 2] = -fx * positions_cam[:, 0] / (depths ** 2)
    J[:, 1, 1] = fy / depths
    J[:, 1, 2] = -fy * positions_cam[:, 1] / (depths ** 2)
    
    # Transform covariances: Σ_2d = J @ R @ Σ_3d @ R^T @ J^T
    # Where R is camera rotation (already applied in positions_cam)
    R_cam = w2c[:3, :3]
    
    covariances_2d = torch.zeros(N_vis, 2, 2, device=device)
    for i in range(N_vis):
        # Rotate covariance to camera space: Σ_cam = R @ Σ_world @ R^T
        cov_cam = R_cam @ covariances_3d_vis[i] @ R_cam.T
        # Project to 2D using Jacobian
        covariances_2d[i] = J[i] @ cov_cam @ J[i].T
    
    # Add small regularization for numerical stability
    covariances_2d = covariances_2d + 1e-4 * torch.eye(2, device=device).unsqueeze(0)
    
    # === STEP 5: Depth sorting (back-to-front for alpha blending) ===
    depth_order = torch.argsort(depths, descending=True)  # Render far to near
    positions_2d = positions_2d[depth_order]
    covariances_2d = covariances_2d[depth_order]
    opacities_vis = opacities_vis[depth_order]
    depths = depths[depth_order]
    
    # === STEP 6: Gaussian splatting ===
    # Create pixel grid
    y_coords, x_coords = torch.meshgrid(
        torch.arange(H, device=device, dtype=torch.float32),
        torch.arange(W, device=device, dtype=torch.float32),
        indexing='ij'
    )
    pixels = torch.stack([x_coords.flatten(), y_coords.flatten()], dim=-1)  # (H*W, 2)
    
    # Render with alpha blending (front-to-back accumulation)
    # For proper alpha blending: accumulate from front to back
    depth_order_f2b = torch.argsort(depths)  # Front to back
    positions_2d = positions_2d[depth_order_f2b]
    covariances_2d = covariances_2d[depth_order_f2b]
    opacities_vis = opacities_vis[depth_order_f2b]
    
    img = torch.zeros(H * W, device=device)
    alpha = torch.ones(H * W, device=device)  # Remaining alpha
    
    num_rendered = 0
    for i in range(positions_2d.shape[0]):
        # Check if Gaussian center is within reasonable bounds
        px, py = positions_2d[i]
        if px < -50 or px > W + 50 or py < -50 or py > H + 50:
            continue  # Skip Gaussians far outside image
        
        # Compute Mahalanobis distance
        try:
            cov_inv = torch.inverse(covariances_2d[i] + 1e-6 * torch.eye(2, device=device))
        except:
            continue  # Skip if covariance is singular
        
        diff = pixels - positions_2d[i]  # (H*W, 2)
        mahalanobis = torch.sum(diff @ cov_inv * diff, dim=-1)  # (H*W,)
        
        # Gaussian kernel (with 3-sigma cutoff for efficiency)
        mask = mahalanobis < 9.0  # 3-sigma cutoff
        g = torch.zeros_like(mahalanobis)
        g[mask] = torch.exp(-0.5 * mahalanobis[mask])
        
        # Alpha blending: I = I + alpha * opacity * color
        # For grayscale: color = 1.0
        contribution = alpha * opacities_vis[i] * g
        img = img + contribution
        
        # Update remaining alpha: alpha = alpha * (1 - opacity * g)
        alpha = alpha * (1.0 - opacities_vis[i] * g).clamp(0, 1)
        num_rendered += 1
    
    # Debug: print rendering stats
    if num_rendered == 0:
        print(f"WARNING: No Gaussians rendered! {positions_2d.shape[0]} Gaussians in view but none contributed.")
    
    # Clamp and reshape
    img = torch.clamp(img, 0, 1).reshape(H, W)
    
    # Convert grayscale to RGB (for now)
    img_rgb = img.unsqueeze(-1).repeat(1, 1, 3)
    
    return img_rgb