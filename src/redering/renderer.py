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
    colors = gaussians.get_colors()  # (N, 3) RGB colors
    
    img = torch.zeros(H * W, 3, device=gaussians.device)  # RGB image
    for idx in range(gaussians.num):
        cov_inv = torch.inverse(covariances[idx] + 1e-6 * torch.eye(2, device=gaussians.device))
        diff = pixels - positions[idx]
        mahalanobis = torch.sum(diff @ cov_inv * diff, dim=-1)
        g = torch.exp(-0.5 * mahalanobis)
        # Add color contribution: (H*W, 3) = (H*W,) * (1,) * (3,)
        contribution = opacities[idx] * g.unsqueeze(-1) * colors[idx].unsqueeze(0)
        img = img + contribution
    
    return torch.clamp(img.reshape(H, W, 3), 0, 1)

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
    colors = gaussians.get_colors()  # (N, 3) RGB colors
    
    # === STEP 2: Transform to camera space ===
    pose = camera_dict['pose']  # camera-to-world (4, 4)
    
    # World-to-camera transform
    if isinstance(pose, torch.Tensor):
        w2c = torch.inverse(pose).to(device)
    else:
        w2c = torch.from_numpy(np.linalg.inv(pose)).float().to(device)
    
    # Transform positions: p_cam = R @ p_world + t
    # positions_3d is (N, 3) as row vectors, w2c[:3, :3] is (3, 3), w2c[:3, 3] is (3,)
    # For row vectors: p_cam = (R @ p_world^T)^T + t = p_world @ R^T + t
    # w2c[:3, :3] is the world-to-camera rotation matrix R_w2c
    # For row vectors: p_cam = p_world @ R_w2c^T + t_w2c
    # But wait - if w2c is the full transform matrix, then w2c[:3, :3] is R_w2c
    # The correct formula for row vectors is: p_cam = p_world @ R_w2c^T + t_w2c
    # However, we need to be careful: if the pose is camera-to-world, then w2c = inv(pose)
    # and w2c[:3, :3] is indeed R_w2c. So the formula should be correct.
    # BUT - let's try without the transpose to see if that fixes the issue:
    # Actually, the standard formula for transforming a point p (as column vector) is:
    # p_cam = R_w2c @ p_world + t_w2c
    # For row vectors: p_cam = (R_w2c @ p_world^T)^T + t_w2c = p_world @ R_w2c^T + t_w2c
    # So positions_cam = positions_3d @ w2c[:3, :3].T + w2c[:3, 3] is correct.
    # However, if the issue is that Z values are too small, maybe we need to check the coordinate system.
    # Let's try the direct matrix multiplication approach:
    positions_cam = (w2c[:3, :3] @ positions_3d.T + w2c[:3, 3:4]).T  # (N, 3)
    
    # === STEP 3: Visibility culling ===
    # LLFF uses OpenGL convention: camera looks down -Z axis, so negative Z = in front
    # Try negative Z first (LLFF convention)
    visible_mask = positions_cam[:, 2] < -0.01
    
    # If no Gaussians in front, try positive Z (other conventions)
    if visible_mask.sum() == 0:
        visible_mask = positions_cam[:, 2] > 0.01
        if visible_mask.sum() > 0:
            # Using positive Z convention, no flip needed
            pass
    
    # If still no visible, try using all Gaussians (might be coordinate system issue)
    if visible_mask.sum() == 0:
        z_min, z_max = positions_cam[:, 2].min().item(), positions_cam[:, 2].max().item()
        print(f"WARNING: No Gaussians visible with standard culling! Camera Z range: [{z_min:.3f}, {z_max:.3f}]")
        print(f"  Trying with ALL Gaussians (might be coordinate system issue)...")
        # Use all Gaussians - we'll handle depth with absolute value
        visible_mask = torch.ones(positions_cam.shape[0], dtype=torch.bool, device=device)
    
    # Filter visible Gaussians
    positions_cam_vis = positions_cam[visible_mask]
    covariances_3d_vis = covariances_3d[visible_mask]
    opacities_vis = opacities[visible_mask]
    colors_vis = colors[visible_mask]
    
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
    # Handle both OpenGL (-Z forward) and standard (+Z forward) conventions
    z_values = positions_cam_vis[:, 2]
    
    # Convert to positive depth for projection
    # For OpenGL (Z < 0): use -Z (positive depth)
    # For standard (Z > 0): use Z (positive depth)
    # This ensures we always divide by a positive depth
    depths = torch.where(z_values < 0, -z_values, z_values)
    
    # Don't clamp depths too aggressively - use actual depth values
    # Only clamp very small values that would cause numerical issues
    # The issue is likely scene scale, not depth values
    depths = torch.clamp(depths, min=0.01)  # Very small minimum to avoid division by zero
    
    # Debug: check depth range on first render
    if hasattr(render_gaussians_3d, '_debug_count'):
        render_gaussians_3d._debug_count += 1
    else:
        render_gaussians_3d._debug_count = 1
    
    if render_gaussians_3d._debug_count <= 3:
        print(f"Debug projection (call {render_gaussians_3d._debug_count}):")
        print(f"  World positions range: X=[{positions_3d[:, 0].min().item():.3f}, {positions_3d[:, 0].max().item():.3f}], Y=[{positions_3d[:, 1].min().item():.3f}, {positions_3d[:, 1].max().item():.3f}], Z=[{positions_3d[:, 2].min().item():.3f}, {positions_3d[:, 2].max().item():.3f}]")
        print(f"  Camera space positions range: X=[{positions_cam_vis[:, 0].min().item():.3f}, {positions_cam_vis[:, 0].max().item():.3f}], Y=[{positions_cam_vis[:, 1].min().item():.3f}, {positions_cam_vis[:, 1].max().item():.3f}], Z=[{z_values.min().item():.3f}, {z_values.max().item():.3f}]")
        print(f"  Depth range: [{depths.min().item():.3f}, {depths.max().item():.3f}]")
        print(f"  Focal length: fx={fx:.1f}, fy={fy:.1f}")
        print(f"  Principal point: cx={cx:.1f}, cy={cy:.1f}")
        print(f"  Image size: {W}x{H}")
        # Check a sample projection
        sample_idx = 0
        if len(positions_cam_vis) > 0:
            sample_x = positions_cam_vis[sample_idx, 0].item()
            sample_y = positions_cam_vis[sample_idx, 1].item()
            sample_z = depths[sample_idx].item()
            print(f"  Sample projection: cam_pos=({sample_x:.3f}, {sample_y:.3f}, {sample_z:.3f})")
            print(f"    -> 2D: ({sample_x/sample_z * fx + cx:.1f}, {sample_y/sample_z * fy + cy:.1f})")
    
    # Project to 2D image coordinates
    # Standard perspective projection: x = (X/Z) * fx + cx
    # But check if we need to normalize by image size first (some LLFF formats)
    # For now, use standard projection
    x_2d = (positions_cam_vis[:, 0] / depths) * fx + cx
    y_2d = (positions_cam_vis[:, 1] / depths) * fy + cy
    positions_2d = torch.stack([x_2d, y_2d], dim=-1)  # (N, 2)
    
    # Debug: if projections are way outside, the scene scale might be wrong
    if render_gaussians_3d._debug_count <= 3:
        x_ratio = (positions_cam_vis[:, 0] / depths).abs().max()
        y_ratio = (positions_cam_vis[:, 1] / depths).abs().max()
        if x_ratio > 2.0 or y_ratio > 2.0:
            print(f"  WARNING: Large X/Z or Y/Z ratios detected!")
            print(f"    Max |X/Z|: {x_ratio:.2f}, Max |Y/Z|: {y_ratio:.2f}")
            print(f"    This suggests scene scale mismatch. Expected ratios < 1.0 for points in view.")
            print(f"    Consider: 1) Scaling scene coordinates, 2) Adjusting focal length, 3) Better Gaussian initialization")
    
    if render_gaussians_3d._debug_count <= 3:
        print(f"  Projected 2D range: X=[{x_2d.min().item():.1f}, {x_2d.max().item():.1f}], Y=[{y_2d.min().item():.1f}, {y_2d.max().item():.1f}]")
        print()
    
    # Project 3D covariance to 2D (Jacobian approximation)
    # J = [[fx/z, 0, -fx*x/z²],
    #      [0, fy/z, -fy*y/z²]]
    N_vis = positions_cam_vis.shape[0]  # Use filtered count
    J = torch.zeros(N_vis, 2, 3, device=device)
    # Use the same depths as projection (already positive)
    J[:, 0, 0] = fx / depths
    J[:, 0, 2] = -fx * positions_cam_vis[:, 0] / (depths ** 2)
    J[:, 1, 1] = fy / depths
    J[:, 1, 2] = -fy * positions_cam_vis[:, 1] / (depths ** 2)
    
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
    
    # === STEP 5: Depth sorting (front-to-back for alpha blending) ===
    # Sort by absolute depth (closest first for proper alpha blending)
    depth_order = torch.argsort(torch.abs(depths))  # Front to back
    positions_2d = positions_2d[depth_order]
    covariances_2d = covariances_2d[depth_order]
    opacities_vis = opacities_vis[depth_order]
    colors_vis = colors_vis[depth_order]
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
    img = torch.zeros(H * W, 3, device=device)  # RGB image
    alpha = torch.ones(H * W, device=device)  # Remaining alpha
    
    num_rendered = 0
    for i in range(positions_2d.shape[0]):
        # Check if Gaussian center is within reasonable bounds
        # Use a larger margin to catch more Gaussians (some may have large covariances)
        px, py = positions_2d[i]
        # Calculate approximate radius based on covariance
        cov_diag = torch.diagonal(covariances_2d[i])
        radius = torch.sqrt(cov_diag.max()) * 3.0  # 3-sigma radius
        
        # VERY lenient bounds check - allow Gaussians that are even partially in view
        # Use a large fixed margin to catch more Gaussians
        # The issue is projections are way outside, so we need a huge margin
        margin = max(radius.item() * 2.0, 200.0)  # At least 200 pixels margin, or 2x radius
        if px < -margin or px > W + margin or py < -margin or py > H + margin:
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
        
        # Alpha blending: I = I + alpha * opacity * g * color
        # contribution shape: (H*W, 3) = (H*W, 1) * (1,) * (H*W,) * (3,)
        # Use proper alpha blending: accumulate color weighted by remaining alpha
        alpha_weight = alpha.unsqueeze(-1)  # (H*W, 1)
        opacity_weight = opacities_vis[i]  # scalar
        gaussian_weight = g.unsqueeze(-1)  # (H*W, 1)
        color = colors_vis[i].unsqueeze(0)  # (1, 3)
        
        contribution = alpha_weight * opacity_weight * gaussian_weight * color
        img = img + contribution
        
        # Update remaining alpha: alpha = alpha * (1 - opacity * g)
        # Clamp to prevent numerical issues
        alpha = alpha * (1.0 - opacity_weight * g).clamp(0.0, 1.0)
        num_rendered += 1
    
    # Debug: print rendering stats
    if num_rendered == 0:
        print(f"WARNING: No Gaussians rendered! {positions_2d.shape[0]} Gaussians in view but none contributed.")
        print(f"  Opacity range: [{opacities_vis.min().item():.4f}, {opacities_vis.max().item():.4f}]")
        print(f"  Color range: [{colors_vis.min().item():.4f}, {colors_vis.max().item():.4f}]")
        print(f"  Position 2D range: X=[{positions_2d[:, 0].min().item():.1f}, {positions_2d[:, 0].max().item():.1f}], Y=[{positions_2d[:, 1].min().item():.1f}, {positions_2d[:, 1].max().item():.1f}]")
        print(f"  Image size: {W}x{H}")
        # Count how many are within bounds
        in_bounds = ((positions_2d[:, 0] >= -50) & (positions_2d[:, 0] <= W + 50) & 
                     (positions_2d[:, 1] >= -50) & (positions_2d[:, 1] <= H + 50)).sum().item()
        print(f"  Gaussians within bounds (with 50px margin): {in_bounds} / {positions_2d.shape[0]}")
        # Return a small non-zero image to help debug
        return torch.ones(H, W, 3, device=device) * 0.1
    
    # Debug: if very few rendered, print info
    if num_rendered < 10 and render_gaussians_3d._debug_count <= 3:
        print(f"WARNING: Only {num_rendered} Gaussians rendered out of {positions_2d.shape[0]} in view!")
        print(f"  Position 2D range: X=[{positions_2d[:, 0].min().item():.1f}, {positions_2d[:, 0].max().item():.1f}], Y=[{positions_2d[:, 1].min().item():.1f}, {positions_2d[:, 1].max().item():.1f}]")
        print(f"  Image size: {W}x{H}")
        in_bounds = ((positions_2d[:, 0] >= -100) & (positions_2d[:, 0] <= W + 100) & 
                     (positions_2d[:, 1] >= -100) & (positions_2d[:, 1] <= H + 100)).sum().item()
        print(f"  Gaussians within 100px margin: {in_bounds} / {positions_2d.shape[0]}")
    
    # Clamp and reshape to (H, W, 3)
    img = torch.clamp(img, 0, 1).reshape(H, W, 3)
    
    # Debug: check if image is too dark
    if img.max() < 0.01:
        print(f"WARNING: Rendered image is very dark! Max value: {img.max().item():.6f}")
        print(f"  Rendered {num_rendered} Gaussians")
        print(f"  Opacity range: [{opacities_vis.min().item():.4f}, {opacities_vis.max().item():.4f}]")
        print(f"  Color range: [{colors_vis.min().item():.4f}, {colors_vis.max().item():.4f}]")
    
    return img