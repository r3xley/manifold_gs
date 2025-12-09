# src/visualization/rerun_viz.py
import numpy as np
import torch
import rerun as rr

def init_rerun(app_name="3DGS Rerun"):
    """
    Initialize and spawn the Rerun UI.
    """
    rr.init(app_name, spawn=True)

def _rotation_matrix_to_quat(R):
    """
    Convert a 3x3 rotation matrix to quaternion (x, y, z, w).
    Uses Shepperd's method for numerical stability.
    """
    R = np.asarray(R, dtype=float)
    
    # Ensure R is a proper rotation matrix (orthonormalize if needed)
    U, _, Vt = np.linalg.svd(R)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    
    trace = np.trace(R)
    
    if trace > 0:
        s = np.sqrt(trace + 1.0) * 2  # s = 4 * qw
        qw = 0.25 * s
        qx = (R[2, 1] - R[1, 2]) / s
        qy = (R[0, 2] - R[2, 0]) / s
        qz = (R[1, 0] - R[0, 1]) / s
    elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2  # s = 4 * qx
        qw = (R[2, 1] - R[1, 2]) / s
        qx = 0.25 * s
        qy = (R[0, 1] + R[1, 0]) / s
        qz = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2  # s = 4 * qy
        qw = (R[0, 2] - R[2, 0]) / s
        qx = (R[0, 1] + R[1, 0]) / s
        qy = 0.25 * s
        qz = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2  # s = 4 * qz
        qw = (R[1, 0] - R[0, 1]) / s
        qx = (R[0, 2] + R[2, 0]) / s
        qy = (R[1, 2] + R[2, 1]) / s
        qz = 0.25 * s
    
    # Normalize quaternion
    q = np.array([qx, qy, qz, qw], dtype=float)
    q = q / (np.linalg.norm(q) + 1e-10)
    
    # Return quaternion in xyzw order (Rerun expects xyzw)
    return q

def log_camera_frustum(name, K, pose4x4, image_size=(640, 480), color=[0.8,0.2,0.2]):
    """
    Logs a camera in 3D space using Rerun's Transform3D + Points3D for frustum corners.
    pose4x4: camera->world transform (torch tensor or numpy array)
    K: 3x3 intrinsic (numpy array or torch tensor)
    image_size: (width, height)
    color: rgb float list [0..1] or int list [0..255] (will be normalized)
    """
    if isinstance(pose4x4, torch.Tensor):
        pose = pose4x4.cpu().numpy()
    else:
        pose = np.asarray(pose4x4)
    
    if isinstance(K, torch.Tensor):
        K = K.cpu().numpy()
    else:
        K = np.asarray(K)

    # Normalize color to [0, 1] if it's in [0, 255] range
    if isinstance(color, (list, tuple)) and len(color) == 3:
        if all(isinstance(c, (int, np.integer)) and c > 1 for c in color):
            color = [c / 255.0 for c in color]

    # pose is camera->world transform
    pos = pose[:3, 3]
    R = pose[:3, :3]

    quat = _rotation_matrix_to_quat(R)

    rr.log(
        f"{name}/transform",
        rr.Transform3D(
            translation=pos.tolist(),
            rotation=rr.Quaternion(xyzw=quat.tolist()),
        )
    )

    # Log frustum corners as points
    w, h = image_size[0], image_size[1]
    fx, fy = float(K[0,0]), float(K[1,1])
    cx, cy = float(K[0,2]), float(K[1,2])
    near, far = 0.1, 3.0  # More reasonable near/far planes
    
    # Generate frustum corners in camera space
    corners_cam = []
    for z in [near, far]:
        for x_sign in [-1, 1]:
            for y_sign in [-1, 1]:
                # Use image bounds to compute frustum corners
                x_pixel = w if x_sign > 0 else 0
                y_pixel = h if y_sign > 0 else 0
                xv = (x_pixel - cx) / fx * z
                yv = (y_pixel - cy) / fy * z
                corners_cam.append([xv, yv, z])
    
    corners_cam = np.array(corners_cam)
    corners_world = (R @ corners_cam.T).T + pos[None, :]
    
    # Log frustum corners
    rr.log(
        f"{name}/frustum_corners", 
        rr.Points3D(
            positions=corners_world.tolist(), 
            colors=[color] * len(corners_world),
            radii=0.01
        )
    )

def log_gaussian_points(name, gaussians, radius=0.02):
    """
    Log Gaussian centers as Points3D.
    gaussians.pos: torch tensor (N,3)
    """
    pts = gaussians.pos.detach().cpu().numpy()
    rr.log(name, rr.Points3D(positions=pts.tolist(), radii=radius))

def log_gaussians_as_ellipsoids(name_root, gaussians, max_objects=1000, colors=None, oriented=True):
    """
    Log Gaussians as ellipsoids.

    By default (oriented=True) this function logs oriented ellipsoids using rr.Ellipsoids3D with
    proper rotation. This preserves the manifold structure of the Gaussians.
    
    Args:
      - name_root: string prefix for logging (e.g. "Gaussians/ellipsoids")
      - gaussians: object with .pos (N,3) torch tensor and .get_covariance_matrices() -> (N,3,3) torch tensor
      - max_objects: maximum number of gaussians to log
      - colors: None or list of RGB values. If list of 3 ints (0-255), will be normalized to [0,1]
      - oriented: bool: if True, use oriented ellipsoids (default True for manifold Gaussians)
    """
    pos = gaussians.pos.detach().cpu().numpy()
    covs = gaussians.get_covariance_matrices().detach().cpu().numpy()
    N = pos.shape[0]
    Nlog = min(N, max_objects)

    # Normalize colors if needed
    normalized_color = None
    if colors is not None:
        if isinstance(colors, (list, tuple)) and len(colors) == 3:
            if all(isinstance(c, (int, np.integer)) and c > 1 for c in colors):
                # Convert from [0, 255] to [0, 1]
                normalized_color = [c / 255.0 for c in colors]
            else:
                normalized_color = list(colors)

    if not oriented:
        # Use axis-aligned ellipsoids (faster but loses orientation)
        centers = []
        half_sizes = []
        ellipsoid_colors = []

        for i in range(Nlog):
            Sigma = covs[i]
            # Ensure symmetric
            Sigma = 0.5 * (Sigma + Sigma.T)
            eigvals, eigvecs = np.linalg.eigh(Sigma)

            # Numerical safety: clamp eigenvalues to small positive value
            eigvals = np.clip(eigvals, 1e-6, None)
            axes_lengths = np.sqrt(eigvals)  # semi-axis lengths (std dev along principal axes)

            centers.append(pos[i].tolist())
            half_sizes.append(axes_lengths.tolist())
            ellipsoid_colors.append(normalized_color if normalized_color else None)

        # Filter out None colors
        final_colors = ellipsoid_colors if normalized_color else None
        
        rr.log(
            name_root,
            rr.Ellipsoids3D(
                centers=centers,
                half_sizes=half_sizes,
                colors=final_colors,
            ),
        )
        return

    # Oriented mode: log individual ellipsoids with transforms
    # For performance, we batch log transforms and then log ellipsoids at origin
    # Note: This approach logs each ellipsoid individually to preserve orientation
    for i in range(Nlog):
        Sigma = covs[i]
        # Ensure symmetric
        Sigma = 0.5 * (Sigma + Sigma.T)
        eigvals, eigvecs = np.linalg.eigh(Sigma)

        # Numerical safety: clamp eigenvalues
        eigvals = np.clip(eigvals, 1e-6, None)
        axes_lengths = np.sqrt(eigvals)

        # Ensure right-handed coordinate system
        if np.linalg.det(eigvecs) < 0:
            eigvecs[:, -1] *= -1

        # Convert rotation matrix to quaternion
        R = eigvecs  # columns are principal axes
        quat = _rotation_matrix_to_quat(R)

        # Log transform for this ellipsoid
        ellipsoid_name = f"{name_root}/gaussian_{i:06d}"
        rr.log(
            ellipsoid_name,
            rr.Transform3D(
                translation=pos[i].tolist(),
                rotation=rr.Quaternion(xyzw=quat.tolist()),
            )
        )
        
        # Log ellipsoid at origin (will be transformed by parent)
        # Use half_sizes as the scale
        ellipsoid_color = normalized_color if normalized_color else [0.5, 0.5, 0.5]
        rr.log(
            f"{ellipsoid_name}/ellipsoid",
            rr.Ellipsoids3D(
                centers=[[0, 0, 0]],
                half_sizes=[axes_lengths.tolist()],
                colors=[ellipsoid_color],
            ),
        )
