import torch
import torch.nn as nn

class ManifoldGaussian3D(nn.Module):
    """Enhanced manifold Gaussians with better regularization"""
    def __init__(self, n_gaussians, device='cuda'):
        super().__init__()
        self.num = n_gaussians
        self.device = device
        
        self.pos = nn.Parameter(torch.randn(n_gaussians, 3, device=device) * 0.15)
        self.log_scales = nn.Parameter(torch.randn(n_gaussians, 3, device=device) * 0.1 - 2.5)
        quats = torch.zeros(n_gaussians, 4, device=device)
        quats[:, 0] = 1.0  # w component = 1 (no rotation)
        self.quaternions = nn.Parameter(quats)
        # Initialize opacity higher so Gaussians are more visible initially
        self.logit_opacity = nn.Parameter(torch.ones(n_gaussians, device=device) * (-0.5))
    
    def get_scales(self):
        return torch.exp(self.log_scales)
    
    def get_rotations(self):
        return self.quaternions / (self.quaternions.norm(dim=1, keepdim=True) + 1e-8)
    
    def get_opacity(self):
        return torch.sigmoid(self.logit_opacity)
    
    def get_covariance_matrices(self):
        scales = self.get_scales()
        rotations = self.get_rotations()
        R = self.quaternion_to_rotation_matrix(rotations)
        S = torch.diag_embed(scales)
        return R @ S @ S.transpose(-2, -1) @ R.transpose(-2, -1)
    
    @staticmethod
    def quaternion_to_rotation_matrix(quaternions):
        w, x, y, z = quaternions[:, 0], quaternions[:, 1], quaternions[:, 2], quaternions[:, 3]
        R = torch.zeros(quaternions.shape[0], 3, 3, device=quaternions.device)
        R[:, 0, 0] = 1 - 2*(y**2 + z**2)
        R[:, 0, 1] = 2*(x*y - w*z)
        R[:, 0, 2] = 2*(x*z + w*y)
        R[:, 1, 0] = 2*(x*y + w*z)
        R[:, 1, 1] = 1 - 2*(x**2 + z**2)
        R[:, 1, 2] = 2*(y*z - w*x)
        R[:, 2, 0] = 2*(x*z - w*y)
        R[:, 2, 1] = 2*(y*z + w*x)
        R[:, 2, 2] = 1 - 2*(x**2 + y**2)
        return R