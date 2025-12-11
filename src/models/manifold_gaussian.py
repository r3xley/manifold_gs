import torch
import torch.nn as nn

class ManifoldGaussian3D(nn.Module):
    def __init__(self, n_gaussians, device='cuda'):
        super().__init__()
        self.num = n_gaussians
        self.device = device
        
        # CRITICAL FIX: Initialize at viewing distance
        # If your camera looks down the -Z axis (standard), place Gaussians at Z ≈ -3 to -5
        # This keeps them in front of camera but not too close
        
        # Option 1: Sphere of Gaussians at proper depth
        self.pos = nn.Parameter(
            torch.randn(n_gaussians, 3, device=device) * 0.3 + 
            torch.tensor([0.0, 0.0, -4.0], device=device)  # Center at Z=-4
        )
        
        # Option 2: If you don't know camera orientation, at least constrain Z
        # Uncomment this instead if camera setup is different:
        # pos_init = torch.randn(n_gaussians, 3, device=device) * 0.3
        # pos_init[:, 2] = -4.0 + torch.randn(n_gaussians, device=device) * 0.5  # Z around -4
        # self.pos = nn.Parameter(pos_init)
        
        # Scales should be appropriate for viewing distance
        # At distance 4, scale of 0.1-0.3 is reasonable
        self.log_scales = nn.Parameter(
            torch.randn(n_gaussians, 3, device=device) * 0.1 - 1.8  # exp(-1.8) ≈ 0.165
        )
        
        quats = torch.zeros(n_gaussians, 4, device=device)
        quats[:, 0] = 1.0
        self.quaternions = nn.Parameter(quats)
        
        # High initial opacity
        self.logit_opacity = nn.Parameter(torch.ones(n_gaussians, device=device) * 2.0)
        
        # Bright, varied colors
        self.rgb_logit = nn.Parameter(
            torch.randn(n_gaussians, 3, device=device) * 0.5 + 2.0
        )
        
    def get_scales(self):
        return torch.exp(self.log_scales)
    
    def get_rotations(self):
        return self.quaternions / (self.quaternions.norm(dim=1, keepdim=True) + 1e-8)
    
    def get_opacity(self):
        return torch.sigmoid(self.logit_opacity)
    
    def get_colors(self):
        """Get RGB colors, clamped to [0, 1]"""
        return torch.sigmoid(self.rgb_logit)
    
    def get_covariance_matrices(self):
        scales = self.get_scales()
        rotations = self.get_rotations()
        R = self.quaternion_to_rotation_matrix(rotations)
        S = torch.diag_embed(scales).to(R.dtype)  # Ensure same dtype as R
        return R @ S @ S.transpose(-2, -1) @ R.transpose(-2, -1)
    
    @staticmethod
    def quaternion_to_rotation_matrix(quaternions):
        w, x, y, z = quaternions[:, 0], quaternions[:, 1], quaternions[:, 2], quaternions[:, 3]
        R = torch.zeros(quaternions.shape[0], 3, 3, device=quaternions.device, dtype=quaternions.dtype)
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