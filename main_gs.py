import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from src.models.manifold_gaussian import ManifoldGaussian3D
from src.training.train import train_improved_gaussians
from src.redering.renderer import render_improved

# -----------------------------
# PSNR function
# -----------------------------
def psnr(img1, img2):
    mse = F.mse_loss(img1, img2)
    if mse == 0:
        return float('inf')
    return 20 * torch.log10(torch.tensor(1.0) / torch.sqrt(mse))

# -----------------------------
# Vanilla Gaussian 2D
# -----------------------------
class VanillaGaussian2D(nn.Module):
    def __init__(self, n_gaussians, device='cuda'):
        super().__init__()
        self.num = n_gaussians
        self.device = device
        self.pos = nn.Parameter(torch.randn(n_gaussians, 2, device=device) * 0.5)
        self.log_scales = nn.Parameter(torch.randn(n_gaussians, device=device) * 0.1 - 1.5)
        self.logit_opacity = nn.Parameter(torch.ones(n_gaussians, device=device) * (-1.5))

    def get_scales(self):
        return torch.exp(self.log_scales)

    def get_opacity(self):
        return torch.sigmoid(self.logit_opacity)

# -----------------------------
# Vanilla rendering
# -----------------------------
def render_vanilla(gaussians, img_size=64):
    H, W = img_size, img_size
    xx, yy = torch.meshgrid(
        torch.linspace(-1, 1, W, device=gaussians.device),
        torch.linspace(-1, 1, H, device=gaussians.device),
        indexing='xy'
    )
    pixels = torch.stack([xx.flatten(), yy.flatten()], dim=-1)
    scales = gaussians.get_scales()
    positions = gaussians.pos
    opacities = gaussians.get_opacity()
    img = torch.zeros(H * W, device=gaussians.device)

    for idx in range(gaussians.num):
        diff = pixels - positions[idx]
        g = torch.exp(-0.5 * torch.sum(diff**2, dim=-1) / (scales[idx]**2 + 1e-8))
        img += opacities[idx] * g

    return torch.clamp(img.reshape(H, W), 0, 1)

# -----------------------------
# Train vanilla
# -----------------------------
def train_vanilla_gaussians(gaussians, target, num_steps=500, lr=0.015):
    optimizer = torch.optim.Adam(gaussians.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=150, gamma=0.5)

    for step in range(num_steps):
        optimizer.zero_grad()
        rendered = render_vanilla(gaussians, img_size=target.shape[0])
        loss = ((rendered - target)**2).mean()
        loss.backward()
        optimizer.step()
        scheduler.step()
    return gaussians

# -----------------------------
# Complex target generator
# -----------------------------
def create_hard_target(img_size=64, n_blobs=15, device='cuda'):
    H, W = img_size, img_size
    xx, yy = torch.meshgrid(
        torch.linspace(-1, 1, W, device=device),
        torch.linspace(-1, 1, H, device=device),
        indexing='xy'
    )
    target = torch.zeros(H, W, device=device)

    for _ in range(n_blobs):
        # Random center
        cx, cy = (2*torch.rand(2, device=device) - 1)
        # Random anisotropic scales
        sx, sy = 0.05 + 0.15*torch.rand(2, device=device)
        # Random rotation angle
        angle = 360 * torch.rand(1, device=device)
        theta = angle / 180 * 3.1415926
        cos_t, sin_t = torch.cos(theta), torch.sin(theta)

        x = xx - cx
        y = yy - cy
        # Rotate coordinates
        xr = cos_t * x + sin_t * y
        yr = -sin_t * x + cos_t * y

        # Random opacity for each blob
        opacity = 0.5 + 0.5*torch.rand(1, device=device)

        g = opacity * torch.exp(-0.5 * ((xr/sx)**2 + (yr/sy)**2))
        target += g

    return torch.clamp(target, 0, 1)

# -----------------------------
# Main comparison
# -----------------------------
def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    n_gaussians = 20
    img_size = 64
    num_steps = 500

    # Complex target
    target = create_hard_target(img_size=img_size, device=device)

    # -----------------------------
    # Manifold Gaussian
    # -----------------------------
    manifold_gaussians = ManifoldGaussian3D(n_gaussians, device=device)
    print("Training Manifold Gaussian...")
    train_improved_gaussians(manifold_gaussians, target, num_steps=num_steps, lr=0.015)
    manifold_rendered = render_improved(manifold_gaussians, img_size=img_size)
    psnr_manifold = psnr(target.cpu(), manifold_rendered.detach().cpu())
    print(f"PSNR Manifold: {psnr_manifold:.2f} dB")

    # -----------------------------
    # Vanilla Gaussian
    # -----------------------------
    vanilla_gaussians = VanillaGaussian2D(n_gaussians, device=device)
    print("Training Vanilla Gaussian...")
    train_vanilla_gaussians(vanilla_gaussians, target, num_steps=num_steps, lr=0.015)
    vanilla_rendered = render_vanilla(vanilla_gaussians, img_size=img_size)
    psnr_vanilla = psnr(target.cpu(), vanilla_rendered.detach().cpu())
    print(f"PSNR Vanilla: {psnr_vanilla:.2f} dB")

    # -----------------------------
    # Plot comparison
    # -----------------------------
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.title("Target")
    plt.imshow(target.cpu(), cmap='viridis')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.title(f"Manifold (PSNR={psnr_manifold:.2f} dB)")
    plt.imshow(manifold_rendered.detach().cpu(), cmap='viridis')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title(f"Vanilla (PSNR={psnr_vanilla:.2f} dB)")
    plt.imshow(vanilla_rendered.detach().cpu(), cmap='viridis')
    plt.axis('off')

    plt.show()

if __name__ == "__main__":
    main()
