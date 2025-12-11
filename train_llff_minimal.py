"""
Minimal memory training script - for systems with limited RAM
"""
import torch
import numpy as np
from pathlib import Path
from data.load_lff import LLFFDataset
from src.models.manifold_gaussian import ManifoldGaussian3D
from src.redering.renderer import render_gaussians_3d
from train_llff import initialize_gaussians_from_colmap, train, visualize_render

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}\n")
    
    # MINIMAL MEMORY SETTINGS
    scene = "data/nerf_llff_data/fern"
    downscale = 16.0  # Very small images
    n_gaussians = 2000  # Few Gaussians
    num_epochs = 20  # Fewer epochs
    
    print("="*80)
    print("MINIMAL MEMORY TRAINING")
    print("="*80)
    print(f"Gaussians: {n_gaussians}")
    print(f"Downscale: {downscale}x")
    print(f"Epochs: {num_epochs}")
    print("="*80 + "\n")
    
    # Load dataset
    ds = LLFFDataset(scene_root=scene, downscale=downscale, device=device)
    print(f"✓ Loaded {len(ds)} views, resolution {ds.H}x{ds.W}\n")
    
    # Initialize Gaussians (with reduced COLMAP loading)
    gaussians = initialize_gaussians_from_colmap(
        ds, 
        n_gaussians=n_gaussians, 
        device=device, 
        scene_root=scene
    )
    print()
    
    # Train (NO OpenGL visualization to save memory)
    gaussians, history = train(
        gaussians, ds,
        num_epochs=num_epochs,
        lr=0.0001,
        device=device,
        output_dir='outputs/fern_minimal',
        visualize_opengl=False,  # Disabled for memory
        opengl_interval=999  # Never visualize
    )
    
    print("\n" + "="*80)
    print("✅ TRAINING COMPLETE!")
    print("="*80)

if __name__ == "__main__":
    main()
