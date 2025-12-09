# 3D Gaussian Splatting Pipeline

## Quick Start

Run the training pipeline:

```bash
python train_gaussian_splatting.py
```

## What It Does

1. **Loads LLFF dataset** (fern scene by default)
2. **Initializes Gaussians** at scene center
3. **Trains** using multi-view optimization
4. **Visualizes results** with comparison images
5. **Saves outputs** to `outputs/fern/`

## Output Structure

```
outputs/fern/
├── renders/              # Training progress renders (every 5 epochs)
│   ├── epoch_000_view_00.png
│   ├── epoch_005_view_00.png
│   └── ...
├── final_renders/        # Final renders for all views
│   ├── epoch_999_view_00.png
│   ├── epoch_999_view_01.png
│   └── ...
└── training_curves.png   # Loss and PSNR curves
```

## Visualization Features

Each render includes:
- **Ground Truth**: Original image
- **Predicted**: Rendered image with PSNR
- **Error Map**: Absolute difference (hot colormap)
- **Difference**: Signed difference (GT - Pred)

## Customization

Edit `train_gaussian_splatting.py` to:
- Change scene: `scene = "data/nerf_llff_data/your_scene"`
- Adjust number of Gaussians: `n_gaussians = 20000`
- Modify training: `num_epochs`, `lr`, etc.
- Change downscale: `downscale=2.0` (in dataset loading)

## Troubleshooting

If you see black renders:
1. Check that Gaussians are initialized near scene center
2. Verify camera poses are correct
3. Try adjusting initialization spread in `initialize_gaussians()`

## Extending the Pipeline

The code is modular:
- `initialize_gaussians()`: Change initialization strategy
- `train()`: Modify training loop, add losses, etc.
- `visualize_comparison()`: Customize visualization
- `compute_metrics()`: Add SSIM, LPIPS, etc.

