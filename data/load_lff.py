# src/datasets/llff.py
import os
import glob
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

def load_poses_bounds_numpy(scene_root):
    """LLFF default: poses_bounds.npy produced by LLFF data prep.
    Returns poses (N, 3, 4) and bounds (N).
    """
    pth = os.path.join(scene_root, "poses_bounds.npy")
    if not os.path.exists(pth):
        raise FileNotFoundError(f"{pth} not found. Run LLFF preprocessing or supply poses.")
    data = np.load(pth)  # (N, 17) often
    poses = data[:, :15].reshape(-1, 3, 5)[:, :, :4]  # sometimes vary; try safely
    bounds = data[:, 15:]
    # Many variants exist; you may need to adapt parsing depending on your file.
    return poses, bounds

def load_llff_images(scene_root, downscale=1.0):
    img_dir = os.path.join(scene_root, "images")
    if not os.path.isdir(img_dir):
        # fallback: many datasets have images in the root
        img_dir = scene_root
    files = sorted(glob.glob(os.path.join(img_dir, "*.JPG")) + glob.glob(os.path.join(img_dir, "*.png")))
    imgs = []
    for f in files:
        im = Image.open(f).convert("RGB")
        if downscale != 1.0:
            new_size = (int(im.width / downscale), int(im.height / downscale))
            im = im.resize(new_size, Image.LANCZOS)
        imgs.append(np.array(im))
    return imgs, files

class LLFFDataset(Dataset):
    """
    Minimal LLFF dataset loader.
    Expects scene folder with:
      - poses_bounds.npy (or poses.txt)
      - images/ (jpg/png)
    Returns:
      dict with 'images' (H,W,3 numpy), 'K' (3x3), 'poses' (N,4,4)
    """
    def __init__(self, scene_root, downscale=1.0, device='cpu'):
        super().__init__()
        self.scene_root = scene_root
        self.device = device
        self.downscale = downscale

        # Load images
        imgs, files = load_llff_images(scene_root, downscale=downscale)
        if len(imgs) == 0:
            raise RuntimeError("No images found in scene folder.")
        self.images = np.stack(imgs).astype(np.float32) / 255.0  # (N,H,W,3)
        self.file_names = files
        self.N, self.H, self.W, _ = self.images.shape

        # Load poses & intrinsics: try poses_bounds.npy
        try:
            poses_raw, bounds = load_poses_bounds_numpy(scene_root)
            # poses_raw shape handling: hope it becomes (N,3,4) or (N,4,4)
            if poses_raw.shape[0] != self.N:
                # fallback: try reading 'poses.txt' (custom)
                raise RuntimeError("poses size mismatch; please provide poses in poses_bounds.npy with N images")
            # Convert to 4x4 extrinsics
            poses = np.zeros((self.N, 4, 4), dtype=np.float32)
            for i in range(self.N):
                p = poses_raw[i]
                if p.shape == (3,4):
                    poses[i][:3,:4] = p
                    poses[i][3,3] = 1.0
                elif p.shape == (4,4):
                    poses[i] = p
                else:
                    raise RuntimeError("Unsupported pose shape")
            self.poses = poses
        except Exception as e:
            raise RuntimeError("Could not load poses from poses_bounds.npy. Please prepare LLFF data.") from e

        # Simple intrinsics approximation: LLFF uses focal length in poses_bounds or external file.
        # Here we compute a fx approximated as 0.5 * W / tan(fov/2) if fov in poses; else fallback
        # If you have explicit K, replace this with correct intrinsics load.
        # Many LLFF preprocess steps store "hwf" with focal in poses array; try to extract:
        # NOTE: adapt for your LLFF variant; this is a heuristic fallback.
        # We'll create K per view for simplicity (assuming same intrinsics).
        fy = fx = 0.5 * self.W  # fallback guess
        cx = self.W / 2.0
        cy = self.H / 2.0
        K = np.array([[fx, 0, cx],
                      [0, fy, cy],
                      [0, 0, 1]], dtype=np.float32)
        self.K = K

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        img = self.images[idx]  # H,W,3
        pose = self.poses[idx]  # 4x4
        K = self.K
        sample = {
            "image": torch.from_numpy(img).permute(2,0,1).to(self.device),  # 3,H,W
            "pose": torch.from_numpy(pose).to(self.device),                 # 4,4
            "K": torch.from_numpy(K).to(self.device),                       # 3,3
            "idx": idx,
            "file": self.file_names[idx]
        }
        return sample
