import glob, os
scene = "/Users/rexley/Documents/Class/Geometric Methods in ML/3DGS/data/nerf_llff_data/fern"

print("SCENE:", scene)
# common places
candidates = [os.path.join(scene, "images"), scene]
for cand in candidates:
    print("\nChecking:", cand)
    if os.path.isdir(cand):
        for ext in ["*.jpg","*.JPG","*.jpeg","*.JPEG","*.png","*.PNG"]:
            found = sorted(glob.glob(os.path.join(cand, ext)))
            if found:
                print(f"  Found {len(found)} files with pattern {ext} in {cand}")
                print("   example:", found[:5])
    else:
        print("  not a directory")

# recursive search
print("\nRecursive search in scene root:")
rec = []
for ext in ["**/*.jpg","**/*.JPG","**/*.jpeg","**/*.JPEG","**/*.png","**/*.PNG"]:
    rec.extend(sorted(glob.glob(os.path.join(scene, ext), recursive=True)))
print("  Recursive found:", len(rec))
if len(rec):
    print("   example:", rec[:8])


import numpy as np, os
pth = "/Users/rexley/Documents/Class/Geometric Methods in ML/3DGS/data/nerf_llff_data/fern/poses_bounds.npy"
print("File exists:", os.path.exists(pth))
data = np.load(pth, allow_pickle=True)
print("dtype:", data.dtype, "ndim:", data.ndim, "shape:", data.shape)
# print a small sample, safely
if data.size <= 2000:
    print("Contents:\n", data)
else:
    print("First 3 rows (or items):")
    try:
        for i in range(min(3, data.shape[0])):
            print(i, data[i])
    except Exception as e:
        print("Could not print rows directly:", e)
