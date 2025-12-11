"""
COLMAP binary format loader for 3D Gaussian Splatting initialization
Reads sparse reconstruction point clouds from COLMAP
"""
import struct
import numpy as np
from pathlib import Path


def read_next_bytes(fid, num_bytes, format_char_sequence, endian_character="<"):
    """Read and unpack the next bytes from a binary file."""
    data = fid.read(num_bytes)
    return struct.unpack(endian_character + format_char_sequence, data)


def read_points3D_binary(path_to_model_file, max_points=None):
    """
    Read COLMAP points3D.bin file.
    
    Args:
        path_to_model_file: Path to points3D.bin
        max_points: Optional maximum number of points to load (for memory efficiency)
                    If None, loads all. If specified, randomly samples.
    
    Returns:
        points_3d: (N, 3) array of 3D point positions
        colors: (N, 3) array of RGB colors (0-255)
        point_ids: (N,) array of point IDs
    """
    points_3d = []
    colors = []
    point_ids = []
    
    with open(path_to_model_file, "rb") as fid:
        num_points = read_next_bytes(fid, 8, "Q")[0]
        print(f"  File contains {num_points:,} points")
        
        # If max_points specified and we have more, randomly sample
        if max_points is not None and num_points > max_points:
            print(f"  Sampling {max_points} points (skipping {num_points - max_points:,})...")
            # Use numpy random choice - but do it in chunks to avoid memory issues
            # For very large files, use a simpler approach: every Nth point
            if num_points > 1000000:  # Very large - use stride sampling
                stride = num_points // max_points
                print(f"  Using stride sampling (every {stride}th point)")
                indices_to_load = list(range(0, num_points, stride))[:max_points]
            else:
                # Random sampling for smaller files
                indices_to_load = sorted(np.random.choice(num_points, max_points, replace=False))
            indices_to_load = set(indices_to_load)
            target_idx = 0
        else:
            indices_to_load = None
            print(f"  Loading all {num_points:,} points...")
        
        loaded = 0
        for i in range(num_points):
            try:
                # Read point header (43 bytes: Q + 3*d + 3*B + d)
                # Format: point_id (Q), xyz (3*d), rgb (3*B), error (d)
                binary_point_line_properties = read_next_bytes(
                    fid, num_bytes=43, format_char_sequence="QdddBBBd"
                )
                point_id = binary_point_line_properties[0]
                xyz = np.array(binary_point_line_properties[1:4])
                rgb = np.array(binary_point_line_properties[4:7])
                error = binary_point_line_properties[7]
                
                # Validate point data before reading track
                # Check for reasonable values (COLMAP points should be in reasonable scene bounds)
                if (np.any(np.abs(xyz) > 1e6) or  # Unreasonably large coordinates
                    np.any(np.isnan(xyz)) or np.any(np.isinf(xyz)) or
                    point_id < 0 or point_id > 1e10 or  # Unreasonable point ID
                    np.any(rgb < 0) or np.any(rgb > 255)):  # Invalid RGB
                    # Skip this point - but we still need to read track_length and skip track data
                    track_length_bytes = fid.read(8)
                    if len(track_length_bytes) < 8:
                        break
                    track_length = struct.unpack("<Q", track_length_bytes)[0]
                    if track_length > 0 and track_length < 10000:  # Reasonable limit
                        track_data_size = 8 * track_length  # OFFICIAL FORMAT: 8 bytes per pair
                        fid.seek(track_data_size, 1)
                    continue
                
                # Read track length (8 bytes, uint64) - OFFICIAL COLMAP FORMAT
                # Format matches llff.poses.colmap_read_model.read_points3d_binary
                track_length_bytes = fid.read(8)
                if len(track_length_bytes) < 8:
                    # End of file
                    break
                track_length = struct.unpack("<Q", track_length_bytes)[0]
                
                # Safety check: skip points with extremely large track data
                # Normal COLMAP points have < 100 track elements
                if track_length > 1000:  # Very conservative limit
                    # CRITICAL: Must skip track data bytes to keep file pointer aligned!
                    # OFFICIAL FORMAT: track_length pairs of (image_id: int32, point2D_idx: int32)
                    # Each pair = 8 bytes (2 int32s = 4+4 bytes)
                    # Total track data: 8 * track_length bytes
                    track_data_size = 8 * track_length
                    fid.seek(track_data_size, 1)  # Skip forward by track_data_size bytes
                    continue
                
                # Read track data if present (must read even if we skip the point)
                # OFFICIAL COLMAP FORMAT: track_length pairs of (image_id: int32, point2D_idx: int32)
                # Format string: "ii"*track_length means track_length pairs of int32
                # Total size: 8 * track_length bytes (2 int32s per pair = 8 bytes)
                if track_length > 0:
                    track_data_size = 8 * track_length
                    track_data = fid.read(track_data_size)
                    if len(track_data) < track_data_size:
                        # Not enough data, file may be corrupted - stop reading
                        print(f"  Warning: Incomplete track data at point {i}, stopping read")
                        break
                
            except (struct.error, ValueError, IOError) as e:
                # If we get here, file pointer is likely misaligned
                # This shouldn't happen with correct track data size (16 bytes)
                # But if it does, we can't easily recover - stop reading
                print(f"  Warning: Error reading point {i}: {e}")
                print(f"  File position: {fid.tell()}, stopping read")
                break
            
            # Only save if we're sampling or if we want all
            should_save = (indices_to_load is None) or (i in indices_to_load)
            
            if should_save:
                points_3d.append(xyz)
                colors.append(rgb)
                point_ids.append(point_id)
                loaded += 1
                if loaded % 1000 == 0:
                    print(f"    Loaded {loaded}/{max_points if max_points else num_points} points...")
                
                if indices_to_load is not None and loaded >= len(indices_to_load):
                    break  # Got all we need
    
    print(f"  ✓ Loaded {len(points_3d)} points")
    return np.array(points_3d), np.array(colors), np.array(point_ids)


def read_images_binary(path_to_model_file):
    """
    Read COLMAP images.bin file.
    
    Returns:
        images: dict mapping image_id to image data
    """
    images = {}
    with open(path_to_model_file, "rb") as fid:
        num_reg_images = read_next_bytes(fid, 8, "Q")[0]
        
        for _ in range(num_reg_images):
            binary_image_properties = read_next_bytes(
                fid, num_bytes=64, format_char_sequence="idddddddi"
            )
            image_id = binary_image_properties[0]
            qvec = np.array(binary_image_properties[1:5])
            tvec = np.array(binary_image_properties[5:8])
            camera_id = binary_image_properties[8]
            
            image_name = ""
            current_char = read_next_bytes(fid, 1, "c")[0]
            while current_char != b"\x00":
                image_name += current_char.decode("utf-8")
                current_char = read_next_bytes(fid, 1, "c")[0]
            
            num_points2D = read_next_bytes(fid, num_bytes=8, format_char_sequence="Q")[0]
            x_y_id_s = read_next_bytes(
                fid, num_bytes=24 * num_points2D, format_char_sequence="ddq" * num_points2D
            )
            
            images[image_id] = {
                'qvec': qvec,
                'tvec': tvec,
                'camera_id': camera_id,
                'name': image_name,
                'xys': np.column_stack([x_y_id_s[0::3], x_y_id_s[1::3]]),
                'point3D_ids': np.array(x_y_id_s[2::3])
            }
    
    return images


def read_cameras_binary(path_to_model_file):
    """
    Read COLMAP cameras.bin file.
    
    Returns:
        cameras: dict mapping camera_id to camera data
    """
    cameras = {}
    with open(path_to_model_file, "rb") as fid:
        num_cameras = read_next_bytes(fid, 8, "Q")[0]
        
        for _ in range(num_cameras):
            camera_properties = read_next_bytes(fid, num_bytes=24, format_char_sequence="iiQQ")
            camera_id = camera_properties[0]
            model_id = camera_properties[1]
            width = camera_properties[2]
            height = camera_properties[3]
            
            num_params = {
                1: 3,  # SIMPLE_PINHOLE
                2: 4,  # PINHOLE
                3: 8,  # SIMPLE_RADIAL
                4: 9,  # RADIAL
                5: 12, # OPENCV
            }.get(model_id, 0)
            
            params = read_next_bytes(fid, num_bytes=8 * num_params, format_char_sequence="d" * num_params)
            
            cameras[camera_id] = {
                'model_id': model_id,
                'width': width,
                'height': height,
                'params': np.array(params)
            }
    
    return cameras


def load_colmap_points(scene_root, max_points=None):
    """
    Load COLMAP sparse reconstruction points.
    
    Args:
        scene_root: Path to scene directory containing sparse/0/ subdirectory
        max_points: Optional maximum number of points to load (for memory efficiency)
                    If None, loads all points. If specified, randomly subsamples.
        
    Returns:
        points_3d: (N, 3) array of 3D point positions in world coordinates
        colors: (N, 3) array of RGB colors (0-255, will be normalized to 0-1)
        point_ids: (N,) array of point IDs
        images: dict of image metadata (optional, for debugging)
    """
    scene_path = Path(scene_root)
    sparse_path = scene_path / "sparse" / "0"
    
    if not sparse_path.exists():
        raise FileNotFoundError(f"COLMAP sparse reconstruction not found at {sparse_path}")
    
    points3D_path = sparse_path / "points3D.bin"
    images_path = sparse_path / "images.bin"
    cameras_path = sparse_path / "cameras.bin"
    
    if not points3D_path.exists():
        raise FileNotFoundError(f"points3D.bin not found at {points3D_path}")
    
    # Load points
    print(f"Loading COLMAP points from {points3D_path}...")
    try:
        # Pass max_points to the binary reader for efficient sampling
        # Use very small default if not specified to avoid memory issues
        if max_points is None:
            max_points = 1000  # Safe default
        points_3d, colors, point_ids = read_points3D_binary(str(points3D_path), max_points=max_points)
    except MemoryError:
        print(f"❌ Memory error loading COLMAP points")
        print(f"   Try loading with max_points parameter to limit memory usage")
        raise
    except Exception as e:
        print(f"❌ Error loading COLMAP points: {e}")
        raise
    
    n_total = len(points_3d)
    print(f"  Loaded {n_total} COLMAP points")
    
    # Subsample if max_points specified
    if max_points is not None and n_total > max_points:
        print(f"  Subsampling to {max_points} points (for memory efficiency)...")
        indices = np.random.choice(n_total, max_points, replace=False)
        points_3d = points_3d[indices]
        colors = colors[indices]
        point_ids = point_ids[indices]
        print(f"  Using {len(points_3d)} points")
    
    # Normalize colors to [0, 1]
    colors = colors.astype(np.float32) / 255.0
    
    # Load images (optional, for debugging or color refinement)
    # Skip for now to save memory
    images = None
    # if images_path.exists():
    #     try:
    #         images = read_images_binary(str(images_path))
    #     except Exception as e:
    #         print(f"Warning: Could not load images.bin: {e}")
    
    # Load cameras (optional, for debugging)
    # Skip for now to save memory
    cameras = None
    # if cameras_path.exists():
    #     try:
    #         cameras = read_cameras_binary(str(cameras_path))
    #     except Exception as e:
    #         print(f"Warning: Could not load cameras.bin: {e}")
    
    print(f"✓ Loaded {len(points_3d)} COLMAP points")
    print(f"  Point positions range: X=[{points_3d[:, 0].min():.3f}, {points_3d[:, 0].max():.3f}], "
          f"Y=[{points_3d[:, 1].min():.3f}, {points_3d[:, 1].max():.3f}], "
          f"Z=[{points_3d[:, 2].min():.3f}, {points_3d[:, 2].max():.3f}]")
    print(f"  Color range: R=[{colors[:, 0].min():.3f}, {colors[:, 0].max():.3f}], "
          f"G=[{colors[:, 1].min():.3f}, {colors[:, 1].max():.3f}], "
          f"B=[{colors[:, 2].min():.3f}, {colors[:, 2].max():.3f}]")
    
    return points_3d, colors, point_ids, images


def transform_colmap_to_llff(points_3d):
    """
    Transform COLMAP coordinates to LLFF coordinate system if needed.
    
    COLMAP: [right, down, forward] = [x, -y, -z]
    LLFF: [down, right, backward] = [-y, x, z]
    
    This transformation may or may not be needed depending on how COLMAP was run.
    Test first before applying.
    
    Args:
        points_3d: (N, 3) points in COLMAP coordinates
        
    Returns:
        points_3d_transformed: (N, 3) points in LLFF coordinates
    """
    # For now, return as-is. We'll test if transformation is needed.
    # If needed, the transformation would be:
    # x_llff = -y_colmap
    # y_llff = x_colmap  
    # z_llff = z_colmap
    # return np.column_stack([-points_3d[:, 1], points_3d[:, 0], points_3d[:, 2]])
    
    return points_3d
