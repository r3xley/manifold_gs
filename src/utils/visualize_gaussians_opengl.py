"""
Visualize Gaussians using OpenGL with PyOpenGL
Better performance and quality than matplotlib for 3D ellipsoids
"""
import numpy as np
import torch
import os
import sys

# Add parent directory to path for imports
_script_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.join(_script_dir, '../..')
sys.path.insert(0, os.path.abspath(_project_root))

try:
    from data.load_colmap import load_colmap_points
    from src.models.manifold_gaussian import ManifoldGaussian3D
except ImportError:
    # Try alternative import path
    import importlib.util
    spec = importlib.util.spec_from_file_location("manifold_gaussian", 
        os.path.join(_project_root, "src/models/manifold_gaussian.py"))
    if spec:
        manifold_gaussian = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(manifold_gaussian)
        ManifoldGaussian3D = manifold_gaussian.ManifoldGaussian3D

try:
    from OpenGL.GL import *
    from OpenGL.GLU import *
    import pygame
    from pygame.locals import *
    OPENGL_AVAILABLE = True
except ImportError:
    OPENGL_AVAILABLE = False
    print("⚠️  PyOpenGL/pygame not available. Install with: pip install PyOpenGL pygame")

def colmap_points_to_gaussians(scene_root, max_points=10000, device='cpu'):
    """Convert COLMAP points to Gaussian parameters"""
    print(f"Loading COLMAP points from {scene_root}...")
    
    try:
        points, colors, ids, _ = load_colmap_points(scene_root, max_points=max_points)
        print(f"✓ Loaded {len(points)} COLMAP points")
    except Exception as e:
        print(f"❌ Error loading COLMAP points: {e}")
        return None
    
    n_points = len(points)
    gaussians = ManifoldGaussian3D(n_gaussians=n_points, device=device)
    
    with torch.no_grad():
        gaussians.pos.data = torch.from_numpy(points).float().to(device)
        
        colors_clamped = np.clip(colors, 0.01, 0.99)
        colors_logit = np.log(colors_clamped / (1 - colors_clamped))
        gaussians.rgb_logit.data = torch.from_numpy(colors_logit).float().to(device)
        
        # ANISOTROPIC SCALES for elongated ellipsoids (manifold Gaussians)
        # Make ellipsoids longer in one direction to look like proper Gaussians, not dots
        # Different scales for x, y, z axes create anisotropic ellipsoids
        log_scales = torch.zeros(n_points, 3, device=device)
        # Make them elongated: longer in X, medium in Y, shorter in Z
        # This creates visible ellipsoids that look like proper manifold Gaussians
        log_scales[:, 0] = 0.8  # exp(0.8) ≈ 2.23 - longer axis
        log_scales[:, 1] = 0.2  # exp(0.2) ≈ 1.22 - medium axis
        log_scales[:, 2] = -0.2  # exp(-0.2) ≈ 0.82 - shorter axis
        # Add some random variation for more realistic look
        log_scales += torch.randn(n_points, 3, device=device) * 0.1
        gaussians.log_scales.data = log_scales
        
        # Random rotations for variety (not all aligned)
        quats = torch.randn(n_points, 4, device=device) * 0.1
        quats[:, 0] = 1.0  # Keep w component dominant (near identity)
        # Normalize quaternions
        quat_norms = torch.norm(quats, dim=1, keepdim=True)
        quats = quats / (quat_norms + 1e-8)
        gaussians.quaternions.data = quats
        
        # High opacity for visibility
        gaussians.logit_opacity.data = torch.ones(n_points, device=device) * 2.0  # sigmoid(2) ≈ 0.88
    
    # Print scale statistics
    with torch.no_grad():
        scales = gaussians.get_scales()
        print(f"✓ Converted {n_points} COLMAP points to Gaussians")
        print(f"  Scale statistics:")
        print(f"    X: mean={scales[:, 0].mean().item():.3f}, range=[{scales[:, 0].min().item():.3f}, {scales[:, 0].max().item():.3f}]")
        print(f"    Y: mean={scales[:, 1].mean().item():.3f}, range=[{scales[:, 1].min().item():.3f}, {scales[:, 1].max().item():.3f}]")
        print(f"    Z: mean={scales[:, 2].mean().item():.3f}, range=[{scales[:, 2].min().item():.3f}, {scales[:, 2].max().item():.3f}]")
        print(f"  Opacity: mean={gaussians.get_opacity().mean().item():.3f}")
        print(f"  Colors: R=[{gaussians.get_colors()[:, 0].min().item():.3f}, {gaussians.get_colors()[:, 0].max().item():.3f}], "
              f"G=[{gaussians.get_colors()[:, 1].min().item():.3f}, {gaussians.get_colors()[:, 1].max().item():.3f}], "
              f"B=[{gaussians.get_colors()[:, 2].min().item():.3f}, {gaussians.get_colors()[:, 2].max().item():.3f}]")
    
    return gaussians

def draw_ellipsoid_opengl(center, radii, rotation_matrix, color, alpha):
    """Draw an ellipsoid using OpenGL"""
    glPushMatrix()
    
    # Translate to center
    glTranslatef(float(center[0]), float(center[1]), float(center[2]))
    
    # Apply rotation (rotation_matrix columns are principal axes)
    # Build 4x4 transformation matrix (column-major for OpenGL)
    transform = np.eye(4, dtype=np.float32)
    transform[:3, :3] = rotation_matrix
    # OpenGL expects column-major, so transpose
    glMultMatrixf(transform.T.flatten())
    
    # Scale by radii
    glScalef(float(radii[0]), float(radii[1]), float(radii[2]))
    
    # Set color with alpha from manifold Gaussian
    # Use colors directly (they're already from COLMAP RGB)
    glColor4f(float(color[0]), float(color[1]), float(color[2]), float(alpha))
    
    # Draw unit sphere (will be transformed to ellipsoid)
    quadric = gluNewQuadric()
    gluQuadricNormals(quadric, GLU_SMOOTH)
    gluQuadricDrawStyle(quadric, GLU_FILL)
    gluSphere(quadric, 1.0, 20, 20)
    gluDeleteQuadric(quadric)
    
    glPopMatrix()

def visualize_gaussians_opengl(gaussians, max_gaussians=2000):
    """Visualize Gaussians using OpenGL"""
    if not OPENGL_AVAILABLE:
        print("❌ OpenGL not available. Install: pip install PyOpenGL pygame")
        return
    
    with torch.no_grad():
        positions = gaussians.pos.cpu().numpy()
        colors = gaussians.get_colors().cpu().numpy()
        opacity = gaussians.get_opacity().cpu().numpy()
        covariances = gaussians.get_covariance_matrices().cpu().numpy()
    
    n_plot = min(len(positions), max_gaussians)
    if n_plot < len(positions):
        indices = np.random.choice(len(positions), n_plot, replace=False)
        positions = positions[indices]
        colors = colors[indices]
        opacity = opacity[indices]
        covariances = covariances[indices]
        print(f"  Rendering {n_plot} out of {len(gaussians.pos)} Gaussians")
    
    # Initialize Pygame and OpenGL
    pygame.init()
    screen = pygame.display.set_mode((1280, 720), DOUBLEBUF | OPENGL)
    pygame.display.set_caption("Gaussian Splatting - COLMAP Points")
    
    # OpenGL setup
    glEnable(GL_DEPTH_TEST)
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    
    # Set background color (light gray instead of black)
    glClearColor(0.2, 0.2, 0.25, 1.0)
    
    # Disable lighting to show true colors (or use color material mode)
    glDisable(GL_LIGHTING)
    glEnable(GL_COLOR_MATERIAL)
    glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
    
    # Camera setup
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(45, 1280/720, 0.1, 1000.0)
    
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()
    
    # Compute center and extent of scene
    center_pos = positions.mean(axis=0)
    extent = np.max(positions.max(axis=0) - positions.min(axis=0))
    
    # Position camera to view the scene
    # Move back based on scene extent
    camera_distance = extent * 2.0
    glTranslatef(0, 0, -camera_distance)
    glTranslatef(-center_pos[0], -center_pos[1], -center_pos[2])
    
    # Rotation state
    rotation_x = 20  # Start with slight angle
    rotation_y = 45
    zoom = 1.0
    
    # Store camera distance for zoom
    camera_distance = extent * 2.0
    
    clock = pygame.time.Clock()
    running = True
    
    print("Controls:")
    print("  Mouse drag: Rotate")
    print("  Scroll: Zoom")
    print("  ESC: Quit")
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
            elif event.type == pygame.MOUSEMOTION:
                if pygame.mouse.get_pressed()[0]:
                    rotation_y += event.rel[0] * 0.5
                    rotation_x += event.rel[1] * 0.5
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 4:  # Scroll up
                    zoom *= 1.1
                elif event.button == 5:  # Scroll down
                    zoom /= 1.1
        
        # Clear
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()
        
        # Apply camera transform
        glTranslatef(0, 0, -camera_distance * zoom)
        glRotatef(rotation_x, 1, 0, 0)
        glRotatef(rotation_y, 0, 1, 0)
        glTranslatef(-center_pos[0], -center_pos[1], -center_pos[2])
        
        # Draw ellipsoids with proper covariance, opacity, and colors
        for i in range(n_plot):
            cov = covariances[i]
            cov = 0.5 * (cov + cov.T)  # Ensure symmetric
            
            # Eigenvalue decomposition to get principal axes and radii
            eigvals, eigvecs = np.linalg.eigh(cov)
            eigvals = np.clip(eigvals, 1e-6, None)
            radii = np.sqrt(eigvals)  # Standard deviations along principal axes
            
            # Ensure right-handed coordinate system
            if np.linalg.det(eigvecs) < 0:
                eigvecs[:, -1] *= -1
            
            # Draw with proper color and opacity from manifold Gaussian
            draw_ellipsoid_opengl(positions[i], radii, eigvecs, colors[i], opacity[i])
        
        pygame.display.flip()
        clock.tick(60)
    
    pygame.quit()

if __name__ == "__main__":
    scene_root = "data/nerf_llff_data/fern"
    if len(sys.argv) > 1:
        scene_root = sys.argv[1]
    
    max_points = 10000
    if len(sys.argv) > 2:
        max_points = int(sys.argv[2])
    
    max_gaussians = 2000
    if len(sys.argv) > 3:
        max_gaussians = int(sys.argv[3])
    
    device = 'cpu'
    
    gaussians = colmap_points_to_gaussians(scene_root, max_points=max_points, device=device)
    if gaussians is not None:
        visualize_gaussians_opengl(gaussians, max_gaussians=max_gaussians)
