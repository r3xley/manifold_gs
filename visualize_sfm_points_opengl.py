"""
Visualize raw COLMAP/SFM 3D points using OpenGL
Simple point cloud visualization
"""
import numpy as np
from data.load_colmap import load_colmap_points
import os
import sys

try:
    from OpenGL.GL import *
    from OpenGL.GLU import *
    import pygame
    from pygame.locals import *
    OPENGL_AVAILABLE = True
except ImportError:
    OPENGL_AVAILABLE = False
    print("⚠️  PyOpenGL/pygame not available. Install with: pip install PyOpenGL pygame")

def visualize_sfm_points_opengl(scene_root, max_points=10000, point_size=3.0):
    """
    Visualize COLMAP/SFM points as colored spheres in OpenGL
    
    Args:
        scene_root: Path to scene directory
        max_points: Maximum number of points to load
        point_size: Size of each point (in pixels/OpenGL units)
    """
    if not OPENGL_AVAILABLE:
        print("❌ OpenGL not available. Install: pip install PyOpenGL pygame")
        return
    
    print(f"Loading COLMAP points from {scene_root}...")
    
    try:
        points, colors, ids, _ = load_colmap_points(scene_root, max_points=max_points)
        print(f"✓ Loaded {len(points)} COLMAP points")
    except Exception as e:
        print(f"❌ Error loading COLMAP points: {e}")
        return
    
    # Initialize Pygame and OpenGL
    pygame.init()
    screen = pygame.display.set_mode((1280, 720), DOUBLEBUF | OPENGL)
    pygame.display.set_caption("COLMAP/SFM 3D Points")
    
    # OpenGL setup
    glEnable(GL_DEPTH_TEST)
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    
    # Set background color (light gray)
    glClearColor(0.2, 0.2, 0.25, 1.0)
    
    # Disable lighting to show true colors
    glDisable(GL_LIGHTING)
    glEnable(GL_COLOR_MATERIAL)
    glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
    
    # Enable point smoothing for better appearance
    glEnable(GL_POINT_SMOOTH)
    glHint(GL_POINT_SMOOTH_HINT, GL_NICEST)
    
    # Camera setup
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(45, 1280/720, 0.1, 1000.0)
    
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()
    
    # Compute center and extent of scene
    center_pos = points.mean(axis=0)
    extent = np.max(points.max(axis=0) - points.min(axis=0))
    
    # Position camera to view the scene
    camera_distance = extent * 2.0
    glTranslatef(0, 0, -camera_distance)
    glTranslatef(-center_pos[0], -center_pos[1], -center_pos[2])
    
    # Rotation state
    rotation_x = 20  # Start with slight angle
    rotation_y = 45
    zoom = 1.0
    
    clock = pygame.time.Clock()
    running = True
    
    print(f"\nRendering {len(points)} points")
    print("Controls:")
    print("  Mouse drag: Rotate")
    print("  Scroll: Zoom")
    print("  ESC: Quit")
    
    # Pre-compute point positions and colors
    point_positions = points.astype(np.float32)
    point_colors = colors.astype(np.float32)
    
    # Brighten colors slightly
    point_colors = np.clip(point_colors * 1.2, 0, 1)
    
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
        
        # Set point size
        glPointSize(point_size)
        
        # Draw points using immediate mode (simple but works)
        glBegin(GL_POINTS)
        for i in range(len(point_positions)):
            pos = point_positions[i]
            col = point_colors[i]
            glColor3f(float(col[0]), float(col[1]), float(col[2]))
            glVertex3f(float(pos[0]), float(pos[1]), float(pos[2]))
        glEnd()
        
        # Alternative: Draw as small spheres for better visibility
        # This is slower but looks better
        if point_size > 2.0:
            quadric = gluNewQuadric()
            sphere_radius = extent * 0.002 * (point_size / 3.0)
            for i in range(len(point_positions)):
                pos = point_positions[i]
                col = point_colors[i]
                glPushMatrix()
                glTranslatef(float(pos[0]), float(pos[1]), float(pos[2]))
                glColor3f(float(col[0]), float(col[1]), float(col[2]))
                gluSphere(quadric, sphere_radius, 8, 8)
                glPopMatrix()
            gluDeleteQuadric(quadric)
        
        pygame.display.flip()
        clock.tick(60)
    
    pygame.quit()
    print("\n✓ Visualization closed")

if __name__ == "__main__":
    scene_root = "data/nerf_llff_data/fern"
    if len(sys.argv) > 1:
        scene_root = sys.argv[1]
    
    max_points = 10000
    if len(sys.argv) > 2:
        max_points = int(sys.argv[2])
    
    point_size = 3.0
    if len(sys.argv) > 3:
        point_size = float(sys.argv[3])
    
    visualize_sfm_points_opengl(scene_root, max_points=max_points, point_size=point_size)
