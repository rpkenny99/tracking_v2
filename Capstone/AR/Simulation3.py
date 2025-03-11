import sys
import numpy as np
import cv2
from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *

# Initialize global variables for perspective data
viewer_position = [0, 0, 5]  # Initial position (x, y, z)
viewer_orientation = [0, 0, 0]  # Initial orientation (pitch, yaw, roll)
depth_scale_factor = 0.0001  # Scale for depth calibration
countV = 0

# Function to update perspective data with depth correction
def update_perspective():
    global viewer_position, countV

    countV += 1
    if countV == 1000:
        countV = 0
    elif countV >= 500:
        viewer_position[1] -= 0.01
        # viewer_position[2] += depth_scale_factor  # Adjust depth dynamically
    else:
        viewer_position[1] += 0.01
        # viewer_position[2] -= depth_scale_factor  # Adjust depth dynamically

# OpenGL initialization
def init():
    glClearColor(0.0, 0.0, 0.0, 1.0)  # Black background
    glEnable(GL_DEPTH_TEST)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(45, 1024 / 768, 0.1, 100.0)  # Extend far plane for better depth
    glMatrixMode(GL_MODELVIEW)

# Function to draw the physical object
def draw_physical_object():
    glEnable(GL_BLEND)  # Enable transparency
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)  # Transparency mode

    glColor4f(0.0, 1.0, 0.0, 1.0)  # Green color (Fully Opaque for Wireframe)

    glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)  # Wireframe mode

    glBegin(GL_QUADS)

    # Front face
    glVertex3f(-1.0, -1.0,  0.5)
    glVertex3f( 1.0, -1.0,  0.5)
    glVertex3f( 1.0,  1.0,  0.5)
    glVertex3f(-1.0,  1.0,  0.5)

    # Back face
    glVertex3f(-1.0, -1.0, -0.5)
    glVertex3f(-1.0,  1.0, -0.5)
    glVertex3f( 1.0,  1.0, -0.5)
    glVertex3f( 1.0, -1.0, -0.5)

    glEnd()
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)  # Restore solid mode

# Function to draw the overlay
def draw_overlay():
    glColor4f(1.0, 0.0, 0.0, 1.0)  # Red color
    glBegin(GL_QUADS)
    glVertex3f(-1.0, -1.0, 0.01)
    glVertex3f(1.0, -1.0, 0.01)
    glVertex3f(1.0, 1.0, 0.01)
    glVertex3f(-1.0, 1.0, 0.01)
    glEnd()

# Display function
def display():
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glLoadIdentity()

    # Apply viewer perspective transformations
    gluLookAt(viewer_position[0], viewer_position[1], viewer_position[2],
              0, 0, 0,  # Looking at the origin
              0, 1, 0)  # Up direction

    # Apply rotation based on orientation
    glRotatef(viewer_orientation[0], 1, 0, 0)  # Pitch
    glRotatef(viewer_orientation[1], 0, 1, 0)  # Yaw
    glRotatef(viewer_orientation[2], 0, 0, 1)  # Roll

    # Draw physical object and overlay
    draw_physical_object()
    draw_overlay()

    glutSwapBuffers()

# Idle function for continuous update
def idle():
    update_perspective()
    glutPostRedisplay()

def keyboard(key, x, y):
    if key == b'\x1b':  # ESC key
        window_id = glutGetWindow()
        glutDestroyWindow(window_id)
        sys.exit(0)

# Main function
def main():
    glutInit(sys.argv)
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH)
    glutInitWindowSize(glutGet(GLUT_SCREEN_WIDTH), glutGet(GLUT_SCREEN_HEIGHT))
    glutCreateWindow("AR Depth Calibration")
    init()
    glutDisplayFunc(display)
    glutKeyboardFunc(keyboard)
    glutIdleFunc(idle)
    glutMainLoop()

if __name__ == "__main__":
    main()
