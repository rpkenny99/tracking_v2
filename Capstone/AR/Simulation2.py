import sys
import numpy as np
import cv2
from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *
# import viewer_control2 

# Initialize global variables for perspective data
viewer_position = [0, 0, 5]  # Starting position (x, y, z)
viewer_orientation = [0, 0, 0]  # Starting orientation (pitch, yaw, roll)

# Function to simulate receiving perspective data
def update_perspective():
    global viewer_position, viewer_orientation
    # viewer_control2.update_perspective()
    # Simulate perspective change (you will replace this with real data)
    # Move slightly to the right viewer_position[0] += 0.01  
    # Yaw rotation viewer_orientation[1] += 0.1  
    

    


# OpenGL initialization
def init():
    glClearColor(0.0, 0.0, 0.0, 1.0)  # Black background
    glEnable(GL_DEPTH_TEST)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(45, 800/600, 0.1, 50.0)
    glMatrixMode(GL_MODELVIEW)

# Draw the physical object (a simple square)
def draw_physical_object():
    glColor3f(0.0, 1.0, 0.0)  # Green color
    glBegin(GL_QUADS)
    glVertex3f(-1.0, -1.0, 0.0)
    glVertex3f(1.0, -1.0, 0.0)
    glVertex3f(1.0, 1.0, 0.0)
    glVertex3f(-1.0, 1.0, 0.0)
    glEnd()

# Draw the overlay image
def draw_overlay():
    glColor4f(1.0, 0.0, 0.0, 0.5)  # Red color with transparency
    glBegin(GL_QUADS)
    glVertex3f(-1.0, -1.0, 0.01)  # Slightly in front of the object
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
        window_id = glutGetWindow()  # Get the current window ID
        glutDestroyWindow(window_id)  # Destroy the window
        sys.exit(0)  # Exit the program cleanly

# Main function
def main():
    glutInit(sys.argv)
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH)
    glutInitWindowSize(800, 600)
    glutCreateWindow("AR Projection Simulation")
    init()
    glutDisplayFunc(display)
    glutKeyboardFunc(keyboard)

    glutIdleFunc(idle)
    glutMainLoop()

if __name__ == "__main__":
    main()
