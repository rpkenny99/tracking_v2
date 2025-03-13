import sys
import numpy as np
import cv2
from OpenGL.GL import *
from OpenGL.GLUT import * # glutInit, glutInitDisplayMode, GLUT_DOUBLE, GLUT_RGB, GLUT_DEPTH, glutCreateWindow
from OpenGL.GLU import *
import os
sys.path.append(os.path.join("Capstone", "Tracking"))
import markerDetectionFrame
import queue
import threading
import multiprocessing
# import random
import logging
# import viewer_control2

tracking_queue = None


# Initialize global variables for perspective data
viewer_position = [0, 0, 10]  # Starting position (x, y, z)
viewer_orientation = [0, 0, 0]  # Starting orientation (pitch, yaw, roll)
countV = 0

tracking_queue = queue.Queue()

# Function to simulate receiving perspective data
def update_perspective():
    global viewer_position, viewer_orientation, countV, tracking_queue
    # countV += 1
    # while True:
    #     item = queue.get()               # blocking get
    #     if item is None:               # <-- sentinel
    #         break
    #     # Otherwise, process item
    #     print(f"Consumed: {item}")
    #     x_val, y_val, z_val, pitch_val, roll_val, yaw_val = item
    #     viewer_position[0], viewer_position[1], viewer_position[2], viewer_orientation[0], viewer_orientation[1], viewer_orientation[2] = item
    #     # viewer_position, viewer_orientation = item




    # viewer_control2.update_perspective()
    # Simulate perspective change (you will replace this with real data)
    # Move slightly to the right 
    # viewer_position[1] += 0.01  
    # Yaw rotation 
    # viewer_orientation[1] += 0.1 

    # if countV == 1000:
    #     countV = 00

    # elif countV >= 500:
    #     viewer_position[0] -= 0.01 
        
    # else:
    #     viewer_position[0] += 0.01

    
    if not tracking_queue.empty():
        # Check if there is data in the queue (non-blocking)
        data = tracking_queue.get()
        print(f"{data=}\n")
        if data is not None:
            if data[0] is not None and data[1] is not None:
                rVec, tVec = data  # Extract rotation and translation vectors

                # Update viewer position based on marker tracking
                viewer_position[0] = tVec[0][0][0] / 100  # Scale for OpenGL
                viewer_position[1] = tVec[0][0][1] / 100
                viewer_position[2] = tVec[0][0][2] / 100

                # Convert rotation vector to Euler angles (approximation)
                viewer_orientation[0] = rVec[0][0][0] * (180 / np.pi)  # Pitch
                viewer_orientation[1] = rVec[0][0][1] * (180 / np.pi)  # Yaw
                viewer_orientation[2] = rVec[0][0][2] * (180 / np.pi)  # Roll
            else:
                return
        else:
            window_id = glutGetWindow()  # Get the current window ID
            glutDestroyWindow(window_id)  # Destroy the window
            sys.exit(0)  # Exit the program cleanly

    else:
        return  # No new data, keep previous values
    

    


# OpenGL initialization
def init():
    glClearColor(0.0, 0.0, 0.0, 1.0)  # Black background
    glEnable(GL_DEPTH_TEST)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(45, 800/600, 0.1, 50.0)
    # gluPerspective(45, 1024/768, 1.0, 50.0)
    glMatrixMode(GL_MODELVIEW)

# Draw the physical object (a simple square)

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

    # Left face
    glVertex3f(-1.0, -1.0, -0.5)
    glVertex3f(-1.0, -1.0,  0.5)
    glVertex3f(-1.0,  1.0,  0.5)
    glVertex3f(-1.0,  1.0, -0.5)

    # Right face
    glVertex3f( 1.0, -1.0, -0.5)
    glVertex3f( 1.0,  1.0, -0.5)
    glVertex3f( 1.0,  1.0,  0.5)
    glVertex3f( 1.0, -1.0,  0.5)

    # Top faceq3
    glVertex3f(-1.0,  1.0, -0.5)
    glVertex3f(-1.0,  1.0,  0.5)
    glVertex3f( 1.0,  1.0,  0.5)
    glVertex3f( 1.0,  1.0, -0.5)

    # Bottom face
    glVertex3f(-1.0, -1.0, -0.5)
    glVertex3f( 1.0, -1.0, -0.5)
    glVertex3f( 1.0, -1.0,  0.5)
    glVertex3f(-1.0, -1.0,  0.5)

    glEnd()

    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)  # Restore solid mode


# def draw_physical_object():
#     glColor3f(0.0, 1.0, 0.0)  # Green color
#     glBegin(GL_QUADS)
#     glVertex3f(-1.0, -1.0, 0.0)
#     glVertex3f(1.0, -1.0, 0.0)
#     glVertex3f(1.0, 1.0, 0.0)
#     glVertex3f(-1.0, 1.0, 0.0)
#     glEnd()

# Draw the overlay image
def draw_overlay():
    glColor4f(1.0, 0.0, 0.0, 1.0)  # Red color with transparency
    glBegin(GL_QUADS)
    glVertex3f(-1.0, -1.0, 0.01)  # Slightly in front of the object
    glVertex3f(1.0, -1.0, 0.01)
    glVertex3f(1.0, 1.0, 0.01)
    glVertex3f(-1.0, 1.0, 0.01)
    glEnd()

def draw_vein_overlay():
    glColor4f(1.0, 0.0, 0.0, 0.8)  # Red, semi-transparent veins
    glLineWidth(2.0)  # Adjust thickness of the veins

    glBegin(GL_LINES)
    # Example vein structure (you can add more complexity)
    glVertex3f(-0.5, 0.0, 0.01)
    glVertex3f(0.0, 0.3, 0.01)

    glVertex3f(0.0, 0.3, 0.01)
    glVertex3f(0.3, 0.5, 0.01)

    glVertex3f(0.0, 0.3, 0.01)
    glVertex3f(-0.3, 0.6, 0.01)

    glVertex3f(-0.3, 0.6, 0.01)
    glVertex3f(-0.5, 0.8, 0.01)

    glVertex3f(0.3, 0.5, 0.01)
    glVertex3f(0.5, 0.7, 0.01)
    glEnd()


# Display function
def display():
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glLoadIdentity()

    # Apply viewer perspective transformations
    gluLookAt(viewer_position[0], viewer_position[1], viewer_position[2],
              0, 0, 0,  # Looking at the origin
              0, 1, 0)  # Up direction, orientation

    # Apply rotation based on orientation
    glRotatef(viewer_orientation[0], 1, 0, 0)  # Pitch
    glRotatef(viewer_orientation[1], 0, 1, 0)  # Yaw
    glRotatef(viewer_orientation[2], 0, 0, 1)  # Roll

    # Draw physical object and overlay
    draw_physical_object()
    # draw_overlay()
    draw_vein_overlay()

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

#  def main(queue):
#     while True:
#         item = queue.get()               # blocking get
#         if item is None:               # <-- sentinel
#             break
#         # Otherwise, process item
#         print(f"Consumed: {item}")
#         x_val, y_val, z_val, pitch_val, roll_val, yaw_val = item

        

# Main function
def mainProjection(tracking):
    global tracking_queue
    tracking_queue = tracking
    # os.environ["DISPLAY"] = ":0"
    glutInit(sys.argv[:])
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH)
    glutInitWindowSize(glutGet(GLUT_SCREEN_WIDTH), glutGet(GLUT_SCREEN_HEIGHT))     # if not full screen, use 800,600
    glutCreateWindow(b"AR Projection Simulation")
    init()
    glutDisplayFunc(display)
    glutKeyboardFunc(keyboard)

    glutIdleFunc(idle)
    glutMainLoop()

if __name__ == "__main__":
    tracking = queue.Queue()
    mainProjection(tracking)
