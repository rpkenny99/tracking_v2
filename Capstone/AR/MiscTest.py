# import viewer_control2

# headPosition = viewer_control2()
# print("Head Position:", headPosition)

# import cv2
# cv2.imshow("Frame", image)
# cv2.waitKey(0)

# from OpenGL.GL import *
# print("OpenGL is working!")


# import cv2
# import numpy as np

# # Create blank images for each window
# img1 = np.zeros((300, 400, 3), dtype=np.uint8)  # Black window
# img2 = np.ones((300, 400, 3), dtype=np.uint8) * 255  # White window

# cv2.namedWindow("Window 1")
# cv2.namedWindow("Window 2")

# while True:
#     cv2.imshow("Window 1", img1)
#     cv2.imshow("Window 2", img2)

#     # Press 'q' to exit both windows
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cv2.destroyAllWindows()

import cv2
import numpy as np
import threading
from OpenGL.GL import *
from OpenGL.GLUT import *
from OpenGL.GLU import *

# Global flag to close both windows properly
exit_flag = False

def opencv_window():
    """Create an OpenCV window displaying a black screen."""
    global exit_flag
    img = np.zeros((400, 400, 3), dtype=np.uint8)

    cv2.namedWindow("OpenCV Window")

    while not exit_flag:
        cv2.imshow("OpenCV Window", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            exit_flag = True
            break

    cv2.destroyAllWindows()

def opengl_window():
    """Create an OpenGL window rendering a red triangle."""
    global exit_flag

    def draw():
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()

        # Draw a red triangle
        glBegin(GL_TRIANGLES)
        glColor3f(1.0, 0.0, 0.0)  # Red color
        glVertex2f(-0.5, -0.5)
        glVertex2f(0.5, -0.5)
        glVertex2f(0.0, 0.5)
        glEnd()

        glutSwapBuffers()

    def update(value):
        """Keep updating the OpenGL window."""
        global exit_flag
        if exit_flag:
            glutDestroyWindow(glutGetWindow())
            return
        glutPostRedisplay()
        glutTimerFunc(16, update, 0)

    def init():
        """Initialize OpenGL settings."""
        glClearColor(0.0, 0.0, 0.0, 1.0)  # Black background
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluOrtho2D(-1, 1, -1, 1)  # Set coordinate system
        glMatrixMode(GL_MODELVIEW)

    glutInit()
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB)
    glutInitWindowSize(400, 400)
    glutInitWindowPosition(500, 100)
    glutCreateWindow(b"OpenGL Window")
    init()
    glutDisplayFunc(draw)
    glutTimerFunc(16, update, 0)
    glutMainLoop()

# Create OpenCV window in a separate thread
opencv_thread = threading.Thread(target=opencv_window)
opencv_thread.start()

# Run OpenGL in the main thread
opengl_window()

# Wait for OpenCV thread to finish before exiting
opencv_thread.join()
