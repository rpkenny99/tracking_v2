import sys
import cv2
import numpy as np
from PyQt6.QtWidgets import QApplication
from PyQt6.QtGui import QImage
# Import the main UI class from feedback3.py
from feedback3 import FeedbackUI
from queue import Queue, Empty

def qpixmap_to_numpy(pixmap):
    """
    Convert a QPixmap to a NumPy array (BGR format for OpenCV).
    """
    # Convert the QPixmap to a QImage using the proper enum for RGBA8888.
    image = pixmap.toImage().convertToFormat(QImage.Format.Format_RGBA8888)
    width = image.width()
    height = image.height()
    ptr = image.bits()
    ptr.setsize(height * width * 4)
    arr = np.array(ptr).reshape(height, width, 4)
    # Convert RGBA to BGR (which OpenCV uses)
    return cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)

def apply_projection_transform(image, angle_deg=25):
    """
    Applies a perspective transformation to simulate a projection plane
    rotated by angle_deg relative to the source screen.
    """
    h, w = image.shape[:2]
    angle_rad = np.deg2rad(angle_deg)
    # Compute the new width for the top edge (foreshortened by cosine of the angle)
    new_top_width = w * np.cos(angle_rad)
    # Calculate offset to center the top edge
    offset = (w - new_top_width) / 2

    # Source points (corners of the original image)
    src_pts = np.float32([[0, 0],
                          [w, 0],
                          [w, h],
                          [0, h]])
    # Destination points with the top edge "squeezed"
    dst_pts = np.float32([[offset, 0],
                          [w - offset, 0],
                          [w, h],
                          [0, h]])
    
    # Compute the perspective transform matrix and apply it
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped = cv2.warpPerspective(image, M, (w, h))
    return warped

def main():
    app = QApplication(sys.argv)


    # Just for demonstration purposes, set the selected vein and point
    selected_vein = "Left Vein"
    selected_point = "Point A"
    my_queue = Queue()

    # Instantiate the UI from feedback3.py (adjust constructor parameters if needed)
    feedback_ui = FeedbackUI(selected_vein, selected_point, work_queue=my_queue)  
    feedback_ui.show()

    # Process events so that the window gets rendered
    app.processEvents()
    
    # Grab the display (entire window) as a QPixmap
    pixmap = feedback_ui.grab()
    original_image = qpixmap_to_numpy(pixmap)

    # Apply the transformation (25° tilt in this example)
    transformed_image = apply_projection_transform(original_image, angle_deg=25)

    # Display the transformed image in an OpenCV window (simulating the projector output)
    cv2.imshow("Projected UI", transformed_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
